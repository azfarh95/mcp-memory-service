# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel index storage: Voyage (cloud) primary + local MiniLM fallback.

One sqlite-vec file per embedding backend — MiniLM vectors are 384-dim, Voyage
1024-dim, and sqlite-vec bakes the dimension into the vec0 schema, so the two
vector spaces can never share a table. This wrapper keeps BOTH indexes live:

- **Writes** go to both backends (secondary first — it is the durable, always-
  available source of truth; primary best-effort). If the Voyage write fails,
  the memory still lands in the local index and is marked ``embed_pending``;
  ``scripts/maintenance/voyage_backfill.py`` catches the primary up later.
- **Semantic reads** prefer the primary; on Voyage failure they fall back to
  the secondary index and flag every result (never a silent vector-space swap).
- **Non-embedding reads** (by tag/time/hash) are served from the secondary,
  which always holds the complete memory set.
- **Deletes/updates** apply to both.

Anything not explicitly overridden is delegated to the SECONDARY via
``__getattr__`` (the primary and secondary are both full
``SqliteVecMemoryStorage`` instances).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..models import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)

# After a primary write/read failure, skip the primary for this long: one
# Voyage outage must not add multi-second retry storms to every call. The
# backfill script is the recovery path, not per-request retries.
_PRIMARY_COOLDOWN_SECONDS = 60.0

# The Voyage index must cover this share of the local corpus before its results
# are trusted on their own. Below it, the primary is "behind" (a fresh install or
# an unfinished backfill) and searches are served from local with the degraded
# flag rather than returning a silently short result set.
_PRIMARY_MIN_COVERAGE = 0.95

# Readiness is a COUNT over the corpus; re-probing on every semantic read would add
# tens of ms per query on a 41K-row store. Cache the verdict briefly.
_READY_CACHE_SECONDS = 30.0


class ParallelSqliteVecStorage:
    """Voyage-primary storage wrapper over two SqliteVecMemoryStorage instances.

    Deliberately duck-typed (not a MemoryStorage subclass): the base class gives
    every interface method a default body, which would shadow the __getattr__
    delegation and silently serve empty results instead of delegating. The
    wrappers below implement the interface explicitly; everything else
    delegates to the secondary. Call sites only use type hints, never
    isinstance checks.
    """

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.primary_last_error: Optional[str] = None
        self._primary_down_until: float = 0.0
        self._ready_cached: bool = True
        self._ready_cached_at: float = 0.0

    # ------------------------------------------------------------------ init

    async def initialize(self) -> None:
        await self.secondary.initialize()
        await self.primary.initialize()

    async def close(self) -> None:
        for child in (self.primary, self.secondary):
            closer = getattr(child, "close", None)
            if closer:
                try:
                    result = closer()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.warning("Error closing child storage: %s", exc)

    # ------------------------------------------------------- failure helpers

    def _primary_available(self) -> bool:
        return time.time() >= self._primary_down_until

    def _primary_ready(self) -> bool:
        """Is the Voyage index actually usable for searching?

        An error-free empty result is only trustworthy if the primary holds
        EMBEDDINGS for the same corpus the local index holds. Counting memory rows
        is not enough: the backfill copies rows first and embeds second, so a run
        that stopped after a failed batch leaves a fully-populated ``memories``
        table with no vectors — coverage would read 100% and every search would
        return [] for memories that exist locally (review round 3, 2026-11-15).
        Gate on embedded rows vs. live memories instead.
        """
        # A child may expose its own readiness verdict (the test doubles do).
        ready_attr = getattr(self.primary, "ready", None)
        if isinstance(ready_attr, bool):
            return ready_attr
        # Counting 41K rowids on every semantic read would add tens of ms per
        # query. Re-probe at most once per interval; writes/backfill change the
        # numbers, and a stale "ready" self-corrects on the next window.
        now = time.time()
        if self._ready_cached_at and (now - self._ready_cached_at) < _READY_CACHE_SECONDS:
            return self._ready_cached
        try:
            primary_vectors = self.primary.conn.execute(
                """SELECT COUNT(*) FROM memory_embeddings me
                   JOIN memories m ON m.rowid = me.rowid
                   WHERE m.deleted_at IS NULL"""
            ).fetchone()[0]
            secondary_rows = self.secondary.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
            ).fetchone()[0]
        except Exception as exc:
            logger.debug("Primary readiness check failed (%s); assuming not ready", exc)
            # Cache the negative verdict too, or a broken primary re-runs a failing
            # COUNT on every single semantic read.
            return self._cache_ready(False)
        # An empty local corpus has nothing to serve, so the primary is trivially
        # "ready" and must not be reported as lagging.
        if secondary_rows == 0:
            self.primary_last_error = None
            return self._cache_ready(True)
        if primary_vectors < secondary_rows * _PRIMARY_MIN_COVERAGE:
            self.primary_last_error = (
                f"voyage index behind: {primary_vectors}/{secondary_rows} vectors embedded "
                "(run scripts/maintenance/voyage_backfill.py)"
            )
            return self._cache_ready(False)
        # Coverage recovered: drop a stale lag notice so health/stats stop lying.
        if self.primary_last_error and "voyage index behind" in self.primary_last_error:
            self.primary_last_error = None
        return self._cache_ready(True)

    def _cache_ready(self, value: bool) -> bool:
        self._ready_cached = value
        self._ready_cached_at = time.time()
        return value

    def _mark_primary_down(self, error: str) -> None:
        self.primary_last_error = error
        self._primary_down_until = time.time() + _PRIMARY_COOLDOWN_SECONDS
        logger.warning("Voyage primary unavailable (%s); using local index", error)

    async def _mark_embed_pending(self, memory: Memory) -> None:
        """Flag a memory that never reached the Voyage index.

        The in-memory metadata mutation keeps the live object truthful; the
        update_memory_metadata write persists the flag on the secondary row so
        operators (and the backfill) can find outage-time writes. Best-effort:
        the memory itself is already safely stored locally either way.
        """
        try:
            memory.metadata = {**(memory.metadata or {}), "embed_pending": "voyage"}
        except Exception:
            pass
        try:
            await self.secondary.update_memory_metadata(
                memory.content_hash, {"embed_pending": "voyage"}
            )
        except Exception as exc:
            logger.warning("Could not persist embed_pending flag on %s: %s", memory.content_hash, exc)

    @staticmethod
    def _flag_degraded(results: List[MemoryQueryResult]) -> List[MemoryQueryResult]:
        for result in results or []:
            try:
                if result.memory is not None:
                    result.memory.metadata = {
                        **(result.memory.metadata or {}),
                        "_degraded_backend": "local",
                    }
            except Exception:
                pass
        return results

    def _is_embed_failure(self, message: str) -> bool:
        """Both store paths report embedding failure in their own words: the
        single store says 'Failed to generate embedding', store_batch says
        'Batch embedding generation failed'. Match on the shared substring, in
        one place, so a wording change in either cannot silently disable the
        primary-down detection (review finding, 2026-11-15)."""
        text = (message or "").lower()
        return "embedding" in text and ("fail" in text or "unavailable" in text)

    # --------------------------------------------------------------- writes

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False, store: str = "default") -> Tuple[bool, str]:
        """Write to the local index always; mirror into the Voyage index best-effort."""
        result = await self.secondary.store(memory, skip_semantic_dedup=skip_semantic_dedup, store=store)
        if not result[0]:
            return result

        if self._primary_available():
            try:
                primary_result = await self.primary.store(memory, skip_semantic_dedup=True, store=store)
                if not primary_result[0] and self._is_embed_failure(primary_result[1]):
                    self._mark_primary_down(primary_result[1])
                    await self._mark_embed_pending(memory)
            except Exception as exc:
                self._mark_primary_down(str(exc))
                await self._mark_embed_pending(memory)
        return result

    async def store_batch(self, memories: List[Memory], store: str = 'default') -> List[Tuple[bool, str]]:
        secondary_results = await self.secondary.store_batch(memories, store=store)
        if self._primary_available():
            try:
                primary_results = await self.primary.store_batch(memories, store=store)
                failed = [i for i, (ok, msg) in enumerate(primary_results)
                          if not ok and self._is_embed_failure(msg)]
                if failed:
                    self._mark_primary_down(primary_results[failed[0]][1])
                    # Mark every failure, not just the first, so the backfill can find them.
                    for i in failed:
                        await self._mark_embed_pending(memories[i])
            except Exception as exc:
                self._mark_primary_down(str(exc))
                for memory in memories:
                    await self._mark_embed_pending(memory)
        return secondary_results

    # ------------------------------------------------------ semantic reads

    async def _semantic_from_primary(self, query, n_results, kind: str, **kwargs):
        """Run one semantic read against the primary; None means "use the secondary".

        The child swallows embedder failures and returns [] (it logs), and an empty
        list is also a legitimate no-match — so two facts decide it:
          - a per-call failure signal (`embedding_last_error`), read once, so a
            concurrent success cannot clear it mid-check;
          - corpus readiness, because an empty/stale index answers [] for memories
            that plainly exist locally.
        """
        if not self._primary_available() or not self._primary_ready():
            return None
        # Snapshot BEFORE the call: a concurrent successful embed on the same child
        # clears its message, so comparing messages after the await can hide a
        # failure that happened during it (review round 3). The counter only grows.
        failures_before = getattr(self.primary, "embedding_failure_count", 0)
        try:
            results = await getattr(self.primary, kind)(query, n_results, **kwargs)
        except Exception as exc:
            self._mark_primary_down(str(exc))
            return None
        failures_after = getattr(self.primary, "embedding_failure_count", 0)
        if failures_after > failures_before:
            self._mark_primary_down(
                getattr(self.primary, "embedding_last_error", None) or "embedding failed"
            )
            return None
        return results

    async def retrieve(self, query: str, n_results: int = 5, **kwargs) -> List[MemoryQueryResult]:
        results = await self._semantic_from_primary(query, n_results, "retrieve", **kwargs)
        if results is not None:
            return results
        return self._flag_degraded(await self.secondary.retrieve(query, n_results, **kwargs))

    async def recall(self, query: Optional[str] = None, n_results: int = 5, **kwargs) -> List[MemoryQueryResult]:
        # A time-only recall (query=None) is a NON-embedding read: the secondary is
        # the complete index, and the primary can legitimately be behind it.
        if query is None:
            return await self.secondary.recall(query, n_results, **kwargs)
        results = await self._semantic_from_primary(query, n_results, "recall", **kwargs)
        if results is not None:
            return results
        return self._flag_degraded(await self.secondary.recall(query, n_results, **kwargs))

    async def recall_memory(self, query: str, n_results: int = 5) -> List[Memory]:
        # Non-embedding read (returns Memory, not MemoryQueryResult): delegate to the
        # complete local index. Routing it through recall() returned the wrong type
        # and would break memory-shaped access in callers (review, 2026-11-16).
        return await self.secondary.recall_memory(query, n_results)

    async def search(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        results = await self._semantic_from_primary(query, n_results, "search")
        if results is not None:
            return results
        return self._flag_degraded(await self.secondary.search(query, n_results))

    # --------------------------------------------------------- maintenance

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        secondary_result = await self.secondary.delete(content_hash)
        if self._primary_available():
            try:
                await self.primary.delete(content_hash)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return secondary_result

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        count, msg = await self.secondary.delete_by_tag(tag)
        if self._primary_available():
            try:
                await self.primary.delete_by_tag(tag)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return count, msg

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str, List[str]]:
        count, msg, hashes = await self.secondary.delete_by_tags(tags)
        if self._primary_available():
            try:
                await self.primary.delete_by_tags(tags)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return count, msg, hashes

    async def purge_deleted(self, older_than_days: int = 30) -> int:
        count = await self.secondary.purge_deleted(older_than_days)
        if self._primary_available():
            try:
                await self.primary.purge_deleted(older_than_days)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return count

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        result = await self.secondary.update_memory_metadata(content_hash, updates, preserve_timestamps)
        if self._primary_available():
            try:
                await self.primary.update_memory_metadata(content_hash, updates, preserve_timestamps)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return result

    async def update_memory(self, memory: Memory) -> bool:
        result = await self.secondary.update_memory(memory)
        if self._primary_available():
            try:
                await self.primary.update_memory(memory)
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return result

    # ---------------------------------------------------- status / health

    async def health_probe(self) -> bool:
        return await self.secondary.health_probe()

    def get_backend_type(self) -> str:
        return "sqlite_vec+voyage_parallel"

    async def get_stats(self) -> Dict[str, Any]:
        stats = await self.secondary.get_stats()
        stats["storage_backend"] = self.get_backend_type()
        stats["parallel_index"] = {
            "primary_backend": self.primary.get_backend_type(),
            "primary_model": getattr(self.primary, "embedding_model_name", None),
            "primary_dimension": getattr(self.primary, "embedding_dimension", None),
            "primary_last_error": self.primary_last_error,
            "primary_available": self._primary_available(),
            "secondary_model": getattr(self.secondary, "embedding_model_name", None),
            "secondary_dimension": getattr(self.secondary, "embedding_dimension", None),
        }
        return stats

    @property
    def embedding_model_name(self) -> str:
        primary_model = getattr(self.primary, "embedding_model_name", "voyage")
        secondary_model = getattr(self.secondary, "embedding_model_name", "local")
        return f"voyage-parallel(primary={primary_model}, fallback={secondary_model})"

    # ---------------------------------------------------------- delegation

    @property
    def max_content_length(self) -> Optional[int]:
        return self.secondary.max_content_length

    @property
    def supports_chunking(self) -> bool:
        return self.secondary.supports_chunking

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None):
        """Non-embedding read: served from the secondary (complete index)."""
        return await self.secondary.search_by_tag(tags, time_start)

    async def search_by_tags(self, tags: List[str], **kwargs):
        """Non-embedding read: served from the secondary (complete index)."""
        return await self.secondary.search_by_tags(tags, **kwargs)

    async def get_by_hash(self, content_hash: str):
        return await self.secondary.get_by_hash(content_hash)

    async def get_by_exact_content(self, content: str):
        return await self.secondary.get_by_exact_content(content)

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Maintenance: dedup the secondary (source of truth), then mirror the purge."""
        result = await self.secondary.cleanup_duplicates()
        if self._primary_available():
            try:
                await self.primary.cleanup_duplicates()
            except Exception as exc:
                self._mark_primary_down(str(exc))
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the secondary (the complete index).

        Non-embedding reads (by tag/time/hash, timestamps, counts...) are served
        from the local index because every write lands there first; the Voyage
        index can legitimately lag after an outage until the backfill runs.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.secondary, name)
