"""Tests for the parallel Voyage/local index wrapper.

Uses in-memory fakes for the two child storages: the wrapper's contract is
orchestration (dual-write, fallback, flagging), which the fakes pin down
without needing sqlite-vec or a Voyage key.
"""

import pytest

from mcp_memory_service.storage.parallel import ParallelSqliteVecStorage


class FakeMemory:
    def __init__(self, content):
        self.content = content
        self.content_hash = f"hash-{content}"
        self.metadata = {}


class FakeQueryResult:
    def __init__(self, content):
        self.memory = FakeMemory(content)


class FakeChild:
    """Records calls; configurable failures per method."""

    def __init__(self, name, backend_type):
        self.name = name
        self.backend_type = backend_type
        self.calls = []
        self.stored = []
        self.returns_empty = False
        self.embedding_last_error = None
        self.embedding_failure_count = 0
        self.ready = True   # mirrors _primary_ready()'s corpus-coverage verdict
        self.embed_error = None  # when set, store returns the embed-failure string
        self.retrieve_error = None
        self.db_path = f"/tmp/{name}.db"
        self.embedding_model_name = name
        self.embedding_dimension = 1024 if name == "primary" else 384

    def get_backend_type(self):
        return self.backend_type if hasattr(self, "backend_type") else f"fake-{self.name}"

    async def initialize(self):
        self.calls.append("initialize")

    def close(self):
        self.calls.append("close")

    async def store(self, memory, skip_semantic_dedup=False, store=None):
        self.calls.append(("store", memory.content))
        if self.embed_error:
            return False, f"Failed to generate embedding: {self.embed_error}"
        self.stored.append(memory.content)
        return True, "stored"

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.calls.append(("update_memory_metadata", content_hash, dict(updates)))
        return True, "updated"

    async def store_batch(self, memories, store='default'):
        self.calls.append(("store_batch", store))
        if self.embed_error:
            return [(False, f"Batch embedding generation failed: {self.embed_error}")] * len(memories)
        for m in memories:
            self.stored.append(m.content)
        return [(True, "stored")] * len(memories)

    async def delete(self, content_hash):
        self.calls.append(("delete", content_hash))
        return True, "deleted"

    async def retrieve(self, query, n_results=5, **kwargs):
        self.calls.append(("retrieve", query))
        if self.retrieve_error:
            self.embedding_failure_count += 1
            self.embedding_last_error = self.retrieve_error
            raise RuntimeError(self.retrieve_error)
        if self.returns_empty:
            # Mirrors the real mixin: an embed failure is swallowed into [] and
            # recorded as a monotonic counter increment, not a raised exception.
            if self.embedding_last_error:
                self.embedding_failure_count += 1
            return []
        return [FakeQueryResult(f"hit-for:{query}")]

    async def search(self, query, n_results=5):
        return await self.retrieve(query, n_results)

    async def recall(self, query=None, n_results=5, **kwargs):
        self.calls.append(("recall", query))
        if self.returns_empty:
            return []
        return [FakeQueryResult(f"recall-for:{query}")]

    async def get_stats(self):
        return {"total_memories": len(self.stored), "storage_backend": f"fake-{self.name}"}

    async def health_probe(self):
        return True


def make_memory(content):
    # The wrapper only touches .content/.metadata — a duck-typed memory avoids
    # depending on the real Memory dataclass's required content_hash field.
    return FakeMemory(content)


@pytest.mark.asyncio
async def test_store_writes_both_backends():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    ok, msg = await wrapper.store(make_memory("hello"))
    assert ok
    assert "hello" in secondary.stored
    assert "hello" in primary.stored


@pytest.mark.asyncio
async def test_store_survives_primary_failure_with_pending_marker():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.embed_error = "voyage 503"
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    memory = make_memory("survivor")
    ok, msg = await wrapper.store(memory)
    assert ok  # local write succeeded — the write is not lost
    assert "survivor" in secondary.stored
    assert "survivor" not in primary.stored
    assert memory.metadata.get("embed_pending") == "voyage"
    assert wrapper.primary_last_error and "voyage 503" in wrapper.primary_last_error


@pytest.mark.asyncio
async def test_primary_cooldown_skips_after_failure():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.embed_error = "voyage 503"
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.store(make_memory("first"))
    primary_calls_before = len([c for c in primary.calls if c[0] == "store"])
    await wrapper.store(make_memory("second"))
    assert len([c for c in primary.calls if c[0] == "store"]) == primary_calls_before  # cooldown: skipped


@pytest.mark.asyncio
async def test_retrieve_prefers_primary_and_flags_fallback():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    results = await wrapper.retrieve("query text")
    assert results[0].memory.content == "hit-for:query text"
    assert results[0].memory.metadata == {}  # primary served: no degradation flag
    assert ("retrieve", "query text") in primary.calls


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_secondary_with_flag():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.retrieve_error = "voyage unreachable"
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    results = await wrapper.retrieve("query text")
    assert results[0].memory.content == "hit-for:query text"
    assert results[0].memory.metadata.get("_degraded_backend") == "local"
    assert wrapper.primary_last_error and "voyage unreachable" in wrapper.primary_last_error


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_primary_returns_empty_after_embed_error():
    """A Voyage outage makes the mixin return [] (it catches and logs); an empty
    list alone is also 'no matches', so the wrapper must use the child's
    embedding_last_error to tell them apart (review finding 2026-11-15)."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.returns_empty = True
    primary.embedding_last_error = "voyage 503 after retries"
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    results = await wrapper.retrieve("query text")
    assert results[0].memory.content == "hit-for:query text"
    assert results[0].memory.metadata.get("_degraded_backend") == "local"


@pytest.mark.asyncio
async def test_retrieve_empty_without_error_stays_empty():
    """Genuine no-match must not be reported as a degraded backend."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.returns_empty = True
    primary.embedding_last_error = None
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    assert await wrapper.retrieve("query text") == []
    assert "retrieve" not in [c[0] for c in secondary.calls if c]


@pytest.mark.asyncio
async def test_store_batch_detects_batch_worded_embed_failure():
    """store_batch words the failure differently ('Batch embedding generation
    failed'); primary-down detection must not depend on the single-store wording."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.embed_error = "voyage unreachable"
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.store_batch([make_memory("one"), make_memory("two")])
    assert wrapper.primary_last_error, "batch embed failure was not detected"
    assert wrapper._primary_available() is False


@pytest.mark.asyncio
async def test_store_batch_forwards_store_partition():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.store_batch([make_memory("p")], store="work")
    assert ("store_batch", "work") in secondary.calls


@pytest.mark.asyncio
async def test_recall_accepts_time_only_query():
    """Base recall allows query=None (time-only recall); the wrapper must match."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    results = await wrapper.recall(None, 5, start_timestamp=1.0)
    assert results


@pytest.mark.asyncio
async def test_delete_applies_to_both():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.delete("abc123")
    assert ("delete", "abc123") in primary.calls
    assert ("delete", "abc123") in secondary.calls


@pytest.mark.asyncio
async def test_stale_primary_index_falls_back_even_with_no_error():
    """A fresh/partially-backfilled Voyage index answers [] for memories that
    exist locally. No error is raised, so readiness (corpus coverage) is the only
    thing that stops the silent shortfall (review round 2)."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.ready = False
    primary.returns_empty = True
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    results = await wrapper.retrieve("query text")
    assert results[0].memory.content == "hit-for:query text"
    assert results[0].memory.metadata.get("_degraded_backend") == "local"


@pytest.mark.asyncio
async def test_unready_primary_is_not_queried_at_all():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    primary.ready = False
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.retrieve("query text")
    assert ("retrieve", "query text") not in primary.calls


@pytest.mark.asyncio
async def test_time_only_recall_goes_to_secondary():
    """query=None is a non-embedding read: the complete (local) index serves it."""
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.recall(None, 5, start_timestamp=1.0)
    assert ("recall", None) in secondary.calls
    assert ("recall", None) not in primary.calls


class FakeConn:
    """Minimal .execute() stand-in for the readiness SQL (no sqlite needed)."""

    def __init__(self, vector_count, row_count):
        self.vector_count = vector_count
        self.row_count = row_count

    def execute(self, sql, *args):
        class _Cur:
            def __init__(self, value):
                self.value = value

            def fetchone(self):
                return (self.value,)

        if "memory_embeddings" in sql:
            return _Cur(self.vector_count)
        return _Cur(self.row_count)


@pytest.mark.asyncio
async def test_primary_ready_uses_vector_count_not_row_count():
    """Rows copied but not embedded must read as NOT ready (round-3 blocker A)."""
    primary = FakeChild("primary", "fake-voyage")
    secondary = FakeChild("secondary", "fake-local")
    del primary.ready   # exercise the real SQL path instead of the test hook
    primary.conn = FakeConn(vector_count=0, row_count=1000)   # rows yes, vectors no
    secondary.conn = FakeConn(vector_count=0, row_count=1000)
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    assert wrapper._primary_ready() is False
    assert "voyage index behind" in (wrapper.primary_last_error or "")


@pytest.mark.asyncio
async def test_primary_ready_when_vectors_cover_the_corpus():
    primary = FakeChild("primary", "fake-voyage")
    secondary = FakeChild("secondary", "fake-local")
    del primary.ready
    primary.conn = FakeConn(vector_count=1000, row_count=1000)
    secondary.conn = FakeConn(vector_count=0, row_count=1000)
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    assert wrapper._primary_ready() is True
    assert wrapper.primary_last_error is None


@pytest.mark.asyncio
async def test_stats_reports_parallel_info():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    stats = await wrapper.get_stats()
    assert stats["storage_backend"] == "sqlite_vec+voyage_parallel"
    assert stats["parallel_index"]["primary_backend"] == "fake-voyage"
    assert stats["parallel_index"]["primary_available"] is True
    assert "voyage-parallel" in wrapper.embedding_model_name


@pytest.mark.asyncio
async def test_delegated_read_goes_to_secondary():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    async def tags(): return ["a", "b"]
    secondary.get_all_tags = tags
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    assert await wrapper.get_all_tags() == ["a", "b"]


@pytest.mark.asyncio
async def test_initialize_initializes_both():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.initialize()
    assert "initialize" in primary.calls
    assert "initialize" in secondary.calls
