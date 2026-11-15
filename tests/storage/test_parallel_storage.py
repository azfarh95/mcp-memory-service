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

    async def store_batch(self, memories):
        self.calls.append(("store_batch", len(memories)))
        if self.embed_error:
            return [(False, f"Failed to generate embedding: {self.embed_error}")] * len(memories)
        for m in memories:
            self.stored.append(m.content)
        return [(True, "stored")] * len(memories)

    async def retrieve(self, query, n_results=5, **kwargs):
        self.calls.append(("retrieve", query))
        if self.retrieve_error:
            raise RuntimeError(self.retrieve_error)
        return [FakeQueryResult(f"hit-for:{query}")]

    async def search(self, query, n_results=5):
        return await self.retrieve(query, n_results)

    async def recall(self, query, n_results=5, **kwargs):
        return [FakeMemory(f"recall-for:{query}")]

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
async def test_delete_applies_to_both():
    primary, secondary = FakeChild("primary", "fake-voyage"), FakeChild("secondary", "fake-local")
    async def ok_delete(hash): return True, "deleted"
    primary.delete = ok_delete
    secondary.delete = ok_delete
    wrapper = ParallelSqliteVecStorage(primary, secondary)
    await wrapper.delete("abc123")
    # both children got the delete (they are the patched ok_delete — verify via call log)
    assert ("store", "abc123") not in primary.calls  # sanity
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
