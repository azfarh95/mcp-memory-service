"""Regression tests for the four defects the v12.10.0 live cutover exposed.

Each test fails without its fix:

1. ``get_stats()`` called ``child.get_backend_type()``, which v12 removed -> the health
   endpoint reported ``status: error`` and every statistic read 0.
2. The backfill wrote ``store = NULL`` into ``memory_embeddings``, where ``store`` is a
   vec0 PARTITION KEY and ``retrieve()`` filters ``store = ?`` inside the vec0 query, so
   a fully-populated index returned zero hits.
3. The wrapper's ``store()``/``store_batch()`` lacked v12's ``recreate`` kwarg, which
   ``memory_service`` passes unconditionally -> every write raised TypeError.
4. ``retrieve()`` passed ``input_type=`` unconditionally to ``_generate_embedding``,
   breaking backends/tests that monkeypatch the older one-argument signature.
"""

import sqlite3

import pytest

from mcp_memory_service.storage.parallel import ParallelSqliteVecStorage


class Child:
    """Minimal child double. Deliberately has NO get_backend_type (v12 shape)."""

    def __init__(self, name):
        self.name = name
        self.calls = []
        self.conn = None
        self.embedding_model_name = name
        self.embedding_dimension = 1024 if name == "primary" else 384
        self.embedding_failure_count = 0
        self.embedding_last_error = None

    async def initialize(self):
        pass

    async def store(self, memory, skip_semantic_dedup=False, store="default", recreate=None):
        self.calls.append(("store", store, recreate))
        return True, "stored"

    async def store_batch(self, memories, store="default", recreate=None):
        self.calls.append(("store_batch", store, recreate))
        return [(True, "stored")] * len(memories)

    async def get_stats(self):
        return {"total_memories": 17123, "storage_backend": "sqlite-vec"}

    async def health_probe(self):
        return True


def wrapper():
    return ParallelSqliteVecStorage(primary=Child("primary"), secondary=Child("secondary"))


@pytest.mark.asyncio
async def test_get_stats_survives_a_child_without_get_backend_type():
    """Defect 1: v12 removed get_backend_type(); stats must still report."""
    w = wrapper()
    stats = await w.get_stats()
    assert stats["total_memories"] == 17123
    assert stats["parallel_index"]["primary_backend"]  # derived, not raised
    assert "error" not in stats


@pytest.mark.asyncio
async def test_store_forwards_the_recreate_kwarg():
    """Defect 3: memory_service passes recreate= unconditionally in v12."""
    w = wrapper()
    ok, _ = await w.store(object(), recreate=False)
    assert ok
    assert ("store", "default", False) in w.secondary.calls
    assert ("store", "default", False) in w.primary.calls


@pytest.mark.asyncio
async def test_store_batch_forwards_the_recreate_kwarg():
    w = wrapper()
    await w.store_batch([object()], store="work", recreate=True)
    assert ("store_batch", "work", True) in w.secondary.calls


def test_embed_text_tolerates_a_legacy_one_argument_override():
    """Defect 4: a monkeypatched _generate_embedding(text) must not raise TypeError."""
    from mcp_memory_service.storage.mixins.embeddings import call_generate_embedding

    class Legacy:
        def _generate_embedding(self, text):  # old signature, no input_type
            return [0.0, 1.0]

    assert call_generate_embedding(Legacy(), "hello", input_type="query") == [0.0, 1.0]

    class Modern:
        def _generate_embedding(self, text, input_type="document"):
            return [float(len(text)), float(input_type == "query")]

    assert call_generate_embedding(Modern(), "hello", input_type="query") == [5.0, 1.0]


def test_backfill_writes_the_vec0_partition_key(tmp_path):
    """Defect 2: a vector written with store NULL is invisible to 'store = ?' queries."""
    sqlite_vec = pytest.importorskip("sqlite_vec")
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "voyage_backfill_under_test", repo / "scripts" / "maintenance" / "voyage_backfill.py"
    )
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)

    db = tmp_path / "voyage.db"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE memories (rowid INTEGER PRIMARY KEY, content TEXT, store TEXT DEFAULT 'default',"
        " embedding_pending INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE memory_embeddings USING vec0("
        "content_embedding FLOAT[4] distance_metric=cosine, store TEXT partition key)"
    )
    conn.execute("INSERT INTO memories (rowid, content, store) VALUES (1, 'x', 'work')")
    conn.commit()

    class Model:
        class _NP:
            @staticmethod
            def zeros(shape, dtype=None):
                import numpy as np

                return np.zeros(shape, dtype=dtype)

        np = _NP

        def encode(self, texts, convert_to_numpy=True, input_type="document"):
            import numpy as np

            return np.zeros((len(texts), 4), dtype=np.float32)

    assert vb.embed_and_store(conn, Model(), [(1, "x")], str(db)) is True
    stores = [r[0] for r in conn.execute("SELECT store FROM memory_embeddings")]
    assert stores == ["work"], f"partition key not copied: {stores}"
