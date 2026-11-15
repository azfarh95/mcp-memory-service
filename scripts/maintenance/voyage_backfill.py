#!/usr/bin/env python3
"""Backfill / catch-up the Voyage parallel index from the local sqlite-vec DB.

Creates the voyage index file if missing (a schema copy of the local DB with a
FLOAT[1024] embedding table), syncs memory rows that exist only in the local
DB, then embeds every voyage row that lacks a vector — resumable: progress is
committed per batch, so an interrupted run (or an outage exhausted-retries
run) simply continues where it stopped.

Usage:
    python scripts/maintenance/voyage_backfill.py [--dry-run] [--limit N]
        [--batch-size N] [--local PATH] [--voyage PATH]

Model/dimension come from the same env vars as the service
(MCP_VOYAGE_MODEL, MCP_VOYAGE_OUTPUT_DIM, MCP_VOYAGE_API_KEY / keyring).
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mcp_memory_service.embeddings.voyage_embeddings import VoyageEmbeddingModel  # noqa: E402

# Hashes copied per INSERT batch: keeps bind variables far under SQLite's limit.
_COPY_CHUNK = 400


def default_db_paths() -> tuple:
    from mcp_memory_service.config import SQLITE_VEC_PATH, VOYAGE_DATABASE_PATH

    return SQLITE_VEC_PATH, VOYAGE_DATABASE_PATH


def ensure_voyage_schema(local_conn: sqlite3.Connection, voyage_conn: sqlite3.Connection,
                         dimension: int, dry_run: bool) -> None:
    """Create the voyage memories/embeddings schema if absent, stamped to dimension."""
    voyage_conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
            content_embedding FLOAT[{dimension}] distance_metric=cosine
        )"""
    )
    # Copy the local memories table DDL verbatim so both files hold identical rows.
    cols = [(r[1], r[2]) for r in local_conn.execute("PRAGMA table_info(memories)")]
    if not cols:
        raise SystemExit("Local DB has no memories table")
    local_ddl = local_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()[0]
    has_memories = voyage_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not has_memories:
        voyage_conn.execute(local_ddl)  # CREATE TABLE (no IF NOT EXISTS upstream) — run once on a fresh file
    # Stamp the backend so startup can verify the file matches the active model.
    voyage_conn.execute(
        "CREATE TABLE IF NOT EXISTS backend_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    voyage_conn.execute(
        "INSERT OR REPLACE INTO backend_metadata VALUES ('embedding_model', ?)",
        (f"voyage:{os.getenv('MCP_VOYAGE_MODEL', 'voyage-4')}",),
    )
    voyage_conn.execute(
        "INSERT OR REPLACE INTO backend_metadata VALUES ('embedding_dimension', ?)",
        (str(dimension),),
    )
    if not dry_run:
        voyage_conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", help="Local (source) sqlite-vec DB path")
    parser.add_argument("--voyage", help="Voyage (target) sqlite-vec DB path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Max rows to embed this run")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    local_path, voyage_path = default_db_paths()
    local_path = args.local or local_path
    voyage_path = args.voyage or voyage_path
    if not local_path or not os.path.exists(local_path):
        raise SystemExit(f"Local DB not found: {local_path}")

    # Open the voyage file read-only under --dry-run: sqlite3.connect() creates the
    # file, so a "dry" run would otherwise leave a DB behind (review round 2).
    if args.dry_run:
        print(f"Local DB: {local_path}")
        if not voyage_path or not os.path.exists(voyage_path):
            print(f"Voyage DB: {voyage_path} (would be created)")
            return
        print(f"Voyage DB: {voyage_path} (exists)")
        try:
            # as_uri() percent-encodes Windows backslashes/#/? — a raw f-string URI
            # breaks on those (review round 3).
            ro = sqlite3.connect(Path(voyage_path).resolve().as_uri() + "?mode=ro", uri=True)
            todo = ro.execute(
                """SELECT COUNT(*) FROM memories m
                   WHERE m.deleted_at IS NULL
                     AND NOT EXISTS (SELECT 1 FROM memory_embeddings me WHERE me.rowid = m.rowid)"""
            ).fetchone()[0]
            print(f"Voyage rows missing embeddings: {todo}")
            ro.close()
        except sqlite3.Error as exc:
            print(f"Voyage DB not yet initialised: {exc}")
        return

    local_conn = sqlite3.connect(local_path)
    voyage_conn = sqlite3.connect(voyage_path or (local_path + ".voyage.db"))
    # vec0 virtual tables need the sqlite-vec extension loaded on this connection.
    import sqlite_vec

    voyage_conn.enable_load_extension(True)
    sqlite_vec.load(voyage_conn)
    voyage_conn.enable_load_extension(False)

    model = VoyageEmbeddingModel(max_retries=8, retry_base_delay=5.0)
    ensure_voyage_schema(local_conn, voyage_conn, model.embedding_dimension, False)

    # 1. Sync memory rows that exist only in the local DB (outage-time writes).
    local_hashes = {r[0] for r in local_conn.execute("SELECT content_hash FROM memories WHERE deleted_at IS NULL")}
    voyage_hashes = {r[0] for r in voyage_conn.execute("SELECT content_hash FROM memories WHERE deleted_at IS NULL")}
    missing = local_hashes - voyage_hashes
    # Bulk-copy rows missing from the voyage index. deleted_at is copied too, so a
    # memory soft-deleted while the primary was down does not resurrect there.
    live_local_hashes = {
        r[0] for r in local_conn.execute("SELECT content_hash FROM memories WHERE deleted_at IS NULL")
    }
    stale = [h for h in voyage_hashes if h not in live_local_hashes]
    if stale:
        print(f"Rows deleted locally but live in the voyage index (to soft-delete): {len(stale)}")
        now = time.time()
        voyage_conn.executemany(
            "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
            [(now, h) for h in stale],
        )
        voyage_conn.commit()
    if missing:
        # Chunk the copy: one bind variable per hash blows SQLite's variable limit
        # (999 by default) on the first real backfill (review round 2).
        local_cols = [r[1] for r in local_conn.execute("PRAGMA table_info(memories)")]
        voyage_cols = {r[1] for r in voyage_conn.execute("PRAGMA table_info(memories)")}
        shared = [c for c in local_cols if c in voyage_cols]
        col_list = ",".join(shared)
        # A row soft-deleted in the voyage index but live locally must come back:
        # INSERT OR IGNORE would leave it dead forever and never re-embed it.
        updates = ",".join(f"{c}=excluded.{c}" for c in shared if c not in ("id", "content_hash"))
        stmt = (
            f"INSERT INTO memories ({col_list}) VALUES ({','.join('?' * len(shared))}) "
            f"ON CONFLICT(content_hash) DO UPDATE SET {updates}"
        )
        missing_list = list(missing)
        for start in range(0, len(missing_list), _COPY_CHUNK):
            chunk = missing_list[start : start + _COPY_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = local_conn.execute(
                f"SELECT * FROM memories WHERE content_hash IN ({placeholders})", tuple(chunk)
            ).fetchall()
            voyage_conn.executemany(
                stmt,
                [[dict(zip(local_cols, row))[c] for c in shared] for row in rows],
            )
        voyage_conn.commit()

    # 2. Embed voyage rows lacking vectors, resumably.
    #    Two-phase on purpose: a live cursor scanning `memories` while the loop
    #    INSERTs into `memory_embeddings` on the same connection would see its own
    #    writes shift the NOT EXISTS, and fetching every content string up front
    #    spikes memory on a 41K-row corpus. So: ids first, content per batch.
    todo_ids = [
        r[0]
        for r in voyage_conn.execute(
            """SELECT m.rowid FROM memories m
               WHERE m.deleted_at IS NULL
                 AND NOT EXISTS (SELECT 1 FROM memory_embeddings me WHERE me.rowid = m.rowid)"""
        ).fetchall()
    ]
    todo = len(todo_ids)
    print(f"Voyage rows missing embeddings: {todo}")
    if not todo:
        return

    batch, done, failures = [], 0, 0
    t0 = time.time()
    for start in range(0, todo, args.batch_size):
        if args.limit and done >= args.limit:
            break
        chunk = todo_ids[start : start + args.batch_size]
        placeholders = ",".join("?" * len(chunk))
        batch = voyage_conn.execute(
            f"SELECT rowid, content FROM memories WHERE rowid IN ({placeholders}) "
            "AND deleted_at IS NULL",
            tuple(chunk),
        ).fetchall()
        if not embed_and_store(voyage_conn, model, batch):
            failures += len(batch)
            print("  stopping after the first failed batch — rerun to resume")
            break
        done += len(batch)
        rate = done / max(time.time() - t0, 0.001)
        print(f"  {done}/{todo} embedded ({rate:.0f} rows/s, ETA {(todo - done) / max(rate, 0.01) / 60:.1f} min)")

    print(f"Done: {done} embedded, {failures} failed (resumable — rerun to continue)")


def embed_and_store(conn: sqlite3.Connection, model: VoyageEmbeddingModel, batch) -> bool:
    """Embed one batch of (rowid, content) and write vectors; commit on success."""
    try:
        vectors = model.encode([c for _, c in batch], convert_to_numpy=True, input_type="document")
        conn.executemany(
            "INSERT OR REPLACE INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
            [(rowid, vectors[i].tobytes()) for i, (rowid, _) in enumerate(batch)],
        )
        conn.commit()
        return True
    except Exception as exc:  # adapter RuntimeError, or a sqlite write error
        print(f"  batch failed ({exc}); committing progress and stopping — rerun to resume")
        try:
            conn.commit()
        except sqlite3.Error:
            pass
        return False


if __name__ == "__main__":
    main()
