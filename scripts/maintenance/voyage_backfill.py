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
import asyncio
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mcp_memory_service.embeddings.voyage_embeddings import VoyageEmbeddingModel  # noqa: E402


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

    local_path, voyage_path = default_db_paths()
    local_path = args.local or local_path
    voyage_path = args.voyage or voyage_path
    if not local_path or not os.path.exists(local_path):
        raise SystemExit(f"Local DB not found: {local_path}")

    local_conn = sqlite3.connect(local_path)
    voyage_conn = sqlite3.connect(voyage_path or (local_path + ".voyage.db"))
    # vec0 virtual tables need the sqlite-vec extension loaded on this connection.
    import sqlite_vec

    voyage_conn.enable_load_extension(True)
    sqlite_vec.load(voyage_conn)
    voyage_conn.enable_load_extension(False)

    model = VoyageEmbeddingModel(max_retries=8, retry_base_delay=5.0)
    ensure_voyage_schema(local_conn, voyage_conn, model.embedding_dimension, args.dry_run)

    # 1. Sync memory rows that exist only in the local DB (outage-time writes).
    local_hashes = {r[0] for r in local_conn.execute("SELECT content_hash FROM memories WHERE deleted_at IS NULL")}
    voyage_hashes = {r[0] for r in voyage_conn.execute("SELECT content_hash FROM memories WHERE deleted_at IS NULL")}
    missing = local_hashes - voyage_hashes
    print(f"Rows only in local DB (to copy): {len(missing)}")
    if missing and not args.dry_run:
        placeholders = ",".join("?" * len(missing))
        rows = local_conn.execute(
            f"SELECT * FROM memories WHERE content_hash IN ({placeholders})", tuple(missing)
        ).fetchall()
        local_cols = [r[1] for r in local_conn.execute("PRAGMA table_info(memories)")]
        voyage_cols = {r[1] for r in voyage_conn.execute("PRAGMA table_info(memories)")}
        shared = [c for c in local_cols if c in voyage_cols]
        col_list = ",".join(shared)
        voyage_conn.executemany(
            f"INSERT OR IGNORE INTO memories ({col_list}) VALUES ({','.join('?' * len(shared))})",
            [[dict(zip(local_cols, row))[c] for c in shared] for row in rows],
        )
        voyage_conn.commit()

    # 2. Embed voyage rows lacking vectors, resumably.
    todo = voyage_conn.execute(
        """SELECT COUNT(*) FROM memories m
           WHERE m.deleted_at IS NULL
             AND NOT EXISTS (SELECT 1 FROM memory_embeddings me WHERE me.rowid = m.rowid)"""
    ).fetchone()[0]
    print(f"Voyage rows missing embeddings: {todo}")
    if args.dry_run:
        return

    batch, done, failures = [], 0, 0
    cursor = voyage_conn.execute(
        """SELECT m.rowid, m.content FROM memories m
           WHERE m.deleted_at IS NULL
             AND NOT EXISTS (SELECT 1 FROM memory_embeddings me WHERE me.rowid = m.rowid)"""
    )
    t0 = time.time()
    for rowid, content in cursor:
        batch.append((rowid, content))
        if len(batch) >= args.batch_size:
            if not embed_and_store(voyage_conn, model, batch):
                failures += len(batch)
            else:
                done += len(batch)
                rate = done / max(time.time() - t0, 0.001)
                print(f"  {done}/{todo} embedded ({rate:.0f} rows/s, ETA {(todo - done) / max(rate, 0.01) / 60:.1f} min)")
            batch = []
            if args.limit and done >= args.limit:
                break
    if batch and not (args.limit and done >= args.limit):
        if not embed_and_store(voyage_conn, model, batch):
            failures += len(batch)
        else:
            done += len(batch)

    print(f"Done: {done} embedded, {failures} failed (resumable — rerun to continue)")


def embed_and_store(conn: sqlite3.Connection, model: VoyageEmbeddingModel, batch) -> bool:
    """Embed one batch of (rowid, content) and write vectors; commit on success."""
    try:
        vectors = model.encode([c for _, c in batch], convert_to_numpy=True, input_type="document")
    except RuntimeError as exc:
        print(f"  batch failed ({exc}); committing progress and stopping — rerun to resume")
        conn.commit()
        return False
    conn.executemany(
        "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
        [(rowid, vectors[i].tobytes()) for i, (rowid, _) in enumerate(batch)],
    )
    conn.commit()
    return True


if __name__ == "__main__":
    main()
