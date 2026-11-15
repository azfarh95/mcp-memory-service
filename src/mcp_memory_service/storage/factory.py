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

"""
Shared storage backend factory for the MCP Memory Service.

This module provides a single, shared factory function for creating storage backends,
eliminating code duplication between the MCP server and web interface initialization.
"""

import logging
from typing import Type

from ..compat import _sanitize_log_value
from .base import MemoryStorage

logger = logging.getLogger(__name__)


def _fallback_to_sqlite_vec() -> Type[MemoryStorage]:
    """
    Helper function to fallback to SQLite-vec storage when other backends fail to import.

    Returns:
        SqliteVecMemoryStorage class
    """
    logger.warning("Falling back to SQLite-vec storage")
    from .sqlite_vec import SqliteVecMemoryStorage
    return SqliteVecMemoryStorage


def get_storage_backend_class() -> Type[MemoryStorage]:
    """
    Get storage backend class based on configuration.

    Returns:
        Storage backend class
    """
    from ..config import STORAGE_BACKEND

    backend = STORAGE_BACKEND.lower()

    if backend == "sqlite-vec" or backend == "sqlite_vec":
        from .sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage
    elif backend == "cloudflare":
        try:
            from .cloudflare import CloudflareStorage
            return CloudflareStorage
        except ImportError as e:
            logger.error("Failed to import Cloudflare storage: %s", _sanitize_log_value(e))
            raise
    elif backend == "hybrid":
        try:
            from .hybrid import HybridMemoryStorage
            return HybridMemoryStorage
        except ImportError as e:
            logger.error("Failed to import Hybrid storage: %s", _sanitize_log_value(e))
            return _fallback_to_sqlite_vec()
    elif backend == "milvus":
        try:
            from .milvus import MilvusMemoryStorage
            return MilvusMemoryStorage
        except ImportError as e:
            logger.error("Failed to import Milvus storage: %s", _sanitize_log_value(e))
            raise
    else:
        logger.warning("Unknown storage backend '%s', defaulting to SQLite-vec", _sanitize_log_value(backend))
        from .sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage


async def create_storage_instance(sqlite_path: str, server_type: str = None) -> MemoryStorage:
    """
    Create and initialize storage backend instance based on configuration.

    Args:
        sqlite_path: Path to SQLite database file (used for SQLite-vec and Hybrid backends)
        server_type: Optional server type identifier ("mcp" or "http") to control hybrid sync ownership

    Returns:
        Initialized storage backend instance
    """
    from ..config import (
        STORAGE_BACKEND, EMBEDDING_MODEL_NAME,
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY,
        HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE, HYBRID_SYNC_OWNER,
        MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION_NAME
    )

    logger.info("Creating storage backend instance (sqlite_path: %s, server_type: %s)...", _sanitize_log_value(sqlite_path), _sanitize_log_value(server_type))

    # Check if we should override hybrid backend based on sync ownership (v8.27.0+)
    effective_backend = STORAGE_BACKEND
    if STORAGE_BACKEND == 'hybrid' and server_type and HYBRID_SYNC_OWNER != 'both':
        if HYBRID_SYNC_OWNER != server_type:
            logger.info(
                "Sync ownership configured for '%s' but this is '%s' server. "
                "Using SQLite-vec storage instead of Hybrid to avoid duplicate sync queues.",
                _sanitize_log_value(HYBRID_SYNC_OWNER), _sanitize_log_value(server_type)
            )
            effective_backend = 'sqlite_vec'

    # Get storage class based on effective configuration
    if effective_backend == 'sqlite_vec':
        # Intentional switch to SQLite-vec (not a fallback/error case)
        from .sqlite_vec import SqliteVecMemoryStorage
        StorageClass = SqliteVecMemoryStorage
    else:
        # Use configured backend (hybrid or cloudflare)
        StorageClass = get_storage_backend_class()

    # Create storage instance based on backend type
    if StorageClass.__name__ == "SqliteVecMemoryStorage":
        from ..config import EMBED_BACKEND, PARALLEL_INDEX, VOYAGE_DATABASE_PATH
        from .mixins.embeddings import EmbeddingsMixin

        effective_embed, reason = EmbeddingsMixin.resolve_embed_backend(sqlite_path)

        if (EMBED_BACKEND == "voyage" and effective_embed == "voyage" and reason == "key"
                and PARALLEL_INDEX and VOYAGE_DATABASE_PATH):
            # Parallel index: Voyage primary in its own file (1024-dim) with the classic
            # local index (384-dim) kept live as the fallback and complete source of truth.
            from .parallel import ParallelSqliteVecStorage

            primary = StorageClass(
                db_path=VOYAGE_DATABASE_PATH,
                embedding_model=EMBEDDING_MODEL_NAME,
                embed_backend="voyage",
            )
            secondary = StorageClass(
                db_path=sqlite_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                embed_backend="local",
            )
            storage = ParallelSqliteVecStorage(primary=primary, secondary=secondary)
            logger.info(
                "Initialized parallel SQLite-vec storage: voyage at %s (primary) + local at %s (fallback)",
                _sanitize_log_value(VOYAGE_DATABASE_PATH), _sanitize_log_value(sqlite_path))
        elif EMBED_BACKEND == "voyage" and reason == "key" and not (PARALLEL_INDEX and VOYAGE_DATABASE_PATH):
            # A key exists and voyage was requested, but the parallel index is off (or
            # there is no voyage path). Honour voyage on the primary path rather than
            # silently dropping to local and blaming a missing key (review, 2026-11-16).
            storage = StorageClass(
                db_path=sqlite_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                embed_backend="voyage",
            )
            logger.warning(
                "Voyage backend active WITHOUT the local fallback index (parallel index "
                "disabled or no voyage path resolvable). Set MCP_MEMORY_PARALLEL_INDEX=1 for "
                "the fallback, or accept that a Voyage outage loses semantic search.")
        elif EMBED_BACKEND == "voyage" and reason == "explicit-no-key":
            # Operator named voyage but there is no key: honour it so the misconfiguration
            # is visible, and point at the voyage file (the local one is 384-dim).
            storage = StorageClass(
                db_path=VOYAGE_DATABASE_PATH or sqlite_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                embed_backend="voyage",
            )
            logger.warning(
                "MCP_MEMORY_EMBED_BACKEND=voyage was explicitly set but no Voyage API key is "
                "configured; storage creation will fail with an actionable error.")
        else:
            # Ambient default with no key: run local so semantic search keeps working.
            storage = StorageClass(
                db_path=sqlite_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                embed_backend="local" if EMBED_BACKEND == "voyage" else None,
            )
            if EMBED_BACKEND == "voyage":
                logger.warning(
                    "Voyage is the default embedding backend but no API key is configured - "
                    "using local embeddings. Set MCP_VOYAGE_API_KEY or the keyring entry, or set "
                    "MCP_MEMORY_EMBED_BACKEND=local to silence this warning.")
            logger.info("Initialized SQLite-vec storage at %s", _sanitize_log_value(sqlite_path))

    elif StorageClass.__name__ == "CloudflareStorage":
        storage = StorageClass(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY
        )
        logger.info("Initialized Cloudflare storage with vectorize index: %s", _sanitize_log_value(CLOUDFLARE_VECTORIZE_INDEX))

    elif StorageClass.__name__ == "MilvusMemoryStorage":
        storage = StorageClass(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            collection_name=MILVUS_COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        logger.info(
            "Initialized Milvus storage (uri=%s, collection=%s)",
            _sanitize_log_value(MILVUS_URI), _sanitize_log_value(MILVUS_COLLECTION_NAME)
        )

    elif StorageClass.__name__ == "HybridMemoryStorage":
        # Prepare Cloudflare configuration dict
        cloudflare_config = None
        if all([CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID]):
            cloudflare_config = {
                'api_token': CLOUDFLARE_API_TOKEN,
                'account_id': CLOUDFLARE_ACCOUNT_ID,
                'vectorize_index': CLOUDFLARE_VECTORIZE_INDEX,
                'd1_database_id': CLOUDFLARE_D1_DATABASE_ID,
                'r2_bucket': CLOUDFLARE_R2_BUCKET,
                'embedding_model': CLOUDFLARE_EMBEDDING_MODEL,
                'large_content_threshold': CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                'max_retries': CLOUDFLARE_MAX_RETRIES,
                'base_delay': CLOUDFLARE_BASE_DELAY
            }

        storage = StorageClass(
            sqlite_db_path=sqlite_path,
            embedding_model=EMBEDDING_MODEL_NAME,
            cloudflare_config=cloudflare_config,
            sync_interval=HYBRID_SYNC_INTERVAL,
            batch_size=HYBRID_BATCH_SIZE
        )
        logger.info("Initialized hybrid storage with SQLite at %s", _sanitize_log_value(sqlite_path))

    else:
        # Unknown storage backend - this should not happen as get_storage_backend_class
        # already handles unknown backends by falling back to SQLite-vec
        raise ValueError(f"Unsupported storage backend class: {StorageClass.__name__}")

    # Initialize storage backend
    await storage.initialize()
    logger.info("Storage backend %s initialized successfully", _sanitize_log_value(StorageClass.__name__))

    return storage