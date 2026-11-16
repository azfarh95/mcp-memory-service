"""EmbeddingsMixin: embedding model initialization, generation, and fallback."""

import hashlib
import inspect
import logging
import math
import os
import struct
import sys
import traceback
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ...compat import _sanitize_log_value
from ...utils.system_detection import get_torch_device
from ..shared import _embedding_cache_get, _embedding_cache_put, _embedding_cache_size, _embedding_cache_clear

logger = logging.getLogger(__name__)

# Module-level caches
def call_generate_embedding(owner, text: str, input_type: str = "document"):
    """Call ``owner._generate_embedding`` with input_type when the callable accepts it.

    Module-level, not a mixin method: retrieve-style code is exercised in tests by
    minimal stubs that provide only ``_generate_embedding`` and do not inherit
    EmbeddingsMixin, and a helper they cannot see is a new failure mode (CI, 2026-11-16).

    Tests and third-party backends monkeypatch the older one-argument signature;
    passing the keyword unconditionally raised "unexpected keyword argument
    'input_type'". The signature is inspected rather than a TypeError caught, because
    a TypeError can equally come from inside the embedding call and would then be
    swallowed.
    """
    func = owner._generate_embedding
    try:
        accepts = "input_type" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return func(text, input_type=input_type)
    return func(text)


_MODEL_CACHE = {}
_DIMENSION_CACHE = {}

# Module-level flag: emit hash-fallback warning only once per process
_HASH_FALLBACK_WARNED = False


def clear_model_caches() -> dict:
    """Clear embedding model caches to free memory."""
    import gc  # inline import: only needed by this cache-clearing helper

    global _MODEL_CACHE, _DIMENSION_CACHE

    model_count = len(_MODEL_CACHE)
    embedding_count = _embedding_cache_clear()

    _MODEL_CACHE.clear()
    _DIMENSION_CACHE.clear()

    collected = gc.collect()

    logger.info(
        "Model caches cleared - Models: %s, Embeddings: %s, GC collected: %s objects",
        model_count, embedding_count, collected,
    )

    return {
        "models_cleared": model_count,
        "embeddings_cleared": embedding_count,
        "gc_collected": collected
    }


def get_model_cache_stats() -> dict:
    """Get statistics about the model cache."""
    return {
        "model_count": len(_MODEL_CACHE),
        "model_keys": list(_MODEL_CACHE.keys()),
        "embedding_count": _embedding_cache_size()
    }


import re


class _HashEmbeddingModel:
    """Deterministic, pure-Python embedding fallback."""

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = int(embedding_dimension)

    def encode(self, texts: List[str], convert_to_numpy: bool = False):
        vectors = [self._embed_one(text) for text in texts]
        if convert_to_numpy:
            try:
                import numpy as np
                return np.asarray(vectors, dtype=np.float32)
            except Exception:
                return vectors
        return vectors

    def _embed_one(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.embedding_dimension

        floats: List[float] = []
        counter = 0
        needed = self.embedding_dimension
        text_bytes = text.encode("utf-8", errors="ignore")

        while len(floats) < needed:
            digest = hashlib.sha256(text_bytes + b"\x1f" + struct.pack("<I", counter)).digest()
            counter += 1
            for i in range(0, len(digest) - 3, 4):
                (val,) = struct.unpack("<i", digest[i : i + 4])
                floats.append(val / 2147483648.0)
                if len(floats) >= needed:
                    break

        return floats


class EmbeddingsMixin:
    """Mixin providing embedding model initialization and generation."""

    def _get_existing_db_embedding_dimension(self) -> int | None:
        """Read the embedding dimension from an existing vec0 table, or None if not present."""
        try:
            if not self.conn:
                return None
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
            )
            row = cursor.fetchone()
            if not row:
                return None
            match = re.search(r'FLOAT\[(\d+)\]', row[0])
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.debug("Could not read embedding dimension from sqlite_master: %s", _sanitize_log_value(exc))
        return None

    async def get_db_embedding_dimension(self) -> int | None:
        """Get the embedding dimension from the underlying database, or None if not set/present."""
        if hasattr(self, '_run_in_thread'):
            return await self._run_in_thread(self._get_existing_db_embedding_dimension)
        return self._get_existing_db_embedding_dimension()

    @staticmethod
    def _voyage_key_present() -> bool:
        """True when a Voyage key resolves from env or the OS keyring (no network)."""
        from ...config.embedding import (
            VOYAGE_KEYRING_SERVICE,
            VOYAGE_KEYRING_USERNAME,
        )
        try:
            from ...embeddings.voyage_embeddings import VoyageEmbeddingModel
            return bool(VoyageEmbeddingModel._resolve_api_key(
                VOYAGE_KEYRING_SERVICE, VOYAGE_KEYRING_USERNAME))
        except Exception as exc:
            logger.debug("Could not resolve Voyage key: %s", _sanitize_log_value(exc))
            return False

    @staticmethod
    def resolve_embed_backend(db_path=None) -> tuple:
        """Decide the effective (backend, reason) for ONE construction.

        The single decision point shared by the factory AND this ladder, so a keyless
        install degrades identically whether the store came from create_storage_instance
        or a direct SqliteVecMemoryStorage() call:
          - a key exists                      -> ('voyage', 'key')
          - no key, backend unset (ambient)   -> ('local', 'no-key')
          - no key, MCP_MEMORY_EMBED_BACKEND=voyage explicitly -> ('voyage', 'explicit-no-key')
        """
        requested = os.environ.get('MCP_MEMORY_EMBED_BACKEND', '').strip().lower()
        if requested not in ('', 'voyage', 'local'):
            raise ValueError(
                f"MCP_MEMORY_EMBED_BACKEND must be 'voyage' or 'local', got {requested!r}")
        if requested == 'local':
            return ('local', 'explicit-local')
        if EmbeddingsMixin._voyage_key_present():
            return ('voyage', 'key')
        return ('voyage', 'explicit-no-key') if requested == 'voyage' else ('local', 'no-key')

    async def _initialize_embedding_model(self):
        """Initialize the embedding model (ONNX or SentenceTransformer based on configuration)."""
        global _MODEL_CACHE, _HASH_FALLBACK_WARNED

        is_docker = getattr(self, '_is_docker_environment', lambda: False)()
        if is_docker:
            logger.info("🐳 Docker environment detected - adjusting model loading strategy")

        try:
            # Voyage cloud backend. Effective backend comes from the shared resolver so a
            # keyless ambient default runs local instead of failing to start.
            embed_backend = getattr(self, 'embed_backend_override', None)
            if not embed_backend:
                embed_backend, _reason = self.resolve_embed_backend(getattr(self, 'db_path', None))
                if _reason == 'no-key':
                    logger.warning(
                        "Voyage is the default embedding backend but no API key is configured "
                        "(MCP_VOYAGE_API_KEY or the keyring entry) - using local embeddings for "
                        "%s. Set the key to enable the cloud index, or set "
                        "MCP_MEMORY_EMBED_BACKEND=local to silence this warning.",
                        _sanitize_log_value(getattr(self, 'db_path', 'this store')))
            if embed_backend == 'voyage':
                from ...embeddings.voyage_embeddings import get_voyage_embedding_model
                from ...config.embedding import VOYAGE_MODEL, VOYAGE_OUTPUT_DIM
                if not self._voyage_key_present():
                    raise RuntimeError(
                        "Voyage embedding backend was selected but no API key is available now "
                        "(one was present when this store was constructed). Set "
                        "MCP_VOYAGE_API_KEY, or the keyring entry named by "
                        "MCP_VOYAGE_KEYRING_SERVICE/MCP_VOYAGE_KEYRING_USERNAME, or set "
                        "MCP_MEMORY_EMBED_BACKEND=local.")
                cache_key = f"voyage_{VOYAGE_MODEL}_{VOYAGE_OUTPUT_DIM}"
                if cache_key in _MODEL_CACHE:
                    self.embedding_model = _MODEL_CACHE[cache_key]
                    self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                else:
                    voyage_model = get_voyage_embedding_model(max_retries=4, retry_base_delay=0.5)
                    self.embedding_model = voyage_model
                    self.embedding_dimension = voyage_model.embedding_dimension
                    _MODEL_CACHE[cache_key] = voyage_model
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                self.embedding_model_name = f"voyage:{VOYAGE_MODEL}"
                # Upstream's namespaced cache key, extended with the model identity.
                self._embedding_cache_namespace = f"voyage_{VOYAGE_MODEL}_{VOYAGE_OUTPUT_DIM}"
                existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
                if existing_dim is not None and existing_dim != self.embedding_dimension:
                    raise RuntimeError(
                        f"Voyage embedding dimension ({self.embedding_dimension}) does not match "
                        f"the existing database schema FLOAT[{existing_dim}] "
                        f"({getattr(self, 'db_path', 'this database')}). Use the parallel index "
                        "(MCP_MEMORY_PARALLEL_INDEX=1, the default) so Voyage gets its own file, "
                        "or run scripts/maintenance/voyage_backfill.py to migrate.")
                logger.info("Voyage embedding backend configured. Model: %s, dimension: %s",
                            VOYAGE_MODEL, self.embedding_dimension)
                return
            external_api_url = os.environ.get('MCP_EXTERNAL_EMBEDDING_URL')
            if external_api_url:
                storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
                if storage_backend in ('hybrid', 'cloudflare'):
                    logger.warning(
                        "⚠️  External embedding API not supported with '%s' backend. "
                        "External APIs only work with 'sqlite_vec' backend. "
                        "The '%s' backend will use its default embedding method. "
                        "Falling back to local models (ONNX/SentenceTransformer) for SQLite-vec component.",
                        _sanitize_log_value(storage_backend), _sanitize_log_value(storage_backend),
                    )
                    external_api_url = None

            if external_api_url:
                logger.info("Using external embedding API: %s", _sanitize_log_value(external_api_url))
                try:
                    from ...embeddings.external_api import get_external_embedding_model

                    DEFAULT_EXTERNAL_EMBEDDING_MODEL = 'nomic-embed-text'
                    external_model_name = os.environ.get('MCP_EXTERNAL_EMBEDDING_MODEL', DEFAULT_EXTERNAL_EMBEDDING_MODEL)
                    external_api_key = os.environ.get('MCP_EXTERNAL_EMBEDDING_API_KEY')
                    cache_key = f"external_{external_api_url}_{external_model_name}_{external_api_key}"

                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        # Also on the cache-hit path, not just the fresh connect
                        # below: the global model cache means this is the normal
                        # path for every process after the first, so setting the
                        # name only on a cold start would leave health reporting
                        # wrong almost everywhere.
                        self.embedding_model_name = external_model_name
                        # Set embedding cache namespace to distinguish different external endpoints
                        self._embedding_cache_namespace = cache_key
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info("Using cached external embedding model")
                        return

                    ext_model = get_external_embedding_model(external_api_url, external_model_name)
                    self.embedding_model = ext_model
                    # Without this the attribute keeps the constructor's local
                    # default, and every health surface that reads it reports a
                    # model that is not the one producing the vectors (#254).
                    self.embedding_model_name = external_model_name
                    # Set embedding cache namespace to distinguish different external endpoints
                    self._embedding_cache_namespace = cache_key
                    self.embedding_dimension = ext_model.embedding_dimension
                    _MODEL_CACHE[cache_key] = ext_model
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension

                    if self.embedding_dimension != 384:
                        logger.warning(
                            "⚠️  External embedding dimension (%s) differs from "
                            "default ONNX dimension (384). Ensure this matches your database schema "
                            "or you may encounter errors. To fix: delete your database or use a "
                            "compatible model.",
                            self.embedding_dimension,
                        )

                    logger.info("External embedding API connected. Dimension: %s", self.embedding_dimension)
                    return
                except (ConnectionError, RuntimeError, ImportError) as e:
                    existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
                    dim_detail = (
                        f" The existing database uses dimension {existing_dim}."
                        f" Falling back to a local model would cause a dimension mismatch and"
                        f" corrupt all store/search operations."
                        if existing_dim is not None else ""
                    )
                    raise RuntimeError(
                        f"External embedding API at {external_api_url} is unreachable: {e}."
                        f"{dim_detail}"
                        f" Ensure your embedding service is running before starting mcp-memory-service."
                    ) from e

            use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() not in ('0', 'false', 'no')

            if use_onnx:
                logger.info("Attempting to use ONNX embeddings (PyTorch-free)")
                try:
                    from ...embeddings import get_onnx_embedding_model
                    from ...embeddings.onnx_embeddings import ONNX_AVAILABLE as _onnx_ok

                    cache_key = f"onnx_{self.embedding_model_name}"
                    if _onnx_ok and cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        # Set embedding cache namespace for ONNX models
                        self._embedding_cache_namespace = cache_key
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info("Using cached ONNX embedding model")
                        return
                    onnx_model = get_onnx_embedding_model(self.embedding_model_name)
                    if onnx_model:
                        # The ONNX path currently serves only all-MiniLM-L6-v2 (384-dim).
                        # If a different model was requested, say so loudly instead of
                        # logging plain success — otherwise the requested model is silently
                        # ignored and the user is unaware they are getting MiniLM-384 (#143).
                        requested_base = (self.embedding_model_name or '').split('/')[-1]
                        served_default = getattr(onnx_model, '_is_default_model', True)
                        if requested_base != 'all-MiniLM-L6-v2' and served_default:
                            logger.warning(
                                "ONNX backend does not honor the requested embedding model '%s'; "
                                "it only serves all-MiniLM-L6-v2 (%s-dim). The requested model is "
                                "being ignored. Set MCP_MEMORY_USE_ONNX=0 to load it via "
                                "SentenceTransformers, or configure an external embedding API "
                                "(MCP_EXTERNAL_EMBEDDING_URL).",
                                _sanitize_log_value(self.embedding_model_name), onnx_model.embedding_dimension,
                            )
                        self.embedding_model = onnx_model
                        # Set embedding cache namespace for ONNX models
                        self._embedding_cache_namespace = cache_key
                        self.embedding_dimension = onnx_model.embedding_dimension
                        _MODEL_CACHE[cache_key] = onnx_model
                        _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info("ONNX embedding model loaded successfully. Dimension: %s", self.embedding_dimension)
                        return
                    else:
                        logger.warning("ONNX model creation failed, falling back to SentenceTransformer")
                except ImportError as e:
                    logger.warning("ONNX dependencies not available: %s", _sanitize_log_value(e))
                except ValueError as e:
                    raise RuntimeError(f"Invalid ONNX embedding configuration: {e}") from e
                except Exception as e:
                    logger.warning("Failed to initialize ONNX embeddings: %s", _sanitize_log_value(e))

            # Check SentenceTransformer availability
            # Use sys.modules introspection to allow test patches on SENTENCE_TRANSFORMERS_AVAILABLE
            _st_mod = sys.modules.get(__name__, None)
            _st_flag = getattr(_st_mod, 'SENTENCE_TRANSFORMERS_AVAILABLE', False) if _st_mod else SENTENCE_TRANSFORMERS_AVAILABLE
            _st_available = _st_flag or SentenceTransformer is not None
            if not _st_available:
                if not (_HASH_FALLBACK_WARNED or getattr(self, '_hash_fallback_warned', False)):
                    logger.warning(
                        "No embedding backend available; using hash embeddings (reduced quality). "
                        "Install ML dependencies for semantic search: pip install mcp-memory-service[ml]. "
                        "The hash fallback is refused on databases that already contain memories; "
                        "set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
                    )
                    _HASH_FALLBACK_WARNED = True
                    self._hash_fallback_warned = True
                await self._initialize_hash_embedding_fallback()
                return

            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                # Set embedding cache namespace for SentenceTransformer models
                self._embedding_cache_namespace = cache_key
                if cache_key in _DIMENSION_CACHE:
                    self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                elif hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                    dim = self.embedding_model.get_sentence_embedding_dimension()
                    if dim:
                        self.embedding_dimension = dim
                        _DIMENSION_CACHE[cache_key] = dim
                elif hasattr(self.embedding_model, 'embedding_dimension'):
                    self.embedding_dimension = self.embedding_model.embedding_dimension
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                logger.info("Using cached embedding model: %s", _sanitize_log_value(self.embedding_model_name))
                return

            device = get_torch_device()

            logger.info("Loading embedding model: %s", _sanitize_log_value(self.embedding_model_name))
            logger.info("Using device: %s", device)

            hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            model_cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
            if os.path.exists(model_cache_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.info("📦 Found cached model - enabling offline mode")

            try:
                hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
                cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
                if os.path.exists(cache_path):
                    snapshots_path = os.path.join(cache_path, "snapshots")
                    if os.path.exists(snapshots_path):
                        snapshot_dirs = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshot_dirs:
                            model_path = os.path.join(snapshots_path, snapshot_dirs[0])
                            logger.info("Loading model from cache: %s", _sanitize_log_value(model_path))
                            self.embedding_model = SentenceTransformer(model_path, device=device)
                        else:
                            raise FileNotFoundError("No snapshot found")
                    else:
                        raise FileNotFoundError("No snapshots directory")
                else:
                    raise FileNotFoundError("No cache found")
            except FileNotFoundError as cache_error:
                logger.warning("Model not in cache: %s", _sanitize_log_value(cache_error))
                try:
                    logger.info("Attempting to download model from Hugging Face...")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                except OSError as download_error:
                    error_msg = str(download_error)
                    if any(phrase in error_msg.lower() for phrase in ['connection', 'network', 'couldn\'t connect', 'huggingface.co']):
                        docker_help = self._get_docker_network_help() if is_docker else ""
                        raise RuntimeError(
                            f"🔌 Model Download Error: Cannot connect to huggingface.co\n"
                            f"{'='*60}\n"
                            f"The model '{self.embedding_model_name}' needs to be downloaded but the connection failed.\n"
                            f"{docker_help}"
                            f"\n💡 Solutions:\n"
                            f"1. Mount pre-downloaded models as a volume:\n"
                            f"   # On host machine, download the model first:\n"
                            f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{self.embedding_model_name}')\"\n"
                            f"   \n"
                            f"   # Then run container with cache mount:\n"
                            f"   docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...\n"
                            f"\n"
                            f"2. Configure Docker network (if behind proxy):\n"
                            f"   docker run -e HTTPS_PROXY=your-proxy -e HTTP_PROXY=your-proxy ...\n"
                            f"\n"
                            f"3. Use offline mode with pre-cached models:\n"
                            f"   docker run -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ...\n"
                            f"\n"
                            f"4. Use host network mode (if appropriate for your setup):\n"
                            f"   docker run --network host ...\n"
                            f"\n"
                            f"📚 See docs: https://github.com/doobidoo/mcp-memory-service/blob/main/docs/deployment/docker.md#model-download-issues\n"
                            f"{'='*60}"
                        ) from download_error
                    else:
                        raise
            except Exception as cache_error:
                logger.warning("Failed to load from cache: %s", _sanitize_log_value(cache_error))
                logger.info("Attempting normal model loading...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]

            # Set embedding cache namespace for SentenceTransformer models  
            self._embedding_cache_namespace = cache_key
            
            _MODEL_CACHE[cache_key] = self.embedding_model
            _DIMENSION_CACHE[cache_key] = self.embedding_dimension

            logger.info("✅ Embedding model loaded successfully. Dimension: %s", self.embedding_dimension)

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("Failed to initialize embedding model: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            if not (_HASH_FALLBACK_WARNED or getattr(self, '_hash_fallback_warned', False)):
                logger.warning(
                    "No embedding backend available; using hash embeddings (reduced quality). "
                    "Install ML dependencies for semantic search: pip install mcp-memory-service[ml]. "
                    "The hash fallback is refused on databases that already contain memories; "
                    "set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
                )
                _HASH_FALLBACK_WARNED = True
                self._hash_fallback_warned = True
            await self._initialize_hash_embedding_fallback()

    def _count_existing_rows(self) -> tuple:
        """Return (memories, embedding_rows), treating missing tables as empty.

        Both counts matter, for different reasons. The emptiness decision needs
        their sum, so a database holding real embeddings is never treated as
        empty. The error message needs them apart: they are separate tables of
        roughly equal size on a normally-populated database, so reporting the
        sum as a memory count overstated it by about a factor of two (#228).
        """
        counts = []
        for table in ("memories", "memory_embeddings"):
            count = 0
            try:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                if row and row[0]:
                    count = int(row[0])
            except Exception as exc:
                # Missing table (fresh DB) or unreadable — treat as empty for this table.
                logger.debug("Could not count rows in %s: %s", table, _sanitize_log_value(exc))
            counts.append(count)
        return counts[0], counts[1]

    def _describe_existing_rows(self) -> str:
        """Phrase the row counts for an error message, without inventing a total."""
        memories, embedding_rows = self._count_existing_rows() if self.conn else (0, 0)
        return f"{memories} memories and {embedding_rows} embedding rows"

    def _hash_fallback_allowed(self) -> bool:
        """Decide whether writing hash pseudo-vectors into this database is safe.

        Policy:
          - MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1/true/yes force-allows the fallback.
          - MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=0/false/no force-refuses it.
          - Unset: allow only when the database is effectively empty (no existing
            ``memories`` or ``memory_embeddings`` rows). A database that already
            contains real memories is protected, because hash pseudo-vectors would
            silently poison semantic search with no per-row marker to find them later.

        Raises RuntimeError with an actionable message when the fallback is refused.
        """
        override = os.environ.get('MCP_MEMORY_ALLOW_HASH_EMBEDDINGS', '').strip().lower()
        if override in ('1', 'true', 'yes'):
            return True
        if override in ('0', 'false', 'no'):
            raise RuntimeError(
                "The configured embedding backend is unavailable and "
                "MCP_MEMORY_ALLOW_HASH_EMBEDDINGS is set to refuse the hash fallback. "
                "Refusing to write hash pseudo-vectors into a database holding "
                f"{self._describe_existing_rows()}. Install a real embedding backend "
                "(pip install 'mcp-memory-service[sqlite-ml]' or 'mcp-memory-service[ml]') "
                "or set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
            )

        # Default policy: allow only on an effectively-empty database.
        memories, embedding_rows = self._count_existing_rows() if self.conn else (0, 0)
        if memories + embedding_rows == 0:
            return True
        raise RuntimeError(
            "The configured embedding backend is unavailable; refusing to write hash "
            f"pseudo-vectors into a database holding {memories} memories and "
            f"{embedding_rows} embedding rows "
            "(this would permanently poison semantic search with no per-row marker to "
            "find affected rows later). Install a real embedding backend "
            "(pip install 'mcp-memory-service[sqlite-ml]' or 'mcp-memory-service[ml]') "
            "or set MCP_MEMORY_ALLOW_HASH_EMBEDDINGS=1 to override for testing."
        )

    async def _initialize_hash_embedding_fallback(self):
        """Initialize hash embedding model, matching existing DB dimension if possible."""
        # Guard: never silently poison a database that already holds real memories.
        # Called synchronously (a quick COUNT read) — initialization is single-threaded,
        # and this raises RuntimeError when the fallback is refused.
        self._hash_fallback_allowed()

        existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
        if existing_dim and existing_dim != self.embedding_dimension:
            logger.warning(
                "Adjusting hash embedding dimension from %s to %s to match existing database schema.",
                self.embedding_dimension, existing_dim,
            )
            self.embedding_dimension = existing_dim
        self.embedding_model = _HashEmbeddingModel(self.embedding_dimension)
        # Set embedding cache namespace for hash fallback.
        # Reserved prefix (with '::' separator, which no configured model/provider
        # namespace uses) so a real model literally named "hash_384" cannot collide
        # with the fallback's pseudo-vectors in the shared cache (Greptile #1367).
        self._embedding_cache_namespace = f"__hash_fallback__::{self.embedding_dimension}"
        self.embedding_backend_degraded = True

    def _generate_embedding(self, text: str, input_type: str = "document") -> List[float]:
        """Generate embedding for text.

        ``input_type`` (query vs document) only affects asymmetric cloud models like
        Voyage; local models ignore it. ``embedding_failure_count`` is a monotonic
        signal for callers that must tell "embedder down" from "no matches": the
        retrieve paths catch our exception and return [], which is otherwise
        indistinguishable from an empty result set.
        """
        self.embedding_last_error = None
        if not self.embedding_model:
            self._record_embedding_failure("No embedding model available")
            raise RuntimeError("No embedding model available. Ensure sentence-transformers is installed and model is loaded.")

        try:
            if self.enable_cache:
                # Use namespace to distinguish different providers/endpoints with same model name
                # Only use the cached namespace if it's for the same model name (for external APIs)
                # or if it's a compound namespace that inherently includes model identity
                namespace = self.embedding_model_name  # Default to current model name
                
                if hasattr(self, '_embedding_cache_namespace') and self._embedding_cache_namespace:
                    cached_namespace = self._embedding_cache_namespace
                    
                    # For external APIs: namespace is "external_url_model_key" - use it if model matches
                    if cached_namespace.startswith('external_') and self.embedding_model_name in cached_namespace:
                        namespace = cached_namespace
                    # For ONNX: namespace is "onnx_model" - use it if model matches  
                    elif cached_namespace.startswith('onnx_') and cached_namespace == f"onnx_{self.embedding_model_name}":
                        namespace = cached_namespace
                    # For hash fallback: namespace is "__hash_fallback__::dimension"
                    # (reserved prefix that includes the DB vector dimension, so two
                    # fallback stores with the same model name but different dimensions
                    # do not share a cache entry — Greptile #1367).
                    elif cached_namespace.startswith('__hash_fallback__::'):
                        namespace = cached_namespace
                    # For regular models: namespace should be model name - only use if it matches
                    elif cached_namespace == self.embedding_model_name:
                        namespace = cached_namespace
                    # Otherwise: use current model name (handles model name changes in tests)
                
                cache_key = f"{namespace}::{input_type}::{text}"
                cached = _embedding_cache_get(cache_key)
                if cached is not None:
                    return cached

            if getattr(self.embedding_model, 'supports_input_type', False):
                embedding = self.embedding_model.encode(
                    [text], convert_to_numpy=True, input_type=input_type)[0]
            else:
                embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            if hasattr(embedding, "tolist"):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)

            if not embedding_list:
                raise ValueError("Generated embedding is empty")

            if len(embedding_list) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding_list)}")

            if not all(isinstance(x, (int, float)) and not math.isnan(x) and x != float('inf') and x != float('-inf') for x in embedding_list):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")

            if self.enable_cache:
                _embedding_cache_put(cache_key, embedding_list)

            return embedding_list

        except Exception as e:
            self._record_embedding_failure(str(e))
            logger.error("Failed to generate embedding: %s", _sanitize_log_value(e))
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    def _record_embedding_failure(self, message: str) -> None:
        """Record a failed embed call: monotonic count plus the latest message."""
        self.embedding_failure_count = getattr(self, "embedding_failure_count", 0) + 1
        self.embedding_last_error = message
