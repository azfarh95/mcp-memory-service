"""Embedding model configuration — ONNX and model selection."""
import os
import logging

from .base import BASE_DIR

logger = logging.getLogger(__name__)

# ONNX Configuration
USE_ONNX = os.getenv('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')
if USE_ONNX:
    logger.info("ONNX embeddings enabled - using PyTorch-free embedding generation")
    # ONNX model cache directory
    ONNX_MODEL_CACHE = os.path.join(BASE_DIR, 'onnx_models')
    os.makedirs(ONNX_MODEL_CACHE, exist_ok=True)

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('MCP_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Voyage cloud embedding backend (parallel local/cloud path).
# 'voyage' = Voyage AI API as the primary embedder, local ONNX kept in parallel as
# the fallback index; 'local' = the classic local-only path.
EMBED_BACKEND = os.getenv('MCP_MEMORY_EMBED_BACKEND', 'voyage').strip().lower()
if EMBED_BACKEND not in ('voyage', 'local'):
    raise ValueError(
        f"MCP_MEMORY_EMBED_BACKEND must be 'voyage' or 'local', got {EMBED_BACKEND!r}"
    )

VOYAGE_MODEL = os.getenv('MCP_VOYAGE_MODEL', 'voyage-4')
VOYAGE_OUTPUT_DIM = int(os.getenv('MCP_VOYAGE_OUTPUT_DIM', '1024'))
VOYAGE_BATCH_SIZE = int(os.getenv('MCP_VOYAGE_BATCH_SIZE', '64'))
VOYAGE_MAX_CHARS = int(os.getenv('MCP_VOYAGE_MAX_CHARS', '8000'))
VOYAGE_API_KEY = os.getenv('MCP_VOYAGE_API_KEY') or None
VOYAGE_KEYRING_SERVICE = os.getenv('MCP_VOYAGE_KEYRING_SERVICE', 'sentinel-jarl')
VOYAGE_KEYRING_USERNAME = os.getenv('MCP_VOYAGE_KEYRING_USERNAME', 'voyage_memory_api_key')

