"""Voyage AI cloud embedding adapter.

Mimics the SentenceTransformer interface (``encode`` +
``get_sentence_embedding_dimension``) like :mod:`.external_api`, but talks to
the Voyage AI ``/v1/embeddings`` endpoint. Voyage is not OpenAI-compatible:
the request carries ``input_type`` (``query`` vs ``document``) and an explicit
``output_dimension``, so it gets its own adapter rather than a variant of
:class:`~mcp_memory_service.embeddings.external_api.ExternalEmbeddingModel`.

Design notes (mirrors the proven client in the estate's retrieval-mcp):

- **No constructor network probe.** The dimension is requested explicitly via
  ``output_dimension``, so startup never blocks on an API round-trip. Failures
  surface on first use as :class:`RuntimeError` — callers decide the fallback
  policy (the storage layer's parallel wrapper degrades to the local index and
  says so; nothing silently swaps vector spaces).
- **Retries only 429 and 5xx** with exponential backoff capped at 60 s,
  honoring ``Retry-After`` when present. Any other 4xx fails immediately.
- **The API key is never logged** — error detail has the key redacted.
- **Batching** with per-text truncation to ``MCP_VOYAGE_MAX_CHARS``.

Configuration:
    - ``MCP_VOYAGE_API_KEY``: explicit API key (wins)
    - else the OS keyring entry ``MCP_VOYAGE_KEYRING_SERVICE`` /
      ``MCP_VOYAGE_KEYRING_USERNAME`` (defaults: ``sentinel-jarl`` /
      ``voyage_memory_api_key`` — the estate's shared Voyage credential)
    - ``MCP_VOYAGE_MODEL`` (default ``voyage-4``)
    - ``MCP_VOYAGE_OUTPUT_DIM`` (default ``1024``)
    - ``MCP_VOYAGE_BATCH_SIZE`` (default ``64``), ``MCP_VOYAGE_MAX_CHARS`` (default ``8000``)
"""

import json
import logging
import os
import time
from typing import List, Optional, Union

import numpy as np
import requests

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingModel:
    """Adapter producing Voyage embeddings behind the SentenceTransformer-ish contract.

    Attributes:
        model_name: Voyage model id (e.g. ``voyage-4``).
        embedding_dimension: Requested output dimension (default 1024).
        supports_input_type: Always True — lets the storage layer know that
            ``encode`` accepts an ``input_type`` argument (``query`` vs
            ``document``) that meaningfully changes the vector.
    """

    supports_input_type = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_chars: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 8,
        retry_base_delay: float = 5.0,
    ):
        from ..config.embedding import (
            VOYAGE_BATCH_SIZE,
            VOYAGE_KEYRING_SERVICE,
            VOYAGE_KEYRING_USERNAME,
            VOYAGE_MAX_CHARS,
            VOYAGE_MODEL,
            VOYAGE_OUTPUT_DIM,
        )

        self.model_name = model_name or VOYAGE_MODEL
        self.embedding_dimension = int(output_dimension or VOYAGE_OUTPUT_DIM)
        self.batch_size = int(batch_size or VOYAGE_BATCH_SIZE)
        self.max_chars = int(max_chars or VOYAGE_MAX_CHARS)
        self.timeout = timeout
        self.max_retries = max_retries
        # Backoff base: 5 s suits a batch backfill riding out free-tier rate
        # limits; interactive paths want ~0.5 s so a Voyage outage degrades to
        # the local index in seconds instead of stalling every call for minutes.
        self.retry_base_delay = retry_base_delay
        self.api_key = api_key or self._resolve_api_key(
            VOYAGE_KEYRING_SERVICE, VOYAGE_KEYRING_USERNAME
        )

    @staticmethod
    def _resolve_api_key(service: str, username: str) -> Optional[str]:
        """API key from the env, else the OS keyring.

        The keyring lookup is best-effort: it is skipped entirely when the
        ``keyring`` package is missing or the backend errors, so the adapter
        works upstream with only ``MCP_VOYAGE_API_KEY`` set.
        """
        key = os.getenv("MCP_VOYAGE_API_KEY") or ""
        if key:
            return key
        try:
            import keyring

            key = keyring.get_password(service, username) or ""
        except Exception as exc:  # no keyring backend, locked vault, etc.
            logger.debug("Voyage keyring lookup failed (%s); relying on env var", exc)
            key = ""
        return key or None

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        input_type: str = "document",
        **kwargs,
    ) -> np.ndarray:
        """Embed texts via Voyage. Raises RuntimeError on failure (never returns junk).

        Args:
            sentences: Single text or list of texts.
            convert_to_numpy: Return a float32 ndarray (default) instead of lists.
            input_type: ``document`` for stored content, ``query`` for search
                queries — Voyage models are asymmetric-trained.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        texts = [t[: self.max_chars] for t in sentences]
        if not texts:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            part = texts[i : i + self.batch_size]
            vectors = self._post(part, input_type)
            if (
                vectors is None
                or len(vectors) != len(part)
                or any(len(v) != self.embedding_dimension for v in vectors)
            ):
                raise RuntimeError(
                    f"Voyage embedding request failed after {self.max_retries} retries "
                    f"({len(part)} texts, model={self.model_name})"
                )
            all_embeddings.extend(vectors)

        if convert_to_numpy:
            return np.array(all_embeddings, dtype=np.float32)
        return all_embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, texts: List[str], input_type: str) -> Optional[List[List[float]]]:
        """One (retried) Voyage /v1/embeddings call. None on exhausted retries."""
        if not self.api_key:
            logger.error(
                "No Voyage API key: set MCP_VOYAGE_API_KEY or the keyring entry "
                "(see MCP_VOYAGE_KEYRING_SERVICE/MCP_VOYAGE_KEYRING_USERNAME)"
            )
            return None
        payload = {
            "model": self.model_name,
            "input": texts,
            "input_type": input_type,
            "output_dimension": self.embedding_dimension,
        }
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    VOYAGE_API_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    rows = sorted(data["data"], key=lambda item: item.get("index", 0))
                    return [row["embedding"] for row in rows]

                if response.status_code == 429 or response.status_code >= 500:
                    wait = min(60.0, self.retry_base_delay * (2**attempt))
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = min(60.0, float(retry_after))
                    logger.warning(
                        "Voyage %s, retrying in %.0fs (%d/%d)",
                        response.status_code, wait, attempt + 1, self.max_retries,
                    )
                    time.sleep(wait)
                    continue

                # Any other 4xx: fail immediately, key redacted.
                logger.error(
                    "Voyage error %s: %s",
                    response.status_code,
                    response.text.replace(self.api_key or "", "<key>")[:200],
                )
                return None

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Voyage request failed (%s), retrying (%d/%d)",
                    type(exc).__name__, attempt + 1, self.max_retries,
                )
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("Voyage returned an unexpected response shape: %s", exc)
                return None
        return None


def get_voyage_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dimension: Optional[int] = None,
) -> VoyageEmbeddingModel:
    """Factory mirroring get_external_embedding_model / get_onnx_embedding_model."""
    return VoyageEmbeddingModel(
        model_name=model_name,
        api_key=api_key,
        output_dimension=output_dimension,
    )
