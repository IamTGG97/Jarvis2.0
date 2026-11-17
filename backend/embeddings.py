"""Local text embeddings through Ollama (``nomic-embed-text`` by default).

nomic-embed-text is trained with task prefixes: stored text is embedded with
``search_document:`` and questions with ``search_query:``. Using the right
prefix measurably improves retrieval, so callers choose the kind explicitly.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

EmbeddingKind = Literal["document", "query"]
_PREFIXES: dict[str, str] = {
    "document": "search_document: ",
    "query": "search_query: ",
}


class EmbeddingError(RuntimeError):
    """Raised when the local embedding model could not produce vectors."""


def embedding_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", "").strip() or "nomic-embed-text"


def embeddings_enabled() -> bool:
    return os.getenv("MEMORY_EMBEDDINGS", "on").strip().lower() not in {"off", "0", "false", "no"}


def turn_document_text(transcript: str, response: str) -> str:
    """The text that represents one stored conversation turn."""
    return f"User: {transcript}\nJarvis: {response}"


def embed_texts(texts: list[str], kind: EmbeddingKind) -> list[list[float]]:
    """Embed texts synchronously. Raises EmbeddingError on any failure."""
    if not texts:
        return []
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = embedding_model()
    try:
        response = httpx.post(
            f"{base_url}/api/embed",
            json={
                "model": model,
                "input": [_PREFIXES[kind] + text for text in texts],
                # "0" unloads the embedding model right away so it is never
                # resident at the same time as the chat model on low-RAM machines.
                "keep_alive": os.getenv("OLLAMA_EMBED_KEEP_ALIVE", "0"),
                "truncate": True,
            },
            timeout=float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "30")),
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings")
    except (httpx.HTTPError, ValueError) as error:
        raise EmbeddingError(f"Embedding with '{model}' failed: {error}") from error

    if (
        not isinstance(vectors, list)
        or len(vectors) != len(texts)
        or not all(isinstance(vector, list) and vector for vector in vectors)
    ):
        raise EmbeddingError(f"Embedding with '{model}' returned an unexpected payload.")
    return [[float(value) for value in vector] for vector in vectors]


def embed_text(text: str, kind: EmbeddingKind) -> list[float]:
    return embed_texts([text], kind)[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
