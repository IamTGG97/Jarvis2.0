"""Conversation-memory orchestration: retrieval (keyword + optional embeddings)
and saving turns with their embeddings.

Failure policy for embeddings (documented in docs/overnight/memory-engineer.md):
if the local embedding model fails, retrieval falls back to keyword search and
the caller receives an explicit ``warning`` string that the API returns to the
UI. Turns are still saved without a vector; ``backfill_embeddings.py`` fills
them in later. Database failures are never swallowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import database
from embeddings import (
    EmbeddingError,
    embed_text,
    embedding_model,
    embeddings_enabled,
    turn_document_text,
)

logger = logging.getLogger(__name__)

RETRIEVAL_WARNING = (
    "Semantic memory search is unavailable right now, so Jarvis used keyword "
    "search only. Check that Ollama has the embedding model installed."
)
SAVE_WARNING = (
    "This turn was saved without a semantic-search embedding. Run "
    "backend/backfill_embeddings.py once Ollama's embedding model is available."
)


@dataclass
class RetrievedMemory:
    turns: list[dict[str, str]]
    mode: str  # "hybrid" or "keyword"
    warning: str | None = None


def retrieve_memory(session_id: UUID, transcript: str) -> RetrievedMemory:
    """Find the turns to show the LLM. Raises database errors unchanged."""
    query_vector = None
    warning = None
    if embeddings_enabled():
        try:
            query_vector = embed_text(transcript, "query")
        except EmbeddingError:
            logger.warning("Query embedding failed; using keyword-only memory search", exc_info=True)
            warning = RETRIEVAL_WARNING

    turns = database.find_conversation_context(
        session_id,
        transcript,
        query_embedding=query_vector,
        embedding_model=embedding_model() if query_vector else None,
    )
    return RetrievedMemory(turns, "hybrid" if query_vector else "keyword", warning)


def remember_turn(session_id: UUID, transcript: str, response: str) -> str | None:
    """Save a turn (with its embedding when possible). Returns a warning if the
    embedding could not be produced. Raises database errors unchanged."""
    vector = None
    warning = None
    if embeddings_enabled():
        try:
            vector = embed_text(turn_document_text(transcript, response), "document")
        except EmbeddingError:
            logger.warning("Turn embedding failed; saving without a vector", exc_info=True)
            warning = SAVE_WARNING

    database.save_conversation_turn(
        session_id,
        transcript,
        response,
        embedding=vector,
        embedding_model=embedding_model() if vector else None,
    )
    return warning
