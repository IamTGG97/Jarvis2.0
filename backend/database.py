import logging
import os
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# Raising these to 4 recent + 5 relevant did not improve recall on the memory
# eval and added unrelated turns that the small model sometimes mixed into its
# answers, so the original 3 + 3 stays.
RECENT_TURN_LIMIT = 3
RELEVANT_TURN_LIMIT = 3
# Ollama runs with num_ctx 2048. Reserve room for the system prompt (~250
# tokens), the new question, and the 160-token answer; history gets the rest
# with a safety margin because the token estimate is approximate.
CONTEXT_TOKEN_BUDGET = 1200


def _connect() -> psycopg.Connection:
    required_settings = ("DB_NAME", "DB_USER", "DB_PASSWORD")
    missing_settings = [
        setting for setting in required_settings if not os.getenv(setting)
    ]
    if missing_settings:
        settings = ", ".join(missing_settings)
        raise RuntimeError(f"Missing database configuration: {settings}.")

    settings = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
        "row_factory": dict_row,
    }
    try:
        return psycopg.connect(**settings)
    except psycopg.errors.ConnectionTimeout:
        # On this low-memory machine a connect right after Ollama unloads the
        # chat model occasionally stalls past the timeout. One retry avoids
        # losing the just-generated turn; a second failure is surfaced.
        logger.warning("PostgreSQL connection timed out; retrying once")
        return psycopg.connect(**settings)


def initialize_database() -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                session_id UUID NOT NULL
                    REFERENCES conversation_sessions(id) ON DELETE CASCADE,
                transcript TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                search_vector TSVECTOR GENERATED ALWAYS AS (
                    setweight(to_tsvector('english'::regconfig, transcript), 'A')
                    || setweight(to_tsvector('english'::regconfig, response), 'B')
                ) STORED
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS conversation_turns_session_created_idx
            ON conversation_turns (session_id, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS conversation_turns_search_idx
            ON conversation_turns USING GIN (search_vector)
            """
        )


def find_conversation_context(
    session_id: UUID,
    transcript: str,
) -> list[dict[str, str]]:
    """Return the turns to show the LLM: relevant older turns, then recent turns.

    Both groups are in chronological order. Recent turns are deduplicated from
    search results by row id (not text, so a repeated question cannot hide a
    different earlier answer), and the whole context is trimmed to fit
    ``CONTEXT_TOKEN_BUDGET``.
    """
    with _connect() as connection, connection.cursor() as cursor:
        recent_turns = _recent_turns(cursor, session_id, RECENT_TURN_LIMIT)
        relevant_turns = _search_turns(
            cursor,
            session_id,
            transcript,
            RELEVANT_TURN_LIMIT,
            exclude_ids=[turn["id"] for turn in recent_turns],
        )

    selected_recent, selected_relevant = fit_context_budget(
        recent_turns, relevant_turns, CONTEXT_TOKEN_BUDGET
    )
    selected_relevant.sort(key=lambda turn: (turn["created_at"], turn["id"]))
    return [
        {"transcript": turn["transcript"], "response": turn["response"]}
        for turn in selected_relevant + selected_recent
    ]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: about four characters per token for English."""
    return len(text) // 4 + 1


def _turn_tokens(turn: dict) -> int:
    # A few extra tokens cover the chat template's per-message role markers.
    return estimate_tokens(turn["transcript"]) + estimate_tokens(turn["response"]) + 8


def fit_context_budget(
    recent_turns: list[dict],
    relevant_turns: list[dict],
    budget: int,
) -> tuple[list[dict], list[dict]]:
    """Keep the newest recent turn and the best search hit first, then fill the
    remaining budget alternating between newer recent turns and better hits."""
    newest_first = list(reversed(recent_turns))
    queues = [newest_first, list(relevant_turns)]
    kept: tuple[list[dict], list[dict]] = ([], [])
    used = 0
    while any(queues):
        for index, queue in enumerate(queues):
            if not queue:
                continue
            turn = queue.pop(0)
            cost = _turn_tokens(turn)
            if used + cost > budget:
                continue
            kept[index].append(turn)
            used += cost
    return list(reversed(kept[0])), kept[1]


def _recent_turns(
    cursor: psycopg.Cursor,
    session_id: UUID,
    limit: int,
) -> list[dict]:
    cursor.execute(
        """
        SELECT id, transcript, response, created_at
        FROM conversation_turns
        WHERE session_id = %s
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (session_id, limit),
    )
    return list(reversed(cursor.fetchall()))


# Words that describe the act of remembering rather than the topic. Left in the
# OR query they match unrelated turns ("Can you *say* hello in French?").
# Values are PostgreSQL English stems as produced by to_tsvector.
MEMORY_QUERY_STOP_LEXEMES = [
    "say", "said", "tell", "told", "rememb", "mention", "know", "recal",
    "forget", "forgot", "ask", "jarvi", "think", "earlier", "previous",
]

_SEARCH_SQL = """
    WITH query_terms AS (
        SELECT lexeme
        FROM unnest(to_tsvector('english', %(query)s))
        WHERE lexeme <> ALL(%(stop_lexemes)s)
    ),
    search_query AS (
        SELECT to_tsquery(
            'simple',
            string_agg(quote_literal(lexeme), ' | ')
        ) AS terms
        FROM query_terms
    )
    SELECT turn.id, turn.transcript, turn.response, turn.created_at,
        ts_rank(turn.search_vector, search_query.terms) AS rank
    FROM conversation_turns AS turn
    CROSS JOIN search_query
    WHERE turn.session_id = %(session_id)s
        AND search_query.terms IS NOT NULL
        AND turn.search_vector @@ search_query.terms
        AND turn.id <> ALL(%(exclude_ids)s)
    ORDER BY rank DESC, turn.created_at DESC, turn.id DESC
    LIMIT %(limit)s
"""


def _search_turns(
    cursor: psycopg.Cursor,
    session_id: UUID,
    query: str,
    limit: int,
    exclude_ids: list[int] | None = None,
) -> list[dict]:
    """Keyword search: any meaningful query word may match; turns that match
    more (and rarer, transcript-weighted) words rank higher."""
    cursor.execute(
        _SEARCH_SQL,
        {
            "query": query,
            "stop_lexemes": MEMORY_QUERY_STOP_LEXEMES,
            "session_id": session_id,
            "exclude_ids": exclude_ids or [],
            "limit": limit,
        },
    )
    return cursor.fetchall()


def search_conversation_turns(
    session_id: UUID,
    query: str,
    limit: int,
) -> list[dict]:
    """Return the session's turns ranked by keyword relevance (best first)."""
    with _connect() as connection, connection.cursor() as cursor:
        return _search_turns(cursor, session_id, query, limit)


def save_conversation_turn(
    session_id: UUID,
    transcript: str,
    response: str,
) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO conversation_sessions (id)
            VALUES (%s)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            (session_id,),
        )
        cursor.execute(
            """
            INSERT INTO conversation_turns (session_id, transcript, response)
            VALUES (%s, %s, %s)
            """,
            (session_id, transcript, response),
        )
