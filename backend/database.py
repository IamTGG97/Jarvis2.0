import logging
import os
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

RECENT_TURN_LIMIT = 3
RELEVANT_TURN_LIMIT = 3


def _connect() -> psycopg.Connection:
    required_settings = ("DB_NAME", "DB_USER", "DB_PASSWORD")
    missing_settings = [
        setting for setting in required_settings if not os.getenv(setting)
    ]
    if missing_settings:
        settings = ", ".join(missing_settings)
        raise RuntimeError(f"Missing database configuration: {settings}.")

    return psycopg.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=5,
        row_factory=dict_row,
    )


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
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT transcript, response
            FROM conversation_turns
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (session_id, RECENT_TURN_LIMIT),
        )
        recent_turns = list(reversed(cursor.fetchall()))
        recent_transcripts = {turn["transcript"] for turn in recent_turns}

        cursor.execute(
            """
            WITH search_query AS (
                SELECT replace(
                    plainto_tsquery('english', %s)::text,
                    ' & ',
                    ' | '
                )::tsquery AS terms
            )
            SELECT transcript, response
            FROM conversation_turns
            CROSS JOIN search_query
            WHERE session_id = %s
                AND search_vector @@ search_query.terms
            ORDER BY ts_rank(search_vector, search_query.terms) DESC,
                created_at DESC,
                id DESC
            LIMIT %s
            """,
            (transcript, session_id, RECENT_TURN_LIMIT + RELEVANT_TURN_LIMIT),
        )
        relevant_turns = [
            turn
            for turn in cursor.fetchall()
            if turn["transcript"] not in recent_transcripts
        ][:RELEVANT_TURN_LIMIT]

    return recent_turns + relevant_turns


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
