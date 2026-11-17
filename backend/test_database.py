import os
import unittest
from unittest.mock import patch
from uuid import UUID

from datetime import datetime, timedelta, timezone

import psycopg

from database import (
    find_conversation_context,
    KEYWORD_CANDIDATE_LIMIT,
    backfill_embeddings,
    fit_context_budget,
    reciprocal_rank_fusion,
    initialize_database,
    save_conversation_turn,
    search_conversation_turns,
)

TEST_SESSION_ID = UUID("fdde5172-0d09-4eb6-8502-ad4a65448b19")
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _turn(turn_id: int, transcript: str, response: str = "answer", minute: int = 0) -> dict:
    return {
        "id": turn_id,
        "transcript": transcript,
        "response": response,
        "created_at": BASE_TIME + timedelta(minutes=minute),
    }


class FakeCursor:
    def __init__(self, results: list[list[dict[str, str]]] | None = None) -> None:
        self.results = results or []
        self.executions: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        self.executions.append((query, parameters))

    def fetchall(self) -> list[dict[str, str]]:
        return self.results.pop(0)


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.fake_cursor


class DatabaseTests(unittest.TestCase):
    def test_requires_database_settings(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"DB_NAME": "", "DB_USER": "", "DB_PASSWORD": ""},
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "Missing database configuration"),
        ):
            from database import _connect

            _connect()

    def test_connect_retries_once_after_timeout(self) -> None:
        settings = {"DB_NAME": "db", "DB_USER": "user", "DB_PASSWORD": "pw"}
        timeout = psycopg.errors.ConnectionTimeout("connection timeout expired")
        with (
            patch.dict(os.environ, settings),
            patch("database.psycopg.connect", side_effect=[timeout, "connection"]) as connect,
            self.assertLogs("database", level="WARNING"),
        ):
            from database import _connect

            self.assertEqual(_connect(), "connection")
        self.assertEqual(connect.call_count, 2)

    def test_connect_surfaces_second_timeout(self) -> None:
        settings = {"DB_NAME": "db", "DB_USER": "user", "DB_PASSWORD": "pw"}
        timeout = psycopg.errors.ConnectionTimeout("connection timeout expired")
        with (
            patch.dict(os.environ, settings),
            patch("database.psycopg.connect", side_effect=[timeout, timeout]),
            self.assertLogs("database", level="WARNING"),
            self.assertRaises(psycopg.errors.ConnectionTimeout),
        ):
            from database import _connect

            _connect()

    def test_creates_session_turn_and_search_tables(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)

        with patch("database._connect", return_value=connection):
            initialize_database()

        sql = "\n".join(query for query, _ in cursor.executions)
        self.assertIn("CREATE TABLE IF NOT EXISTS conversation_sessions", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS conversation_turns", sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS conversation_turns_search_idx", sql)

    def test_context_puts_relevant_turns_before_recent_turns(self) -> None:
        recent_newest_first = [
            _turn(10, "Latest question", minute=10),
            _turn(9, "Recent question", minute=9),
        ]
        search_ranked = [
            _turn(2, "Best older match", minute=2),
            _turn(1, "Oldest weaker match", minute=1),
        ]
        cursor = FakeCursor([recent_newest_first, search_ranked])
        connection = FakeConnection(cursor)

        with patch("database._connect", return_value=connection):
            result = find_conversation_context(TEST_SESSION_ID, "matching question")

        self.assertEqual(
            [turn["transcript"] for turn in result],
            [
                "Oldest weaker match",
                "Best older match",
                "Recent question",
                "Latest question",
            ],
        )
        self.assertEqual(set(result[0]), {"transcript", "response"})
        search_parameters = cursor.executions[1][1]
        self.assertEqual(search_parameters["session_id"], TEST_SESSION_ID)
        # Recent turns are excluded from search by id, not by transcript text.
        self.assertEqual(search_parameters["exclude_ids"], [9, 10])
        self.assertIn("say", search_parameters["stop_lexemes"])

    def test_repeated_question_does_not_hide_older_answer(self) -> None:
        recent = [_turn(5, "What is my name?", minute=5)]
        older_same_text = [_turn(1, "What is my name?", "You said George.", minute=1)]
        cursor = FakeCursor([recent, older_same_text])

        with patch("database._connect", return_value=FakeConnection(cursor)):
            result = find_conversation_context(TEST_SESSION_ID, "What is my name?")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["response"], "You said George.")

    def test_budget_keeps_best_hit_and_newest_turn(self) -> None:
        recent = [_turn(i, f"recent {i}", "r" * 400, minute=i) for i in range(1, 5)]
        relevant = [_turn(100 + i, f"hit {i}", "h" * 400, minute=0) for i in range(5)]

        kept_recent, kept_relevant = fit_context_budget(recent, relevant, 250)

        self.assertEqual([t["id"] for t in kept_recent], [4])
        self.assertEqual([t["id"] for t in kept_relevant], [100])

    def test_budget_keeps_everything_when_it_fits(self) -> None:
        recent = [_turn(1, "a", minute=1), _turn(2, "b", minute=2)]
        relevant = [_turn(3, "c", minute=0)]

        kept_recent, kept_relevant = fit_context_budget(recent, relevant, 1000)

        self.assertEqual(kept_recent, recent)
        self.assertEqual(kept_relevant, relevant)

    def test_search_conversation_turns_returns_ranked_rows(self) -> None:
        ranked = [{"transcript": "Match", "response": "Answer"}]
        cursor = FakeCursor([ranked])

        with patch("database._connect", return_value=FakeConnection(cursor)):
            result = search_conversation_turns(TEST_SESSION_ID, "match", 5)

        self.assertEqual(result, ranked)
        parameters = cursor.executions[0][1]
        self.assertEqual(parameters["query"], "match")
        self.assertEqual(parameters["session_id"], TEST_SESSION_ID)
        # Keyword search fetches a fixed candidate pool, then trims to the limit.
        self.assertEqual(parameters["limit"], KEYWORD_CANDIDATE_LIMIT)

    def test_initialize_adds_embedding_columns_without_extensions(self) -> None:
        cursor = FakeCursor()
        with patch("database._connect", return_value=FakeConnection(cursor)):
            initialize_database()

        sql = "\n".join(query for query, _ in cursor.executions)
        self.assertIn("ADD COLUMN IF NOT EXISTS embedding REAL[]", sql)
        self.assertNotIn("CREATE EXTENSION", sql)

    def test_reciprocal_rank_fusion_prefers_turns_found_by_both(self) -> None:
        keyword = [_turn(1, "a"), _turn(2, "b"), _turn(3, "c")]
        vector = [_turn(4, "d"), _turn(2, "b")]

        fused = reciprocal_rank_fusion([keyword, vector])

        self.assertEqual([turn["id"] for turn in fused][:3], [2, 4, 1])

    def test_hybrid_context_uses_vector_hits_above_threshold(self) -> None:
        recent = [_turn(10, "Latest", minute=10)]
        keyword_hits: list[dict] = []
        vector_rows = [
            {**_turn(1, "My team is the Bulls.", minute=1), "embedding": [1.0, 0.0]},
            {**_turn(2, "Weather chat", minute=2), "embedding": [0.0, 1.0]},
        ]
        cursor = FakeCursor([recent, keyword_hits, vector_rows])

        with patch("database._connect", return_value=FakeConnection(cursor)):
            result = find_conversation_context(
                TEST_SESSION_ID,
                "Which squad do I root for?",
                query_embedding=[0.9, 0.1],
                embedding_model="nomic-embed-text",
            )

        self.assertEqual(
            [turn["transcript"] for turn in result],
            ["My team is the Bulls.", "Latest"],
        )
        vector_parameters = cursor.executions[2][1]
        self.assertEqual(vector_parameters["model"], "nomic-embed-text")
        self.assertEqual(vector_parameters["exclude_ids"], [10])

    def test_keyword_only_context_skips_vector_query(self) -> None:
        cursor = FakeCursor([[], []])
        with patch("database._connect", return_value=FakeConnection(cursor)):
            find_conversation_context(TEST_SESSION_ID, "hello")
        self.assertEqual(len(cursor.executions), 2)

    def test_backfill_embeds_missing_turns_in_batches(self) -> None:
        batch = [{"id": 7, "transcript": "Q", "response": "A"}]
        cursor = FakeCursor([batch, []])
        cursor.executemany = lambda query, rows: cursor.executions.append((query, rows))

        with (
            patch("database._connect", return_value=FakeConnection(cursor)),
            patch("database.embed_texts", return_value=[[0.5, 0.5]]) as embed,
        ):
            updated = backfill_embeddings(batch_size=1)

        self.assertEqual(updated, 1)
        embed.assert_called_once_with(["User: Q\nJarvis: A"], "document")
        self.assertEqual(cursor.executions[1][1], [([0.5, 0.5], "nomic-embed-text", 7)])

    def test_saves_turn_and_creates_session_if_needed(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)

        with patch("database._connect", return_value=connection):
            save_conversation_turn(TEST_SESSION_ID, "Question", "Answer")

        self.assertEqual(len(cursor.executions), 2)
        self.assertIn("INSERT INTO conversation_sessions", cursor.executions[0][0])
        self.assertEqual(cursor.executions[0][1], (TEST_SESSION_ID,))
        self.assertIn("INSERT INTO conversation_turns", cursor.executions[1][0])
        self.assertEqual(
            cursor.executions[1][1],
            (TEST_SESSION_ID, "Question", "Answer", None, None),
        )


if __name__ == "__main__":
    unittest.main()
