import os
import unittest
from unittest.mock import patch
from uuid import UUID

from database import (
    find_conversation_context,
    initialize_database,
    save_conversation_turn,
    search_conversation_turns,
)

TEST_SESSION_ID = UUID("fdde5172-0d09-4eb6-8502-ad4a65448b19")


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

    def test_creates_session_turn_and_search_tables(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)

        with patch("database._connect", return_value=connection):
            initialize_database()

        sql = "\n".join(query for query, _ in cursor.executions)
        self.assertIn("CREATE TABLE IF NOT EXISTS conversation_sessions", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS conversation_turns", sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS conversation_turns_search_idx", sql)

    def test_search_returns_recent_and_distinct_relevant_turns(self) -> None:
        recent_newest_first = [
            {"transcript": "Latest question", "response": "Latest answer"},
            {"transcript": "Recent question", "response": "Recent answer"},
        ]
        search_ranked = [
            {"transcript": "Latest question", "response": "Latest answer"},
            {"transcript": "Earlier matching question", "response": "Earlier answer"},
        ]
        cursor = FakeCursor([recent_newest_first, search_ranked])
        connection = FakeConnection(cursor)

        with patch("database._connect", return_value=connection):
            result = find_conversation_context(TEST_SESSION_ID, "matching question")

        self.assertEqual(
            result,
            [
                {"transcript": "Recent question", "response": "Recent answer"},
                {"transcript": "Latest question", "response": "Latest answer"},
                {
                    "transcript": "Earlier matching question",
                    "response": "Earlier answer",
                },
            ],
        )
        self.assertIn("plainto_tsquery", cursor.executions[1][0])
        self.assertEqual(cursor.executions[1][1][1], TEST_SESSION_ID)

    def test_search_conversation_turns_returns_ranked_rows(self) -> None:
        ranked = [{"transcript": "Match", "response": "Answer"}]
        cursor = FakeCursor([ranked])

        with patch("database._connect", return_value=FakeConnection(cursor)):
            result = search_conversation_turns(TEST_SESSION_ID, "match", 5)

        self.assertEqual(result, ranked)
        self.assertEqual(cursor.executions[0][1], ("match", TEST_SESSION_ID, 5))

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
            (TEST_SESSION_ID, "Question", "Answer"),
        )


if __name__ == "__main__":
    unittest.main()
