import asyncio
import os
import unittest
from unittest.mock import patch

import database
from retention import run_retention_once, start_retention_task


class RetentionTests(unittest.IsolatedAsyncioTestCase):
    def test_retention_is_off_by_default(self) -> None:
        with patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": ""}):
            self.assertIsNone(database.retention_days())
        with patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": "0"}):
            self.assertIsNone(database.retention_days())

    def test_invalid_retention_value_is_an_error(self) -> None:
        with (
            patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": "thirty"}),
            self.assertRaisesRegex(RuntimeError, "whole number"),
        ):
            database.retention_days()

    def test_run_once_purges_when_enabled(self) -> None:
        with (
            patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": "30"}),
            patch("retention.database.purge_expired_sessions", return_value=2) as purge,
        ):
            self.assertEqual(run_retention_once(), 2)
        purge.assert_called_once_with(30)

    def test_run_once_does_nothing_when_off(self) -> None:
        with (
            patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": ""}),
            patch("retention.database.purge_expired_sessions") as purge,
        ):
            self.assertIsNone(run_retention_once())
        purge.assert_not_called()

    async def test_task_only_starts_when_enabled(self) -> None:
        with patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": ""}):
            self.assertIsNone(start_retention_task())

        with (
            patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": "7"}),
            patch("retention.database.purge_expired_sessions", return_value=0) as purge,
            self.assertLogs("retention", level="INFO"),
        ):
            task = start_retention_task()
            self.assertIsNotNone(task)
            for _ in range(100):
                if purge.called:
                    break
                await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        purge.assert_called_with(7)

    def test_purge_rejects_non_positive_days(self) -> None:
        with self.assertRaises(ValueError):
            database.purge_expired_sessions(0)


class SessionDatabaseTests(unittest.TestCase):
    def test_delete_session_reports_whether_it_existed(self) -> None:
        from test_database import FakeConnection, FakeCursor

        cursor = FakeCursor()
        cursor.rowcount = 1
        with patch("database._connect", return_value=FakeConnection(cursor)):
            self.assertTrue(database.delete_session(database.UUID(int=1)))
        self.assertIn("DELETE FROM conversation_sessions", cursor.executions[0][0])

    def test_get_session_turns_returns_none_for_unknown_session(self) -> None:
        from test_database import FakeConnection, FakeCursor

        cursor = FakeCursor()
        cursor.fetchone = lambda: None
        with patch("database._connect", return_value=FakeConnection(cursor)):
            self.assertIsNone(database.get_session_turns(database.UUID(int=1)))

    def test_list_sessions_includes_preview_and_count(self) -> None:
        from test_database import FakeConnection, FakeCursor

        cursor = FakeCursor([[{"id": 1}]])
        with patch("database._connect", return_value=FakeConnection(cursor)):
            self.assertEqual(database.list_sessions(5), [{"id": 1}])
        query, parameters = cursor.executions[0]
        self.assertIn("turn_count", query)
        self.assertIn("preview", query)
        self.assertEqual(parameters["limit"], 5)


if __name__ == "__main__":
    unittest.main()
