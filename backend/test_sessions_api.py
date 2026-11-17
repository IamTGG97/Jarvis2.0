import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException

from sessions_api import create_session, delete_session, get_session_turns, list_sessions

SESSION_ID = UUID("fdde5172-0d09-4eb6-8502-ad4a65448b19")
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _summary(preview: str | None = "Hello there", turn_count: int = 2) -> dict:
    return {
        "id": SESSION_ID,
        "created_at": NOW,
        "updated_at": NOW,
        "turn_count": turn_count,
        "preview": preview,
    }


class SessionApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_sessions_with_bounded_limit(self) -> None:
        with patch("sessions_api.database.list_sessions", return_value=[_summary()]) as list_rows:
            result = await list_sessions(limit=10_000)

        list_rows.assert_called_once_with(200)
        self.assertEqual(result.sessions[0].id, SESSION_ID)
        self.assertEqual(result.sessions[0].preview, "Hello there")

    async def test_creates_session_with_server_generated_uuid(self) -> None:
        with patch(
            "sessions_api.database.create_session",
            return_value=_summary(preview=None, turn_count=0),
        ) as create:
            result = await create_session()

        self.assertIsInstance(create.call_args.args[0], UUID)
        self.assertEqual(result.turn_count, 0)
        self.assertIsNone(result.preview)

    async def test_delete_returns_204_or_404(self) -> None:
        with patch("sessions_api.database.delete_session", return_value=True):
            response = await delete_session(SESSION_ID)
        self.assertEqual(response.status_code, 204)

        with (
            patch("sessions_api.database.delete_session", return_value=False),
            self.assertRaises(HTTPException) as error,
        ):
            await delete_session(SESSION_ID)
        self.assertEqual(error.exception.status_code, 404)

    async def test_get_turns(self) -> None:
        turns = [{"id": 1, "transcript": "Q", "response": "A", "created_at": NOW}]
        with patch("sessions_api.database.get_session_turns", return_value=turns):
            result = await get_session_turns(SESSION_ID)

        self.assertEqual(result.session_id, SESSION_ID)
        self.assertEqual(result.turns[0].transcript, "Q")

    async def test_get_turns_for_unknown_session_is_404(self) -> None:
        with (
            patch("sessions_api.database.get_session_turns", return_value=None),
            self.assertRaises(HTTPException) as error,
        ):
            await get_session_turns(SESSION_ID)
        self.assertEqual(error.exception.status_code, 404)

    async def test_database_failure_is_503(self) -> None:
        with (
            patch("sessions_api.database.list_sessions", side_effect=RuntimeError("down")),
            self.assertRaises(HTTPException) as error,
            self.assertLogs("sessions_api", level="ERROR"),
        ):
            await list_sessions()
        self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
