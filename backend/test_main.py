from io import BytesIO
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException, UploadFile

from main import RespondRequest, app, respond, transcribe


class SpeechApiTests(unittest.IsolatedAsyncioTestCase):
    def test_routes_are_registered(self) -> None:
        route_paths = {route.path for route in app.routes}

        self.assertIn("/api/health", route_paths)
        self.assertIn("/api/transcribe", route_paths)
        self.assertIn("/api/respond", route_paths)

    async def test_rejects_empty_audio(self) -> None:
        upload = UploadFile(file=BytesIO(b""), filename="empty.webm")

        with self.assertRaises(HTTPException) as error:
            await transcribe(upload)

        self.assertEqual(error.exception.status_code, 400)

    async def test_transcribes_audio_locally(self) -> None:
        upload = UploadFile(file=BytesIO(b"audio"), filename="utterance.webm")
        with patch("main._transcribe_audio", return_value="recognized words"):
            result = await transcribe(upload)

        self.assertEqual(result, {"transcript": "recognized words"})

    async def test_rejects_blank_transcript(self) -> None:
        with self.assertRaises(HTTPException) as error:
            await respond(RespondRequest(transcript="  \n "))

        self.assertEqual(error.exception.status_code, 400)

    async def test_generates_response_with_local_ollama(self) -> None:
        request_data: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            request_data["url"] = str(request.url)
            request_data["body"] = await request.aread()
            return httpx.Response(
                200,
                json={"message": {"content": "Hello there."}},
                request=request,
            )

        local_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with (
            patch.dict(
                os.environ,
                {
                    "OLLAMA_BASE_URL": "http://localhost:11434",
                    "OLLAMA_MODEL": "qwen2.5:1.5b",
                },
            ),
            patch("main.httpx.AsyncClient", return_value=local_client),
        ):
            result = await respond(RespondRequest(transcript="  Say hello.  "))

        self.assertEqual(result, {"response": "Hello there."})
        self.assertEqual(request_data["url"], "http://localhost:11434/api/chat")
        self.assertIn(b'"model":"qwen2.5:1.5b"', request_data["body"])
        self.assertIn(b'"content":"Say hello."', request_data["body"])

    async def test_reports_ollama_not_running(self) -> None:
        with (
            patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://127.0.0.1:1"}),
            patch("main.httpx.AsyncClient", side_effect=httpx.ConnectError("offline")),
            self.assertRaises(HTTPException) as error,
        ):
            await respond(RespondRequest(transcript="Hello"))

        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("Ollama is not running", error.exception.detail)
if __name__ == "__main__":
    unittest.main()
