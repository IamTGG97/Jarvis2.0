from contextlib import asynccontextmanager
import io
import logging
import os
import threading
from typing import AsyncIterator
from uuid import UUID

import av
import httpx
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from database import find_conversation_context, initialize_database, save_conversation_turn
from memory_prompt import build_chat_messages

load_dotenv()

logger = logging.getLogger(__name__)
transcription_model = None
transcription_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await run_in_threadpool(initialize_database)
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Could not initialize PostgreSQL conversation memory")
        raise RuntimeError(
            "Could not connect to PostgreSQL or initialize the Jarvis memory tables. "
            "Check DB_* settings in the backend .env file."
        ) from error
    yield


app = FastAPI(title="Jarvis2.0 Local API", lifespan=lifespan)


class RespondRequest(BaseModel):
    transcript: str
    session_id: UUID


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _transcribe_audio(audio: bytes) -> str:
    from faster_whisper import WhisperModel, decode_audio

    global transcription_model
    with transcription_lock:
        if transcription_model is None:
            model_name = os.getenv("WHISPER_MODEL", "tiny.en")
            transcription_model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
            )

        samples = decode_audio(io.BytesIO(audio), sampling_rate=16_000)
        segments, _ = transcription_model.transcribe(
            samples,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")

    try:
        transcript = await run_in_threadpool(_transcribe_audio, audio)
    except ImportError as error:
        logger.exception("Local speech recognition dependencies are unavailable")
        raise HTTPException(
            status_code=503,
            detail="Local speech recognition is not installed. Install backend requirements.",
        ) from error
    except (
        OSError,
        RuntimeError,
        av.error.FFmpegError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
    ) as error:
        logger.exception("Local speech recognition failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Local transcription failed. Check that the speech model downloaded "
                "successfully and that the audio format is supported."
            ),
        ) from error

    return {"transcript": transcript}


def _ollama_chat_options() -> dict[str, float | int]:
    options: dict[str, float | int] = {"num_ctx": 2048, "num_predict": 160}
    temperature = os.getenv("OLLAMA_TEMPERATURE", "").strip()
    if temperature:
        options["temperature"] = float(temperature)
    return options


async def _generate_response(
    transcript: str,
    conversation_context: list[dict[str, str]],
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    messages = build_chat_messages(transcript, conversation_context)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            result = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": messages,
                    "options": _ollama_chat_options(),
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "0"),
                },
            )
            result.raise_for_status()
            answer = result.json().get("message", {}).get("content", "").strip()
    except httpx.ConnectError as error:
        logger.warning("Could not connect to the local Ollama service")
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start Ollama and make sure its local service is available.",
        ) from error
    except httpx.TimeoutException as error:
        logger.exception("Ollama response timed out")
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond. Try again or use a smaller local model.",
        ) from error
    except httpx.HTTPStatusError as error:
        logger.exception("Ollama returned an unsuccessful response")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned HTTP {error.response.status_code}. "
                f"Make sure the '{model}' model is installed with `ollama pull {model}`."
            ),
        ) from error
    except ValueError as error:
        logger.exception("Ollama returned invalid JSON")
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response.",
        ) from error

    if not answer:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
    return answer


@app.post("/api/respond")
async def respond(request: RespondRequest) -> dict[str, str]:
    transcript = request.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")

    try:
        conversation_context = await run_in_threadpool(
            find_conversation_context,
            request.session_id,
            transcript,
        )
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Could not retrieve conversation memory")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Check PostgreSQL and DB_* settings.",
        ) from error

    response = await _generate_response(transcript, conversation_context)

    try:
        await run_in_threadpool(
            save_conversation_turn,
            request.session_id,
            transcript,
            response,
        )
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Could not save conversation turn")
        raise HTTPException(
            status_code=503,
            detail="Jarvis responded but could not save this turn to conversation memory.",
        ) from error

    return {"response": response}