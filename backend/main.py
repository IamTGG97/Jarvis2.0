import asyncio
from contextlib import asynccontextmanager
import io
import json
import logging
import os
import threading
import time
from typing import AsyncIterator
from uuid import UUID

import av
import httpx
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUpload

import reasoning
import vision
from database import initialize_database
from memory import remember_turn, retrieve_memory
from memory_prompt import build_chat_messages
from retention import start_retention_task
from sessions_api import router as sessions_router
from vision_api import (
    ALLOWED_CONTENT_TYPES,
    MAX_IMAGE_BYTES,
    TOO_LARGE_DETAIL,
    read_form_in_memory,
)
from vision_api import router as vision_router

load_dotenv()

logger = logging.getLogger(__name__)
transcription_model = None
transcription_lock = threading.Lock()
# One vision-language model call at a time: two would not fit in RAM.
vlm_lock = asyncio.Lock()


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
    retention_task = start_retention_task()
    try:
        yield
    finally:
        if retention_task is not None:
            retention_task.cancel()


app = FastAPI(title="Jarvis2.0 Local API", lifespan=lifespan)
app.include_router(sessions_router)
app.include_router(vision_router)


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
    vision_context: str | None = None,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    messages = build_chat_messages(transcript, conversation_context, vision_context)

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


async def _retrieve(session_id: UUID, transcript: str):
    try:
        return await run_in_threadpool(retrieve_memory, session_id, transcript)
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Could not retrieve conversation memory")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Check PostgreSQL and DB_* settings.",
        ) from error


async def _save(session_id: UUID, transcript: str, response: str) -> str | None:
    """Save the text turn only (never the image). Returns an embedding warning."""
    try:
        return await run_in_threadpool(remember_turn, session_id, transcript, response)
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Could not save conversation turn")
        raise HTTPException(
            status_code=503,
            detail="Jarvis responded but could not save this turn to conversation memory.",
        ) from error


async def _answer_with_vlm(
    transcript: str,
    decision: reasoning.RouteDecision,
    image,
    detections: list[reasoning.Detection],
) -> reasoning.Answer:
    focus, note = reasoning.choose_vlm_focus(decision, detections)
    region = reasoning.crop_for(image, focus)
    cropped = region is not image
    try:
        settings = reasoning.vlm_settings()
        image_b64 = await run_in_threadpool(reasoning.encode_for_vlm, region, settings["max_edge"])
        prompt = reasoning.build_vlm_prompt(transcript, focus, cropped)
        async with vlm_lock:
            raw = await reasoning.ask_vlm(prompt, image_b64, settings)
            if not raw:
                # Small VLMs sometimes stop at once on terse questions; one
                # rephrased retry, then give up honestly.
                raw = await reasoning.ask_vlm(
                    reasoning.retry_prompt(transcript, focus), image_b64, settings
                )
    except reasoning.VlmError as error:
        logger.warning("Vision-language model failed: %s", error)
        fallback = reasoning.vlm_failure_answer(decision, detections, str(error))
        if fallback is None:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        return fallback
    if not raw:
        message = "The vision model gave no answer for this picture."
        fallback = reasoning.vlm_failure_answer(decision, detections, message)
        if fallback is None:
            raise HTTPException(status_code=502, detail=message + " Try asking another way.")
        return fallback
    text = reasoning.tidy_vlm_answer(raw)
    if note:
        text = f"{note} {text}"
    return reasoning.Answer(
        text,
        "vlm",
        True,
        [d.as_reference() for d in focus],
        "medium",
        decision.reason + (" (cropped to the object)" if cropped else ""),
    )


async def _answer(
    session_id: UUID,
    transcript: str,
    image=None,
    detections: list[reasoning.Detection] | None = None,
    vision_warning: str | None = None,
) -> tuple[reasoning.Answer, str | None]:
    """Route the question and answer it. Returns (answer, memory warning).

    Memory: the text route retrieves notes as before. Vision routes answer from
    the frame only (small VLMs get confused by extra text, and retrieval adds
    seconds), but every route saves the text turn so later questions like
    "what color did you say my mug was?" work. Images are never saved.
    """
    decision = reasoning.route_question(transcript, image is not None, detections)
    memory_warning = None
    answer: reasoning.Answer | None = None

    if decision.route == "detections":
        outcome = reasoning.answer_from_detections(decision, detections or [])
        if outcome.escalate and image is not None:
            decision.reason = f"escalated to VLM: {outcome.reason}"
            decision.route = "vlm"
        else:
            answer = outcome.answer

    if decision.route == "vlm":
        answer = await _answer_with_vlm(transcript, decision, image, detections or [])
    elif decision.route == "no_vision":
        answer = reasoning.honest_no_vision(decision)
    elif decision.route == "text":
        memory = await _retrieve(session_id, transcript)
        memory_warning = memory.warning
        response = await _generate_response(
            transcript, memory.turns, reasoning.vision_summary(detections)
        )
        answer = reasoning.Answer(response, "text", False, [], None, decision.reason)

    assert answer is not None
    answer.vision_warning = answer.vision_warning or vision_warning
    save_warning = await _save(session_id, transcript, answer.response)
    return answer, memory_warning or save_warning


@app.post("/api/respond")
async def respond(request: RespondRequest) -> dict[str, str | None]:
    """Answer with conversation memory (text only, no camera data).

    Returns ``{"response": str, "memory_warning": str | None}``. The warning is
    set when semantic (embedding) memory was unavailable for this turn, so the
    UI can say that recall was keyword-only instead of pretending it worked.
    Clearly visual questions ("what do you see?") get an honest "I can't see
    right now" here; use ``POST /api/respond/vision`` to send the camera frame
    and detections.
    """
    transcript = request.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")
    answer, memory_warning = await _answer(request.session_id, transcript)
    return {"response": answer.response, "memory_warning": memory_warning}


VISION_RESPOND_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["transcript", "session_id"],
                    "properties": {
                        "transcript": {"type": "string"},
                        "session_id": {"type": "string", "format": "uuid"},
                        "detections": {
                            "type": "string",
                            "description": "Optional JSON: the /api/detect result for this frame "
                            "(or just its detections list).",
                        },
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "Optional JPEG or PNG camera frame, at most 5 MB. "
                            "Used in memory only, never stored.",
                        },
                    },
                }
            }
        },
    }
}


def _validation_error(field: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=[{"type": "value_error", "loc": ["body", field], "msg": message, "input": None}],
    )


MISSING_TRANSCRIPT = [
    {"type": "missing", "loc": ["body", "transcript"], "msg": "Field required", "input": None}
]


@app.post("/api/respond/vision", openapi_extra=VISION_RESPOND_BODY)
async def respond_vision(request: Request) -> dict:
    """Answer a spoken question using the camera frame and/or detections.

    ``multipart/form-data`` fields: ``transcript`` and ``session_id``
    (required), ``detections`` (optional JSON string, the /api/detect result)
    and ``file`` (optional JPEG/PNG frame). The frame is parsed and decoded in
    memory and dropped after the answer; it is never written to disk, the
    database, or logs. If an image comes without detections, the local
    detector runs on it first.
    """
    started = time.perf_counter()
    form = await read_form_in_memory(request, missing_detail=MISSING_TRANSCRIPT)
    try:
        transcript = form.get("transcript")
        session_raw = form.get("session_id")
        detections_raw = form.get("detections")
        upload = form.get("file")
        if not isinstance(transcript, str):
            raise _validation_error("transcript", "Field required")
        if not isinstance(session_raw, str):
            raise _validation_error("session_id", "Field required")
        if detections_raw is not None and not isinstance(detections_raw, str):
            raise _validation_error("detections", "Must be a JSON text field")
        image_bytes = None
        if isinstance(upload, StarletteUpload):
            content_type = (upload.content_type or "").split(";")[0].strip().lower()
            if content_type and content_type not in ALLOWED_CONTENT_TYPES | {"application/octet-stream"}:
                raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")
            image_bytes = await upload.read(MAX_IMAGE_BYTES + 1)
        elif upload is not None:
            raise _validation_error("file", "Must be a file upload")
    finally:
        await form.close()

    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")
    try:
        session_id = UUID(session_raw.strip())
    except ValueError:
        raise _validation_error("session_id", "Input should be a valid UUID") from None

    detections = None
    if detections_raw and detections_raw.strip():
        try:
            detections = reasoning.parse_detections(json.loads(detections_raw))
        except (ValueError, RecursionError) as error:  # JSONDecodeError is a ValueError
            raise _validation_error("detections", f"Invalid detections: {error}") from None

    image = None
    vision_warning = None
    if image_bytes is not None:
        if not image_bytes:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)
        try:
            image = await run_in_threadpool(vision.decode_image, image_bytes)
        except vision.ImageValidationError as error:
            status = 415 if isinstance(error, vision.UnsupportedImageFormatError) else 400
            raise HTTPException(status_code=status, detail=str(error)) from error
        del image_bytes
        if detections is None:
            try:
                result = await run_in_threadpool(vision.detect, image)
                detections = reasoning.parse_detections(result.as_dict())
            except (vision.ModelUnavailableError, vision.DetectionError, ValueError,
                    RuntimeError, OSError) as error:
                # The VLM can still look at the image; say detection is missing.
                logger.warning("Object detection failed inside respond/vision: %s", error)
                vision_warning = f"Object detection was unavailable: {error}"

    answer, memory_warning = await _answer(
        session_id, transcript, image, detections, vision_warning
    )
    return {
        "response": answer.response,
        "memory_warning": memory_warning,
        "route": answer.route,
        "used_vision": answer.used_vision,
        "objects_referenced": answer.objects_referenced,
        "confidence": answer.confidence,
        "reason": answer.reason,
        "vision_warning": answer.vision_warning,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
