"""HTTP endpoint for local object detection: ``POST /api/detect``.

Input: ``multipart/form-data`` with one JPEG or PNG image in field ``file``
(at most 5 MB). The image is decoded in memory, run through the local YOLOX
model, and discarded. It is never written to disk, the database, or logs.

Output::

    {
      "detections": [
        {"label": "person", "confidence": 0.91,
         "box": {"x": 0.46, "y": 0.08, "width": 0.53, "height": 0.91}}
      ],
      "person_present": true,
      "image_width": 960,
      "image_height": 742,
      "inference_ms": 140.2
    }

Boxes are normalized to the original image (0-1, top-left origin). People are
reported as presence only: label "person", a box, and a confidence.

Why the body is parsed by hand: FastAPI's ``UploadFile`` parameter lets
Starlette spool every upload larger than 1 MB into a temporary file on disk,
and it buffers the whole body before the endpoint can check the size. Camera
frames must never touch disk, so this endpoint streams the body with a hard
size cap into an in-memory-only parser instead.
"""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

import vision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vision"])

MAX_IMAGE_BYTES = 5 * 1024 * 1024
# Multipart boundaries and part headers add a little on top of the image.
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png"}
TOO_LARGE_DETAIL = "Image must be 5 MB or smaller."
MISSING_FILE_DETAIL = [
    {"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}
]


class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionOut(BaseModel):
    label: str
    confidence: float
    box: Box


class DetectResponse(BaseModel):
    detections: list[DetectionOut]
    person_present: bool
    image_width: int
    image_height: int
    inference_ms: float


class _RequestTooLarge(Exception):
    pass


class _InMemoryMultiPartParser(MultiPartParser):
    # The body is capped at MAX_REQUEST_BYTES before parsing, so keeping the
    # whole file part in RAM is bounded. Starlette's default (1 MB) would roll
    # larger frames over into a temp file on disk.
    spool_max_size = MAX_REQUEST_BYTES + 1


async def _capped_body(request: Request) -> AsyncIterator[bytes]:
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_REQUEST_BYTES:
            raise _RequestTooLarge
        yield chunk


async def read_form_in_memory(request: Request, missing_detail: object = MISSING_FILE_DETAIL):
    """Parse a multipart body (at most one file, 5 MB + 64 KB in total) fully
    in memory. Also used by ``POST /api/respond/vision``. The caller must
    ``await form.close()``."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise HTTPException(status_code=422, detail=missing_detail)

    parser = _InMemoryMultiPartParser(
        request.headers, _capped_body(request), max_files=1, max_fields=10
    )
    try:
        return await parser.parse()
    except _RequestTooLarge:
        raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL) from None
    except MultiPartException as error:
        raise HTTPException(status_code=400, detail=f"Malformed upload: {error.message}") from error


async def _read_image_upload(request: Request) -> tuple[str, bytes]:
    """Return (content type, bytes) of the ``file`` field, entirely in memory."""
    form = await read_form_in_memory(request)
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail=MISSING_FILE_DETAIL)
        content_type = (upload.content_type or "").split(";")[0].strip().lower()
        data = await upload.read(MAX_IMAGE_BYTES + 1)
    finally:
        await form.close()
    return content_type, data


@router.post(
    "/detect",
    response_model=DetectResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "JPEG or PNG image, at most 5 MB.",
                            }
                        },
                    }
                }
            },
        }
    },
)
async def detect(request: Request) -> dict:
    content_type, data = await _read_image_upload(request)
    if content_type and content_type not in ALLOWED_CONTENT_TYPES | {"application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)

    try:
        result = await run_in_threadpool(vision.detect_bytes, data)
    except vision.ImageValidationError as error:
        status = 415 if isinstance(error, vision.UnsupportedImageFormatError) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except vision.ModelUnavailableError as error:
        logger.warning("Object detection model unavailable: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        # Bad VISION_* setting (e.g. VISION_CONFIDENCE outside 0-1).
        logger.error("Invalid object detection setting: %s", error)
        raise HTTPException(
            status_code=503,
            detail=f"Object detection is misconfigured: {error}",
        ) from error
    except (vision.DetectionError, RuntimeError, OSError) as error:
        logger.exception("Local object detection failed")
        raise HTTPException(
            status_code=502,
            detail="Local object detection failed. Check the backend log and the model file.",
        ) from error

    return result.as_dict()
