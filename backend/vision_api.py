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
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import vision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vision"])

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png"}


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


@router.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> dict:
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES | {"application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )

    # Read one byte past the limit so oversized uploads are rejected without
    # loading arbitrarily large bodies into memory.
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

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
