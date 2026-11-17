"""Local object detection with a YOLOX ONNX model (Apache-2.0) on onnxruntime.

Person handling is presence-only: the detector returns the COCO label "person"
and a box, nothing else. There are no face crops, identities, or embeddings,
and images are decoded in memory only (never written to disk or logged).

See docs/overnight/phase3-detector-research.md for why YOLOX was chosen and how
its output tensor is laid out.
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Exact artifacts verified on 2026-11-17 (see the research doc).
MODELS: dict[str, dict[str, str | int]] = {
    "yolox_tiny": {
        "filename": "yolox_tiny.onnx",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        "input_size": 416,
    },
    "yolox_nano": {
        "filename": "yolox_nano.onnx",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "sha256": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        "input_size": 416,
    },
}
DEFAULT_MODEL = "yolox_tiny"

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
PERSON_LABEL = "person"

STRIDES = (8, 16, 32)
PAD_VALUE = 114
NMS_IOU = 0.45
MAX_DETECTIONS = 50
# Reject huge images before decoding pixels (decompression-bomb guard).
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(ValueError):
    """The upload is not a usable JPEG/PNG image (maps to HTTP 4xx)."""


class ModelUnavailableError(RuntimeError):
    """Model weights are missing or the model name is unknown (maps to 503)."""


class DetectionError(RuntimeError):
    """Inference failed on a valid image (maps to 502)."""


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # Normalized to the original image: 0-1, top-left origin.
    x: float
    y: float
    width: float
    height: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "width": round(self.width, 4),
                "height": round(self.height, 4),
            },
        }


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float

    @property
    def person_present(self) -> bool:
        return any(d.label == PERSON_LABEL for d in self.detections)

    def as_dict(self) -> dict:
        return {
            "detections": [d.as_dict() for d in self.detections],
            "person_present": self.person_present,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 1),
        }


# ---------------------------------------------------------------- settings


def model_name() -> str:
    return os.getenv("VISION_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def confidence_threshold() -> float:
    raw = os.getenv("VISION_CONFIDENCE", "").strip()
    if not raw:
        return 0.35
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError("VISION_CONFIDENCE must be between 0 and 1.")
    return value


def models_dir() -> Path:
    raw = os.getenv("VISION_MODELS_DIR", "").strip()
    return Path(raw) if raw else MODELS_DIR


def model_spec(name: str) -> dict[str, str | int]:
    try:
        return MODELS[name]
    except KeyError as error:
        raise ModelUnavailableError(
            f"Unknown VISION_MODEL '{name}'. Choose one of: {', '.join(sorted(MODELS))}."
        ) from error


def model_path(name: str) -> Path:
    return models_dir() / str(model_spec(name)["filename"])


def missing_weights_message(name: str) -> str:
    return (
        f"Object detection model weights are missing ({model_path(name)}). "
        f"Run: .\\.venv\\Scripts\\python.exe backend\\download_vision_model.py --model {name}"
    )


# ---------------------------------------------------------------- image input


def decode_image(data: bytes) -> Image.Image:
    """Decode JPEG/PNG bytes in memory into an RGB image. Nothing touches disk."""
    if not data:
        raise ImageValidationError("The uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(data))
        image_format = image.format
        width, height = image.size
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError("The upload is not a readable JPEG or PNG image.") from error
    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError("Only JPEG and PNG images are supported.")
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            f"Image dimensions {width}x{height} are not supported "
            f"(maximum {MAX_IMAGE_PIXELS:,} pixels)."
        )
    try:
        return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImageValidationError("The image data is corrupt or truncated.") from error


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float]:
    """YOLOX preprocessing.

    Resize keeping aspect ratio so the image fits in ``size``x``size``, paste it
    at the top-left of a canvas filled with 114, convert RGB to BGR, keep raw
    0-255 values (no mean/std), and return a ``[1, 3, size, size]`` float32
    tensor plus the resize ratio needed to map boxes back.
    """
    width, height = image.size
    ratio = min(size / width, size / height)
    new_w = max(1, round(width * ratio))
    new_h = max(1, round(height * ratio))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.float32)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas[:new_h, :new_w] = np.asarray(resized, dtype=np.float32)
    bgr = canvas[:, :, ::-1]
    tensor = np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])
    return tensor, ratio


# ---------------------------------------------------------------- decoding


def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for stride in STRIDES:
        cells = size // stride
        ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grids.append(np.stack((xs, ys), axis=-1).reshape(-1, 2))
        strides.append(np.full((cells * cells, 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_yolox(
    output: np.ndarray,
    input_size: int,
    ratio: float,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn raw YOLOX output ``[N, 85]`` into boxes, scores, and class ids.

    Rows are grid cells for strides 8, 16, 32 in row-major order. Columns are
    ``tx, ty, tw, th, objectness, 80 class scores`` (already sigmoid). Returned
    boxes are ``x1, y1, x2, y2`` in original-image pixels, filtered by
    ``objectness * class_score >= conf_threshold``.
    """
    rows = output.reshape(-1, output.shape[-1])
    grids, strides = _grids(input_size)
    if rows.shape[0] != grids.shape[0] or rows.shape[1] != 5 + len(COCO_LABELS):
        raise DetectionError(
            f"Unexpected YOLOX output shape {tuple(output.shape)} for input size {input_size}."
        )
    centers = (rows[:, :2] + grids) * strides
    sizes = np.exp(np.clip(rows[:, 2:4], -20, 20)) * strides
    class_scores = rows[:, 4:5] * rows[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= conf_threshold
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)[keep] / ratio
    return boxes, scores[keep], class_ids[keep]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = NMS_IOU) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best score first."""
    order = scores.argsort()[::-1]
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    """Class-wise NMS, clip to the image, normalize to 0-1, sort by confidence."""
    detections: list[Detection] = []
    for class_id in np.unique(class_ids):
        idx = np.where(class_ids == class_id)[0]
        for k in nms(boxes[idx], scores[idx]):
            x1, y1, x2, y2 = boxes[idx[k]]
            x1 = float(np.clip(x1, 0, image_width)) / image_width
            x2 = float(np.clip(x2, 0, image_width)) / image_width
            y1 = float(np.clip(y1, 0, image_height)) / image_height
            y2 = float(np.clip(y2, 0, image_height)) / image_height
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    label=COCO_LABELS[int(class_id)],
                    confidence=float(scores[idx[k]]),
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections[:MAX_DETECTIONS]


# ---------------------------------------------------------------- model


_session = None
_session_name: str | None = None
_session_lock = threading.Lock()


def _load_session(name: str):
    """Load the ONNX session once (lazy, thread-safe; same pattern as Whisper)."""
    global _session, _session_name
    if _session is not None and _session_name == name:
        return _session
    path = model_path(name)
    if not path.is_file():
        raise ModelUnavailableError(missing_weights_message(name))

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = int(os.getenv("VISION_THREADS", "2"))
    options.inter_op_num_threads = 1
    _session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    _session_name = name
    return _session


def unload_model() -> None:
    """Drop the loaded session (used by tests and to free RAM)."""
    global _session, _session_name
    with _session_lock:
        _session = None
        _session_name = None


def detect(image: Image.Image) -> DetectionResult:
    """Run detection on an RGB image and return normalized detections."""
    name = model_name()
    input_size = int(model_spec(name)["input_size"])
    threshold = confidence_threshold()
    width, height = image.size
    tensor, ratio = letterbox(image, input_size)
    with _session_lock:
        session = _load_session(name)
        started = time.perf_counter()
        try:
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: tensor})[0]
        except Exception as error:  # onnxruntime raises its own exception types
            raise DetectionError(f"ONNX inference failed: {type(error).__name__}") from error
        inference_ms = (time.perf_counter() - started) * 1000
    boxes, scores, class_ids = decode_yolox(output, input_size, ratio, threshold)
    detections = postprocess(boxes, scores, class_ids, width, height)
    return DetectionResult(detections, width, height, inference_ms)


def detect_bytes(data: bytes) -> DetectionResult:
    return detect(decode_image(data))
