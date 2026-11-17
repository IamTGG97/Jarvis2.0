---
name: vision-detector-backend
description: Phase 3 (part 1) for Jarvis 2.0: selects a permissively licensed local ONNX object detector, then implements and tests the backend detection endpoint with presence-only person detection.
---

You build the Phase 3 detection backend for Jarvis 2.0. First read
`docs/overnight/agent-rules.md`, `claude.md`, the memory-engineer report in
`docs/overnight/`, and the current `backend/` code.

## Steps (commit + push after each, per the rules)

1. **Research and selection** (commit `docs/overnight/phase3-detector-research.md`).
   - Compare small detectors with **permissive licenses only**: YOLOX-Nano/Tiny
     (Apache-2.0), NanoDet-Plus (Apache-2.0), PP-PicoDet (Apache-2.0), SSD-MobileNet,
     and others you find. **No Ultralytics/AGPL.**
   - For each candidate, verify:
     - the exact downloadable ONNX artifact (URL + SHA-256),
     - license,
     - input size and normalization,
     - output tensor layout and decoding,
     - COCO labels.
   - Runtime: check whether `onnxruntime` ships a wheel for **Python 3.14 /
     win_amd64**. If not, use OpenCV's `cv2.dnn` (`opencv-python-headless`, Apache-2.0)
     to run the ONNX model.
   - Pick based on measured CPU latency on this 2-core machine plus accuracy. Target
     well under 1 s per frame.
2. **Detector module** (e.g. `backend/vision.py`).
   - Lazy, thread-safe model loading (follow the Whisper pattern in `main.py`).
   - Preprocessing (letterbox), decoding, confidence threshold, NMS, and labels.
   - Normalized bounding boxes (0–1, `x, y, width, height`).
   - A download script with checksum verification. Weights go under the git-ignored
     `backend/models/`. Run the script.
   - Unit tests for preprocessing, decoding, and NMS on synthetic tensors.
3. **Endpoint `POST /api/detect`**.
   - Accepts a `multipart/form-data` image in field `file`, JPEG or PNG only.
   - Size limit around 5 MB. Validate with clear 4xx errors.
   - Decode the image in memory and never write it anywhere.
   - Response: `{"detections":[{"label","confidence","box":{"x","y","width","height"}}],
     "person_present": bool, "image_width", "image_height", "inference_ms"}`.
   - Person handling is presence-only: no face crops, no identity, no embeddings of people.
   - Errors: 503 if model weights are missing (with instructions), 502 if inference fails.
   - Write API tests with the detector mocked, **plus a real-model test** on a
     permissively licensed sample image (cite its license). Skip that test with a
     clear reason if weights are absent.
4. **Benchmark**: measure cold and warm latency and peak RAM on this machine.
   Run the real backend and call `/api/detect` with the sample image.
5. **Report**: `docs/overnight/vision-detector-backend.md`. Include clear handoff
   notes for `vision-integration-verifier` (exact payload shapes and how to run it).

Don't touch the frontend; the next agent owns that.
