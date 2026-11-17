# Vision detector backend: overnight report (Phase 3, part 1)

Date: 2026-11-17. Machine: AMD Athlon Gold 3150U (2 cores), 5.9 GB RAM, no GPU,
Windows 11, Python 3.14.0.

Commits on `main`:

| Step | Commit | Message |
|---|---|---|
| 1. Research and selection | `19ca92f` | add phase 3 detector research and selection |
| 2. Detector module + download script | `c397249` | add yolox onnx detector module and weight download script |
| 3. `POST /api/detect` | `64ff1a4` | add object detection endpoint |
| 4. Benchmark | `df6a6a0` | add detection benchmark and optional detector unload |
| 5. Report (this file) + README | final commit | add vision detector backend report |

## 1. What I built

- **`POST /api/detect`**: send one JPEG or PNG photo, get back the objects in it
  (80 COCO classes such as person, laptop, cup, bottle, chair), each with a
  confidence and a box, plus a simple `person_present` yes/no.
- `backend/vision.py`: the detector. It decodes the image in memory, resizes it
  for the model ("letterbox"), runs YOLOX-Tiny with onnxruntime, turns the raw
  output into boxes, removes duplicate boxes (NMS), and normalizes boxes to 0-1.
  The model is loaded the first time it is needed and then reused, guarded by a
  lock (the same pattern as Whisper in `main.py`).
- `backend/vision_api.py`: the FastAPI router (keeps `main.py` thin; `main.py`
  only gained two lines to include it).
- `backend/download_vision_model.py`: downloads the weights into the git-ignored
  `backend/models/` folder and checks their SHA-256 before using them. I ran it;
  both `yolox_tiny.onnx` and `yolox_nano.onnx` are on this machine and verified.
- `backend/bench_vision.py`: measures latency (and RAM, given the server's process
  id) against a running backend.
- `backend/test_data/office_public_domain.jpg`: a public-domain FEMA photo used by
  the real-model test (license in `backend/test_data/README.md`).
- `docs/overnight/phase3-detector-research.md`: the model comparison.
- `.env.example`: `VISION_MODEL`, `VISION_CONFIDENCE`, `VISION_THREADS`,
  `VISION_MODELS_DIR`, `VISION_KEEP_LOADED`. All have defaults in code.
- `.gitignore`: `backend/models/`. README: setup step 5 and an "Object detection"
  section. `claude.md` (untracked) updated: status, API, settings, checklist, map.

## 2. Technologies

- **YOLOX-Tiny ONNX** (Megvii, **Apache-2.0**), the official release file
  `yolox_tiny.onnx` (20 MB). YOLOX-Nano (same license, 3.7 MB) is a supported
  lighter option.
- **onnxruntime 1.31.0** (MIT). It was already installed because faster-whisper
  depends on it, and it works on Python 3.14 / Windows, so the OpenCV fallback was
  not needed. I added it to `requirements.txt` explicitly because we now use it
  directly.
- **Pillow 13** (MIT-CMU, permissive): the only new package, used to read JPEG/PNG
  in memory. It has a Python 3.14 Windows wheel.
- NumPy (already installed) for preprocessing and decoding.

## 3. Decisions and reasoning

- **Why YOLOX-Tiny.** I downloaded seven candidate models, hashed them, and ran
  each on this laptop with two test photos. YOLOX-Tiny had the best published
  accuracy of the ones that worked (32.8 COCO mAP), the most confident and complete
  results on both photos (all three people, four laptops, bottle, cup, table), and
  about 0.15 s per image. Full comparison in the research doc.
- **Why not "YOLO" from Ultralytics.** YOLOv5/v8/v11 are AGPL-3.0 and YOLOv6/7/9 are
  GPL-3.0, which the project rules forbid. YOLOX is a YOLO-style model under
  Apache-2.0.
- **Keep the model loaded (default).** It costs about 75 MB of RAM, about 7% of
  what the chat model uses, and saves about 0.1-0.6 s per frame. Because RAM is
  tight, `VISION_KEEP_LOADED=off` unloads it after every request (measured below).
- **Presence-only people.** The model only knows the class "person". The API
  returns label, confidence, and box, and nothing else. There are no face crops,
  identities, or embeddings, and a test checks that no extra fields appear.
- **Images never touch disk.** The upload is read into memory (at most 5 MB + 1
  byte, so huge uploads are rejected without buffering them), decoded with
  Pillow from a `BytesIO`, and dropped. A test patches `open`/`os.open` to fail
  and runs the real decode path to prove it. Nothing about the image is logged.
- **Input checks.** Content type must be JPEG/PNG (or unspecified), and the bytes
  must actually decode as JPEG/PNG (a GIF renamed `.png` gets 415). Images over 40
  megapixels are rejected before the pixels are decoded (decompression-bomb guard).
- **Errors are honest.** Missing weights give 503 with the exact download command.
  A bad `VISION_*` setting gives 503 "misconfigured". An inference failure gives
  502. There is no fake empty result.
- **Boxes are normalized** (0-1, `x, y, width, height`, top-left origin) so the
  frontend can draw them over a video element of any size.
- **`inference_ms`** is the model forward pass only. It excludes decoding,
  letterboxing, post-processing, and the one-time model load. Wall time is
  measured separately below.

## 4. Tradeoffs and alternatives considered

- **PP-PicoDet-S 416** (Apache-2.0): about as accurate, faster (~85 ms), and found
  more small objects. I passed on it because the file is on a Baidu CDN (less
  reliable to download), it needs a more complex decoder, and its confidences were
  lower. It is the first thing to try if latency matters more.
- **YOLOX-Nano**: 3-4x faster (~30 ms inference, ~56 ms per request) and lighter,
  but less accurate (25.8 mAP). It is one setting away: `VISION_MODEL=yolox_nano`
  (then run the download script with `--model yolox_nano`; both are already
  downloaded here).
- **D-FINE-N** (Apache-2.0, 42.8 mAP on paper): the Hugging Face ONNX export found
  no people in the office photo with any preprocessing I tried, so I could not
  verify it. It is worth retrying with an official export.
- **NanoDet-Plus** (fast, less confident) and **SSD-MobileNet v1** (old, 33 s load)
  were also rejected.
- **OpenCV `cv2.dnn`** was not needed, because onnxruntime works on Python 3.14.
- **Unloading after every request** saves only about 45 MB (the allocator keeps
  some memory) and roughly doubles per-request time, so it is off by default.

## 5. Measurements

All on this laptop. Free physical RAM was only about 440 MB at idle tonight (VS
Code and other apps open; Ollama had no model loaded).

**Real backend** (`uvicorn` started as documented, `backend\bench_vision.py`,
office photo 960x742, 10 warm requests, memory = uvicorn worker working set):

| Setting | Cold first request | Warm wall time (median) | Inference (median) | Worker RAM |
|---|---|---|---|---|
| yolox_tiny (default), run 1 | 719 ms (152 ms inference) | 162 ms (139-214) | 134 ms | 84 MB → 160 MB (peak 167) = **+76 MB** |
| yolox_tiny, run 2 (fresh server) | 742 ms | 139 ms (123-156) | 114 ms | not measured (wrong pid) |
| yolox_tiny, `VISION_KEEP_LOADED=off` | 480 ms | 261 ms (248-274) | 117 ms | 116 MB between requests (peak 163) |
| yolox_nano | 318 ms | 56 ms (52-64) | 31 ms | 84 MB → 135 MB (peak 142) |

The cold request includes loading the model (~0.5 s).

**Effect on the voice/memory flow** (real `/api/respond` with real Ollama
`qwen2.5:1.5b` and PostgreSQL, throwaway session ids, deleted afterwards; a
background job sampled free RAM every 0.5 s):

| Condition | `/api/respond` time | Lowest free RAM during the answer |
|---|---|---|
| Detector not loaded ("Just say hello.") | 18.4 s | 136 MB |
| Detector loaded ("Just say hello.") | 19.0 s | 9 MB |
| Detector loaded (longer question) | 24.5 s | 10 MB |

Answer time did not change in a meaningful way, and there were no PostgreSQL
timeouts. Free RAM got very low, though. These are single runs on a machine with
other apps open, so the free-RAM numbers are noisy (the 127 MB gap is bigger than
the detector itself). Windows trims the idle detector's memory while qwen loads.
The first detection afterwards took 379 ms instead of ~160 ms while those pages
came back. If the morning demo feels sluggish, set `VISION_KEEP_LOADED=off` or
`VISION_MODEL=yolox_nano`, and close other apps.

**Accuracy (sanity check, not a benchmark).** Office photo: person 0.92 / 0.82 /
0.77 (all three people), laptops 0.88 / 0.86 / 0.69 / 0.44, bottle 0.90, cup 0.59,
dining table 0.52. Street photo: car 0.72, two persons 0.71 / 0.68 (the woman at
the far left was missed), and the child's bicycle labeled "motorcycle" 0.68. The
published COCO mAP is 32.8 (from the YOLOX README, not measured here).

## 6. Tests

Backend suite: **100 tests, all passing**
(`.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v`).

Mocked or synthetic (no model needed):
- `test_vision.py`: letterbox scale, padding (114), and BGR order; YOLOX decoding
  of a synthetic tensor into exact pixel boxes; objectness × class threshold; wrong
  output shape is an error; NMS suppress/keep; class-wise NMS plus clipping and
  normalization; JPEG/PNG accepted; empty, garbage, GIF, BMP, and truncated input
  rejected; pixel limit; detect with a fake session; inference error, missing
  weights (message names the download script), and unknown model; confidence
  setting validation; `VISION_KEEP_LOADED=off` unloads; the download script
  verifies, skips re-download, and discards on checksum mismatch.
- `test_vision_api.py` (FastAPI `TestClient` on the vision router): route present
  in `app.openapi()["paths"]`; exact response shape; person fields limited to
  label, confidence, and box; `person_present` false; 422 missing field; 400 empty
  or garbage; 413 oversized (detector not called); 415 wrong type and GIF bytes
  sent as PNG; 503 missing weights and bad setting; 502 inference failure; no file
  opened during a request (real decode path, fake session).

Real model (runs because the weights exist; skipped with a reason otherwise):
- `RealModelDetectTests`: the real YOLOX-Tiny on the public-domain office photo
  finds at least 2 people and a laptop, keeps boxes inside 0-1, and runs under
  1 s. A blank 640x480 image gives `person_present: false`.

Real-service checks I ran by hand (server started, then stopped):
- `/api/health` ok; `/api/detect` with the office photo returned 200 (results
  above); a GIF got 415, a 6 MB upload 413, an invalid image 400, and a missing
  field 422.
- `/api/respond` still works with the detector loaded (real Ollama and PostgreSQL;
  test sessions deleted, 204).

Not tested: a real webcam frame (no camera tonight), and Whisper transcription
with the detector loaded (no audio input; the transcription code was not changed).

## 7. Known limitations and open issues

- **The frontend does not call `/api/detect` yet.** The next agent owns that.
- COCO's 80 classes only (no "phone case", "pen", brands, or text). "Phone" is
  `cell phone`; "TV/monitor" is `tv`. Small or far-away objects are often missed,
  and similar classes get confused (the child's bicycle was labeled "motorcycle").
- Free RAM gets very low while qwen loads (see Measurements).
- **Found outside my scope (not fixed):** with no vision wired in, qwen answered
  "What can you see on my desk?" with "I can see your computer and your phone on
  your desk". That is a hallucination. Phase 4 should either pass real detections
  into the prompt or tell the model it cannot see.
- No auth or rate limit on `/api/detect` (local single-user only, like the rest).
- `inference_ms` excludes decode, pre/post-processing, and model load (about
  20-30 ms extra when warm).

## 8. Morning manual checklist

1. `.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`
   (adds Pillow; already installed here).
2. `.\.venv\Scripts\python.exe backend\download_vision_model.py` should print
   "already present and verified".
3. Start the backend, then run
   `curl.exe -F "file=@backend/test_data/office_public_domain.jpg;type=image/jpeg" http://127.0.0.1:8000/api/detect`.
   Expect `person_present: true`, persons, laptops, and a bottle.
4. Take a photo of your own desk with your phone or webcam app, save it as a JPEG,
   and send it the same way. Check the labels make sense. Then delete the photo
   yourself; the backend never saved it.
5. `.\.venv\Scripts\python.exe backend\bench_vision.py` should show a warm median
   of roughly 0.15 s.
6. Have a normal voice conversation in the UI. It should work exactly as before.

## 9. Handoff notes for `vision-integration-verifier`

**Endpoint**: `POST /api/detect` (through the Vite proxy as `/api/detect`).

Request: `multipart/form-data` with one field, **`file`**, containing a JPEG or PNG
of at most 5 MB. From the browser, `canvas.toBlob(cb, "image/jpeg", 0.85)` and then
`form.append("file", blob, "frame.jpg")`. Do not set `Content-Type` yourself; let
`fetch` add the multipart boundary.

Success, 200:

```json
{
  "detections": [
    {"label": "person", "confidence": 0.9177,
     "box": {"x": 0.4393, "y": 0.0581, "width": 0.5597, "height": 0.9419}},
    {"label": "laptop", "confidence": 0.8769,
     "box": {"x": 0.1535, "y": 0.2798, "width": 0.1935, "height": 0.2095}}
  ],
  "person_present": true,
  "image_width": 960,
  "image_height": 742,
  "inference_ms": 114.8
}
```

- `box` values are fractions of the **uploaded** image (0-1, top-left origin). To
  draw over a `<video>` with `object-fit: cover`, account for the crop. Multiply by
  the displayed size only if the frame and the display have the same aspect ratio.
- `detections` is sorted by confidence, at most 50, threshold `VISION_CONFIDENCE`
  (0.35). It can be empty (`[]`, `person_present: false`), which is not an error.
- `confidence` has 4 decimals, `inference_ms` has 1.
- Labels are COCO names: `person`, `cell phone`, `cup`, `bottle`, `laptop`, `tv`,
  `keyboard`, `mouse`, `book`, `chair`, and so on (full list in
  `vision.COCO_LABELS`).

Errors (all `{"detail": ...}`; 422 has FastAPI's list form):

| Status | When |
|---|---|
| 400 | empty file, or the bytes are not a readable JPEG/PNG |
| 413 | larger than 5 MB |
| 415 | content type, or the actual format, is not JPEG/PNG |
| 422 | no `file` field |
| 503 | weights missing (detail contains the download command) or bad `VISION_*` setting |
| 502 | inference failed |

**Weights**: `.\.venv\Scripts\python.exe backend\download_vision_model.py`
(`--model yolox_nano` or `--model all` for others, `--force` to re-download).
Files go to `backend\models\` (git-ignored) and are SHA-256 checked:
- `yolox_tiny.onnx` `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`
- `yolox_nano.onnx` `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`

**How to run and verify**:
- Backend: `.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000`
- Benchmark: `.\.venv\Scripts\python.exe backend\bench_vision.py --runs 10`. Add
  `--pid <uvicorn worker pid>` for RAM. The venv `python.exe` launches a child
  process, and the **child** is the worker (its parent is the launcher; a conhost
  child also appears, so filter on `Name='python.exe'`).
- Tests: `test_vision.py` and `test_vision_api.py`. The real-model test skips
  itself if weights are absent.

**Gotchas**:
- The first request after start (or after Windows trims memory) is slower:
  0.3-0.75 s, versus ~0.15 s warm.
- Requests are serialized by a lock (2-core CPU). Do not fire overlapping
  detections; send at most one frame at a time and wait for the answer.
- The frame is captured at ≤1024 px long edge today, which is fine (the model sees
  416x416 anyway). Sending ~640 px JPEGs would cut upload and decode time a little.
- Do not save frames anywhere (privacy rule); there is nothing to clean up on the
  server.
- UI: the conversation menu is top-right (`.session-menu`, z-index 2) and the
  camera LED is top-left. Keep detection overlays/labels clear of those, or put them
  below the menu's z-index.
- Phase 4: qwen currently claims to "see" things. Feed it the detection labels, or
  tell it it has no vision, before demoing visual questions.
