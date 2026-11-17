# Vision integration verifier: overnight report (Phase 3, part 2)

Date: 2026-11-17. Machine: AMD Athlon Gold 3150U (2 cores), 5.9 GB RAM, no GPU,
Windows 11, Python 3.14.0, Node 17.3, Chrome (headless, for checks only).

Commits on `main`:

| Step | Commit | Message |
|---|---|---|
| 1. Independent review + backend fixes | `c1b4cb0` | fix detector upload spooling, bomb, mpo, exif and corrupt weight handling |
| 2. Frontend wiring | `b0de2d9` | wire captured frame to object detection and show results |
| 3. Verification + frontend tests | `3a21382` | add dependency-free frontend tests for detection helpers |
| 4. Report (this file) + README | final commit | add vision integration verifier report |

## 1. What I built

**Backend fixes (step 1).** I re-read the detector agent's code and re-ran
everything myself instead of trusting its report. The core detector is correct:
box decoding, grid order, letterbox scaling, and NMS all check out (I drew the
boxes onto the test photo and they sit on the right objects, and a mirrored copy
of the photo gives mirrored boxes). I found and fixed five real bugs:

| Bug | What happened | Fix |
|---|---|---|
| **Uploads over 1 MB were written to disk** | FastAPI's `UploadFile` lets Starlette spool any file part over 1 MB into a temporary file on disk. That breaks the "camera frames never touch disk" rule. The old "never written to disk" test only used a tiny image, so it never triggered this. It also buffered the *whole* body (even 6 MB) before checking the size. | `vision_api.py` now streams the body with a hard cap (5 MB + 64 KB) into an in-memory-only multipart parser. Oversized bodies get 413 as soon as the cap is crossed, or straight away from `Content-Length`. |
| **Decompression bomb gave HTTP 500** | A 24 KB PNG claiming 20000x10000 pixels makes Pillow raise `DecompressionBombError`, which is not an `OSError`, so it escaped as a plain-text 500. | Caught and returned as 400 "not supported (maximum ... pixels)". |
| **Corrupt weights gave HTTP 500** | onnxruntime's `InvalidProtobuf` is not a `RuntimeError`, so a broken model file escaped as a 500. | Model load errors become 503 with the re-download command (`--force`). |
| **Phone JPEGs rejected (415)** | Many phone cameras save multi-picture JPEGs, which Pillow reports as `MPO`. The morning checklist tells you to try a phone photo, which would have failed. | `MPO` is accepted as JPEG. |
| **EXIF rotation ignored** | Phone photos often store pixels sideways plus a "rotate 90" tag. Boxes and `image_width/height` were relative to the sideways pixels, not the picture you see. | `ImageOps.exif_transpose` before detection. (Browser canvas frames have no EXIF, so the app itself was not affected.) |

Also: requests with two files now get 400, non-multipart bodies get the same 422
shape as before, and the OpenAPI docs still show the `file` field.

**Frontend (step 2).**
- `src/detection.ts`: the `/api/detect` client (`detectFrame`), `canvasToJpeg`
  (a promise around `canvas.toBlob`), error mapping, and the pure box math:
  `coverLayout` (where an `object-fit: cover` frame lands, including the cropped
  edges) and `projectBox` (normalized box to on-screen pixels, mirrored to match
  the video's `scaleX(-1)`, clipped to the visible area).
- `src/DetectionOverlay.tsx`: draws the boxes with labels and confidence
  (people in a warm color, objects in green), plus a small summary chip at the
  top center: "Seen: person ×3, laptop ×4, bottle" and "Person in view" / "No
  person". While waiting it says "Looking…"; on failure it shows a short amber
  note ("Detector model missing", "Detector not reachable", ...) with the full
  server message as a tooltip.
- `src/App.tsx`: on mic press the frame is drawn to the canvas as before, then
  `startFrameDetection()` encodes and uploads it **without being awaited**, so the
  microphone starts at once. Requests are chained so only one is in flight; a
  frame that a newer capture replaced while it was waiting is never sent. The
  old `toDataURL` copy of the frame (only used for an aria attribute) is gone,
  which removes a synchronous JPEG encode from the click path.
- `src/App.css`: overlay styles. Boxes sit under the LED, menu, voice text and
  mic button. Labels of boxes that start in the top 58 px are pushed down inside
  the box so they don't collide with the LED or menu, and labels at the right
  edge are right-aligned so they stay on screen. Works at 390 px wide.

**Tests (step 3).** `src/detection.test.ts` + `tsconfig.test.json` +
`npm run test:frontend`: compiles the helpers with the existing TypeScript and
runs them with Node. No new packages.

## 2. Technologies

- Nothing new at runtime. The backend uses Starlette's own `MultiPartParser`
  (already installed with FastAPI) with a larger in-memory limit, and Pillow's
  `ImageOps` (already installed).
- Frontend tests use only TypeScript (already a dev dependency) and Node. I tried
  to add Vitest, but installing packages was blocked in this unattended session,
  so I wrote a tiny assert-based runner instead (about 15 lines).
- Headless Chrome (already installed) was used for checks only, driven from
  Python over the Chrome DevTools Protocol with the `websockets` package that
  uvicorn already installs. That script lives in my scratch folder, not in the repo.

## 3. Decisions and reasoning

- **Parse the upload by hand instead of keeping `UploadFile`.** The privacy rule
  says frames never touch disk. The only way to guarantee that with Starlette is
  to control the spool size, which is a class attribute of the parser. A subclass
  used only by this endpoint keeps other endpoints unchanged.
- **Detection runs beside listening, never in front of it.** `canvas.toBlob`
  copies the pixels immediately and encodes in the background, and the upload is
  a separate promise. In the browser check, listening began while the detector
  was still working on every run.
- **One request at a time, newest frame wins.** The backend serializes
  detections behind a lock anyway. Chaining the requests in the browser means a
  slow request can't pile up others, and an outdated frame is skipped.
- **Mirror the boxes, not the labels.** The video is shown mirrored (like a
  mirror) but the captured frame is not. Rather than flipping the overlay with CSS
  (which would also flip the text), `projectBox` mirrors the x coordinate.
- **Boxes fade after 12 s, the summary stays.** Boxes describe the instant of
  capture, while the live video keeps moving. Keeping them forever would mislead
  and clutter the view. The "Seen" summary stays until the next press.
- **Detection errors are visible but harmless.** Failures show a short note and
  are logged to the browser console. Transcription and answers carry on (checked
  in the browser with a forced 503).
- **Detections are kept for Phase 4 but not sent to the LLM.** That is Phase 4's
  job (routing and prompt design). See the handoff notes.

## 4. Tradeoffs and alternatives considered

- **Draw over a frozen still vs. the live video.** A frozen still lines up
  perfectly but makes the app feel paused. I drew over the live video (same
  geometry) and fade the boxes instead.
- **Vitest** would be the usual choice, but package installation was blocked
  tonight. The plain runner covers the pure helpers and the fetch client (with a
  fake `fetch`). It cannot render React components. If Vitest is added later,
  `detection.test.ts` converts easily.
- **Restricting Pillow to JPEG/PNG decoders** (`Image.open(formats=...)`) would
  shrink the parsing attack surface, but it would turn "GIF sent as PNG" from 415
  into 400. I left it and noted it below.
- **JPEG draft decoding** would cut memory for huge uploads (a 40 MP image is
  about 120 MB decoded). The app sends at most 1024 px, so I left it.

## 5. Measurements

All on this laptop with only about 260-280 MB of RAM free (VS Code, Chrome and
other apps open; no Ollama model loaded during the vision checks).

**Browser end to end** (headless Chrome, fake camera playing the public-domain
office photo at 960x720, real Vite dev server, real backend; "capture to result"
is measured inside the app from the moment the frame is drawn to when the result
is in state: JPEG encode + upload through the Vite proxy + decode + inference +
JSON):

| Run | Capture to result | Model inference |
|---|---|---|
| First press after backend start (cold: model load, memory pressure) | 8.1 s | 363 ms |
| First press on a fresh page, model already loaded | 1.1-1.6 s | 0.74-1.2 s |
| Warm, 8 presses in a row | median **385 ms** (235-661) | median 245 ms (158-577) |

- Time from the click to "listening": median about 0.14 s warm. It never
  waited for detection: in every normal run the chip still said "Looking…" when
  listening began.
- Backend alone, same session (`bench_vision.py --runs 10`): warm median 157 ms
  wall (133-203), 129 ms inference; worker working set 115 MB. This matches the
  detector agent's numbers.
- Inference took longer inside the browser test than in the bench because Chrome,
  Vite and the backend were all competing for 2 cores and very little free RAM.

**Detections in the browser** matched the backend: person ×3, laptop ×4, bottle,
dining table, cup, keyboard (keyboard at 0.36 appears only on the 960x720 resize).
The boxes line up with the mirrored, cropped video on both desktop (1280x800)
and phone (390x844, cover crops the sides) screenshots.

## 6. Tests

**Mocked / synthetic unit tests**
- Backend suite: **111 tests, all pass** (was 100). New in `test_vision.py`:
  decompression-bomb header, MPO phone JPEG accepted, EXIF orientation applied
  (checks a marker pixel moved), corrupt weights become `ModelUnavailableError`.
  New in `test_vision_api.py`: a 1.9 MB frame is processed with `open`, `os.open`
  and `tempfile.TemporaryFile` all patched to fail (this test fails on the old
  code), chunked oversized body with no `Content-Length` gets 413 and never
  reaches the detector, non-multipart body 422, two files 400, bomb 400 not 500,
  corrupt weights 503 JSON, OpenAPI still documents `file`. I ran the new tests
  against the previous commit's `vision.py`/`vision_api.py`: 9 of them fail
  there, as they should.
- Frontend: `npm run test:frontend`, **14 tests pass**: cover layout (portrait
  phone and wide desktop), plain scaling, mirroring, the center stays at the
  center, boxes in the cropped margin dropped and partial boxes clipped (also
  when mirrored), invalid sizes, summary grouping and capping, error mapping, and
  `detectFrame` with a fake `fetch` (multipart POST without a manual
  Content-Type, HTTP errors, malformed body, unreachable backend).
- `npm run build` and `npm run lint` pass.

**Real-service checks** (servers started, then stopped)
- Real uvicorn + real model: office photo 200 (same boxes as before), a 2.4 MB
  noise PNG 200, bomb PNG 400, 6 MB upload 413, GIF 415, wrong field 422, JSON
  body 422.
- Headless Chrome with a fake camera and fake microphone: real `/api/detect`
  and real `/api/transcribe`; `/api/respond` was intercepted with a canned
  answer so no LLM ran and **no database rows were created**. 16 presses in total
  (desktop and phone sizes). A forced 503 from `/api/detect` showed "Detector
  model missing" and the voice flow still finished normally.
- **Not tested**: a physical webcam and microphone, real lighting, and the full
  flow with qwen answering. The fake microphone plays a beep, so Whisper always
  returned "No speech detected".

## 7. Known limitations and open issues

- **Answers don't use vision yet.** With no vision context, qwen invents what it
  "sees" (reported by the detector agent). Phase 4 must pass detections in, or
  tell the model it cannot see.
- Boxes are drawn over the live video and fade after 12 s. If you move a lot
  right after pressing the mic, they won't match the current picture.
- COCO's 80 classes only. In the test photo the hand sanitizer was labeled "cup"
  and the printer "laptop". Small objects are often missed.
- The first detection after the backend starts can take 1-8 s on this machine
  (model load plus memory pressure).
- **Out of scope, not fixed:** `/api/transcribe` in `backend/main.py` still uses
  `UploadFile`, so recorded audio over 1 MB (roughly a minute of speech) is
  briefly spooled to a temp file on disk. That conflicts with the "no raw audio
  on disk" rule. The same in-memory parser pattern from `vision_api.py` would fix it.
- `Image.open` still lets Pillow inspect headers of every format it knows before
  the JPEG/PNG check. Only the header is read (pixels are decoded only for
  JPEG/PNG), but restricting `formats=` would be stricter.
- Free RAM was only about 260 MB during the checks. Running Chrome, the backend,
  and qwen together will swap.

## 8. Morning manual checklist

1. Start the backend and `npm run dev` as in the README, then open the page.
2. Point the webcam at your desk (a cup, phone, laptop, or bottle) and press the
   mic. The chip at the top should show "Looking…", and within about half a second
   "Seen: ..." plus "Person in view" if you are in the frame. Boxes should sit on
   the objects. The preview is mirrored and the boxes should follow it: hold a cup
   on one side of your face and its box should be on that same side on screen.
3. Speak normally while it detects. Listening must not be delayed, and the
   answer should arrive as before.
4. Narrow the window to phone width (or use the browser's device toolbar).
   Boxes should still line up, and labels should not cover the LED or menu.
5. Stop the backend and press the mic. The chip should say "Detector not
   reachable" and the voice part will fail with its own error, as before.
   Restart the backend.
6. Optional: rename `backend\models\yolox_tiny.onnx`, press the mic: "Detector
   model missing" (hover for the download command). Rename it back.
7. Try a phone photo with curl (README command). Portrait photos should now
   give an `image_width` smaller than `image_height`. Delete the photo yourself
   afterwards.
8. Different lighting: try a dim room and a backlit window, and note what is missed.

## 9. Handoff notes (Phase 4 agents)

**Latest detection state** (in `App.tsx`, type `VisionState` in `src/detection.ts`):

```ts
type VisionState =
  | { status: 'idle' }
  | { status: 'detecting'; frameId: number }
  | { status: 'ready'; frameId: number; capturedAt: number /* Date.now() */;
      result: DetectionResult; elapsedMs: number /* capture -> result */ }
  | { status: 'error'; frameId: number; message: string; detail: string }

type DetectionResult = {               // exactly the /api/detect JSON
  detections: { label: string; confidence: number;
                box: { x: number; y: number; width: number; height: number } }[]
  person_present: boolean
  image_width: number; image_height: number; inference_ms: number
}
```

- `vision` (React state) drives the UI.
- **Use `frameDetectionRef` inside async code**:
  `frameDetectionRef.current = { frameId, promise }`. The promise **never
  rejects**. It resolves to the `DetectionResult` for that frame, or `null` if
  detection failed or a newer frame replaced it. `transcribeBatch` is a closure
  and would see stale `vision` state, so in Phase 4 do
  `const detection = await frameDetectionRef.current?.promise ?? null` just before
  calling `/api/respond`. It is almost always finished by then, because
  transcription takes longer than detection.
- `frameIdRef.current` is the id of the newest capture. Compare ids if you need
  to be sure a result belongs to the frame the user asked about.

**Reusing the captured frame**: `capturedFrameRef.current` is
`{ frameId, blob, width, height, capturedAt }`. `blob` is the same JPEG (quality
0.85, longest edge ≤1024 px, **not mirrored**) that was sent to `/api/detect`. Send
it to a vision-language model with `FormData.append('file', blob, 'frame.jpg')`.
Don't re-capture: the user may have moved. It lives in memory only; do not
persist it. It is set when the JPEG is ready (a few ms after the click), so read
it after awaiting the detection promise.

**Gotchas**
- Boxes are normalized to the unmirrored frame. "Left" in box terms is the
  user's right in the mirrored preview. If the LLM describes positions, say
  "left/right in the camera image" or flip x (`1 - x - width`) to match what the
  user sees.
- `/api/detect` takes one request at a time (backend lock). Use the existing
  chain (`detectChainRef`) if you add more calls, or the second will queue behind
  the first.
- Labels are COCO names (`cell phone`, `tv`, `dining table`, `couch`); map them
  to natural words before putting them in a prompt. Confidence below ~0.5 is
  often wrong (hand sanitizer read as "cup" at 0.59).
- RAM: the detector holds about 75 MB. A VLM (moondream, granite3.2-vision)
  plus qwen will not fit at the same time on 5.9 GB. Measure, and keep
  `keep_alive: 0`.
- Memory agent's advice still applies: add vision context to the system prompt
  through `memory_prompt.build_chat_messages` with a new parameter, not as fake
  chat turns.
- Backend errors from `/api/detect` are `{"detail": ...}`; 422 has FastAPI's list
  form, and `describeDetectError` handles both.
