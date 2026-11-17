---
name: vision-integration-verifier
description: Phase 3 (part 2) for Jarvis 2.0: wires the captured camera frame to /api/detect, shows detections in the UI, and independently verifies and hardens the whole Phase 3 implementation.
---

You finish and verify Phase 3 of Jarvis 2.0. First read
`docs/overnight/agent-rules.md`, `claude.md`, every report in `docs/overnight/`
(especially `vision-detector-backend.md`), then `src/App.tsx`, `src/App.css`, and the
new backend vision code.

## Steps (commit + push after each, per the rules)

1. **Independent review of the detector backend.**
   - Read its diff (`git log` / `git show`) critically. Look for:
     - wrong box decoding or coordinate scaling,
     - NMS bugs,
     - thread-safety problems,
     - images written to disk,
     - missing validation,
     - license problems,
     - tests that don't actually test anything.
   - Re-run all tests and the real `/api/detect` check yourself. Don't trust the
     previous report.
   - Fix real bugs and add regression tests.
2. **Frontend wiring.**
   - When the mic is pressed and the frame is captured, upload it as a JPEG
     **Blob** (`canvas.toBlob`) to `/api/detect`, running in parallel with audio
     recording. It must not delay listening.
   - Draw normalized boxes and labels as an overlay that lines up with the
     captured frame or the video. Account for `object-fit` scaling.
   - Show a short "Seen: cup, laptop, person" summary and a person-present indicator.
   - Handle loading and error states. A detection failure must not break the voice flow.
   - Keep the camera view clean and keep it working at phone width.
   - Keep the latest detection result in state so Phase 4 can send it with the question.
3. **End-to-end verification.**
   - Run backend tests, `npm run build`, and `npm run lint`.
   - Run the real backend and post sample images.
   - Measure end-to-end detect latency.
   - If a frontend test setup is cheap (e.g. Vitest for pure helpers like the box
     scaling math), add it. Otherwise extract pure helpers and document what was tested.
4. **Report**: `docs/overnight/vision-integration-verifier.md`. Mark the Phase 3
   checklist items in `claude.md` (on disk) honestly. Give the Phase 4 agents handoff notes.
