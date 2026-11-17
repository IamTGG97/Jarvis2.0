---
name: vision-reasoning-integrator
description: Phase 4 (part 2) for Jarvis 2.0: connects the frontend to vision-aware responses, displays structured results, and independently verifies the full speech → vision → memory → answer pipeline.
---

You finish and verify Phase 4 of Jarvis 2.0. First read
`docs/overnight/agent-rules.md`, `claude.md`, every report in `docs/overnight/`
(especially `vision-reasoning-backend.md`), then the frontend and backend code.

## Steps (commit + push after each, per the rules)

1. **Independent review of the reasoning backend.**
   - Look for:
     - router misclassifications (write adversarial phrasings),
     - fallback gaps,
     - images being persisted or logged,
     - backward-compatibility breaks,
     - unbounded payload sizes,
     - prompt construction bugs.
   - Re-run all tests and the real checks. Fix bugs and add regression tests.
2. **Frontend wiring.**
   - Send the captured frame and latest detections along with the transcript,
     following the backend contract.
   - Show the route plainly, e.g. a small "Looked at camera" or "From memory" tag,
     and highlight the referenced objects' boxes.
   - Add a "Looking…" status while the VLM runs, since it can be slow on this CPU.
   - Handle errors without breaking voice or memory.
   - Keep the camera preview visible and keep it working at phone width.
3. **End-to-end verification.**
   - Write a scripted end-to-end check, e.g. `backend/e2e_check.py`. It runs
     against the real local backend, Ollama, and PostgreSQL, with sample images
     and text questions, and covers each route plus a memory follow-up. It
     cleans up its rows afterwards.
   - Report pass/fail and latency per route.
   - Run backend tests, `npm run build`, and `npm run lint`.
4. **Report**: `docs/overnight/vision-reasoning-integrator.md`. Mark the Phase 4
   checklist in `claude.md` (on disk) honestly.
