---
name: vision-reasoning-backend
description: Phase 4 (part 1) for Jarvis 2.0: benchmarks small local vision-language models, then builds question routing that combines speech, detections, memory, and a VLM into a structured answer.
---

You build the Phase 4 reasoning backend for Jarvis 2.0. First read
`docs/overnight/agent-rules.md`, `claude.md`, every report in `docs/overnight/`,
and the current backend code.

## Steps (commit + push after each, per the rules)

1. **VLM benchmark** (commit `docs/overnight/phase4-vlm-benchmark.md`).
   - Compare `moondream` and `granite3.2-vision` (both already pulled in Ollama)
     on this 2-core, 5.9 GB machine. Only try a 3B+ model if free RAM allows,
     and record whether it fit.
   - Use a scripted set of sample images (permissive licenses, cited) with
     questions covering color, visible text/brand, counting, comparison, and
     "what is this".
   - Measure cold and warm latency, peak RAM, and answer correctness (scored
     against expected answers you write down beforehand).
   - Pick the default and make it configurable via `VISION_MODEL`. Check the
     model license.
2. **Router and reasoning module** (e.g. `backend/reasoning.py`).
   - Decide the route for each question:
     - `text`: no vision needed, so use the existing memory + LLM path.
     - `detections`: presence, counting, or "what do you see" questions,
       answered quickly from structured detections.
     - `vlm`: brand, text, color, comparison, or detailed description; send the
       image plus detection context to the VLM.
   - Start with a transparent rule/keyword router. Document whether an LLM
     classifier is worth the extra latency (measure it).
   - Confidence fallback: if relevant detections are low-confidence or missing,
     escalate to the VLM, or say honestly what is uncertain.
   - Multi-object selection: choose the object(s) the question refers to by label
     match, then prominence (size or centrality). State the choice in the answer
     when it's ambiguous.
   - Memory still applies: include retrieved context, and save the text turn
     only, never the image.
3. **API.**
   - Extend the respond flow in a **backward-compatible** way. Text-only requests
     must keep working unchanged.
   - Optionally accept the captured image (multipart, or base64 with a size
     limit) and the client's detections.
   - Return a structured payload, e.g. `{"response","route","used_vision",
     "objects_referenced":[...],"confidence"}`.
   - Never store images. Surface VLM and Ollama errors honestly.
   - Unit tests for the router (many phrasings), fallback logic, object selection,
     and the endpoint with mocks.
   - Run **real** checks through Ollama with sample images.
4. **Report**: `docs/overnight/vision-reasoning-backend.md`, with handoff notes
   (exact request/response contracts) for `vision-reasoning-integrator`.

Don't touch the frontend; the next agent owns it.
