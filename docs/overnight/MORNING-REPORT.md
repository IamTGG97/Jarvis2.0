# Overnight run — morning report (2026-11-18)

Five agents ran overnight. Four finished, and the fifth was stopped early so you
could test before your 11:00 update. The sixth (microphone + memory polish) did
not run.

Each agent's detailed report is in this folder. It covers decisions, tradeoffs,
measurements, and tests, with mocked tests kept separate from real ones.

## Where things stand

| # | Agent | Status | Report |
|---|---|---|---|
| 1 | Memory (Phase 2) | ✅ Done | [memory-engineer.md](memory-engineer.md) |
| 2 | Object detector backend (Phase 3) | ✅ Done | [vision-detector-backend.md](vision-detector-backend.md), [phase3-detector-research.md](phase3-detector-research.md) |
| 3 | Detector UI + review (Phase 3) | ✅ Done | [vision-integration-verifier.md](vision-integration-verifier.md) |
| 4 | Vision reasoning backend (Phase 4) | ✅ Done | [vision-reasoning-backend.md](vision-reasoning-backend.md), [phase4-vlm-benchmark.md](phase4-vlm-benchmark.md) |
| 5 | Vision UI + end-to-end checks (Phase 4) | ⚠️ Stopped after step 2 of 4 | No report; see "Agent 5" below |
| 6 | Microphone + memory polish | ⏸ Not run | — |

**Verified this morning:**
- 192 backend tests and 12 frontend tests pass.
- `npm run build` and `npm run lint` pass.
- Real smoke test: the backend starts and `/api/health` responds.
- `/api/detect` on the sample photo found people, a laptop, a cup, and a bottle
  in about 0.3 s.
- `/api/respond/vision` answered "How many people are here?" with "I count three
  people" in about 4 s.

**Commits:** 21 commits on `main`, all pushed, all authored by George Smith,
with no AI attribution lines.

## What changed, in plain terms

### Phase 2: conversation memory (agent 1)
- **Hybrid search.** Memory now combines keyword search with meaning-based
  search. Turns are converted to vectors by the local `nomic-embed-text` model
  and stored in a PostgreSQL `REAL[]` column, so no database extension is needed.
  The two result lists are merged with *reciprocal rank fusion*: a turn ranked
  high by either method rises to the top.
- **Results, on a repeatable eval of 35 questions:**

  | Metric | Before | After |
  |---|---|---|
  | Right memory found first (recall@1) | 57% | 89% |
  | Correct answers to reworded questions | 8% | 85% |
  | Honest "you never told me" answers | 75% | 100% |

- **Re-check after the Phase 4 prompt change:** recall@1 stayed at 89%. Answer
  accuracy moved from 93% to 89%, which is one question, within normal run-to-run
  noise ([memory-eval/after-vision-prompt-hybrid.json](memory-eval/after-vision-prompt-hybrid.json)).
- **Tradeoff:** each answer takes about 6 s longer, because of the extra
  embedding model loads.
- **Conversation menu.** A new menu at the top right lets you start, switch, and
  delete conversations, with an in-app delete confirmation. There is optional
  `MEMORY_RETENTION_DAYS` auto-cleanup, off by default. There is no login, so
  this is single-user and local only.

### Phase 3: object detection (agents 2 and 3)
- **Model choice.** The detector is **YOLOX-Tiny** (Apache-2.0) running on ONNX
  Runtime. Seven candidates were benchmarked on this laptop. Ultralytics YOLO
  was excluded because its AGPL license would require sharing all the project's
  source code under the same terms.
- **Speed and memory:** about 0.15 s per frame warm, using about 76 MB of RAM.
  Weights are in `backend/models/`, which is git-ignored. If they're missing,
  run `backend/download_vision_model.py`.
- **How it works:** when you press the mic, the frame is sent to `/api/detect`
  in parallel with listening. Boxes are drawn on the video, and a chip shows
  "Seen: …" and "Person in view". Person detection is presence-only, with no
  facial recognition. Images are processed only in memory and never saved.
- **Review.** Agent 3 found and fixed 5 real bugs in agent 2's work:
  - Large uploads were briefly written to a temp file on disk.
  - A crafted "decompression bomb" image crashed the endpoint.
  - Corrupt weights caused a crash.
  - Phone JPEGs were rejected.
  - Sideways photos weren't rotated.

### Phase 4: vision reasoning (agents 4 and 5)
Each question goes to one of four **routes**:
- `detections`: counting and "what do you see" questions. Answered instantly
  from the detector, with no LLM.
- `vlm`: color, text, brand, and description questions. The image goes to a
  local vision-language model.
- `text`: memory and general questions. Uses qwen plus memory, as before.
- `no_vision`: a visual question with no camera frame. Jarvis says honestly that
  it can't see.

Other details:
- **Rule-based router.** The router is transparent rules, not an LLM. A small LLM
  classifier was measured: it got 10 of 26 questions right in about 3–12 s. The
  rules got 26 of 26 in under a millisecond, though on questions they were tuned on.
- **Hallucination fixed.** Before, "what's on my desk?" got an invented answer.
  Now Jarvis answers from real detections or says it can't see.
- **Agent 5's UI work.** The UI now shows a route tag ("Looked at camera" /
  "From memory"), highlights the objects the answer refers to, and shows a
  "Looking closely…" state with elapsed time for slow vision answers.

## ⚠️ Things to know before the demo

1. **The vision-language model doesn't work well on this laptop.**
   - `moondream` takes **2–4 minutes** per answer and often returns nothing.
     Jarvis then gives a clearly labeled detector-only fallback.
   - `granite3.2-vision` doesn't fit in RAM.
   - **For the demo, stick to detector-type questions** ("How many people are
     here?", "What do you see?", "Is there a laptop?") and memory questions.
   - Avoid "What color is…" and "What does the label say?". The code path is
     correct, but this hardware is the bottleneck; [phase4-vlm-benchmark.md](phase4-vlm-benchmark.md)
     has the numbers.
2. **Text and memory answers take about 20–60 s.**
   - RAM (5.9 GB) is the main limit, because models are reloaded on every call.
   - **Close Chrome tabs and other apps before demoing.** It makes a big difference.
3. **Your microphone complaint is not addressed yet.** That was agent 6.
   Speak close to the mic, in a quiet room.
4. **Known privacy gap:** `/api/transcribe` audio over 1 MB (long recordings) is
   briefly spooled to a temp file by FastAPI. Images don't have this problem;
   they were fixed. This is the first item for the next session.

## Agent 5: what was finished vs. not

- **Committed:**
  - `090d5e6` fixes misroutes, escalation fallback, and payload limits.
  - `5ac8905` wires the frontend to the vision endpoint.
  - `a9d3876` is a small route-preview fix I tested and committed this morning.
- **Not finished:**
  - its report,
  - a scripted end-to-end check (`backend/e2e_check.py`),
  - a README update.

  These are saved in `git stash` (`stash@{0}`) and are untested. Restore them later
  with `git stash pop`.
- **Not done:** the full end-to-end run across all routes in a browser.

## Test it yourself (≈10 minutes)

Close other apps first, then use two terminals from the repo root.

```powershell
# Terminal 1 — backend
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```
```powershell
# Terminal 2 — frontend
$env:OPENSSL_CONF = ''
npm run dev
```

Open the printed localhost URL and allow camera and mic. Then:

- [ ] Press the mic and say **"How many people are here?"** Boxes and "Seen: …"
      appear, the answer comes back in a few seconds, and the tag shows the camera route.
- [ ] Hold up a cup or phone and ask **"What do you see?"** The object should be
      listed and highlighted.
- [ ] Say **"My favorite team is the Chicago Bulls."** Then ask something unrelated.
      Then ask **"Which team do I root for?"** It should recall the Bulls (about 20–60 s).
- [ ] Ask **"What's the capital of France?"** You should get a normal text answer
      with no camera claims.
- [ ] Open the **conversation menu** (top right). Start a new conversation, ask
      the team question again, and expect "you haven't mentioned that". Switch
      back, then try deleting a test conversation.
- [ ] Narrow the browser to phone width. The layout should still be usable.
- [ ] (Optional, slow) **"What color is the cup?"** Expect a long "Looking
      closely…" wait and possibly the labeled fallback answer.

## Suggested next steps

1. Run agent 6 (`voice-memory-ux`). It covers the mic accuracy fixes (Whisper
   model, noise suppression, adaptive speech detection, level meter), the audio
   temp-file fix, and latency/RAM tuning.
2. Finish agent 5's remaining work: the e2e script, its report, and a real
   browser run.
3. Decide on the vision-language model for the final demo. The options are a
   machine with more RAM or a GPU, accepting detector-only vision, or an
   approved exception to the local-first rule.
