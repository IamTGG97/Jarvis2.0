# Vision reasoning backend: overnight report (Phase 4, part 1)

Date: 2026-11-17. Machine: AMD Athlon Gold 3150U (2 cores), 5.9 GB RAM, no GPU,
Windows 11, Python 3.14.0, Ollama 0.40.13. Free RAM before loading any model was
usually only **100-250 MB** (VS Code, Chrome and Claude were open, and Windows
"Memory Compression" held 2-3 GB). Every latency number below comes from those
conditions.

Commits on `main`:

| Step | Commit | Message |
|---|---|---|
| 1. VLM benchmark | `1a4203c` | add local vlm benchmark and results |
| 2. Router + reasoning module | `25c8afa` | add question router and vision reasoning module |
| 3. API + real checks | `e7446d7` | add vision respond endpoint |
| 4. Report (this file) + README | final commit | add vision reasoning backend report |

## 1. What I built

- **`backend/bench_vlm.py`**: a repeatable VLM benchmark. It uses 12
  questions with expected answers written down before any model ran, measures
  cold/warm latency, peak RAM of the Ollama processes, and the minimum free
  system RAM. Results are in `docs/overnight/phase4-vlm-benchmark.md`.
- **`backend/reasoning.py`**: the reasoning module (pure functions, no web code):
  - `route_question()`: a transparent keyword router with four routes:
    `text`, `detections`, `vlm`, and `no_vision`.
  - `answer_from_detections()`: answers presence / count / "where is" /
    "what do you see" questions straight from YOLOX results, with no language
    model, in under 1 ms.
  - Confidence fallback: detections at or above 0.5 count as "seen", and weaker
    ones are only ever "might be". If only weak matches exist, the question
    escalates to the VLM when a frame exists; otherwise Jarvis says it isn't sure.
  - Multi-object selection: `select_objects()` / `choose_vlm_focus()` pick the
    object(s) by label, then by prominence (size x centrality x confidence).
    When there are several (two laptops), the answer says which one it used.
  - VLM client (`ask_vlm`, Ollama `/api/generate`, `keep_alive: 0`), crop to the
    chosen object, one "describe ..." retry when the VLM returns nothing, honest
    error messages, and a labeled detector-only fallback when the VLM fails.
  - `vision_summary()`: a short detector summary for qwen's prompt.
- **`backend/memory_prompt.py`**: `build_chat_messages(..., vision_context=None)`.
  It adds a `<vision>` block to the system message (not fake chat turns). With
  no vision data, the block says Jarvis can't see this turn. **This fixes the
  invented "I can see your computer and phone" answers.**
- **`backend/main.py`**:
  - New **`POST /api/respond/vision`** (multipart: `transcript`, `session_id`,
    optional `detections`, optional `file`).
  - `POST /api/respond` (JSON) has the **same request and response shape as
    before**. It now goes through the same router, so clearly visual questions
    get an honest `no_vision` answer and everything else reaches qwen with the
    "no camera" line.
- **`backend/vision_api.py`**: the in-memory multipart parser is now a reusable
  `read_form_in_memory()` (used by `/api/detect` and `/api/respond/vision`).
  `/api/detect` behaves exactly as before.
- **`backend/eval_router.py`**: the rule router vs an LLM classifier (qwen2.5:0.5b)
  on 26 hand-labeled questions.
- Tests: `backend/test_reasoning.py` (43), `backend/test_respond_vision.py` (20),
  plus new cases in `backend/test_memory_prompt.py`.
- `.env.example`, `README.md`, and `claude.md` (on disk, not committed) are updated.

## 2. Technologies

Nothing new was installed. It uses Ollama (already running), Pillow (crop and
resize in memory), httpx, and Starlette's multipart parser (the same in-memory
subclass the detector uses). The VLM default is **moondream** (Apache-2.0,
checked with `ollama show moondream --license`). granite3.2-vision is also
Apache-2.0 but doesn't fit in RAM here.

## 3. Decisions and reasoning

- **New endpoint instead of changing `/api/respond`.** The task asked for full
  backward compatibility. `/api/respond` still takes the same JSON and returns
  exactly `{"response", "memory_warning"}`. The frame needs multipart (the
  browser already has a JPEG Blob), so it goes to a separate path,
  `/api/respond/vision`. Its response is a superset of the old one, so the
  frontend can use one code path.
- **`VLM_MODEL`, not `VISION_MODEL`.** The task said `VISION_MODEL`, but that
  variable already selects the YOLOX detector (`yolox_tiny`). Reusing it would
  break detection, so the VLM uses `VLM_MODEL`.
- **Rule router, not an LLM classifier.** I measured both on 26 labeled questions
  (`eval_router.py`):

  | Router | Correct | Time per question |
  |---|---|---|
  | keyword rules (first run, before seeing the misses) | 24/26 | ~1 ms |
  | keyword rules (after fixing "Why is the sky blue?" and "Is the bottle full or empty?") | 26/26 | ~1-3 ms |
  | qwen2.5:0.5b, `keep_alive: 0` (how this machine has to run) | 9/26 | median 11.8 s (first 46 s) |
  | qwen2.5:0.5b, kept loaded | 10/26 | median 2.7 s (first 19 s) |

  The small LLM labeled almost everything `text`. It would add 3-12 s and be
  wrong more often, so it isn't worth it. (The 26/26 is measured on the set
  that I tuned against, so it is optimistic. The unit tests have about 80 more
  phrasings.)
- **Detection questions don't use any LLM.** "How many people are here?" can be
  answered exactly from structured detections in about a millisecond. That is
  also the honest answer: the detector's count *is* what Jarvis knows. The
  wording is fixed and states the limits ("That's presence only; I don't
  identify people.").
- **An object counts as "seen" at confidence 0.5 or higher.** The detector keeps
  anything at 0.35 or higher, but the earlier agents saw many wrong labels in
  the 0.35-0.6 range. Weaker detections are mentioned as "might also see ...,
  but I'm not sure". Overall `confidence` is `high` when every object used is at
  0.75 or higher, `medium` at 0.5 or higher, otherwise `low`. VLM answers are
  `medium` (the VLM gives no calibrated score), and fallbacks are `low`.
- **Absence doesn't escalate to the VLM.** "Do you see a phone?" with no phone
  detected answers "I don't see a phone right now. Small or partly hidden things
  can be missed..." Sending every "no" to a VLM would add minutes on this
  machine, and the VLM answers short questions badly here. Objects the detector
  *can't* know about ("pens", "keys") go to the VLM.
- **Crop for the VLM.** When the question names a detected object, only that
  object (plus a margin) is sent. The small VLM sees more detail, and the
  answer can't drift to other things in the room.
- **Left/right are flipped to match the screen.** Boxes are relative to the
  unmirrored camera frame, while the preview is mirrored. Positions are worded
  as "on the right side of the screen", matching what the user sees.
- **Memory.** The text route retrieves notes exactly as before. Vision routes
  skip retrieval: the answer comes from the frame, small VLMs get confused by
  extra text, and retrieval costs ~2-5 s here. **Every route saves the text
  turn** (never the image), so a later "what color did you say the bottle
  was?" can be answered from memory.
- **VLM failures.** If the VLM errors, times out, or returns nothing twice,
  and the detector saw something, Jarvis answers: "I couldn't take a close
  look just now because the vision model failed, so I can't answer that
  properly. My object detector only tells me I can see a bottle." This comes
  with `confidence: "low"` and `vision_warning` set to the real error. With
  nothing to fall back on, the API returns the error (502/503/504) with a plain
  message. It never returns a made-up answer.
- **"No camera" prompt wording was measured.** The first version ("you cannot
  see anything right now ... never invent visual details") still produced
  "Sure, I can help with that. Let's check the desk." and "You don't look
  tired." The final version ("IMPORTANT: You have no eyes ...") answered "I
  can't see anything right now." for those. It still answered "What color is my
  car?" from the notes and "What color is a banana?" from general knowledge.

## 4. Tradeoffs and alternatives considered

- **granite3.2-vision** (2.5B): needs 5,167 image tokens for a 960 px photo
  (7x moondream) and never answered within 15 minutes here. It might be the
  better model on a machine with RAM to spare. That is untested.
- **qwen + detector summary instead of the VLM for "describe" questions.** In
  one real check, "What is happening in this picture?" accidentally went to the
  text route. qwen, given only the detector summary, answered sensibly in 51 s,
  while moondream on the same photo failed after 4 minutes. I routed that
  phrasing to the instant detector list instead. If the VLM stays this
  unreliable, sending description questions to qwen + summary is a reasonable
  next step. I didn't do it tonight because the task asks for descriptions to
  go to the VLM.
- **Escalating "not detected" to the VLM**: rejected for latency (see above).
- **Saving the turn in the background** would cut ~3-5 s from every answer,
  but a failed save could then not be reported. I kept the save synchronous,
  as before.
- **base64 JSON upload** instead of multipart: rejected. Multipart reuses the
  hardened in-memory parser and avoids +33% size.

## 5. Measurements

**VLM benchmark**: full tables in `docs/overnight/phase4-vlm-benchmark.md`.
moondream scored **2/12** with plain questions and **3/12** with the
"describe" retry style. Cold load was 129-145 s, warm median 115 s, maximum 680 s.
The Ollama runner crashed on 2 of 12 requests per run. Peak Ollama working set
was 640-690 MB, and free RAM dropped to 3-9 MB. When moondream was already in
RAM, a call took **8-20 s**, but it still often answered with an empty string.
granite3.2-vision answered **0/4** (900 s timeouts, a context overflow at 5,167
tokens, a crash).

**Real end-to-end checks** (real uvicorn, PostgreSQL, Ollama, and YOLOX; office
photo; dedicated test session `6a0c1d7e-3b52-4c1e-9f7a-000000000404`, deleted
afterwards):

| Request | Route | Answer | Wall time |
|---|---|---|---|
| vision: "How many people are here?" | detections (high) | "I count three people. That's presence only; I don't identify people." | 7.8 s |
| vision: "What do you see?" | detections (medium) | "I can see three people, a table, three laptops, a bottle and a cup. (I only notice ...)" | 5.4 s |
| vision: "Where is the bottle?" | detections (high) | "The bottle is on the right side of the screen." | 5.9 s |
| vision: "Do you see a phone?" | detections (medium) | "I don't see a phone right now. Small or partly hidden things can be missed, ..." | 7.0 s |
| vision: "Can you see me?" | detections (high) | "Yes, I can see three people. I can only tell that someone is there, not who it is." | 8.2 s |
| vision: "What is the capital of France?" | text | "The capital of France is Paris." | 60 s |
| vision: "What color is the bottle?" | vlm, then fallback (low) | "I couldn't take a close look just now because the vision model failed ... I can see a bottle." (moondream returned nothing twice) | 254 s |
| vision: "Describe this picture in detail." | vlm (medium) | "In the image, a man is seated at a desk, engrossed in his work on a laptop. The desk is cluttered with ... a water bottle, a coffee cup, and a stack of papers. The man is wearing a blue shirt and glasses ..." (correct) | 138 s |
| vision: "What is happening in this picture?" (after routing fix) | detections (medium) | "I can see three people, a table, three laptops, a bottle and a cup. ..." | 6.3 s |
| respond: "What do you see?" | no_vision | "I can't see anything right now: no camera image or detections came with your question. ..." | 6.6 s |
| respond: "Can you tell me what's on my desk right now?" | text (old wording) | "Sure, I can help with that. Let's check the desk." (**the bug**) | 33 s |
| same, after the router + prompt fix | no_vision | "I can't see anything right now: ..." | 3.9 s |
| respond: "What is the capital of France?" | text | "The capital of France is Paris." | 29-32 s |

Where the time goes: the detector answer itself takes under 1 ms, and YOLOX on
the frame ~0.2-0.5 s. Saving the turn embeds it with `nomic-embed-text`, which
is unloaded after each call (`OLLAMA_EMBED_KEEP_ALIVE=0`), so the save costs
about **4-7 s** on every route. The text route adds a query embedding plus a
cold qwen load (~30-60 s under this memory pressure, vs ~5-10 s reported by
earlier agents with more free RAM). The backend's own memory is unchanged apart
from the decoded frame (about 3 MB for 1024x768) while a request runs.

## 6. Tests

**Mocked / unit tests** (`.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v`):
**176 tests, all pass** (was 111 before tonight's Phase 4 work).
- `test_reasoning.py` (43): the router across ~80 phrasings (text vs
  detections vs vlm vs no_vision, with and without a frame; "How many days are
  in a year?" stays text; "How many red cups" goes to vlm; unknown objects like
  pens go to vlm; place words like "on my desk" are not targets; synonyms
  phone->cell phone, mug->cup, kids->person). Also detection answers (lists,
  counts, uncertain extras, absent objects, low-confidence escalation, person
  wording with no identification, "your bottle", most-prominent of several,
  natural names), object selection and VLM focus (ambiguity note, comparison
  picks two, low confidence ignored), crop and downscale, detections parsing and
  validation, mirrored positions, the vision summary, no-vision messages, VLM
  answer tidying, prompts, settings, and the Ollama client (payload,
  keep_alive 0, 404/500/bad JSON/connect/timeout mapped to honest
  503/502/504 messages).
- `test_respond_vision.py` (20): endpoint contract keys; the detection route
  calls no model and saves the text turn; the VLM route crops and saves text
  only; ambiguity note; empty VLM answer triggers the describe retry; VLM
  failure gives a labeled fallback or a 503 when there's no fallback (nothing
  saved); low-confidence escalation with and without a frame; frame without
  detections runs the detector; detector failure still uses the VLM with a
  warning; the text route gets the detector summary and memory; the no-vision
  answer; 9 validation cases (422/400/415); a JSON body gives 422; 413 for over
  5 MB; **a 1.9 MB frame is processed with temp-file spooling patched to
  fail**; `/api/respond` still returns exactly two keys, and visual questions
  there never reach qwen.
- `test_memory_prompt.py` (+2): the no-vision line is present by default and
  is replaced by the vision context when given.

**Real-service checks** (not mocked): the table in section 5, the VLM benchmark
runs, `eval_router.py --llm qwen2.5:0.5b`, and direct qwen prompt-wording
checks (6 questions x 2 wordings, 4 memory/general questions). The real
uvicorn server was started for the checks and stopped afterwards, and the test
session was deleted.

**Not tested**: a physical webcam or microphone, the browser flow (the frontend
doesn't call the new endpoint yet), and `eval_memory.py` with the new
system-prompt line (it takes a long time on this machine; see the open issues).

## 7. Known limitations and open issues

- **The VLM route is not usable for a live demo on this laptop right now.**
  moondream mostly returns empty answers on Ollama 0.40.13 (even when warm and
  fast), its runner crashed several times, and calls take minutes while RAM is
  short. Jarvis is honest about it, but colors, text and brands are effectively
  unavailable. Ideas: free RAM before demos, try a newer moondream2 build or
  SmolVLM (Apache-2.0), or send description questions to qwen + the detector
  summary.
- The keyword router is English-only and phrase-based. It will misroute some
  wordings. Known gray areas: "What does this word mean?" with a frame
  attached goes to the VLM. "Where is my bottle?" *without* a frame goes to the
  text route, and qwen may answer from an earlier remembered turn ("on the
  right side of the screen"), which may be out of date.
- The `no_vision` answer is fixed text. A non-visual question that happens to
  match a strong visual pattern ("Can you see why this fails?") would get it.
- I didn't re-run the memory eval (`eval_memory.py --e2e`) after adding the
  "no camera" block to the system prompt. A spot check (4 questions) showed
  memory answers still work. **Run the eval to confirm.**
- Every route still pays ~4-7 s to embed and save the turn (existing behavior).
- Answer text for detections is English template text. It's fine for speech,
  but not varied.

## 8. Morning manual checklist

1. Start the backend (README). Run
   `.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v`.
   All tests should pass.
2. Run `.\.venv\Scripts\python.exe backend\eval_router.py` (should print 26/26).
3. Run the README curl example ("How many people are here?" with the office
   photo). It should answer "I count three people ..." with
   `"route": "detections"`. Delete that session afterwards (README command).
4. In the app (after the integrator wires it), ask "What do you see?",
   "How many people are here?", "Where is my cup?", and "What color is my cup?"
   with a cup on the desk. The first three should answer in a few seconds. The
   color question may take minutes and may say the vision model failed. Note
   which.
5. Close VS Code/Chrome and re-run
   `.\.venv\Scripts\python.exe backend\bench_vlm.py --models moondream`
   to see whether moondream works with more free RAM.
6. Run `.\.venv\Scripts\python.exe backend\eval_memory.py --e2e` to confirm
   the new "no camera" prompt line didn't hurt memory answers.

## 9. Handoff notes for `vision-reasoning-integrator`

**Endpoint**: `POST /api/respond/vision` (through the Vite proxy as
`/api/respond/vision`). Use it for every question: it handles text-only
questions too, and the response is a superset of `/api/respond`.

**Request** (`multipart/form-data`; let the browser set the boundary, and
don't set `Content-Type` yourself):

| Field | Required | Value |
|---|---|---|
| `transcript` | yes | the transcript string |
| `session_id` | yes | the conversation UUID |
| `detections` | no | `JSON.stringify(detectionResult)`: the exact `/api/detect` JSON (or just its `detections` array) |
| `file` | no | the captured JPEG Blob: `form.append('file', blob, 'frame.jpg')` |

```ts
const form = new FormData()
form.append('transcript', transcript)
form.append('session_id', sessionId)
const detection = await frameDetectionRef.current?.promise ?? null
if (detection) form.append('detections', JSON.stringify(detection))
const frame = capturedFrameRef.current
if (frame) form.append('file', frame.blob, 'frame.jpg')
const res = await fetch('/api/respond/vision', { method: 'POST', body: form })
```

- Send both when you have them. Without `detections` and with `file`, the
  backend runs YOLOX itself (+0.2-0.5 s, and it queues behind the detector
  lock). With `detections` but no `file`, detection questions work, but
  color/text questions get `no_vision`.
- Body limit: 5 MB + 64 KB total, one file.

**Response 200**:

```json
{
  "response": "I count three people. That's presence only; I don't identify people.",
  "memory_warning": null,
  "route": "detections",
  "used_vision": true,
  "objects_referenced": [
    {"label": "person", "detector_label": "person", "confidence": 0.92,
     "box": {"x": 0.47, "y": 0.06, "width": 0.53, "height": 0.94},
     "position": "on the left side of the screen"}
  ],
  "confidence": "high",
  "reason": "counted detections",
  "vision_warning": null,
  "latency_ms": 6989
}
```

- `route`: `"text" | "detections" | "vlm" | "no_vision"`.
- `used_vision`: false for `text` and `no_vision`.
- `confidence`: `"high" | "medium" | "low" | null` (null on `text`/`no_vision`).
- `objects_referenced`: objects the answer used. `box` is normalized to the
  **unmirrored** frame (same as `/api/detect`), so mirror x (`1 - x - width`)
  if you highlight them. `position` is already worded for the mirrored preview.
  `label` is the natural word ("phone"), and `detector_label` is the COCO
  label ("cell phone").
- `vision_warning`: set when the detector or VLM failed but an answer was still
  given (show it small, e.g. as a tooltip). `memory_warning` means the same as
  on `/api/respond`.
- `reason`: a short debug string ("escalated to VLM: only low-confidence
  matches"). Fine for a dev tooltip, but don't speak it.

**Errors** are `{"detail": ...}`, as everywhere else: 400 (empty transcript,
empty/unreadable image, malformed multipart), 413 (over 5 MB), 415 (not
JPEG/PNG), 422 (missing `transcript`/`session_id`, bad UUID, bad `detections`
JSON; FastAPI-style list `detail`), 503 (memory DB down, Ollama not running,
VLM model not installed, bad `VLM_*` setting), 502 (Ollama/VLM failure), 504
(chat model or VLM timeout). The turn is not saved when an error is returned.

**Gotchas**

- **VLM answers can take 1-5 minutes** on this laptop (`VLM_TIMEOUT_SECONDS=180`
  per attempt, and there can be two attempts). Show a distinct "Looking closely…"
  state when you expect `vlm` (you can't know the route in advance, so start
  that state after ~5 s of waiting). Allow at least ~7 minutes before the fetch
  is aborted, or let the server time out first.
- The VLM is serialized by a lock. Don't fire a second question while one is
  waiting on the VLM. It will queue.
- `/api/respond` (JSON) still works and still returns only
  `{"response","memory_warning"}`. Keep it as the fallback if you need one.
- Never persist the Blob. The backend keeps the image only in memory and saves
  only the transcript and answer.
