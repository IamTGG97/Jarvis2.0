# Memory engineer: overnight report (Phase 2 memory track)

Date: 2026-11-17. Machine: AMD Athlon Gold 3150U (2 cores), 5.9 GB RAM, no GPU,
Python 3.14, PostgreSQL 18, Ollama (`qwen2.5:1.5b`, `nomic-embed-text`).

Commits on `main`:

| Step | Commit | Message |
|---|---|---|
| 1. Eval harness + baseline | `ab629cc` | add memory eval harness and baseline |
| 2. Keyword/prompt fixes | `437ed48` | improve memory retrieval dedup, context budget and prompt |
| 3. Hybrid search | `78dc351` | add hybrid keyword and embedding memory search |
| 4. Sessions + retention | `1e8635c` | add conversation session management and retention |
| 5. Report (this file) | final commit | add memory engineer report and eval results |

## Headline numbers

The same 28 fact questions (15 exact, 13 paraphrased), 4 never-mentioned questions,
and 3 "user shares a new fact" statements, run against a seeded 32-turn
conversation with real PostgreSQL and real Ollama:

| Metric | Baseline (original code) | Final, keyword only | **Final, hybrid (default)** |
|---|---|---|---|
| Search recall@1 (all facts) | 0.571 | 0.571 | **0.893** |
| Search recall@5 (all facts) | 0.571 | 0.571 | **1.000** |
| Paraphrase recall@3 | 0.077 | 0.077 | **0.846** |
| Right turn reached the LLM (context hit) | 0.571 | 0.571 | **0.929** |
| Answer correct, exact questions | 1.000 | 1.000 | **1.000** |
| Answer correct, paraphrased questions | 0.077 | 0.077 | **0.846** |
| Answer correct, all fact questions | 0.571 | 0.571 | **0.929** |
| Never-mentioned questions answered honestly | 0.75 | 0.75 | **1.00** |
| New-fact statements acknowledged (not "you never told me") | not measured | 1.00 | **1.00** |
| Median `/api/respond` wall time (real server, 5 turns) | ~17.9 s* | 17.9 s | **23.9 s** |

\*Keyword-only server timing is the closest stand-in for the original code path.

In plain words: before tonight Jarvis remembered a fact only if you asked with the
same words. Now it also finds it when you rephrase ("Which squad do I root for?"
finds "My favorite basketball team is the Chicago Bulls"), at a cost of about six
extra seconds per answer on this laptop.

## 1. What I built

**Evaluation (step 1)**
- `backend/eval_memory.py`: seeds `backend/memory_eval_fixtures.json` into two
  dedicated sessions (`e7a1e7a1-...e7a1` for the conversation, `...dec0` as a decoy
  session with conflicting facts), asks every question, and reports recall@1/3/5,
  MRR, "context hit" (did the right turn reach the LLM), noise (unrelated turns in
  the top 3), decoy leaks, context size, and latency. `--e2e` also sends each
  question through the real `main._generate_response` and grades the answer.
  `--mode keyword|hybrid` picks the retrieval being measured. It deletes its rows
  before seeding and after finishing.
- Results are stored in `docs/overnight/memory-eval/*.json`
  (`baseline.json`, `step2-keyword-prompt.json`, `final-keyword-only.json`,
  `final-hybrid.json`) with every answer, so you can read what Jarvis actually said.

**Retrieval and prompt (step 2)**
- Recent turns are excluded from search **by row id**, not by transcript text (the
  old text check could hide an older answer if you repeated a question).
- Meta words such as "say", "tell", "remember", "mention", "Jarvis" are dropped
  from the keyword query; they described the act of remembering and matched
  unrelated turns ("Can you *say* good morning in French?").
- A token budget (`CONTEXT_TOKEN_BUDGET = 1200`, estimated at 4 characters per
  token) keeps history inside qwen's 2048-token window even when old answers are
  long. It keeps the newest turn and best search hit first.
- Relevant older turns are placed before the recent turns, both in time order.
- New `backend/memory_prompt.py`: memory is given to the model as a
  `<notes>` block inside the system message, with explicit rules (answer personal
  questions from the notes, say "You haven't mentioned that to me yet" when the
  notes lack the answer, acknowledge new facts). Main.py just calls
  `build_chat_messages()`.
- `OLLAMA_KEEP_ALIVE` (default `"0"`, unchanged behavior) and optional
  `OLLAMA_TEMPERATURE` settings.
- PostgreSQL connections retry once after a connect timeout
  (`DB_CONNECT_TIMEOUT`, default 10 s, was a hard-coded 5 s). See "Known issues".

**Hybrid search (step 3)**
- `backend/embeddings.py`: calls Ollama `POST /api/embed` with
  `nomic-embed-text`, using the `search_document: ` prefix for stored turns and
  `search_query: ` for questions. Raises `EmbeddingError` on any failure.
- `backend/database.py`: nullable `embedding REAL[]` and `embedding_model TEXT`
  columns added with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (no extensions).
  Vector search loads at most the session's 400 newest embedded turns, scores
  cosine similarity in Python, keeps hits >= 0.55 and within 0.10 of the best hit,
  and fuses them with keyword hits using reciprocal rank fusion (k = 60).
- `backend/memory.py`: `retrieve_memory()` and `remember_turn()` orchestrate
  embedding + database work so `main.py` stays small.
- `backend/backfill_embeddings.py`: embeds old turns (or turns saved while the
  embedding model was down). I ran it: your 10 existing turns are now embedded.

**Sessions and retention (step 4)**
- `backend/sessions_api.py` (FastAPI router, included by `main.py`):
  - `GET /api/sessions?limit=50`
  - `POST /api/sessions`
  - `DELETE /api/sessions/{id}` (turns removed by `ON DELETE CASCADE`)
  - `GET /api/sessions/{id}/turns?limit=200`
- `backend/retention.py`: if `MEMORY_RETENTION_DAYS` is set (off by default),
  conversations idle that many days are deleted at startup and every
  `MEMORY_RETENTION_CHECK_HOURS` (default 6).
- Frontend: `src/SessionMenu.tsx` + `src/sessionsApi.ts`. A small round menu
  button at the top right of the camera view opens a panel: "+ New conversation",
  a list (first-message preview, turn count, "5 min ago"), click to switch, and a
  trash icon that turns the row into an inline "Delete ...? [Delete] [Cancel]"
  confirmation (no `window.confirm`). Switching shows that conversation's last
  exchange. If semantic memory fails, a small amber warning appears under the
  answer.

**There is no authentication.** Anyone who can reach the backend can list, read,
and delete every conversation. This is only acceptable for a single-user,
local-only prototype.

## 2. Technologies

- **nomic-embed-text via Ollama** (Apache-2.0): already installed, runs locally,
  768-dimension vectors, about 370 MB loaded. No new Python packages.
- **PostgreSQL `REAL[]` column**: stores vectors without pgvector or any other
  extension, as required.
- **Reciprocal rank fusion (RRF)**: a standard, parameter-light way to merge two
  ranked lists without having to make keyword scores and cosine scores comparable.
- No new frontend dependencies.

## 3. Decisions and reasoning

- **Measure first.** I confirmed each known weakness before touching it:
  - *OR query matches common words*: confirmed. "What did I say my favorite team
    was?" became `'say' | 'favorit' | 'team'`; never-mentioned questions pulled an
    average of 2.25 unrelated turns into the top 3. But exact questions already
    ranked the right turn first 15/15 times, so the OR was not what was hurting
    answers. The real failure was paraphrases (1/13 found).
  - *3 recent + 3 search hits*: confirmed limits. Raising them to 4 + 5 did not
    improve recall and added unrelated turns. In one run the model then said
    "You mentioned your cat's name was Priya" (Priya is the sister). I kept 3 + 3.
  - *Dedup by transcript*: confirmed in code. Fixed (by id) with a unit test.
  - *qwen2.5:1.5b at 2048 context is weak*: confirmed, see the prompt A/B below.
    Context stays well under budget (84 tokens on average, 116 max in the eval).
  - *`keep_alive: "0"` reloads the model each call*: confirmed. Reload costs
    4.1-4.9 s per answer. A warm model answers a short prompt in about 4.2 s
    instead of about 9.6 s. However, with qwen kept loaded the machine had only
    40-440 MB of RAM free, and one eval run that kept it loaded for 5 minutes got
    PostgreSQL connection timeouts. **I kept `"0"` as the default** and made it
    configurable (`OLLAMA_KEEP_ALIVE`). The Phase 3/4 vision models make RAM even
    tighter.
- **Prompt layout.** I ran each variant against the whole question set (not all
  with the same keep-alive, so ignore their latency):

  | Variant | Exact | Paraphrase | Honest "not told" rate |
  |---|---|---|---|
  | A: history replayed as chat messages (old style) | 0.93 | 0.08 | 0.38 |
  | B: history as `<notes>` in the system message + rules | 1.00 | 0.08 | 0.56 |
  | B again (second run) | 1.00 | 0.15 | 0.63 |
  | C: chat messages + same rules | 0.87 | 0.08 | 0.50 |
  | D: B with temperature 0.2 | 1.00 | 0.08 | 0.50 |

  I kept B. The gain is modest and the runs are noisy (only 16 "unsupported"
  questions), but B was never worse than A. Variant C also showed that a concrete
  example in the prompt leaks: it answered "What am I allergic to?" with "You told
  me your favorite team is the Chicago Bulls", copying the old prompt's example.
  The final prompt has no concrete example fact, and a unit test enforces that.
  Temperature 0.2 did not help, so the model default stays.
- **Statements regression caught and fixed.** In the real-server check, the new
  rules made Jarvis answer "My favorite basketball team is the Chicago Bulls."
  with "You haven't mentioned that to me yet." I added a rule ("When the user tells
  you something about themselves, acknowledge it") and three statement cases to the
  eval. All three now pass.
- **Adopted hybrid search** because it measurably helps. Paraphrase answer
  accuracy rose from 0.08 to 0.85 and overall from 0.57 to 0.93, with no loss on
  exact questions. The cost is about 6 s per turn (see Measurements).
- **Similarity floor.** Vector search always returns a "nearest" turn, even when
  nothing is related. On the eval, correct matches scored 0.57-0.85 and unrelated
  turns 0.45-0.68, so the scores overlap. An absolute floor of 0.55 plus "within
  0.10 of the best hit" cut top-3 noise roughly in half (2.04 to 0.96) and raised
  recall@1 from 0.82 to 0.89.
- **Embedding failure policy.** If the query embedding fails (model missing,
  Ollama down), retrieval falls back to keyword search, logs a warning, and
  `/api/respond` returns `memory_warning`, which the UI shows. If the turn's own
  embedding fails, the turn is saved without a vector (so the conversation is not
  lost), the response carries a warning telling you to run the backfill, and the
  backfill fills it in later. Database failures still return 503s as before. I
  chose this over failing the whole request because keyword memory still works
  and the user is told plainly that recall was degraded.
- **Embedding model keep-alive `"0"`** for the same RAM reason: it never sits in
  memory beside qwen. Cold embeddings cost about 2.2 s each instead of about 0.6 s.
  Set `OLLAMA_EMBED_KEEP_ALIVE=2m` if you have spare RAM.
- **Server-generated session ids** for `POST /api/sessions`; the browser still
  keeps the active id in `localStorage` (`jarvis-conversation-session`).

## 4. Tradeoffs and alternatives considered

- **pgvector**: faster and indexable, but the brief forbade new extensions. With
  a bounded scan of 400 turns, scoring in Python costs about 110 ms. That is fine
  for a personal prototype, but it would not scale to many thousands of turns per
  session.
- **numpy for cosine**: it is available (pulled in by faster-whisper), but it is
  not a declared dependency. Pure Python is fast enough at this scale.
- **Synonym lists for keyword search**: rejected. They would overfit to the eval
  questions.
- **Embedding the turn after responding (background task)**: would take about
  2-3 s off the user's wait. I did not do it tonight because the failure warning
  would then arrive after the answer and needs a separate reporting path. It is a
  good next step.
- **Keeping models warm**: about 5 s faster per answer, but it risks swapping and
  database timeouts on 5.9 GB RAM. Left as an opt-in setting.
- **A relevance threshold for keyword hits**: I could not find a threshold that
  separated "brother's name" noise from real hits without losing recall.

## 5. Measurements

How measured: `eval_memory.py` against real PostgreSQL 18 + Ollama on this
machine, with `OLLAMA_KEEP_ALIVE=0` and nothing else heavy running unless noted.
Single runs; Ollama sampling is random, so answer accuracy can vary by a few
questions between runs.

- **Retrieval (final hybrid eval)**: recall@1 0.893, @3 0.929, @5 1.000, MRR
  0.929. Exact questions 1.0 at every k. Paraphrases: @1 0.769, @3 0.846, @5 1.0.
  Decoy-session leaks: 0 in every run. Context averaged 5 turns (~86 tokens).
- **Retrieval latency**: keyword only, median 0.16-0.30 s. Hybrid, median 1.96 s
  (p90 5.5 s), almost all of it the cold query embedding (average 2.16 s; 0.6-0.9 s
  when the model is warm).
- **Python cosine cost**: 400 candidates x 768 dimensions = 112 ms.
- **Backfill**: 34 turns in 18.3 s (about 0.54 s per turn, batches of 16). Your
  10 real turns took 8.8 s.
- **End-to-end `/api/respond` wall time** (real uvicorn server, 5-turn script,
  `server_latency` check): keyword only, median 17.9 s (mean 17.7). Hybrid,
  median 23.9 s (mean 23.8), about +6 s (+33%). That is two cold embeddings
  (question and saved turn) plus model swapping.
- **Ollama chat model** (`qwen2.5:1.5b`, Q4_K_M): about 1.06 GB resident. Load
  4.1-4.9 s per cold call; a warm call is about 4.2 s for a 55-token prompt.
- **RAM**: free memory was about 1.4 GB idle, about 440 MB with qwen loaded, and
  nomic adds about 370 MB. That is why both keep-alives default to `"0"`.
- **Baseline vs final answer latency in the eval** (generation only): median
  18.7 s baseline, 22.2 s hybrid. The eval latencies are noisy because I was
  building and testing other things at the same time; trust the server
  comparison above more.

## 6. Tests

**Mocked unit tests** (64 total, all passing:
`.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v`)
- `test_database.py`: connect retry (once, then surfaced), schema including the
  `REAL[]` columns and no extensions, context ordering, id-based dedup (a repeated
  question no longer hides an older answer), token budget, keyword stop-lexemes,
  RRF ordering, hybrid context with similarity floor, keyword-only path, backfill
  batching, save with embedding.
- `test_embeddings.py`: prefixes, endpoint, error mapping (connect/HTTP/malformed),
  cosine similarity, on/off switch.
- `test_memory.py`: hybrid path, fallback with warning on embedding failure,
  disabled embeddings, database errors not swallowed, save with and without vector.
- `test_memory_prompt.py`: notes order, honest-ignorance rule, acknowledge-facts
  rule, no concrete example fact.
- `test_main.py`: routes (now read from OpenAPI), Ollama options/keep-alive,
  `memory_warning` propagation, existing error paths.
- `test_sessions_api.py`: list/create/delete/turns, 404s, 503 on DB failure,
  limit clamping.
- `test_retention.py`: off by default, invalid value rejected, purge call,
  background task start/cancel, session DB helpers.
- `test_eval_memory.py`: fixture consistency, scoring (including statements),
  summary math.
- Frontend: `npm run build` and `npm run lint` pass (no unit test framework is
  set up for the frontend).

**Real-service checks** (real PostgreSQL + Ollama, test session UUIDs deleted
afterwards; the database ended with only your original session and 10 turns):
- Five eval runs end to end (baseline, step 2, and final keyword vs hybrid,
  plus prompt variants).
- uvicorn + `/api/respond`, 5 turns: the paraphrase "Which squad do I root for?"
  answered "You root for the Chicago Bulls." All 5 saved turns had 768-dimension
  `nomic-embed-text` vectors.
- Embedding failure: a nonexistent embed model and an unreachable Ollama both
  gave keyword mode + the warning text.
- Session endpoints through uvicorn: POST 201, list shows preview/turn count,
  turns in order, unknown id 404, bad UUID 422, DELETE 204 then 404. Retention
  purge deleted a 40-day-old test session and its turns via cascade. (Before
  purging, I checked that no real session was older than 30 days.)
- **Not tested**: the frontend in a real browser, the camera, and the
  microphone. I could not do that unattended. See the morning checklist.

## 7. Known limitations and open issues

- **+6 s per answer** with hybrid memory on this CPU. `MEMORY_EMBEDDINGS=off`
  restores keyword-only speed. A background document embedding would recover
  about 2-3 s.
- **Two paraphrases still miss**: "Is there any food I need to avoid?" (peanut
  allergy) and "What should you put in my morning drink?" (oat milk). The right
  turn ranks 4th, just outside the 3 relevant slots.
- **Noise for never-mentioned questions** remains (about 3 related-but-wrong turns,
  e.g. the sister for "brother"). The new prompt handled all 4 correctly in the
  final run, but a 1.5B model can still mix facts up.
- **"Which university did I go to?"** produced a made-up answer (UCLA/UCSD) in
  most runs, including the baseline and the final keyword-only run, so treat
  honest-ignorance as improved, not solved.
- **Statement replies are sometimes odd**, e.g. "You've mentioned that before.
  You told me your favorite color is green." The eval counts these as acceptable
  acknowledgements.
- **The eval grades answers by keywords**: "I don't have personal preferences"
  counts as admitting ignorance. It is a rough metric; read the JSON answers for
  detail.
- **Low-RAM DB timeouts**: twice, a PostgreSQL connect timed out right after an
  Ollama call while RAM was nearly exhausted. One automatic retry now covers this
  (and fired once during the real checks). If memory pressure grows (vision
  models), saves could still fail. They fail visibly with a 503.
- **No authentication** and no per-user separation (single-user, local only).
- Vector search scans up to 400 newest turns per session; older turns are found
  only by keyword.
- **Changing `OLLAMA_EMBED_MODEL`** makes old vectors ignored (filtered by
  `embedding_model`) until you run the backfill.
- **Out of scope, not fixed**: the README repeats the "first transcription
  downloads tiny.en" note twice. `src/App.tsx` still creates a browser-side UUID
  for the very first session (fine, since the backend upserts sessions on first
  save). The step-1 commit of `eval_memory.py` had a mis-encoded apostrophe in
  answer normalization. It only matters for answers with curly apostrophes and
  was fixed in step 2; the baseline's honesty rate above was recomputed with
  the fixed code.

## 8. Morning manual checklist

1. `ollama list` shows `nomic-embed-text`. Start the backend and frontend as in
   the README.
2. Say "My favorite basketball team is the Chicago Bulls." Jarvis should
   acknowledge it, not say you never mentioned it.
3. Ask two or three unrelated things, then ask "Which squad do I root for?" It
   should say the Bulls.
4. Ask "What's my brother's name?" It should say you haven't mentioned it.
5. Click the menu button (top right). You should see the current conversation
   with a preview. Click "+ New conversation" and ask "Which squad do I root
   for?" again: the new conversation should not know.
6. Open the menu, switch back to the first conversation (its last exchange should
   appear), then delete the new one. Check that the inline confirmation appears
   and Cancel works.
7. Delete the conversation you are in: a fresh empty one should become active.
8. Optional failure check: set `OLLAMA_EMBED_MODEL=nonexistent` in `.env` and
   restart the backend. An answer should show the amber "keyword search only"
   warning. Remove the setting afterwards.
9. Camera and microphone: confirm the camera view is still clean, with only the
   small menu button added, and that voice capture works as before.

## 9. Handoff notes (Phase 3 / Phase 4 agents)

- **`/api/respond` response shape changed**:
  `{"response": str, "memory_warning": str | null}`. The frontend reads
  `memory_warning`; keep it if you change the endpoint.
- **Memory entry points** (keep `main.py` thin):
  - `memory.retrieve_memory(session_id, transcript) -> RetrievedMemory(turns, mode, warning)`
  - `memory.remember_turn(session_id, transcript, response) -> warning | None`
  - `memory_prompt.build_chat_messages(transcript, turns)` builds the Ollama
    messages. If Phase 4 adds vision context, add it to the system prompt via a new
    parameter rather than appending fake chat turns. The eval showed the notes
    layout works better for this model.
- **Session endpoints**: `GET/POST /api/sessions`,
  `DELETE /api/sessions/{id}`, `GET /api/sessions/{id}/turns`. Summaries look
  like `{id, created_at, updated_at, turn_count, preview}`.
- **Gotcha, FastAPI 0.141**: `app.routes` contains `_IncludedRouter` objects
  without `.path`. Tests now read routes from `app.openapi()["paths"]`.
- **Gotcha, RAM**: both Ollama models unload after every call (`keep_alive "0"`).
  Loading a vision model at the same time as qwen will push this 5.9 GB machine
  into swapping. I saw PostgreSQL connect timeouts under that pressure. Measure
  before raising any keep-alive.
- **Gotcha, Ollama calls are sequential per request**: query embedding, then
  chat, then turn embedding. Adding a vision call adds another model load.
- **Frontend**: `SessionMenu` sits at the top right (`.session-menu`, z-index 2);
  the camera LED is top left. The Phase 7 results panel should avoid that corner
  or move the menu. `switchSession()` in `App.tsx` resets transcript/response.
- **Eval**: run `backend\eval_memory.py --e2e` after any prompt or model change.
  It takes about 13-15 minutes on this machine with `--e2e` and about 1 minute
  without. Add new question kinds to `memory_eval_fixtures.json`.
- **New settings** (all in `.env.example` with safe defaults): `DB_CONNECT_TIMEOUT`,
  `OLLAMA_KEEP_ALIVE`, `OLLAMA_TEMPERATURE`, `MEMORY_EMBEDDINGS`,
  `OLLAMA_EMBED_MODEL`, `OLLAMA_EMBED_KEEP_ALIVE`, `OLLAMA_EMBED_TIMEOUT_SECONDS`,
  `MEMORY_RETENTION_DAYS`, `MEMORY_RETENTION_CHECK_HOURS`.
