# Jarvis2.0

Local-first capstone demo: a browser camera preview, microphone capture with
voice-activity detection, local speech-to-text, and a local AI response.
Audio and transcript requests stay on this computer; no paid API key is used.

## One-time setup

1. Make sure PostgreSQL is installed and running. Create a dedicated local
   database user and database. In PowerShell, connect with the PostgreSQL
   administrator account (the password is entered at a hidden prompt):

   ```powershell
   & 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -h 127.0.0.1 -U postgres -d postgres -W
   ```

   At the `psql` prompt, run:

   ```sql
   CREATE ROLE jarvis_app WITH LOGIN;
   \password jarvis_app
   CREATE DATABASE jarvis2 OWNER jarvis_app;
   \q
   ```

   `\password` prompts for the new app user's password without putting it in
   the SQL command or shell history.

2. Copy `.env.example` to `.env` and set `DB_PASSWORD` to the password you just
   created. Keep `.env` private; Git ignores it. The backend creates its
   conversation tables and indexes on startup.

3. Install [Ollama](https://ollama.com/download) and download the small chat
   model and the embedding model used for semantic memory search:

   ```powershell
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

4. Install the Python backend dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

   The first transcription downloads the `tiny.en` speech model automatically.
   That one-time download requires an internet connection; subsequent
   transcription runs locally.

5. Download the object detection model (YOLOX-Tiny, Apache-2.0, about 20 MB).
   The script verifies the SHA-256 checksum and saves it to the git-ignored
   `backend\models\` folder:

   ```powershell
   .\.venv\Scripts\python.exe backend\download_vision_model.py
   ```

No paid API key is required. The first transcription downloads the `tiny.en`
speech model automatically; that one-time download requires an internet
connection.

## Run the demo

Start FastAPI in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another:

```powershell
npm run dev
```

Open the localhost URL printed by Vite and allow camera/microphone access.
Press the microphone button, speak, and pause. After silence is detected, the
recorded batch is transcribed locally and sent to Ollama for a text response.
Completed transcripts and responses are saved in PostgreSQL for the current
conversation. Jarvis includes up to three recent turns and up to three relevant
earlier turns as context for its next response. Audio and camera images are not
stored.

To override defaults, set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `WHISPER_MODEL`
in `.env`. The default chat model is intentionally small for computers with limited
memory; response quality and speed depend on the machine. For a lower-memory
fallback, install `qwen2.5:0.5b` and set `OLLAMA_MODEL=qwen2.5:0.5b`. See
`.env.example` for every optional setting.

### Conversation memory

- **Search:** relevant earlier turns are found with PostgreSQL full-text
  keyword search fused with local `nomic-embed-text` embeddings (meaning-based
  search), so "Which squad do I root for?" finds "My favorite basketball team
  is the Chicago Bulls." Set `MEMORY_EMBEDDINGS=off` for keyword-only search.
  If the embedding model is unavailable, Jarvis falls back to keyword search and
  shows a warning under its answer.
- **Existing turns:** turns saved before embeddings existed (or while the
  embedding model was down) can be embedded with:

  ```powershell
  .\.venv\Scripts\python.exe backend\backfill_embeddings.py
  ```

- **Conversations:** the menu button (top right) lists conversations and lets
  you start a new one, switch, or delete one (with a confirmation step).
- **Retention:** set `MEMORY_RETENTION_DAYS=30` (for example) to delete
  conversations idle for that many days. It is off by default.
- **No accounts:** there is no login. Anyone who can reach the backend can read
  and delete every conversation, so run it only on your own machine.
- **Evaluation:** `.\.venv\Scripts\python.exe backend\eval_memory.py` measures
  recall on a seeded test conversation (add `--e2e` to also grade real Ollama
  answers, `--mode keyword` to compare against keyword-only search). It deletes
  its test rows when done.

### Object detection

`POST /api/detect` accepts one JPEG or PNG image (multipart field `file`, up to
5 MB) and returns COCO object labels, confidences, and boxes normalized to 0-1,
plus `person_present`. It runs locally with YOLOX-Tiny on `onnxruntime`, about
0.15 s per image on a 2-core laptop. Person detection is presence-only (no face
recognition), and images are decoded in memory and never saved. Settings:
`VISION_MODEL` (`yolox_tiny` or the lighter `yolox_nano`), `VISION_CONFIDENCE`,
`VISION_THREADS`, `VISION_KEEP_LOADED`. Try it with the backend running:

```powershell
curl.exe -F "file=@backend/test_data/office_public_domain.jpg;type=image/jpeg" http://127.0.0.1:8000/api/detect
.\.venv\Scripts\python.exe backend\bench_vision.py
```

When you press the microphone button, the app captures one camera frame and
sends it (as a JPEG, in memory) to `/api/detect` while it is already listening.
Boxes and labels are drawn over the camera view for a few seconds, and a small
"Seen: ..." summary with a person indicator appears at the top. If detection
fails, a short note appears there instead and the voice flow carries on.
The app does not send the frame or detections with the question yet; that
wiring is the next step (see below). Until then, `/api/respond` tells the model
it has no camera this turn, so "What do you see?" gets an honest "I can't see
anything right now" instead of an invented answer.

### Answering questions about the camera frame (backend)

`POST /api/respond/vision` takes `multipart/form-data` with `transcript`,
`session_id`, and optionally `detections` (the `/api/detect` JSON) and `file`
(the JPEG/PNG frame, up to 5 MB, kept in memory only). A keyword router picks
one route per question:

- `text`: no vision needed, the normal memory + qwen answer.
- `detections`: "what do you see", "how many people", "is there a cup",
  "where is my bottle". These are answered straight from the detector in
  milliseconds, with weak detections (under 0.5) phrased as "might be".
- `vlm`: color, text/brand, comparison, "what is this". The frame (cropped to
  the object when one is named) goes to a local vision-language model
  (`VLM_MODEL`, default `moondream`, Apache-2.0).
- `no_vision`: a visual question arrived without a frame, so Jarvis says it
  can't see.

The reply is `{"response", "memory_warning", "route", "used_vision",
"objects_referenced", "confidence", "reason", "vision_warning", "latency_ms"}`.
Only the text of the turn is saved. Settings: `VLM_MODEL`, `VLM_KEEP_ALIVE`,
`VLM_TIMEOUT_SECONDS`, `VLM_MAX_IMAGE_EDGE`, `VLM_NUM_PREDICT`. On this 2-core,
5.9 GB laptop the VLM route is slow and unreliable (see
`docs/overnight/phase4-vlm-benchmark.md`), and Jarvis says so when it fails.

```powershell
curl.exe -F "transcript=How many people are here?" -F "session_id=00000000-0000-4000-8000-000000000001" -F "file=@backend/test_data/office_public_domain.jpg;type=image/jpeg" http://127.0.0.1:8000/api/respond/vision
.\.venv\Scripts\python.exe backend\eval_router.py
.\.venv\Scripts\python.exe backend\bench_vlm.py --models moondream
```

(The curl example saves a turn under that session id. Delete it afterwards
from the conversation menu or with `curl.exe -X DELETE
http://127.0.0.1:8000/api/sessions/00000000-0000-4000-8000-000000000001`.)

Frontend helper tests (box math for the mirrored `object-fit: cover` video,
summary text, API errors) run without extra packages:

```powershell
$env:OPENSSL_CONF = ''
npm run test:frontend
```

Run backend tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v
```
