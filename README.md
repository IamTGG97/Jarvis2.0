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

3. Install [Ollama](https://ollama.com/download) and download the small chat model:

   ```powershell
   ollama pull qwen2.5:1.5b
   ```

4. Install the Python backend dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

   The first transcription downloads the `tiny.en` speech model automatically.
   That one-time download requires an internet connection; subsequent
   transcription runs locally.

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
Completed transcripts and responses are saved in PostgreSQL for this browser's
local conversation session. Jarvis includes up to three recent turns and up to
three full-text matches from earlier turns as context for its next response.
Audio and camera images are not stored.

To override defaults, set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `WHISPER_MODEL`
in `.env`. The default chat model is intentionally small for computers with limited
memory; response quality and speed depend on the machine. For a lower-memory
fallback, install `qwen2.5:0.5b` and set `OLLAMA_MODEL=qwen2.5:0.5b`.

Memory currently uses PostgreSQL English full-text keyword search, not semantic
or vector search. The browser keeps one conversation session ID in local
storage; a UI for starting/managing multiple sessions and a data-retention
policy are future work.

Run backend tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v
```
