# Jarvis2.0

Local-first capstone demo: a browser camera preview, microphone capture with
voice-activity detection, local speech-to-text, and a local AI response.
Audio and transcript requests stay on this computer; no paid API key is used.

## One-time setup

1. Install [Ollama](https://ollama.com/download) and download the small chat model:

   ```powershell
   ollama pull qwen2.5:1.5b
   ```

2. Install the Python backend dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

   The first transcription downloads the `tiny.en` speech model automatically.
   That one-time download requires an internet connection; subsequent
   transcription runs locally.

3. Optional: copy `.env.example` to `.env` to override the Ollama URL or model.
   `.env` is ignored by Git, and no API key is required.

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
Press the microphone button, speak, and pause; after silence is detected the
recorded batch is transcribed locally and sent to Ollama for a text response.
The first request may take longer while the speech model is downloaded.

To override defaults, set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `WHISPER_MODEL`
in `.env`. The default chat model is intentionally small for computers with limited
memory; response quality and speed depend on the machine. For a lower-memory
fallback, install `qwen2.5:0.5b` and set `OLLAMA_MODEL=qwen2.5:0.5b`.

Run backend tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v
```
