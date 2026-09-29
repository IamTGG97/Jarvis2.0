import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

load_dotenv()

app = FastAPI(title="Jarvis2.0 Speech API")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is not configured. Set OPENAI_API_KEY in the backend environment.",
        )

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")

    client = AsyncOpenAI(api_key=api_key)
    try:
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename or "utterance.webm", audio, file.content_type or "audio/webm"),
        )
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The speech provider could not transcribe this audio. Check the API key and audio format.",
        ) from error
    finally:
        await client.close()

    return {"transcript": result.text}