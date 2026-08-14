"""AnimalID Universal Bioacoustic Engine.

Standalone Hugging Face Space backend. It never imports or modifies SoundTracksAi.
The model adapter is intentionally isolated so larger specialist models can be
added without changing the mobile app contract.
"""
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from animal_inference import AnimalInferenceEngine

APP_VERSION = "1.0.0"
MAX_AUDIO_BYTES = 25 * 1024 * 1024

app = FastAPI(title="AnimalID Universal Engine", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = AnimalInferenceEngine()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "animalid-universal", "version": APP_VERSION, "engine": engine.status()}


@app.post("/identify-animal")
async def identify_animal(audio: UploadFile = File(...)) -> dict[str, Any]:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Arquivo de áudio ausente.")
    suffix = Path(audio.filename).suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
        raise HTTPException(status_code=415, detail="Formato de áudio não suportado.")

    payload = await audio.read(MAX_AUDIO_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Áudio vazio.")
    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Áudio excede o limite de 25 MB.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            temp_path = tmp.name
        return engine.identify(Path(temp_path))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Falha na inferência acústica.") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/api")
def root() -> dict[str, str]:
    return {"service": "AnimalID Universal Engine", "endpoint": "POST /identify-animal"}
