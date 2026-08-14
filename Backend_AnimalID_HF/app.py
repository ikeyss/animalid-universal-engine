"""Gradio entrypoint for the free Hugging Face Spaces runtime.

The FastAPI endpoint remains available at /identify-animal for the Android app;
the Gradio screen is only a browser-facing diagnostic interface.
"""
import json
from pathlib import Path

import gradio as gr
import uvicorn

from main import app as api_app
from main import engine


def identify_from_ui(audio_path: str | None) -> dict:
    if not audio_path:
        return {"identified": False, "message": "Envie um áudio de até 5 segundos."}
    try:
        return engine.identify(Path(audio_path))
    except Exception as exc:
        return {"identified": False, "message": f"Falha na inferência: {exc}"}


demo = gr.Interface(
    fn=identify_from_ui,
    inputs=gr.Audio(type="filepath", label="Áudio do animal"),
    outputs=gr.JSON(label="Resultado"),
    title="AnimalID Universal Engine",
    description="Teste a identificação bioacústica e consulte o contrato usado pelo aplicativo.",
    api_name="identify_animal_ui",
)

app = gr.mount_gradio_app(api_app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
