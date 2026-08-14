"""Small CPU-only YAMNet adapter for free Render hosting.

YAMNet is used only for conservative sound-class recognition. It is not a
species-level universal model; ambiguous results are rejected instead of
being presented as a species.
"""
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class AnimalInferenceEngine:
    SAMPLE_RATE = 16_000
    WINDOW = 15_600

    def __init__(self) -> None:
        self.model_path = Path(__file__).with_name("yamnet.tflite")
        self.labels_path = Path(__file__).with_name("yamnet_class_map.csv")
        self._interpreter: Any = None
        self._labels: list[str] | None = None

    def status(self) -> dict[str, str]:
        return {"adapter": "yamnet-tflite-cpu", "model_id": "YAMNet AudioSet 521 classes"}

    def identify(self, audio_path: Path) -> dict[str, Any]:
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise ValueError("Arquivo de áudio inválido.")
        try:
            scores = self._infer(audio_path)
            candidates = self._candidates(scores)
        except Exception:
            return self._abstention("Modelo acústico indisponível no servidor.")

        best = candidates[0] if candidates else None
        if not best or best["score"] < 0.50 or best["animal"] is None:
            return self._abstention("Confiança abaixo de 50%; identificação recusada.", candidates)

        confidence = float(best["score"])
        return {
            "identified": True,
            "animal": best["animal"],
            "scientific_name": None,
            "group": best["group"],
            "confidence": confidence,
            "confidence_percent": int(confidence * 100),
            "source": "AnimalID Universal Engine / YAMNet",
            "model": "YAMNet AudioSet 521 classes",
            "candidates": candidates,
            "requires_confirmation": confidence < 0.70,
            "message": "Classe sonora identificada; confirme a espécie no aplicativo." if confidence < 0.70 else "Resultado acústico aceito.",
        }

    def _infer(self, audio_path: Path) -> np.ndarray:
        samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = self._resample(np.asarray(samples, dtype=np.float32), int(rate))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak < 0.01:
            raise ValueError("Áudio silencioso.")
        samples = np.clip(samples * min(1.0 / peak, 8.0), -1.0, 1.0)

        interpreter = self._get_interpreter()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        pooled: np.ndarray | None = None
        step = self.WINDOW // 2
        for offset in range(0, len(samples), step):
            window = np.zeros(self.WINDOW, dtype=np.float32)
            chunk = samples[offset:offset + self.WINDOW]
            window[:len(chunk)] = chunk
            interpreter.set_tensor(input_detail["index"], window)
            interpreter.invoke()
            raw = np.asarray(interpreter.get_tensor(output_detail["index"]), dtype=np.float32)
            if raw.ndim >= 2:
                raw = raw.reshape(-1, raw.shape[-1]).mean(axis=0)
            raw = raw.reshape(-1)
            pooled = raw if pooled is None else np.maximum(pooled, raw)
            if offset + self.WINDOW >= len(samples):
                break
        if pooled is None:
            raise ValueError("Áudio curto demais.")
        return pooled

    def _get_interpreter(self) -> Any:
        if self._interpreter is None:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter  # type: ignore
            self._interpreter = Interpreter(model_path=str(self.model_path), num_threads=2)
            self._interpreter.allocate_tensors()
        return self._interpreter

    def _labels_for(self) -> list[str]:
        if self._labels is None:
            with self.labels_path.open(newline="", encoding="utf-8") as stream:
                self._labels = [row[2] for row in csv.reader(stream) if len(row) >= 3 and row[0] != "index"]
        return self._labels

    def _candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        labels = self._labels_for()
        ranked = np.argsort(scores)[::-1][:8]
        result = []
        for index in ranked:
            label = labels[int(index)] if int(index) < len(labels) else "unknown"
            animal, group = self._animal_label(label)
            result.append({"label": label, "animal": animal, "group": group, "score": float(scores[index])})
        return result

    @staticmethod
    def _animal_label(label: str) -> tuple[str | None, str]:
        value = label.lower()
        rules = {
            "dog": ("Cachorro", "mammal"), "cat": ("Gato", "mammal"),
            "frog": ("Sapo", "amphibian"), "cricket": ("Grilo", "insect"),
            "bee": ("Abelha", "insect"), "wasp": ("Vespa", "insect"),
            "mosquito": ("Mosquito", "insect"), "cicada": ("Cigarra", "insect"),
            "rooster": ("Galo", "bird"), "chicken": ("Galinha", "bird"),
            "duck": ("Pato", "bird"), "goose": ("Ganso", "bird"),
            "bird": ("Ave não especificada", "bird"), "whale": ("Baleia", "marine"),
            "dolphin": ("Golfinho", "marine"), "horse": ("Cavalo", "mammal"),
            "cow": ("Bovino", "mammal"), "pig": ("Porco", "mammal"),
        }
        for keyword, mapped in rules.items():
            if keyword in value:
                return mapped
        if any(word in value for word in ("animal", "mammal", "wild animals", "domestic animals")):
            return None, "unknown"
        return None, "unknown"

    def _abstention(self, message: str, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"identified": False, "animal": None, "scientific_name": None, "group": "unknown", "confidence": 0.0, "confidence_percent": 0, "source": "AnimalID Universal Engine / YAMNet", "model": "YAMNet AudioSet 521 classes", "candidates": candidates or [], "requires_confirmation": False, "message": message}

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == AnimalInferenceEngine.SAMPLE_RATE:
            return samples
        target_size = max(1, int(round(len(samples) * AnimalInferenceEngine.SAMPLE_RATE / source_rate)))
        source_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        return np.interp(target_x, source_x, samples).astype(np.float32)
