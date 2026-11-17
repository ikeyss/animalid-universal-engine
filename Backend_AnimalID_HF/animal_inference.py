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
        scientific_name = self._scientific_name(best) if best else None
        animal_candidates = [candidate for candidate in candidates if candidate["animal"] is not None]
        runner_up = animal_candidates[1] if len(animal_candidates) > 1 else None
        # Se duas espécies diferentes têm pontuação praticamente igual, não
        # escolher uma arbitrariamente. Isso evita transformar vaca em gato,
        # cachorro em raposa etc. por causa de ruído ou áudio reproduzido.
        ambiguous = (
            runner_up is not None
            and runner_up["animal"] != best["animal"]
            and float(best["score"]) - float(runner_up["score"]) < 0.08
        ) if best else False
        if (
            not best
            or best["score"] < 0.50
            or best["animal"] is None
            or not scientific_name
            or ambiguous
        ):
            return self._abstention(
                "Áudio ambíguo, confiança abaixo de 50%, classe não biológica ou espécie sem nome científico.",
                candidates,
            )

        confidence = float(best["score"])
        return {
            "identified": True,
            "animal": best["animal"],
            "scientific_name": scientific_name,
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
        # Não usar máximo entre janelas: uma oscilação curta ou ruído de fundo
        # pode fazer YAMNet eleger gato/cachorro mesmo quando o som dominante é
        # outro animal. A mediana representa o padrão recorrente do áudio e
        # elimina picos isolados de uma janela.
        window_scores: list[np.ndarray] = []
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
            window_scores.append(raw)
            if offset + self.WINDOW >= len(samples):
                break
        if not window_scores:
            raise ValueError("Áudio curto demais.")
        return np.median(np.stack(window_scores, axis=0), axis=0)

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
    def _scientific_name(candidate: dict[str, Any]) -> str | None:
        """Retorna nome científico somente para rótulos suficientemente específicos.

        YAMNet também retorna grupos genéricos (por exemplo, "Bird" ou
        "Insect"). Esses grupos não são espécies e nunca devem entrar no
        catálogo como se fossem nomes científicos.
        """
        label = str(candidate.get("label", "")).lower()
        animal = str(candidate.get("animal", "")).lower()
        value = f"{label} {animal}"
        rules = (
            (("cat", "meow", "purr", "hiss", "gato"), "Felis catus"),
            (("dog", "bark", "yip", "growl", "cachorro"), "Canis lupus familiaris"),
            (("horse", "neigh", "whinny", "cavalo"), "Equus ferus caballus"),
            (("cow", "cattle", "moo", "bovino"), "Bos taurus"),
            (("pig", "oink", "porco"), "Sus scrofa domesticus"),
            (("sheep", "ovelha"), "Ovis aries"),
            (("goat", "cabra"), "Capra hircus"),
            (("elephant", "elefante"), "Loxodonta africana"),
            (("lion", "leão", "leao"), "Panthera leo"),
            (("tiger", "tigre"), "Panthera tigris"),
            (("rooster", "chicken", "crowing", "galo", "galinha"), "Gallus gallus domesticus"),
            (("duck", "pato"), "Anas platyrhynchos"),
            (("goose", "ganso"), "Anser anser"),
        )
        return next((scientific for keywords, scientific in rules if any(k in value for k in keywords)), None)

    @staticmethod
    def _animal_label(label: str) -> tuple[str | None, str]:
        value = label.lower()
        rules = {
            # Mamíferos e vocalizações
            "dog": ("Cachorro", "mammal"), "bark": ("Cachorro", "mammal"),
            "howl": ("Cachorro ou lobo", "mammal"), "cat": ("Gato", "mammal"),
            "meow": ("Gato", "mammal"), "purr": ("Gato", "mammal"),
            "hiss": ("Gato", "mammal"), "horse": ("Cavalo", "mammal"),
            "cow": ("Bovino", "mammal"), "cattle": ("Bovino", "mammal"),
            "moo": ("Bovino", "mammal"), "pig": ("Porco", "mammal"),
            "sheep": ("Ovelha", "mammal"), "goat": ("Cabra", "mammal"),
            "elephant": ("Elefante", "mammal"), "lion": ("Leão", "mammal"),
            "tiger": ("Tigre", "mammal"), "monkey": ("Macaco", "mammal"),
            "rodent": ("Roedor", "mammal"), "bat": ("Morcego", "mammal"),
            # Aves e vocalizações
            "rooster": ("Galo", "bird"), "chicken": ("Galinha", "bird"),
            "duck": ("Pato", "bird"), "goose": ("Ganso", "bird"),
            "crow": ("Corvo", "bird"), "owl": ("Coruja", "bird"),
            "bird": ("Ave não especificada", "bird"),
            # Anfíbios, répteis e vida marinha
            "frog": ("Sapo", "amphibian"), "toad": ("Sapo", "amphibian"),
            "snake": ("Serpente", "reptile"), "lizard": ("Lagarto", "reptile"),
            "whale": ("Baleia", "marine"), "dolphin": ("Golfinho", "marine"),
            # Insetos e artrópodes sonoros
            "cricket": ("Grilo", "insect"), "bee": ("Abelha", "insect"),
            "wasp": ("Vespa", "insect"), "mosquito": ("Mosquito", "insect"),
            "cicada": ("Cigarra", "insect"), "fly": ("Mosca", "insect"),
            "insect": ("Inseto não especificado", "insect"), "beetle": ("Besouro", "insect"),
            "spider": ("Aranha", "arachnid"),
        }
        for keyword, mapped in rules.items():
            if keyword in value:
                return mapped
        # Preserva classes biológicas do AudioSet ainda não traduzidas no
        # dicionário acima, sem transformar rótulos genéricos em espécie.
        biological_groups = (
            (("mammal", "canidae", "feline", "bovine", "equine", "primate", "rodent"), "mammal"),
            (("bird", "avian", "songbird", "waterfowl", "poultry"), "bird"),
            (("insect", "arthropod", "crustacean"), "insect"),
            (("amphibian", "amphib",), "amphibian"),
            (("reptile", "reptilian"), "reptile"),
            (("fish", "marine", "aquatic", "cetacean"), "marine"),
            (("arachnid",), "arachnid"),
        )
        for keywords, group in biological_groups:
            if any(keyword in value for keyword in keywords):
                return label, group
        # Sons humanos, música, veículos, chuva e ruídos não são competência
        # deste aplicativo e devem ser descartados, não convertidos em animal.
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
