"""STT backends: Groq whisper-large-v3-turbo (cloud, ~300 ms) + local faster-whisper."""

from __future__ import annotations

import io

import numpy as np


class STTBackend:
    name = "base"

    async def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        raise NotImplementedError

    async def warm(self) -> None:
        """Optional: preload models/connections before the first turn."""
        return None


def _wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Whisper mishears exchange/ticker words ("IHSG" → "nih HSG", "XAUUSD" →
# "sao usd") unless the vocabulary is primed. On a finance stream that is the
# whole point of the transcript, so bias it explicitly.
FINANCE_PROMPT = (
    "Percakapan trading dan pasar modal Indonesia. Istilah: IHSG, XAUUSD, emas Antam, "
    "harga spot gold, Bitcoin, USD IDR, saham BBCA BBRI TLKM, TradingView, timeframe, "
    "support resistance, OJK, The Fed, obligasi, yield, komoditas, dollar index."
)


class GroqSTT(STTBackend):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo",
                 language: str = "id", initial_prompt: str = FINANCE_PROMPT):
        self.api_key = api_key
        self.model = model
        self.language = language or None
        self.initial_prompt = initial_prompt

    async def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        import httpx

        if not self.api_key:
            raise RuntimeError("Groq: JARVIS_GROQ_API_KEY kosong")
        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if self.language:
            data["language"] = self.language
        if self.initial_prompt:
            data["prompt"] = self.initial_prompt
        files = {"file": ("audio.wav", _wav_bytes(pcm, sr), "audio/wav")}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"},
                             data=data, files=files)
        if r.status_code != 200:
            raise RuntimeError(f"Groq STT {r.status_code}: {r.text[:200]}")
        return (r.json().get("text") or "").strip()


class LocalWhisperSTT(STTBackend):
    """faster-whisper on CPU/CUDA. Model loads lazily on first use."""

    name = "local"

    def __init__(self, model: str = "small", device: str = "cpu", compute: str = "int8",
                 language: str = "id", initial_prompt: str = FINANCE_PROMPT,
                 cpu_threads: int = 4):
        self.model_name = model
        self.device = device
        self.compute = compute
        self.language = language or None
        self.initial_prompt = initial_prompt
        # Measured here: 12 threads is *slower* than 4 for a 3 s clip (3.7 s vs
        # 3.5 s) because OBS/browser already saturate the cores; 4 keeps headroom.
        self.cpu_threads = cpu_threads

    def _load(self):
        from faster_whisper import WhisperModel

        kwargs = {"device": self.device, "compute_type": self.compute}
        if self.device == "cpu" and self.cpu_threads:
            kwargs["cpu_threads"] = self.cpu_threads
        return WhisperModel(self.model_name, **kwargs)

    async def warm(self) -> None:
        """Load the model off the event loop so the first turn doesn't pay it."""
        import asyncio

        if getattr(self, "_model", None) is None:
            self._model = await asyncio.to_thread(self._load)

    async def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        import asyncio

        def _run() -> str:
            model = getattr(self, "_model", None)
            if model is None:
                model = self._model = self._load()
            audio = pcm.astype(np.float32)
            if sr != 16000:  # faster-whisper wants 16 kHz mono
                # np.interp(x, xp, fp): x must be the NEW grid, xp the OLD one.
                target = max(1, int(len(audio) * 16000 / sr))
                audio = np.interp(np.linspace(0.0, 1.0, target),
                                  np.linspace(0.0, 1.0, len(audio)),
                                  audio).astype(np.float32)
            segments, _info = model.transcribe(audio, language=self.language, beam_size=1,
                                               vad_filter=False,
                                               initial_prompt=self.initial_prompt)
            return " ".join(s.text.strip() for s in segments).strip()

        return await asyncio.to_thread(_run)


class NullSTT(STTBackend):
    """No STT available: returns empty so the caller can fall back to typed input."""

    name = "none"

    async def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        return ""


def build_stt(settings) -> STTBackend:
    backend = (settings.stt_backend or "groq").lower()
    threads = getattr(settings, "local_whisper_threads", 4)
    if backend == "groq" and settings.groq_api_key:
        return GroqSTT(settings.groq_api_key, settings.groq_stt_model, settings.stt_language)
    if backend == "local":
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("STT lokal butuh `uv pip install faster-whisper`") from exc
        return LocalWhisperSTT(settings.local_whisper_model, settings.local_whisper_device,
                               settings.local_whisper_compute, settings.stt_language,
                               cpu_threads=threads)
    if backend == "groq":
        # No cloud key: prefer local whisper if installed, else degrade to typed input.
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return NullSTT()
        return LocalWhisperSTT(settings.local_whisper_model, settings.local_whisper_device,
                               settings.local_whisper_compute, settings.stt_language,
                               cpu_threads=threads)
    return NullSTT()
