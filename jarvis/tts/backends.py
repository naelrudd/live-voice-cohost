"""TTS backends. Every backend can *stream* PCM so playback starts on the first
chunk instead of waiting for the whole sentence to be synthesised."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np


@dataclass
class AudioClip:
    pcm: np.ndarray  # shape (n,), float32, mono
    sr: int

    @property
    def duration(self) -> float:
        return len(self.pcm) / self.sr if self.sr else 0.0


class TTSBackend:
    name = "base"
    sample_rate = 24000

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError

    async def close(self) -> None:
        """Release sockets/subprocesses. Overridden by long-lived backends."""
        return None

    async def prime(self) -> None:
        """Optional: called on PTT-down to re-arm a stale connection."""
        return None

    async def synthesize(self, text: str) -> AudioClip:
        """Convenience: drain the stream into one clip (used for warmup acks)."""
        parts: list[np.ndarray] = []
        async for chunk in self.stream(text):
            if chunk.size:
                parts.append(chunk)
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return AudioClip(pcm, self.sample_rate)


class EdgeTTS(TTSBackend):
    """Free Microsoft Edge TTS. Indonesian male voice, no API key.

    Uses a persistent websocket (edge_ws.EdgeWS): a fresh handshake per sentence
    costs 0.4-1.4 s, a reused socket ~0.15-0.25 s to first audio.
    """

    name = "edge"
    sample_rate = 24000

    def __init__(self, voice: str = "id-ID-ArdiNeural", rate: str = "+8%"):
        self.voice = voice
        self.rate = rate
        self._ws = None

    def _client(self):
        if self._ws is None:
            from .edge_ws import EdgeWS
            self._ws = EdgeWS(self.voice, self.rate)
        return self._ws

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def prime(self) -> None:
        if self._ws is not None:
            await self._ws.warm_if_stale()

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        from .mp3 import Mp3FrameDecoder

        text = text.strip()
        if not text:
            return
        dec = Mp3FrameDecoder(sample_rate=self.sample_rate)
        async for data in self._client().stream(text):
            out = dec.feed(data)
            if out is not None and out.size:
                yield out
        tail = dec.flush()
        if tail is not None and tail.size:
            yield tail


class CartesiaTTS(TTSBackend):
    """Cartesia Sonic over WebSocket: raw PCM chunks, ~200-400 ms to first audio."""

    name = "cartesia"
    URL = "wss://api.cartesia.ai/tts/websocket"

    def __init__(self, api_key: str, voice_id: str, model: str = "sonic-2", sr: int = 24000):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = sr

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        import base64
        import json
        import uuid

        import websockets

        if not (self.api_key and self.voice_id):
            raise RuntimeError("Cartesia: JARVIS_CARTESIA_API_KEY / VOICE_ID belum diisi")
        text = text.strip()
        if not text:
            return

        url = f"{self.URL}?api_key={self.api_key}&cartesia_version=2024-06-10"
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({
                "model_id": self.model,
                "transcript": text,
                "voice": {"mode": "id", "id": self.voice_id},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                "language": "id",
                "context_id": str(uuid.uuid4()),
            }))
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "chunk" and msg.get("data"):
                    pcm = np.frombuffer(base64.b64decode(msg["data"]), dtype="<i2")
                    yield pcm.astype(np.float32) / 32768.0
                elif kind == "done":
                    return
                elif kind == "error":
                    raise RuntimeError(f"Cartesia error: {msg.get('error')}")


class SapiTTS(TTSBackend):
    """Offline PowerShell System.Speech fallback. Zero extra deps, robotic voice."""

    name = "sapi"
    sample_rate = 22050

    def __init__(self, rate: int = 1):
        self.rate = rate

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        import soundfile as sf

        text = text.strip()
        if not text:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            path = Path(fh.name)
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.Rate=%d;"
            "$s.SetOutputToWaveFile('%s');"
            "$s.Speak([Console]::In.ReadToEnd());$s.Dispose()"
            % (self.rate, str(path).replace("'", "''"))
        )
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate(text.encode("utf-8"))
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        finally:
            path.unlink(missing_ok=True)
        self.sample_rate = int(sr)
        yield data.mean(axis=1).astype(np.float32)


def build_tts(settings) -> TTSBackend:
    backend = (settings.tts_backend or "edge").lower()
    if backend == "cartesia":
        return CartesiaTTS(settings.cartesia_api_key, settings.cartesia_voice_id,
                           settings.cartesia_model, settings.sample_rate)
    if backend == "sapi":
        return SapiTTS()
    return EdgeTTS(settings.tts_voice, settings.tts_rate)
