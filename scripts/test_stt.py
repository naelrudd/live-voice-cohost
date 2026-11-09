"""STT round-trip check without a microphone: TTS → 16 kHz → STT → text.

    .venv/Scripts/python.exe scripts/test_stt.py                # backend from .env
    .venv/Scripts/python.exe scripts/test_stt.py --local        # force faster-whisper
    .venv/Scripts/python.exe scripts/test_stt.py "buka chart XAUUSD timeframe 4 jam"

Tests the backend the app will actually use (`build_stt`), so a green run here
means the voice loop's STT hop is wired and fast enough, not just that some
model can decode audio.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from jarvis.config import get_settings  # noqa: E402
from jarvis.stt.backends import LocalWhisperSTT, build_stt  # noqa: E402
from jarvis.tts.backends import EdgeTTS  # noqa: E402

DEFAULT_PHRASES = [
    "harga emas dan IHSG sekarang berapa?",
    "buka chart XAUUSD timeframe empat jam",
    "berapa kurs dollar sekarang",
    "menurut lo emas besok naik apa turun?",
]


def to_16k(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample — plenty for a smoke test."""
    if sr == 16000:
        return pcm
    n = int(round(len(pcm) * 16000 / sr))
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pcm)), pcm).astype(np.float32)


async def main() -> None:
    argv = sys.argv[1:]
    force_local = "--local" in argv
    phrases = [a for a in argv if not a.startswith("--")] or DEFAULT_PHRASES
    settings = get_settings()
    tts = EdgeTTS()
    if force_local:
        stt = LocalWhisperSTT(settings.local_whisper_model, settings.local_whisper_device,
                              settings.local_whisper_compute, settings.stt_language,
                              cpu_threads=settings.local_whisper_threads)
    else:
        stt = build_stt(settings)
    await stt.warm()
    print(f"backend: {stt.name}")
    print(f"{'phrase':46} {'audio':>6} {'stt_ms':>7} {'match':>5}  heard")
    warm: list[float] = []
    for text in phrases:
        clip = await tts.synthesize(text)
        t0 = time.perf_counter()
        heard = await stt.transcribe(to_16k(clip.pcm, clip.sr), 16000)
        ms = (time.perf_counter() - t0) * 1000
        if len(warm) or ms < 5000:  # first call may include model load
            warm.append(ms)
        words = text.lower().split()
        hit = sum(w.strip("?:,") in heard.lower() for w in words)
        verdict = "OK" if heard.strip() and hit >= max(1, len(words) // 2) else "LOW"
        print(f"{text[:46]:46} {clip.duration:6.2f} {ms:7.0f} {verdict:>5}  {heard!r}")
    await tts.close()
    if warm:
        print(f"\nwarm stt median {sorted(warm)[len(warm) // 2]:.0f} ms "
              f"(budget < 1200 ms for the voice loop)")


if __name__ == "__main__":
    asyncio.run(main())