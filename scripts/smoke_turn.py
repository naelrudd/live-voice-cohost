"""End-to-end turn without a mic: text → LLM (tools) → streaming TTS → device.

    .venv/Scripts/python.exe scripts/smoke_turn.py "harga emas sekarang berapa?"
    .venv/Scripts/python.exe scripts/smoke_turn.py --save --quiet "..."

Prints the measured latency breakdown and, with --save, writes the answer audio
to logs/turn.wav so the result can be inspected outside the live session.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.audio.io import PlaybackQueue, resolve_output  # noqa: E402
from jarvis.config import get_settings  # noqa: E402
from jarvis.llm.client import LLMClient  # noqa: E402
from jarvis.orchestrator import Orchestrator  # noqa: E402
from jarvis.tts.backends import build_tts  # noqa: E402

T0 = 0.0


class RecordingPlayer(PlaybackQueue):
    """PlaybackQueue that also captures every PCM chunk that reaches the device."""

    def __init__(self, device, record: bool):
        super().__init__(device)
        self.chunks: list[np.ndarray] = []
        self.sr: int | None = None
        self._record = record

    def open_stream(self, sr):
        handle = super().open_stream(sr)
        if not self._record:
            return handle
        orig_feed = handle.feed

        def feed(pcm):
            self.chunks.append(np.asarray(pcm, dtype=np.float32).copy())
            self.sr = sr
            orig_feed(pcm)

        handle.feed = feed  # type: ignore[method-assign]
        return handle


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="jangan cetak tiap event")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    out = resolve_output(s.output_device)
    player = RecordingPlayer(out.index, record=True)

    def on_event(ev: dict) -> None:
        if args.quiet:
            return
        brief = {k: v for k, v in ev.items() if k not in ("type", "result")}
        print(f"  +{(time.perf_counter()-T0)*1000:7.0f}ms · {ev['type']}: {brief}")

    llm = LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model,
                    temperature=s.llm_temperature, max_tokens=s.llm_max_tokens,
                    reasoning_effort=s.llm_reasoning_effort)
    orch = Orchestrator(llm=llm, tts=build_tts(s), player=player,
                        tools_enabled=not args.no_tools, on_event=on_event)

    print(f"out device: {out.index}:{out.name}")
    t0 = time.perf_counter()
    await orch.warmup()
    print(f"warmup (ack TTS, parallel): {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"user: {args.text}")
    global T0
    T0 = time.perf_counter()
    m = await orch.respond(args.text)
    print("\n--- metrics ---")
    for k, v in m.as_dict().items():
        print(f"{k:16} {v}")

    await asyncio.sleep(0.2)  # let the device drain
    if player.chunks:
        audio = np.concatenate(player.chunks)
        print(f"audio out: {audio.size / (player.sr or 24000):.2f}s  sr={player.sr}")
        if args.save:
            import soundfile as sf

            Path("logs").mkdir(exist_ok=True)
            sf.write("logs/turn.wav", audio, player.sr or 24000)
            print("saved logs/turn.wav")
    player.close()
    _ = threading


if __name__ == "__main__":
    asyncio.run(main())
