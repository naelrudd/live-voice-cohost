"""Entrypoint: PTT hotkeys + console text mode + local overlay server.

Run:
    .venv/Scripts/python.exe -m jarvis.main                 # hotkeys + mic
    .venv/Scripts/python.exe -m jarvis.main --text          # typed input
    .venv/Scripts/python.exe -m jarvis.main --no-server     # headless
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import sys
import threading
import time

from .audio.io import PlaybackQueue, Recorder, resolve_input, resolve_output
from .config import Settings, get_settings
from .hotkeys import Binding, HotkeyListener
from .llm.client import LLMClient
from .orchestrator import Orchestrator
from .server import EventBus, create_app
from .stt.backends import build_stt
from .tts.backends import build_tts


def build(settings: Settings, bus: EventBus):
    out = resolve_output(settings.output_device)
    inp = resolve_input(settings.input_device)
    player = PlaybackQueue(out.index)
    llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                    temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                    reasoning_effort=settings.llm_reasoning_effort)
    tts = build_tts(settings)
    stt = build_stt(settings)
    orch = Orchestrator(llm=llm, tts=tts, player=player, stt=stt,
                        tools_enabled=settings.tools_enabled,
                        streamer_name=settings.streamer_name, on_event=bus.emit)
    return player, llm, tts, stt, orch, out, inp


async def serve(settings: Settings, bus: EventBus, state: dict, submit=None,
                ptt_down=None, ptt_up=None) -> asyncio.Task:
    import uvicorn

    bus.bind_loop(asyncio.get_running_loop())
    app = create_app(bus, state, submit, ptt_down=ptt_down, ptt_up=ptt_up)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning")
    server = uvicorn.Server(config)
    return asyncio.create_task(server.serve(), name="http")


async def main_async(args) -> None:
    settings = get_settings()
    bus = EventBus()
    state: dict = {"status": "starting"}
    player, llm, tts, stt, orch, out_dev, in_dev = build(settings, bus)

    print("=" * 62)
    print(" Jarvis-Finance")
    print(f" LLM     : {settings.llm_model} @ {settings.llm_base_url}")
    print(f" STT     : {stt.name}" + ("" if stt.name != "none" else "  (no key → pakai --text)"))
    print(f" TTS     : {tts.name} / {settings.tts_voice}")
    print(f" Out     : {out_dev.index}:{out_dev.name}" + ("  [FALLBACK]" if out_dev.fallback else ""))
    print(f" In      : {in_dev.index}:{in_dev.name}" + ("  [FALLBACK]" if in_dev.fallback else ""))
    print(f" Overlay : http://{settings.host}:{settings.port}/overlay")
    print("=" * 62)

    loop = asyncio.get_running_loop()
    jobs: queue.Queue = queue.Queue()
    recorder = Recorder(in_dev.index) if stt.name != "none" else None
    stop_flag = threading.Event()

    def on_ptt_down():
        if recorder is None or recorder.recording:
            return
        orch.cancel()  # barge-in: kill current speech
        recorder.start()
        print("[ptt] listening… (tahan, bicara, lepas)")
        bus.emit({"type": "listening"})
        # Warm TTS during the ~1 s the user is speaking, so the first answer
        # sentence doesn't pay a cold ~1.4 s synthesis.
        loop.call_soon_threadsafe(lambda: asyncio.create_task(orch.prime_tts()))

    def on_ptt_up():
        if recorder is None or not recorder.recording:
            return
        pcm = recorder.stop()
        t_end = time.perf_counter()
        if pcm.size < 1600:  # <100 ms of audio: treat as accidental tap
            print("[ptt] tap diabaikan (<100 ms)")
            bus.emit({"type": "cancelled"})
            return
        print(f"[ptt] audio {pcm.size/16000:.1f}s → STT")
        jobs.put(("audio", pcm, t_end))

    def on_kill():
        orch.cancel()
        bus.emit({"type": "cancelled"})
        print("[kill] speech + tools dibatalkan")

    listener = HotkeyListener([
        Binding(settings.ptt_key, on_ptt_down, on_ptt_up),
        Binding(settings.ptt_fallback_key, on_ptt_down, on_ptt_up),
        Binding(settings.kill_key, on_kill),
    ])
    listener.start()
    print(f"[hotkey] PTT={settings.ptt_key} / {settings.ptt_fallback_key}   kill={settings.kill_key}")

    http_task = None
    if not args.no_server:
        http_task = await serve(settings, bus, state,
                                lambda text: jobs.put(("text", text, time.perf_counter())),
                                ptt_down=on_ptt_down, ptt_up=on_ptt_up)

    await orch.warmup()
    state["status"] = "ready"
    bus.emit({"type": "ready"})

    def console_reader():
        while not stop_flag.is_set():
            try:
                line = input()
            except EOFError:
                return
            if line.strip():
                jobs.put(("text", line.strip(), time.perf_counter()))

    if args.text:
        threading.Thread(target=console_reader, daemon=True).start()
        print("[input] mode teks aktif — ketik pertanyaan lalu Enter. Ctrl+C untuk keluar.")

    async def worker() -> None:
        while True:
            kind, payload, t_end = await asyncio.to_thread(jobs.get)
            try:
                if kind == "audio":
                    await orch.handle_audio(payload, 16000, t_speech_end=t_end)
                else:
                    await orch.respond(payload)
            except Exception as exc:
                print(f"[error] {type(exc).__name__}: {exc}")
                bus.emit({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    worker_task = asyncio.create_task(worker(), name="worker")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_flag.set()
        worker_task.cancel()
        listener.stop()
        if http_task:
            http_task.cancel()
        try:
            await tts.close()
        except Exception:
            pass
        player.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Jarvis-Finance voice co-host")
    ap.add_argument("--text", action="store_true", help="izinkan input lewat keyboard (tanpa STT)")
    ap.add_argument("--no-server", action="store_true", help="jangan jalankan server overlay")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[exit] bye")
        sys.exit(0)


if __name__ == "__main__":
    main()
