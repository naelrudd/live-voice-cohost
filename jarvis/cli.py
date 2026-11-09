"""Command-line entry point for live-voice-cohost.

Installed as a console script (`cohost`) via `[project.scripts]`, and also
callable directly with `python -m jarvis.cli` before installation.

Subcommands:
  cohost setup          run the interactive configuration wizard (terminal)
  cohost setup --gui    run the wizard as a Tkinter window
  cohost start          start the voice co-host engine
  cohost stop           stop the engine (kills whatever holds the web port)
  cohost doctor         check audio devices, keys and endpoint reachability
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _print_banner() -> None:
    print("=" * 60)
    print("  live-voice-cohost — co-host AI untuk live stream finansial")
    print("=" * 60)


def cmd_setup(args: argparse.Namespace) -> int:
    from .setup import run_cli, run_gui

    if args.gui:
        try:
            run_gui()
            return 0
        except Exception as exc:
            print(f"[setup] GUI tidak dapat dibuka ({exc}), beralih ke CLI...")
    run_cli()
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    if args.browser:
        import webbrowser

        webbrowser.open("http://127.0.0.1:8848/control")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    python = sys.executable
    # exec so Ctrl+C and the TTS/STT signal handlers own the terminal cleanly
    os.execvp(python, [python, "-m", "jarvis.main"])
    return 0  # unreachable


def cmd_stop(args: argparse.Namespace) -> int:
    port = args.port
    if sys.platform == "win32":
        import subprocess

        try:
            net = subprocess.run(
                ["netstat", "-aon"], capture_output=True, text=True, check=True
            )
        except Exception as exc:
            print(f"[stop] netstat gagal: {exc}")
            return 1
        pids: set[str] = set()
        for line in net.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pids.add(parts[-1])
        if not pids:
            print(f"[stop] Tidak ada engine di port {port}.")
            return 0
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            print(f"[stop] Stopped PID {pid}")
        return 0

    print("[stop] Hanya didukung di Windows saat ini. Tutup proses secara manual.")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from .setup import (
        load_current_env,
        list_audio_devices,
        test_groq_key,
        test_llm_connection,
    )

    _print_banner()
    print("  Doctor — cek kesiapan sistem\n")

    env = load_current_env()
    inputs, outputs = list_audio_devices()
    print(f"  Mic terdeteksi : {len(inputs)}")
    for d in inputs[:5]:
        print(f"      - {d}")
    print(f"  Output terdeteksi: {len(outputs)}")
    for d in outputs[:5]:
        print(f"      - {d}")

    base = env.get("JARVIS_LLM_BASE_URL", "")
    model = env.get("JARVIS_LLM_MODEL", "")
    llm_key = (
        env.get("JARVIS_LLM_API_KEY")
        or env.get("JARVIS_GROQ_API_KEY")
        or env.get("OPENROUTER_API_KEY")
        or env.get("OPENAI_API_KEY")
        or ""
    )
    print(f"\n  LLM base : {base or '(belum di-set)'}")
    print(f"  LLM model: {model or '(belum di-set)'}")
    if base:
        ok, msg = test_llm_connection(base, llm_key)
        print(f"  Endpoint : {'OK' if ok else 'GAGAL'} — {msg}")

    groq_key = env.get("JARVIS_GROQ_API_KEY", "")
    print(f"  Groq key : {'ada' if groq_key else 'kosong (fallback STT lokal)'}")
    if groq_key:
        ok, msg = test_groq_key(groq_key)
        print(f"  Groq STT : {'OK' if ok else 'GAGAL'} — {msg}")

    tavily = env.get("TAVILY_API_KEY", "")
    print(f"  Tavily key: {'ada (web search aktif)' if tavily else 'kosong (web_search mati)'}")

    print("\n  Jalankan `cohost setup` untuk konfigurasi, atau `cohost start` untuk mulai.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cohost",
        description="CLI untuk live-voice-cohost: setup, start, stop, doctor.",
    )
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    p_setup = sub.add_parser("setup", help="Jalankan wizard konfigurasi interaktif")
    p_setup.add_argument("--gui", action="store_true", help="Pakai GUI (Tkinter) bukan terminal")
    p_setup.set_defaults(func=cmd_setup)

    p_start = sub.add_parser("start", help="Nyalakan engine voice co-host")
    p_start.add_argument("--browser", action="store_true", help="Buka /control di browser")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Hentikan engine")
    p_stop.add_argument("--port", type=int, default=8848, help="Port web engine (default 8848)")
    p_stop.set_defaults(func=cmd_stop)

    p_doc = sub.add_parser("doctor", help="Cek devices + key + endpoint")
    p_doc.set_defaults(func=cmd_doctor)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())