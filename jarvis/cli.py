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


def _open_control_browser(url: str = "http://127.0.0.1:8848/control") -> bool:
    """Buka control panel di browser yang sedang berjalan.

    Alur: deteksi kimi-webbridge -> nyalakan daemon bila mati -> buka tab baru
    di browser pengguna. Jika bridge tidak ada / gagal, fallback ke
    `webbrowser.open` standar OS supaya `cohost start` tidak pernah buntu.
    """
    import webbrowser

    from . import webbridge

    state, st = webbridge.ensure()

    if state == webbridge.READY:
        if webbridge.navigate(url):
            print("[start] Control panel dibuka di browser aktif via webbridge.")
            return True
        print("[start] Webbridge aktif tapi navigate gagal; pakai browser default.")

    elif state == webbridge.NO_EXTENSION:
        print(
            "[start] kimi-webbridge daemon jalan, ekstensi browser belum tersambung.\n"
            f"        Buka browser sekali, lalu ulangi. Ekstensi: {webbridge.EXTENSION_URL}"
        )

    elif state == webbridge.NOT_INSTALLED:
        print(
            "[start] kimi-webbridge belum terpasang — tab tidak bisa dibuka di browser\n"
            "        yang sedang jalan. Pasang dengan: cohost webbridge --install\n"
            f"        (atau ekstensi manual: {webbridge.EXTENSION_URL})"
        )

    elif state == webbridge.FAILED:
        print(
            "[start] kimi-webbridge terpasang tapi daemon gagal dinyalakan.\n"
            "        Cek: cohost webbridge --status"
        )

    try:
        webbrowser.open(url)
        print("[start] Control panel dibuka via browser default OS.")
        return True
    except Exception as exc:
        print(f"[start] Gagal membuka browser ({exc}). Buka manual: {url}")
        return False


def cmd_start(args: argparse.Namespace) -> int:
    """Nyalakan engine, lalu buka /control setelah port benar-benar siap."""
    import subprocess

    from . import webbridge

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    python = sys.executable
    url = f"http://127.0.0.1:{args.port}/control"

    proc = subprocess.Popen([python, "-m", "jarvis.main"])
    try:
        if args.browser:
            if webbridge.wait_for_port(args.port, timeout=args.wait):
                _open_control_browser(url)
            else:
                print(f"[start] Engine belum siap setelah {args.wait:.0f}s; membuka tab tetap.")
                _open_control_browser(url)
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[start] Berhenti.")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


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


def cmd_webbridge(args: argparse.Namespace) -> int:
    """Deteksi / nyalakan / pasang kimi-webbridge (browser bridge)."""
    from . import webbridge

    _print_banner()
    state, st = webbridge.ensure(auto_start=False)
    print(f"  Status : {webbridge.describe(st)}")

    if args.install:
        print("  Install: menjalankan installer resmi…")
        if not webbridge.install(args.distro):
            return 1
        state, st = webbridge.ensure(auto_start=True)
        print(f"  Sesudah install: {webbridge.describe(st)}")
        return 0

    if args.start or state in (webbridge.NOT_RUNNING, webbridge.FAILED):
        print("  Menyalakan daemon…")
        st = webbridge.start()
        if st:
            state, st = webbridge.ensure(auto_start=False)
        print(f"  Status : {webbridge.describe(st)}")

    if args.status:
        if st:
            for key in ("running", "port", "version", "extension_connected", "extension_id"):
                print(f"    {key}: {st.get(key)}")
        else:
            print("    (daemon tidak menjawab /status)")

    return 0 if webbridge.is_ready(st) else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .setup import (
        load_current_env,
        list_audio_devices,
        test_groq_key,
        test_llm_connection,
    )
    from . import webbridge

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

    st = webbridge.status()
    print(f"  Webbridge : {webbridge.describe(st)}")

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
    p_start.add_argument(
        "--browser",
        dest="browser",
        action="store_true",
        default=True,
        help="Buka /control di browser aktif (default: aktif)",
    )
    p_start.add_argument(
        "--no-browser",
        dest="browser",
        action="store_false",
        help="Jangan buka browser otomatis",
    )
    p_start.add_argument(
        "--port",
        type=int,
        default=8848,
        help="Port web engine (default 8848)",
    )
    p_start.add_argument(
        "--wait",
        type=float,
        default=25.0,
        help="Max detik tunggu engine siap sebelum paksa buka browser (default 25)",
    )
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Hentikan engine")
    p_stop.add_argument("--port", type=int, default=8848, help="Port web engine (default 8848)")
    p_stop.set_defaults(func=cmd_stop)

    p_web = sub.add_parser("webbridge", help="Deteksi / nyalakan / pasang kimi-webbridge")
    p_web.add_argument("--status", action="store_true", help="Tampilkan detail /status")
    p_web.add_argument("--start", action="store_true", help="Nyalakan daemon (idempotent)")
    p_web.add_argument("--install", action="store_true", help="Jalankan installer resmi (via WSL di Windows)")
    p_web.add_argument("--distro", help="Nama distro WSL untuk --install (default: otomatis)")
    p_web.set_defaults(func=cmd_webbridge)

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