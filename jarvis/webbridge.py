"""kimi-webbridge detection, startup and install helper.

kimi-webbridge is a small local daemon (`127.0.0.1:10086`) plus a browser
extension. The daemon drives the user's *already running* browser, so the
control panel can open as a new tab in it instead of spawning a separate
browser window/profile.

Everything here is best-effort and must never crash the engine: callers fall
back to `webbrowser.open` when the bridge is missing, stopped, or has no
extension attached.

Layout (same on every OS, per upstream docs):
  ~/.kimi-webbridge/bin/kimi-webbridge[.exe]   daemon binary
  ~/.kimi-webbridge/                           pid, logs, identity

On Windows the daemon does not run reliably as a native process (a stale
`daemon.pid` makes `start` die), so we start it inside WSL when WSL is
available.

Install:  curl -fsSL https://cdn.kimi.com/webbridge/install.sh | bash
Extension: https://www.kimi.com/features/webbridge
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE_PORT = 10086
BRIDGE_HOST = "127.0.0.1"
STATUS_URL = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/status"
COMMAND_URL = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/command"

INSTALL_URL = "https://cdn.kimi.com/webbridge/install.sh"
EXTENSION_URL = "https://www.kimi.com/features/webbridge"

# ensure() outcomes
READY = "ready"
NO_EXTENSION = "no-extension"
NOT_RUNNING = "not-running"
NOT_INSTALLED = "not-installed"
FAILED = "failed"


def _home() -> Path:
    return Path.home() / ".kimi-webbridge"


def binary_path() -> Path | None:
    """Local daemon binary, or None when kimi-webbridge is not installed."""
    exe = "kimi-webbridge.exe" if sys.platform == "win32" else "kimi-webbridge"
    candidate = _home() / "bin" / exe
    return candidate if candidate.exists() else None


def status(timeout: float = 1.5) -> dict | None:
    """GET /status. None when the daemon is not listening."""
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def is_ready(st: dict | None = None) -> bool:
    """True when the daemon runs *and* a browser extension is attached."""
    st = st if st is not None else status()
    return bool(st and st.get("running") and st.get("extension_connected"))


def _wsl_distros() -> list[str]:
    if sys.platform != "win32" or not shutil.which("wsl"):
        return []
    try:
        proc = subprocess.run(
            ["wsl.exe", "-l", "-q"], capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return []
    raw = proc.stdout or b""
    for enc in ("utf-16-le", "utf-8"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        names = [ln.strip().strip("\x00") for ln in text.splitlines()]
        names = [n for n in names if n]
        if names:
            return names
    return []


def _wsl_start_argv(distro: str) -> list[str]:
    return [
        "wsl.exe", "-d", distro, "--", "bash", "-lc",
        "test -x ~/.kimi-webbridge/bin/kimi-webbridge && "
        "~/.kimi-webbridge/bin/kimi-webbridge start",
    ]


def _wsl_install_argv(distro: str) -> list[str]:
    return [
        "wsl.exe", "-d", distro, "--", "bash", "-lc",
        f"curl -fsSL {INSTALL_URL} | bash",
    ]


def _start_commands() -> list[list[str]]:
    """Every plausible way to start the daemon on this machine."""
    cmds: list[list[str]] = []
    on_path = shutil.which("kimi-webbridge")
    if on_path:
        cmds.append([on_path, "start"])
    local = binary_path()
    if local:
        cmds.append([str(local), "start"])
    for distro in _wsl_distros():
        cmds.append(_wsl_start_argv(distro))
    return cmds


def start(wait_s: float = 20.0) -> dict | None:
    """Start the daemon (idempotent) and wait until /status answers."""
    st = status()
    if st and st.get("running"):
        return st
    for cmd in _start_commands():
        try:
            subprocess.run(cmd, capture_output=True, timeout=25)
        except (OSError, subprocess.SubprocessError):
            continue
        deadline = time.time() + wait_s
        while time.time() < deadline:
            st = status()
            if st and st.get("running"):
                return st
            time.sleep(1.0)
    return status()


def ensure(auto_start: bool = True, wait_s: float = 20.0) -> tuple[str, dict | None]:
    """Detect the bridge, starting the daemon if needed.

    Returns (state, status_dict) where state is one of READY, NO_EXTENSION,
    NOT_RUNNING, NOT_INSTALLED or FAILED.
    """
    st = status()
    if st and st.get("running"):
        return (READY if st.get("extension_connected") else NO_EXTENSION), st

    if binary_path() is None and not _start_commands():
        return NOT_INSTALLED, st

    if not auto_start:
        return NOT_RUNNING, st

    st = start(wait_s)
    if st and st.get("running"):
        return (READY if st.get("extension_connected") else NO_EXTENSION), st
    return FAILED, st


def install(distro: str | None = None) -> bool:
    """Run the upstream installer (needs bash + curl; on Windows that is WSL)."""
    cmd: list[str] | None = None
    if sys.platform == "win32":
        distros = [distro] if distro else _wsl_distros()
        if not distros:
            print("[webbridge] WSL tidak terdeteksi. Install manual di Linux/macOS,")
            print(f"[webbridge] atau pakai ekstensi: {EXTENSION_URL}")
            return False
        cmd = _wsl_install_argv(distros[0])
    else:
        bash = shutil.which("bash")
        if not bash:
            print("[webbridge] bash tidak ditemukan; jalankan manual:")
            print(f"  curl -fsSL {INSTALL_URL} | bash")
            return False
        cmd = [bash, "-lc", f"curl -fsSL {INSTALL_URL} | bash"]

    print(f"[webbridge] Install: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[webbridge] Install gagal: {exc}")
        return False
    if proc.returncode != 0:
        print(f"[webbridge] Installer keluar dengan kode {proc.returncode}.")
        return False
    print("[webbridge] Installer selesai. Pastikan ekstensi browser terpasang:")
    print(f"  {EXTENSION_URL}")
    return True


def navigate(
    url: str,
    session: str = "jarvis-control",
    group_title: str = "Jarvis control",
    timeout: float = 5.0,
) -> bool:
    """Open `url` as a new tab in the user's running browser via the bridge."""
    payload = json.dumps(
        {
            "action": "navigate",
            "args": {"url": url, "newTab": True, "group_title": group_title},
            "session": session,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        COMMAND_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False
    return bool(data.get("ok"))


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 25.0) -> bool:
    """Block until a TCP port accepts connections (engine readiness)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def describe(st: dict | None = None) -> str:
    """One-line human summary for `cohost doctor`."""
    st = st if st is not None else status()
    if not st:
        if binary_path() is None:
            return f"tidak terpasang (install: cohost webbridge --install, ekstensi: {EXTENSION_URL})"
        return "terpasang, daemon mati (jalankan: cohost webbridge --start)"
    if st.get("extension_connected"):
        ver = str(st.get("version", "?")).lstrip("v")
        return f"aktif v{ver} — tab /control dibuka di browser yang sedang jalan"
    return "daemon jalan, ekstensi browser belum tersambung (buka browser, lalu ulangi)"
