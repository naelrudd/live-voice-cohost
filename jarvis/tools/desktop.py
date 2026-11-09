"""Desktop / browser control tools — the "computer use" half of the co-host.

Design rules (same as finance.py):
- Read-only-ish: we can *open* things, never transact. Every action is checked
  against `Settings.deny_action_list` before it runs.
- No screenshot-driven agents here. Opening a URL in a dedicated Chrome profile
  is deterministic, ~300 ms, and keeps the streamer's TradingView/broker logins.
- Every action returns `{"ok": bool, ...}` and the spoken ack is produced by the
  orchestrator, not by the tool.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import get_settings

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# market keyword → TradingView symbol
TV_SYMBOLS: dict[str, str] = {
    "xauusd": "OANDA:XAUUSD",
    "emas": "OANDA:XAUUSD",
    "gold": "OANDA:XAUUSD",
    "xagusd": "OANDA:XAGUSD",
    "perak": "OANDA:XAGUSD",
    "ihsg": "IDX:COMPOSITE",
    "jkse": "IDX:COMPOSITE",
    "bbca": "IDX:BBCA",
    "bbri": "IDX:BBRI",
    "tlkm": "IDX:TLKM",
    "btc": "BINANCE:BTCUSDT",
    "bitcoin": "BINANCE:BTCUSDT",
    "eth": "BINANCE:ETHUSDT",
    "usdidr": "FX_IDC:USDIDR",
    "dolar": "FX_IDC:USDIDR",
    "eurusd": "OANDA:EURUSD",
}

TV_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                "1h": "60", "4h": "240", "1d": "D", "1w": "W"}


def _denied(text: str) -> str | None:
    low = (text or "").lower()
    for word in get_settings().deny_action_list:
        if word and word in low:
            return word
    return None


def _chrome() -> str | None:
    env = get_settings().chrome_path
    if env and Path(env).exists():
        return env
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _profile_dir() -> Path:
    d = Path(get_settings().chrome_profile_dir or (Path.home() / "jarvis-chrome"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def tv_url(query: str, interval: str = "") -> str | None:
    q = (query or "").lower().replace("/", "").replace(" ", "").replace("_", "")
    symbol = next((sym for key, sym in TV_SYMBOLS.items() if key in q), None)
    if not symbol:
        return None
    iv = TV_INTERVALS.get((interval or "").lower(), "")
    suffix = f"?interval={iv}" if iv else ""
    return f"https://www.tradingview.com/chart/{suffix}#symbol={symbol}"


def open_url(url: str, new_window: bool = False) -> dict[str, Any]:
    """Open URL in the dedicated Chrome profile (keeps logins, no new browser)."""
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": f"url tidak valid: {url!r}"}
    blocked = _denied(url)
    if blocked:
        return {"ok": False, "error": f"aksi diblokir guardrail: {blocked!r}"}
    chrome = _chrome()
    if not chrome:
        return {"ok": False, "error": "Chrome/Brave/Edge tidak ditemukan, set JARVIS_CHROME_PATH"}
    args = [chrome, f"--user-data-dir={_profile_dir()}"]
    if new_window:
        args.append("--new-window")
    args.append(url)
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "action": "open_url", "url": url, "at": time.strftime("%H:%M:%S")}


def open_chart(query: str, interval: str = "") -> dict[str, Any]:
    url = tv_url(query, interval)
    if not url:
        return {"ok": False, "error": f"pair/chart tidak dikenal: {query!r}",
                "hint": "contoh: xauusd, ihsg, bbca, btc"}
    res = open_url(url)
    if res.get("ok"):
        res.update({"chart": query, "interval": interval or "default"})
    return res


APP_ALIASES: dict[str, str] = {
    "obs": "obs64.exe",
    "spotify": "Spotify.exe",
    "telegram": "Telegram.exe",
    "whatsapp": "WhatsApp.exe",
    "excel": "EXCEL.EXE",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "kalkulator": "calc.exe",
    "explorer": "explorer.exe",
}


def open_app(name: str) -> dict[str, Any]:
    blocked = _denied(name)
    if blocked:
        return {"ok": False, "error": f"aksi diblokir guardrail: {blocked!r}"}
    exe = APP_ALIASES.get((name or "").strip().lower())
    if not exe:
        return {"ok": False, "error": f"aplikasi tidak dikenal: {name!r}",
                "hint": "yang didukung: " + ", ".join(sorted(APP_ALIASES))}
    try:
        os.startfile(exe)  # noqa: S606 - fixed alias table, no user string
    except Exception:
        try:
            subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "action": "open_app", "app": name, "at": time.strftime("%H:%M:%S")}