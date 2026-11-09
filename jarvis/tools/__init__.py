"""Tool registry: one dispatch point for every capability the co-host has.

Split by risk:
- `finance`  → read-only data (prices, web search). Safe, no confirmation.
- `desktop`  → opens windows/tabs. Reversible, guardrail-checked, no clicks on
               Buy/Sell (see `Settings.deny_action_list`).

Nothing here can transact. That is a hard requirement from the PRD: the agent
must never be able to press a broker's Buy/Sell button.
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import desktop, finance

SCHEMAS: list[dict] = [
    *finance.TOOL_SCHEMAS,
    {
        "type": "function",
        "function": {
            "name": "open_chart",
            "description": ("Buka chart TradingView untuk pair/aset di browser streamer. "
                            "Pakai ini kalau dia minta 'buka chart XAUUSD', 'liat IHSG', "
                            "'ganti timeframe 4 jam'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "pair/aset: xauusd, emas, ihsg, bbca, btc, usdidr"},
                    "interval": {"type": "string",
                                 "description": "timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Buka URL apa pun di browser dedicated (tab baru) — berita, portal, dashboard.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL lengkap http(s)"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Buka aplikasi desktop: obs, spotify, excel, notepad, telegram, kalkulator.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "nama aplikasi"}},
                "required": ["name"],
            },
        },
    },
]


async def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "open_chart":
        return await asyncio.to_thread(desktop.open_chart, str(args.get("query", "")),
                                       str(args.get("interval", "")))
    if name == "open_url":
        return await asyncio.to_thread(desktop.open_url, str(args.get("url", "")))
    if name == "open_app":
        return await asyncio.to_thread(desktop.open_app, str(args.get("name", "")))
    return await finance.run_tool(name, args)


def tool_schemas(enabled: bool) -> list[dict]:
    return SCHEMAS if enabled else []