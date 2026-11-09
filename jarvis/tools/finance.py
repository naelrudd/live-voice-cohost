"""Tool implementations + [OI] function schemas.

Design rules:
- Every tool returns a small dict; failures are explicit (`ok: False`) so the
  model can say "data gagal" instead of hallucinating a number.
- No tool can transact. Read-only by construction.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import httpx

# ---------------------------------------------------------------- price feed

# name fragments → Yahoo Finance symbol
SYMBOLS: dict[str, str] = {
    "xauusd": "GC=F",
    "gold": "GC=F",
    "emas": "GC=F",
    "xau": "GC=F",
    "silver": "SI=F",
    "perak": "SI=F",
    "ihsg": "^JKSE",
    "jkse": "^JKSE",
    "bbca": "BBCA.JK",
    "bbri": "BBRI.JK",
    "tlkm": "TLKM.JK",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "usdidr": "IDR=X",
    "usd": "IDR=X",
    "dollar": "IDR=X",
    "dolar": "IDR=X",
    "oil": "CL=F",
    "minyak": "CL=F",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
}

LABELS = {
    "GC=F": "Harga emas (COMEX gold futures, USD/oz)",
    "SI=F": "Perak (USD/oz)",
    "^JKSE": "IHSG",
    "BTC-USD": "Bitcoin (USD)",
    "ETH-USD": "Ethereum (USD)",
    "IDR=X": "USD/IDR",
    "CL=F": "Minyak WTI (USD/barel)",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
}


def _is_subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def resolve_symbol(query: str) -> str | None:
    """Map free speech to a Yahoo symbol, tolerant of STT mishears.

    Whisper turns "IHSG" into "nih hsg" and "XAUUSD" into "sawus" often enough
    that exact matching drops real requests; a subsequence check keeps
    "dnihsgsekarang" → IHSG working without opening the door to short keys
    ("eth", "btc") matching any sentence by accident.
    """
    q = (query or "").lower().replace("/", "").replace(" ", "")
    if not q:
        return None
    if q.upper() in {v for v in SYMBOLS.values()}:
        return q.upper()
    for key, sym in SYMBOLS.items():
        if key in q:
            return sym
    for key, sym in SYMBOLS.items():
        if len(key) >= 4 and _is_subsequence(key, q):
            return sym
    if q.isupper() and len(q) <= 6:  # raw ticker like BBCA
        return q
    return None


async def get_price(query: str) -> dict[str, Any]:
    """Quote via Yahoo Finance chart endpoint (no API key, ~200-500 ms)."""
    symbol = resolve_symbol(query)
    if not symbol:
        return {"ok": False, "error": f"simbol tidak dikenal: {query!r}",
                "hint": "pakai web_search untuk nama aset seperti emas Antam"}

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "5d", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        async with httpx.AsyncClient(timeout=8, headers=headers) as c:
            r = await c.get(url, params=params)
        if r.status_code != 200:
            return {"ok": False, "error": f"yahoo {r.status_code}", "symbol": symbol}
        result = (r.json().get("chart") or {}).get("result") or []
        if not result:
            return {"ok": False, "error": "no data", "symbol": symbol}
        meta = result[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        currency = meta.get("currency", "")
        change = None
        pct = None
        if price is not None and prev:
            change = round(price - prev, 4)
            pct = round((price - prev) / prev * 100, 2)
        ts = meta.get("regularMarketTime")
        as_of = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else None
        return {
            "ok": True,
            "symbol": symbol,
            "label": LABELS.get(symbol, meta.get("shortName") or symbol),
            "price": price,
            "currency": currency,
            "change": change,
            "change_pct": pct,
            "previous_close": prev,
            "as_of": as_of,
        }
    except Exception as exc:  # network/DNS/timeout
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "symbol": symbol}


# ---------------------------------------------------------------- web search


async def web_search(query: str, max_results: int = 4) -> dict[str, Any]:
    """Tavily search, trimmed to what a talking agent needs."""
    import os

    key = os.environ.get("JARVIS_TAVILY_API_KEY") or os.environ.get("MCP_TAVILY_API_KEY") or ""
    if not key:
        return {"ok": False, "error": "TAVILY key belum diset (JARVIS_TAVILY_API_KEY)"}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post("https://api.tavily.com/search", json={
                "api_key": key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True,
                "country": "indonesia",
            })
        if r.status_code != 200:
            return {"ok": False, "error": f"tavily {r.status_code}: {r.text[:160]}"}
        data = r.json()
        results = [
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "snippet": (it.get("content") or "")[:400],
            }
            for it in (data.get("results") or [])[:max_results]
        ]
        return {"ok": True, "answer": data.get("answer"), "results": results}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- schemas


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Ambil harga pasar terkini dan akurat: emas/XAUUSD, IHSG, saham ID, crypto, USD/IDR, minyak.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nama aset, contoh: 'xauusd', 'ihsg', 'bitcoin', 'usd idr', 'bbca'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Cari berita/regulasi/harga yang tidak ada di feed harga (contoh: harga emas Antam hari ini, aturan OJK terbaru).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pertanyaan pencarian, bahasa Indonesia lebih baik"},
                },
                "required": ["query"],
            },
        },
    },
]


async def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "get_price":
        return await get_price(str(args.get("query", "")))
    if name == "web_search":
        return await web_search(str(args.get("query", "")))
    return {"ok": False, "error": f"tool tidak dikenal: {name}"}


def tool_schemas(enabled: bool) -> list[dict]:
    return TOOL_SCHEMAS if enabled else []
