"""Benchmark TTFT with the real persona prompt + tool schemas.

    .venv/Scripts/python.exe scripts/bench_llm.py [runs]
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.config import get_settings  # noqa: E402
from jarvis.llm.client import LLMClient  # noqa: E402
from jarvis.persona import build_messages  # noqa: E402
from jarvis.tools.finance import tool_schemas  # noqa: E402

MODELS = [
    "inf/cbcn/deepseek-v4.1-flash",
    "free-coding",
    "inf/cbcn/glm-5.3-flash",
    "inf/ocb/deepseek-v4.1-flash",
    "inf/cb/glm-5.3",
]
PROMPTS = [
    "harga emas sekarang berapa?",
    "menurut lo IHSG besok naik apa turun?",
    "kenapa crypto turun malam ini?",
]


async def once(client: LLMClient, prompt: str, with_tools: bool) -> tuple[float | None, int]:
    msgs = build_messages([], prompt)
    t0 = time.perf_counter()
    ttft = None
    n = 0
    async for ev in client.stream(msgs, tool_schemas(with_tools)):
        if ev.kind == "text":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            n += 1
    return ttft, n


async def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s = get_settings()
    print(f"{'model':32} {'tools':6} {'ttft_med':>9} {'ttft_min':>9} {'chunks':>7}")
    for model in MODELS:
        for tools in (True, False):
            client = LLMClient(s.llm_base_url, s.llm_api_key, model,
                               temperature=s.llm_temperature, max_tokens=s.llm_max_tokens,
                               reasoning_effort=s.llm_reasoning_effort)
            ttfts, chunks = [], 0
            for i in range(runs):
                try:
                    t, n = await once(client, PROMPTS[i % len(PROMPTS)], tools)
                except Exception as exc:
                    print(f"{model:32} {str(tools):6} ERR {type(exc).__name__}: {str(exc)[:60]}")
                    ttfts = []
                    break
                if t:
                    ttfts.append(t)
                    chunks = n
            if ttfts:
                print(f"{model:32} {str(tools):6} {statistics.median(ttfts):9.0f} {min(ttfts):9.0f} {chunks:7}")


if __name__ == "__main__":
    asyncio.run(main())
