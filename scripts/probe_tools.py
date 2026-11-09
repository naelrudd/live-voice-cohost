"""Measure time-to-tool-call and time-to-first-text for the voice tool path.

    .venv/Scripts/python.exe scripts/probe_tools.py [runs]
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
    "inf/ocb/deepseek-v4.1-flash",
    "free-coding",
    "inf/cbcn/glm-5.3-flash",
]
EFFORTS = ["", "none"]
PROMPT = "harga emas dan IHSG sekarang berapa?"


async def once(client: LLMClient, tools: bool) -> tuple[float | None, float | None, list[str]]:
    msgs = build_messages([], PROMPT)
    t0 = time.perf_counter()
    t_text = t_tool = None
    names: list[str] = []
    async for ev in client.stream(msgs, tool_schemas(tools)):
        if ev.kind == "text" and t_text is None:
            t_text = (time.perf_counter() - t0) * 1000
        elif ev.kind == "tool_calls" and t_tool is None:
            t_tool = (time.perf_counter() - t0) * 1000
            names = [tc.name for tc in ev.tool_calls]
    return t_text, t_tool, names


async def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    s = get_settings()
    print(f"{'model':30} {'eff':5} {'tools':6} {'text_med':>9} {'tool_med':>9} {'calls'}")
    for model in MODELS:
        for effort in EFFORTS:
            for tools in (True, False):
                client = LLMClient(s.llm_base_url, s.llm_api_key, model,
                                   temperature=s.llm_temperature,
                                   max_tokens=s.llm_max_tokens,
                                   reasoning_effort=effort)
                texts, tools_ms, names = [], [], []
                for _ in range(runs):
                    try:
                        t_text, t_tool, names = await once(client, tools)
                    except Exception as exc:
                        print(f"{model:30} {effort or '-':5} {str(tools):6} ERR "
                              f"{type(exc).__name__}: {str(exc)[:50]}")
                        texts = []
                        break
                    if t_text:
                        texts.append(t_text)
                    if t_tool:
                        tools_ms.append(t_tool)
                if not texts and not tools_ms:
                    continue
                tm = f"{statistics.median(texts):9.0f}" if texts else "        -"
                km = f"{statistics.median(tools_ms):9.0f}" if tools_ms else "        -"
                print(f"{model:30} {effort or '-':5} {str(tools):6} {tm} {km} {','.join(names)}")


if __name__ == "__main__":
    asyncio.run(main())
