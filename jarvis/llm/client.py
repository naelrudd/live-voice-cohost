"""OpenAI-compatible streaming chat client with tool-call assembly + model fallback.

Works against any OpenAI-compatible `/v1/chat/completions` endpoint
(Groq, OpenRouter, Together, vLLM, LM Studio, or a local gateway).

`model` may be a comma-separated list ("primary,fallback1,fallback2").
Calls stream each candidate in order; an empty response (no text, no tool
call) or an HTTP error falls through to the next candidate. This keeps the
voice loop alive on flaky free tiers instead of answering "data belum dapet".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""

    def parsed(self) -> dict[str, Any]:
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass
class StreamEvent:
    """kind: 'text' | 'reasoning' | 'tool_calls' | 'done'"""
    kind: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, *,
                 temperature: float = 0.8, max_tokens: int = 220,
                 reasoning_effort: str = "", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = [m.strip() for m in model.split(",") if m.strip()]
        assert self.models, "at least one model required"
        self.model = self.models[0]  # primary (back-compat attribute)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, messages: list[dict], tools: list[dict] | None,
                 stream: bool, model: str) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def stream(self, messages: list[dict], tools: list[dict] | None = None,
                     client: httpx.AsyncClient | None = None) -> AsyncIterator[StreamEvent]:
        own = client is None
        client = client or httpx.AsyncClient(timeout=self.timeout)
        last_error: Exception | None = None
        try:
            for mi, model in enumerate(self.models):
                last_model = mi == len(self.models) - 1
                acc: dict[int, ToolCall] = {}
                finish: str | None = None
                produced = False
                try:
                    async with client.stream(
                        "POST", f"{self.base_url}/chat/completions",
                        json=self._payload(messages, tools, True, model),
                        headers=self._headers(),
                    ) as r:
                        if r.status_code != 200:
                            body = (await r.aread()).decode("utf-8", "replace")
                            raise RuntimeError(f"LLM {r.status_code}: {body[:300]}")
                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                payload = json.loads(data)
                                choice = payload["choices"][0]
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                            delta = choice.get("delta") or {}
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            reason = delta.get("reasoning_content") or delta.get("reasoning")
                            if reason:
                                yield StreamEvent("reasoning", text=reason)
                            if delta.get("content"):
                                produced = True
                                yield StreamEvent("text", text=delta["content"])
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                slot = acc.setdefault(idx, ToolCall())
                                if tc.get("id"):
                                    slot.id = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot.name = fn["name"]
                                if fn.get("arguments"):
                                    slot.arguments += fn["arguments"]
                    if acc:
                        produced = True
                        yield StreamEvent(
                            "tool_calls",
                            tool_calls=[acc[i] for i in sorted(acc)],
                        )
                    if produced:
                        yield StreamEvent("done", finish_reason=finish)
                        return
                    # Empty response (reasoning only, or nothing) — try next model.
                    last_error = RuntimeError(f"LLM empty response from {model}")
                    if not last_model:
                        continue
                except RuntimeError as exc:
                    last_error = exc
                    if last_model:
                        raise
                    continue
            if last_error:
                raise last_error
            raise RuntimeError("LLM: no models configured")
        finally:
            if own:
                await client.aclose()

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Single non-streaming call (used by smoke tests / warmup). Falls back across models."""
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            for mi, model in enumerate(self.models):
                try:
                    r = await c.post(
                        f"{self.base_url}/chat/completions",
                        json=self._payload(messages, tools, False, model),
                        headers=self._headers(),
                    )
                    if r.status_code == 200:
                        return r.json()
                    last_error = RuntimeError(f"LLM {r.status_code}: {r.text[:300]}")
                except Exception as exc:
                    last_error = exc
                if mi == len(self.models) - 1:
                    break
        raise last_error or RuntimeError("LLM: no models configured")
