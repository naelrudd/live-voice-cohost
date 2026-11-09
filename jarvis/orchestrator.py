"""Orchestrator: text/audio in → LLM streaming → sentence-level TTS → speakers.

Latency budget (voice):
  speech_end → transcript   ~50 ms PTT / 300-500 ms Groq STT
  transcript → LLM first token ~700-1400 ms (OpenAI-compatible, reasoning off)
  first token → first sentence → first audio ~400-600 ms (edge-tts)
Anything above ~1 s of dead air is hidden with a pre-synthesised filler ack.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .llm.client import LLMClient, ToolCall
from .persona import build_messages
from .tools import run_tool, tool_schemas

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

FILLERS = ["Oke,", "Hmm, bentar ya.", "Gue cek dulu."]
TOOL_ACKS = ["Bentar, gue tarik datanya dulu.", "Sabar ya, gue cek data live-nya."]
MAX_FILLER_CLIPS = 3
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
SOFT_BREAK = re.compile(r"(?<=[,;:])\s+")


def split_sentences(buffer: str, flush: bool = False) -> tuple[list[str], str]:
    """Pull complete sentences out of a streaming buffer.

    Long fragments with a comma are emitted early so TTS starts sooner; the
    remainder stays buffered until a real sentence boundary arrives.
    """
    out: list[str] = []
    # Models sometimes emit "\n\n" between spoken sentences; a newline handed to
    # TTS is either read as a pause or dropped, never as the intended sentence break.
    rest = buffer.replace("\r", " ").replace("\n", "  ")
    while True:
        m = SENTENCE_END.search(rest)
        if m:
            head, rest = rest[: m.end()], rest[m.end():]
            if head.strip():
                out.append(head.strip())
            continue
        if len(rest) > 80:
            m = SOFT_BREAK.search(rest)
            if m:
                head, rest = rest[: m.end()], rest[m.end():]
                if head.strip():
                    out.append(head.strip())
                continue
        break
    if flush and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


@dataclass
class TurnMetrics:
    t_speech_end: float = 0.0
    t_transcript: float = 0.0
    t_llm_first_token: float = 0.0
    t_first_audio: float = 0.0
    t_first_sound: float = 0.0
    text: str = ""
    answer: str = ""
    tools: list[str] = field(default_factory=list)
    round_ttft_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def ms(a: float, b: float) -> float | None:
            return round((b - a) * 1000, 1) if a and b else None

        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": self.text,
            "answer": self.answer[:300],
            "tools": self.tools,
            "rounds": self.round_ttft_ms,
            "stt_ms": ms(self.t_speech_end, self.t_transcript),
            "llm_ttft_ms": ms(self.t_transcript, self.t_llm_first_token),
            "first_sound_ms": ms(self.t_speech_end, self.t_first_sound),
            "first_audio_ms": ms(self.t_speech_end, self.t_first_audio),
        }


class Orchestrator:
    def __init__(self, *, llm: LLMClient, tts, player, stt=None,
                 tools_enabled: bool = True, max_tool_rounds: int = 2,
                 streamer_name: str = "Pak Febri",
                 on_event: Callable[[dict], None] | None = None):
        self.llm = llm
        self.tts = tts
        self.player = player
        self.stt = stt
        self.tools_enabled = tools_enabled
        self.max_tool_rounds = max_tool_rounds
        self.streamer_name = streamer_name
        self.history: list[dict] = []
        self.max_history = 12
        self.fillers: list[np.ndarray] = []
        self.tool_acks: list[np.ndarray] = []
        self._filler_sr = 24000
        self.on_event = on_event or (lambda ev: None)
        self._cancelled = False
        self._ack_pending = False

    # ------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Pre-synthesise fillers so the first second of LLM latency is masked."""
        async def one(text: str, bucket: list[np.ndarray]):
            try:
                clip = await self.tts.synthesize(text)
                if clip.pcm.size:
                    bucket.append(clip.pcm)
                    self._filler_sr = clip.sr
                    self.on_event({"type": "warmup", "detail": f"ack: {text}"})
            except Exception as exc:
                self.on_event({"type": "warmup_error", "detail": f"{text}: {exc}"})

        jobs = [(t, self.fillers) for t in FILLERS]
        jobs += [(t, self.tool_acks) for t in TOOL_ACKS]
        # Sequential, not parallel: four simultaneous cold WSS handshakes to the
        # Edge endpoint get throttled and every later sentence pays ~1.3 s instead
        # of ~0.5 s. One after another keeps the rest of the session warm.
        for text, bucket in jobs:
            await one(text, bucket)
        try:
            await asyncio.to_thread(self.player.warm, self.tts.sample_rate)
        except Exception as exc:
            self.on_event({"type": "warmup_error", "detail": f"device: {exc}"})
        if self.stt is not None:
            try:
                await self.stt.warm()  # whisper model load is ~5-50 s on CPU
                self.on_event({"type": "warmup", "detail": "stt"})
            except Exception as exc:
                self.on_event({"type": "warmup_error", "detail": f"stt: {exc}"})
        await self._warm_llm()

    async def _warm_llm(self) -> None:
        """Open the TCP/TLS path to the LLM and pay the model cold-start before
        the first real turn: a cold first call costs 1-2 s more than a warm one."""
        for tools in (False, True):
            try:
                await asyncio.wait_for(
                    self.llm.complete([{"role": "user", "content": "ok"}], tool_schemas(tools)),
                    timeout=25)
                self.on_event({"type": "warmup", "detail": f"llm tools={tools}"})
            except Exception as exc:
                self.on_event({"type": "warmup_error", "detail": f"llm: {exc}"})

    async def prime_tts(self) -> None:
        """Re-arm the TTS socket the moment PTT goes down (~1 s before the
        transcript exists) so a stale connection never lands on the answer."""
        try:
            await self.tts.prime()
        except Exception:
            pass

    def cancel(self) -> None:
        self._cancelled = True
        self.player.interrupt()

    # ------------------------------------------------------------- pipeline
    async def handle_audio(self, pcm: np.ndarray, sr: int,
                           t_speech_end: float | None = None) -> TurnMetrics:
        m = TurnMetrics(t_speech_end=t_speech_end or time.perf_counter())
        if self.stt is None:
            raise RuntimeError("STT tidak dikonfigurasi")
        text = await self.stt.transcribe(pcm, sr)
        m.t_transcript = time.perf_counter()
        m.text = text
        if not text.strip():
            return m
        await self.respond(text, metrics=m)
        return m

    async def respond(self, user_text: str, metrics: TurnMetrics | None = None) -> TurnMetrics:
        m = metrics or TurnMetrics(t_speech_end=time.perf_counter(), t_transcript=time.perf_counter())
        m.text = user_text
        self._cancelled = False
        self._ack_pending = False
        self.on_event({"type": "user", "text": user_text})
        # Warm the Edge TTS path while the LLM is still thinking: a cold first
        # synthesis costs ~1.4 s, a warm one ~0.45 s.
        prime = asyncio.create_task(self.prime_tts())

        messages = build_messages(self.history, user_text, streamer_name=self.streamer_name)
        answer_parts: list[str] = []
        sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
        first_audio = asyncio.Event()

        speaker = asyncio.create_task(self._speak_worker(sentence_q, first_audio, m))
        filler = asyncio.create_task(self._maybe_filler(first_audio, m))

        try:
            for round_no in range(self.max_tool_rounds + 1):
                pending_tools: list[ToolCall] = []
                buffer = ""
                t_round = time.perf_counter()
                round_marked = False
                async for ev in self.llm.stream(messages, tool_schemas(self.tools_enabled and round_no < self.max_tool_rounds)):
                    if self._cancelled:
                        break
                    if ev.kind == "text":
                        if not round_marked:
                            round_marked = True
                            m.round_ttft_ms.append(round((time.perf_counter() - t_round) * 1000, 1))
                        if not m.t_llm_first_token:
                            m.t_llm_first_token = time.perf_counter()
                        answer_parts.append(ev.text)
                        buffer += ev.text
                        sentences, buffer = split_sentences(buffer)
                        for s in sentences:
                            await sentence_q.put(s)
                            self.on_event({"type": "sentence", "text": s})
                    elif ev.kind == "tool_calls":
                        pending_tools = ev.tool_calls
                if self._cancelled:
                    break
                if buffer.strip():
                    await sentence_q.put(buffer.strip())
                if not pending_tools:
                    break
                # ---- execute tools in parallel, feed results back, loop once more
                if not first_audio.is_set() and self.tool_acks:
                    self._ack_pending = True  # filler task voices the ack, no extra task here
                messages.append({
                    "role": "assistant",
                    "content": "".join(answer_parts) or None,
                    "tool_calls": [
                        {"id": tc.id or f"call_{i}", "type": "function",
                         "function": {"name": tc.name, "arguments": tc.arguments or "{}"}}
                        for i, tc in enumerate(pending_tools)
                    ],
                })
                for tc in pending_tools:
                    m.tools.append(tc.name)
                    self.on_event({"type": "tool_call", "name": tc.name, "args": tc.parsed()})
                results = await asyncio.gather(
                    *(run_tool(tc.name, tc.parsed()) for tc in pending_tools),
                    return_exceptions=True,
                )
                for i, (tc, result) in enumerate(zip(pending_tools, results)):
                    if isinstance(result, BaseException):
                        result = {"ok": False, "error": f"{type(result).__name__}: {result}"}
                    self.on_event({"type": "tool_result", "name": tc.name, "ok": result.get("ok"),
                                   "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id or f"call_{i}",
                        "content": json.dumps(result, ensure_ascii=False)[:1600],
                    })
                answer_parts = []
        finally:
            await sentence_q.put(None)
            filler.cancel()
            if not prime.done():
                prime.cancel()
            try:
                await speaker
            except asyncio.CancelledError:
                pass

        answer = "".join(answer_parts).strip()
        m.answer = answer
        if answer:
            self._remember(user_text, answer)
        self._log(m)
        self.on_event({"type": "turn_done", "answer": answer, "metrics": m.as_dict()})
        return m

    async def _speak_worker(self, q: asyncio.Queue, first_audio: asyncio.Event, metrics: TurnMetrics) -> None:
        """Feed every sentence into the device queue as soon as it is synthesised.

        The device thread plays strictly in FIFO order, so we do NOT wait for
        sentence N to finish before synthesising N+1 — that overlap saves ~1-2 s
        per sentence. We only wait for everything to drain at the very end.
        """
        handles = []
        try:
            while True:
                sentence = await q.get()
                if sentence is None:
                    break
                handle = self.player.open_stream(self.tts.sample_rate)
                handles.append(handle)
                try:
                    await self._pump(handle, sentence, first_audio, metrics)
                except Exception as exc:
                    self.on_event({"type": "tts_error", "detail": str(exc)[:200]})
                    handle.end()
        finally:
            for h in handles:
                h.end()
            if handles and not self._cancelled:
                await asyncio.gather(*(asyncio.to_thread(h.done.wait) for h in handles),
                                     return_exceptions=True)

    async def _pump(self, handle, text: str, first_audio: asyncio.Event, metrics: TurnMetrics) -> None:
        """Stream one sentence's PCM into `handle`, ending it when the stream ends."""
        t0 = time.perf_counter()
        self.on_event({"type": "tts_start", "chars": len(text),
                       "at_ms": round((t0 - metrics.t_speech_end) * 1000)})
        started = False
        async for chunk in self.tts.stream(text):
            handle.feed(chunk)
            if not started:
                started = True
                self.on_event({"type": "tts_first_chunk",
                               "ms": round((time.perf_counter() - t0) * 1000),
                               "chars": len(text)})
                if not metrics.t_first_sound:
                    metrics.t_first_sound = time.perf_counter()
                if not first_audio.is_set():
                    first_audio.set()
                    metrics.t_first_audio = time.perf_counter()
        handle.end()

    async def _maybe_filler(self, first_audio: asyncio.Event, metrics: TurnMetrics) -> None:
        """Keep the line alive instead of leaving dead air while LLM + TTS spin up.

        At most `MAX_FILLER_CLIPS` clips: two or three natural bridges sound like a
        thinking person, five in a row sound broken. After that we stay quiet.
        """
        await asyncio.sleep(0.3)
        deadline = time.perf_counter() + 8.0
        spoken = 0
        while (not first_audio.is_set() and time.perf_counter() < deadline
               and spoken < MAX_FILLER_CLIPS):
            if self._ack_pending and self.tool_acks:
                self._ack_pending = False
                pcm = random.choice(self.tool_acks)
                self.on_event({"type": "filler", "detail": "tool_ack",
                               "at_ms": round((time.perf_counter() - metrics.t_speech_end) * 1000)})
            elif self.fillers:
                pcm = self.fillers[spoken % len(self.fillers)]
                self.on_event({"type": "filler", "n": spoken + 1,
                               "at_ms": round((time.perf_counter() - metrics.t_speech_end) * 1000)})
            else:
                return
            spoken += 1
            if not metrics.t_first_sound:
                metrics.t_first_sound = time.perf_counter()
            done = self.player.play_clip(pcm, self._filler_sr)
            await asyncio.to_thread(done.wait, 5.0)
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------- helpers
    def _remember(self, user_text: str, answer: str) -> None:
        # Only keep the spoken answer, not tool JSON, to keep the prompt small.
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _log(self, m: TurnMetrics) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        rec = m.as_dict()
        with (LOG_DIR / "latency.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[turn] stt={rec['stt_ms']}ms llm_ttft={rec['llm_ttft_ms']}ms "
              f"first_audio={rec['first_audio_ms']}ms tools={rec['tools']}")
