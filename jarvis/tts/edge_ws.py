"""Persistent-websocket Edge TTS client.

Why not `edge_tts.Communicate`: it opens a brand new WSS + TLS handshake for
every sentence. Measured on this machine that costs 400-600 ms warm and
1.4-3.9 s whenever the connection has gone cold — and worse, *concurrent*
`Communicate` calls starve each other (two at once → ~4 s each).

This client keeps ONE websocket open for the whole stream session and replays
`speech.config` + `ssml` per utterance on that same socket. Measured first-audio
drops to ~250-350 ms and stays flat across a 5-hour session, which is what the
PRD's <1.5 s budget needs.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import aiohttp

from edge_tts.communicate import (  # reuse the maintained protocol bits
    _SSL_CTX,
    connect_id,
    date_to_string,
    mkssml,
    ssml_headers_plus_data,
)
from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
from edge_tts.data_classes import TTSConfig
from edge_tts.drm import DRM

OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"


class EdgeWS:
    """One long-lived socket, many utterances."""

    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz",
                 volume: str = "+0%", connect_timeout: float = 10.0,
                 receive_timeout: float = 30.0, first_chunk_timeout: float = 2.5,
                 max_turns: int = 4):
        self.config = TTSConfig(voice=voice, rate=rate, volume=volume, pitch=pitch,
                               boundary="SentenceBoundary")
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.first_chunk_timeout = first_chunk_timeout
        # The service drops the socket after ~5 utterances; recycling at 4 keeps
        # the reconnect predictable (~600 ms) instead of hitting MSG_CLOSED mid-turn.
        self.max_turns = max_turns
        self._turns = 0
        self._warm = False
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._lock = asyncio.Lock()
        self.last_activity = 0.0

    async def warm_if_stale(self, idle_s: float = 45.0) -> None:
        """Re-arm a socket that may have been dropped by the server during idle.

        Called on PTT-down: if the last utterance is older than `idle_s`, spend
        ~250 ms now (while the user is still talking) instead of paying a full
        reconnect on the first answer sentence.
        """
        import time as _time
        if _time.monotonic() - self.last_activity < idle_s:
            return
        try:
            async for _ in self.stream("Oke,"):
                pass
        except Exception:
            pass

    # ------------------------------------------------------------- connection
    async def _ensure(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws is not None and self._ws.closed:
            self._ws = None
        if self._ws is not None:
            return self._ws
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=_SSL_CTX, limit=4, ttl_dns_cache=300),
                trust_env=True,
            )
        url = (f"{WSS_URL}&ConnectionId={connect_id()}"
               f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
               f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}")
        self._ws = await asyncio.wait_for(
            self._session.ws_connect(
                url, compress=15, headers=DRM.headers_with_muid(WSS_HEADERS), ssl=_SSL_CTX,
                timeout=aiohttp.ClientWSTimeout(ws_close=self.connect_timeout),
                autoclose=False, autoping=True,
            ),
            timeout=self.connect_timeout,
        )
        return self._ws

    async def close(self) -> None:
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None

    async def _drop(self) -> None:
        try:
            await self.close()
        except Exception:
            pass
        self._warm = False

    # -------------------------------------------------------------- synthesis
    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw MP3 bytes for `text`, reusing the open socket.

        If the first audio frame takes longer than `first_chunk_timeout`, the
        socket is considered poisoned (the service sometimes stalls a turn) and
        the whole utterance is retried once on a fresh connection.
        """
        if not text.strip():
            return
        async with self._lock:
            if self._turns >= self.max_turns:
                await self._drop()
                self._turns = 0
            for attempt in (1, 2):
                try:
                    ws = await self._ensure()
                    # A freshly created socket pays TLS + service warmup, which can
                    # exceed the steady-state budget. Don't mistake that for a stall
                    # and burn the retry on a second cold connection.
                    budget = self.first_chunk_timeout if self._warm else self.first_chunk_timeout * 3
                    await self._send(ws, text)
                    agen = self._read(ws)
                    try:
                        head = await asyncio.wait_for(agen.__anext__(), timeout=budget)
                    except (asyncio.TimeoutError, StopAsyncIteration):
                        await self._drop()
                        if attempt == 2:
                            raise asyncio.TimeoutError("edge tts: stall on both attempts")
                        continue
                    self._warm = True
                    yield head
                    async for chunk in agen:
                        yield chunk
                    import time as _time
                    self.last_activity = _time.monotonic()
                    self._turns += 1
                    return
                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):
                    await self._drop()
                    if attempt == 2:
                        raise

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, text: str) -> None:
        await ws.send_str(
            f"X-Timestamp:{date_to_string()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            '"sentenceBoundaryEnabled":"true","wordBoundaryEnabled":"false"}'
            f',"outputFormat":"{OUTPUT_FORMAT}"'
            "}}}}\r\n"
        )
        await ws.send_str(ssml_headers_plus_data(connect_id(), date_to_string(),
                                                 mkssml(self.config, text)))

    async def _read(self, ws: aiohttp.ClientWebSocketResponse) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.receive_timeout
        while True:
            left = deadline - loop.time()
            if left <= 0:
                raise asyncio.TimeoutError("edge tts: turn.end not received")
            msg = await asyncio.wait_for(ws.receive(), timeout=left)
            if msg.type == aiohttp.WSMsgType.BINARY:
                if len(msg.data) < 2:
                    continue
                header_len = int.from_bytes(msg.data[:2], "big")
                header = msg.data[2:2 + header_len].decode("utf-8", "replace")
                if "Path:audio" not in header:
                    continue
                data = msg.data[2 + header_len:]
                if data:
                    yield data
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if "Path:turn.end" in msg.data:
                    return
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR,
                              aiohttp.WSMsgType.CLOSE):
                raise ConnectionError(f"edge tts socket closed: {msg.type}")