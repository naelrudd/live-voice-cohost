"""Local FastAPI: state endpoint, OBS overlay widget, and an event WebSocket."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

OVERLAY = (Path(__file__).parent / "overlay.html").read_text(encoding="utf-8")
CONTROL = (Path(__file__).parent / "control.html").read_text(encoding="utf-8")


class EventBus:
    """Fan-out orchestrator events to the overlay and the console."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.last: dict = {"type": "boot"}
        self.loop: asyncio.AbstractEventLoop | None = None
        # Types worth seeing in the console: a silent tts_error means the answer
        # is printed in the log but nobody on stream ever heard it.
        self.echo = {"error", "tts_error", "warmup_error", "tool_call", "tool_result"}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def emit(self, event: dict) -> None:
        self.last = event
        if event.get("type") in self.echo:
            print(f"[{event.get('type')}] "
                  + json.dumps({k: v for k, v in event.items() if k != 'type'},
                               ensure_ascii=False, default=str)[:300])
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(event, ensure_ascii=False, default=str)
        for ws in list(self.clients):
            asyncio.create_task(self._send(ws, data))

    async def _send(self, ws: WebSocket, data: str) -> None:
        try:
            await ws.send_text(data)
        except Exception:
            self.clients.discard(ws)


def create_app(bus: EventBus, state: dict, submit=None, ptt_down=None, ptt_up=None) -> FastAPI:
    app = FastAPI(title="Jarvis-Finance", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health():
        return {"ok": True, "state": state}

    @app.get("/state")
    async def get_state():
        return JSONResponse(state)

    @app.post("/say")
    async def say(payload: dict):
        """Inject a typed turn without a microphone (rehearsal / remote test)."""
        text = (payload or {}).get("text", "").strip()
        if not text or submit is None:
            return JSONResponse({"ok": False, "error": "no text or text mode off"}, 400)
        submit(text)
        return {"ok": True, "queued": text}

    @app.post("/ptt/start")
    async def ptt_start():
        if ptt_down is None:
            return JSONResponse({"ok": False, "error": "no mic (stt=none)"}, 400)
        ptt_down()
        return {"ok": True, "recording": True}

    @app.post("/ptt/stop")
    async def ptt_stop():
        if ptt_up is None:
            return JSONResponse({"ok": False, "error": "no mic (stt=none)"}, 400)
        ptt_up()
        return {"ok": True, "recording": False}

    @app.get("/control", response_class=HTMLResponse)
    async def control():
        return CONTROL

    @app.get("/overlay", response_class=HTMLResponse)
    async def overlay():
        return OVERLAY

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        bus.clients.add(ws)
        try:
            await ws.send_text(json.dumps(bus.last, ensure_ascii=False, default=str))
            while True:
                await ws.receive_text()  # keepalive pings from the widget
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            bus.clients.discard(ws)

    return app
