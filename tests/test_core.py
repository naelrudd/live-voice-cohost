import pytest

from jarvis.config import Settings
from jarvis.orchestrator import split_sentences
from jarvis.tools import desktop, tool_schemas
from jarvis.tools.finance import resolve_symbol

# pynput.keyboard needs a real backend (X11 on Linux, win32 native on Windows).
# Headless CI runners have no display, so the import fails there; the hotkey
# tests still run on Windows where the app actually drives the keyboard.
_HOTKEYS_AVAILABLE = True
try:
    import pynput.keyboard  # noqa: F401
except Exception:
    _HOTKEYS_AVAILABLE = False

requires_hotkeys = pytest.mark.skipif(
    not _HOTKEYS_AVAILABLE,
    reason="pynput keyboard backend unavailable (headless CI)",)


def test_deny_actions_parsed():
    s = Settings(deny_actions="buy, sell ,order")
    assert s.deny_action_list == ["buy", "sell", "order"]


def test_auth_headers_empty_without_key():
    assert Settings(llm_api_key="").auth_headers == {}
    assert Settings(llm_api_key="abc").auth_headers == {"Authorization": "Bearer abc"}


def test_split_sentences_partial_buffer():
    out, rest = split_sentences("Harga emas naik. IHSG")
    assert out == ["Harga emas naik."]
    assert rest == "IHSG"


def test_split_sentences_flush():
    out, rest = split_sentences("tanpa titik", flush=True)
    assert out == ["tanpa titik"]
    assert rest == ""


def test_split_sentences_soft_break_on_long_text():
    long = "harga emas malam ini naik cukup tajam karena dolar melemah, " * 3
    out, rest = split_sentences(long)
    assert out, "long comma-joined text must be chunked early"
    assert all(len(s) <= 200 for s in out)


def test_split_sentences_collapses_newlines():
    sents, rest = split_sentences("Emas naik 1 persen.\n\nIHSG juga hijau.", flush=True)
    assert sents == ["Emas naik 1 persen.", "IHSG juga hijau."]
    assert rest == ""
    assert all("\n" not in s for s in sents)


@requires_hotkeys
def test_hotkey_matches_side_specific_modifiers():
    """Windows sends ctrl_l/alt_l (VK 162/164), not ctrl/alt (VK 17/18)."""
    from pynput import keyboard as kb

    from jarvis.hotkeys import Binding, HotkeyListener

    seen: list[str] = []
    hl = HotkeyListener([Binding("ctrl+alt+j", lambda: seen.append("down"),
                                 lambda: seen.append("up"))])
    hl._on_press(kb.Key.ctrl_l)
    hl._on_press(kb.Key.alt_l)
    assert seen == []
    hl._on_press(kb.KeyCode.from_char("j"))
    assert seen == ["down"]
    hl._on_release(kb.KeyCode.from_char("j"))
    assert seen == ["down", "up"]

    # ctrl-held letter arrives as KeyCode(vk=74, char=None), not char='j'
    seen.clear()
    hl._on_press(kb.Key.ctrl_l)
    hl._on_press(kb.Key.alt_l)
    hl._on_press(kb.KeyCode(vk=ord("J"), char=None))
    assert seen == ["down"]
    hl._on_release(kb.KeyCode(vk=ord("J"), char=None))
    assert seen == ["down", "up"]


@requires_hotkeys
def test_hotkey_release_of_modifier_first_does_not_stick():
    from pynput import keyboard as kb

    from jarvis.hotkeys import Binding, HotkeyListener

    seen: list[str] = []
    hl = HotkeyListener([Binding("ctrl+alt+j", lambda: seen.append("down"),
                                 lambda: seen.append("up"))])
    for k in (kb.Key.ctrl_r, kb.Key.alt_gr, kb.KeyCode.from_char("j")):
        hl._on_press(k)
    assert seen == ["down"]
    hl._on_release(kb.Key.alt_gr)   # modifier let go before the letter
    assert seen == ["down", "up"]
    hl._on_release(kb.KeyCode.from_char("j"))  # must not fire a second time
    assert seen == ["down", "up"]


@requires_hotkeys
def test_hotkey_single_key_and_spec_rejects_bad_input():
    from pynput import keyboard as kb

    from jarvis.hotkeys import Binding, HotkeyListener

    seen: list[str] = []
    hl = HotkeyListener([Binding("f13", lambda: seen.append("down"))])
    hl._on_press(kb.Key.f13)
    assert seen == ["down"]
    for bad in ("ctrl", "ctrl+alt+"):
        try:
            HotkeyListener([Binding(bad, lambda: None)])
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


@requires_hotkeys
def test_hotkey_listener_can_start_and_stop():
    from jarvis.hotkeys import Binding, HotkeyListener

    hl = HotkeyListener([Binding("ctrl+alt+j", lambda: None)])
    hl.start()
    try:
        assert hl._listener is not None and hl._listener.running
    finally:
        hl.stop()


def test_resolve_symbol():
    assert resolve_symbol("harga emas sekarang") == "GC=F"
    assert resolve_symbol("IHSG") == "^JKSE"
    assert resolve_symbol("btc") == "BTC-USD"
    assert resolve_symbol("usd idr") == "IDR=X"
    assert resolve_symbol(")(") is None


def test_resolve_symbol_tolerates_stt_mishears():
    # faster-whisper hears "IHSG" as "nih hsg" / "DNI HSG" on CPU models.
    assert resolve_symbol("ihsg sekarang") == "^JKSE"
    assert resolve_symbol("dni hsg") == "^JKSE"
    assert resolve_symbol("berapa kurs dollar") == "IDR=X"
    # short keys must not match by subsequence alone
    assert resolve_symbol("bento harian") is None


# ---------------------------------------------------------------- desktop tools


def test_tv_url_maps_pair_and_interval():
    assert desktop.tv_url("buka xauusd", "4h") == \
        "https://www.tradingview.com/chart/?interval=240#symbol=OANDA:XAUUSD"
    assert desktop.tv_url("ihsg") == "https://www.tradingview.com/chart/#symbol=IDX:COMPOSITE"
    assert desktop.tv_url("saham gak jelas") is None


def test_guardrail_blocks_order_words():
    # The PRD forbids the agent from ever transacting: any tool arg carrying a
    # deny word must fail closed and launch nothing.
    res = desktop.open_url("https://broker.example/buy?symbol=xauusd")
    assert res["ok"] is False and "guardrail" in res["error"]
    assert desktop.open_app("buy")["ok"] is False


def test_open_url_rejects_non_http():
    assert desktop.open_url("calc.exe")["ok"] is False


def test_tool_schemas_are_read_only():
    names = {s["function"]["name"] for s in tool_schemas(True)}
    assert names == {"get_price", "web_search", "open_chart", "open_url", "open_app"}
    assert tool_schemas(False) == []


# ---------------------------------------------------------------- webbridge

from jarvis import webbridge  # noqa: E402


def test_webbridge_status_none_when_daemon_absent(monkeypatch):
    monkeypatch.setattr(webbridge, "STATUS_URL", "http://127.0.0.1:1/status")
    assert webbridge.status(timeout=0.4) is None
    assert webbridge.is_ready() is False


def test_webbridge_ensure_reports_not_installed(monkeypatch):
    monkeypatch.setattr(webbridge, "STATUS_URL", "http://127.0.0.1:1/status")
    monkeypatch.setattr(webbridge, "binary_path", lambda: None)
    monkeypatch.setattr(webbridge, "_start_commands", lambda: [])
    state, st = webbridge.ensure()
    assert state == webbridge.NOT_INSTALLED and st is None


def test_webbridge_ensure_starts_daemon_when_stopped(monkeypatch):
    # daemon down, binary present -> ensure() must attempt a start, no install
    monkeypatch.setattr(webbridge, "STATUS_URL", "http://127.0.0.1:1/status")
    monkeypatch.setattr(webbridge, "binary_path", lambda: __import__("pathlib").Path("/fake/kimi-webbridge"))
    calls = []
    monkeypatch.setattr(webbridge, "start", lambda wait_s=20.0: calls.append(wait_s) or None)
    state, _ = webbridge.ensure()
    assert calls and state == webbridge.FAILED


def test_webbridge_ensure_skips_start_when_asked(monkeypatch):
    monkeypatch.setattr(webbridge, "STATUS_URL", "http://127.0.0.1:1/status")
    monkeypatch.setattr(webbridge, "binary_path", lambda: __import__("pathlib").Path("/fake/x"))
    called = []
    monkeypatch.setattr(webbridge, "start", lambda wait_s=20.0: called.append(1))
    state, _ = webbridge.ensure(auto_start=False)
    assert state == webbridge.NOT_RUNNING and not called


def test_webbridge_navigate_false_when_unreachable(monkeypatch):
    monkeypatch.setattr(webbridge, "COMMAND_URL", "http://127.0.0.1:1/command")
    assert webbridge.navigate("http://127.0.0.1:8848/control") is False


def test_webbridge_install_runs_upstream_script(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(webbridge.sys, "platform", "linux")
    monkeypatch.setattr(webbridge.shutil, "which", lambda name: "/usr/bin/bash" if name == "bash" else None)
    monkeypatch.setattr(webbridge.subprocess, "run", fake_run)
    assert webbridge.install() is True
    assert "install.sh" in seen["cmd"][-1] and "curl -fsSL" in seen["cmd"][-1]


def test_webbridge_install_uses_wsl_on_windows(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(webbridge.sys, "platform", "win32")
    monkeypatch.setattr(webbridge, "_wsl_distros", lambda: ["Debian"])
    monkeypatch.setattr(webbridge.subprocess, "run", fake_run)
    assert webbridge.install() is True
    assert seen["cmd"][:4] == ["wsl.exe", "-d", "Debian", "--"]


def test_webbridge_describe_covers_states(monkeypatch):
    monkeypatch.setattr(webbridge, "status", lambda timeout=1.5: None)
    monkeypatch.setattr(webbridge, "binary_path", lambda: None)
    assert "tidak terpasang" in webbridge.describe(None)
    monkeypatch.setattr(webbridge, "binary_path", lambda: __import__("pathlib").Path("/fake/x"))
    assert "daemon mati" in webbridge.describe(None)
    assert "ekstensi browser belum tersambung" in webbridge.describe(
        {"running": True, "extension_connected": False})
    assert "browser yang sedang jalan" in webbridge.describe(
        {"running": True, "extension_connected": True, "version": "v9"})


# ------------------------------------------------------- setup wizard: bridge

def test_setup_cli_wizard_checks_webbridge(monkeypatch):
    from jarvis import setup as setup_mod
    from jarvis import webbridge

    monkeypatch.setattr(setup_mod, "load_current_env", lambda: {})
    monkeypatch.setattr(setup_mod, "list_audio_devices", lambda: ([], []))
    monkeypatch.setattr(setup_mod, "save_env", lambda values: None)
    monkeypatch.setattr(setup_mod, "test_llm_connection", lambda b, k: (True, "ok"))
    monkeypatch.setattr(setup_mod, "test_groq_key", lambda k: (True, "ok"))
    monkeypatch.setattr(setup_mod, "prompt", lambda msg: "")

    seen = []
    st = {"running": True, "extension_connected": True, "version": "v2.0.22"}

    def fake_ensure(auto_start=True):
        seen.append(auto_start)
        return webbridge.READY, st

    monkeypatch.setattr(webbridge, "ensure", fake_ensure)
    setup_mod.run_cli()
    assert seen == [True], "wizard must probe/start the bridge exactly once"


# ------------------------------------------------- request routing / LLM payload

def test_llm_payload_omits_reasoning_effort_when_not_supported():
    """Groq rejects reasoning_effort='none' with HTTP 400 — the key must be absent."""
    from jarvis.llm.client import LLMClient

    c = LLMClient("http://x", "k", "m", reasoning_effort="none")
    assert "reasoning_effort" not in c._payload([], None, True, "m")
    c = LLMClient("http://x", "k", "m", reasoning_effort="")
    assert "reasoning_effort" not in c._payload([], None, True, "m")
    c = LLMClient("http://x", "k", "m", reasoning_effort="low")
    assert c._payload([], None, True, "m")["reasoning_effort"] == "low"


def test_sanitize_answer_replaces_error_markers():
    from jarvis.orchestrator import _sanitize_answer

    assert _sanitize_answer("[PARSING ERROR]") != "[PARSING ERROR]"
    assert "coba lagi" in _sanitize_answer("")
    assert _sanitize_answer("IHSG naik 0,8 persen.") == "IHSG naik 0,8 persen."


def test_respond_calls_open_url_for_open_page_requests(monkeypatch):
    """'bukain halaman browser ... berita' must reach desktop.open_url, not chit-chat."""
    import asyncio

    from jarvis import orchestrator as orch_mod
    from jarvis.llm.client import StreamEvent, ToolCall
    from jarvis.tools import desktop

    calls: list[str] = []
    monkeypatch.setattr(desktop, "open_url", 
                        lambda url: (calls.append(url), {"ok": True, "url": url})[1])

    class FakeLLM:
        def __init__(self):
            self.round = 0

        async def stream(self, messages, tools=None, client=None):
            self.round += 1
            if self.round == 1:
                yield StreamEvent("tool_calls", tool_calls=[
                    ToolCall(id="1", name="open_url",
                             arguments='{"url": "https://www.reuters.com/world/us/"}')])
            else:
                yield StreamEvent("text", text="Udah gue buka berita US-nya.")
            yield StreamEvent("done", finish_reason="stop")

    class FakeTTS:
        sample_rate = 24000

        async def synthesize(self, text):
            raise RuntimeError("no warmup in test")

        async def prime(self):
            return None

        async def stream(self, text):
            yield b"\x00\x00" * 240

    class FakeHandle:
        def __init__(self):
            import threading
            self.done = threading.Event()

        def feed(self, chunk):
            pass

        def end(self):
            self.done.set()

    class FakePlayer:
        def warm(self, sr):
            pass

        def open_stream(self, sr):
            return FakeHandle()

        def interrupt(self):
            pass

    o = orch_mod.Orchestrator(llm=FakeLLM(), tts=FakeTTS(), player=FakePlayer(),
                              tools_enabled=True)
    m = asyncio.run(o.respond("Bukain halaman browser baru tentang berita US terbaru."))
    assert calls == ["https://www.reuters.com/world/us/"], calls
    assert m.tools == ["open_url"]
    assert "buka berita" in m.answer.lower()


def _stub_audio(monkeypatch, spoken: list):
    """Fake TTS/player that records whatever the orchestrator tries to speak."""
    from jarvis import orchestrator as orch_mod

    class FakeTTS:
        sample_rate = 24000
        async def synthesize(self, text):
            raise RuntimeError("no warmup in test")
        async def prime(self):
            return None
        async def stream(self, text):
            spoken.append(text)
            yield b"\x00\x00" * 240

    class FakeHandle:
        def __init__(self):
            import threading
            self.done = threading.Event()
        def feed(self, chunk):
            pass
        def end(self):
            self.done.set()

    class FakePlayer:
        def warm(self, sr):
            pass
        def open_stream(self, sr):
            return FakeHandle()
        def interrupt(self):
            pass

    return orch_mod.Orchestrator(llm=None, tts=FakeTTS(), player=FakePlayer(),
                                 tools_enabled=True)


def test_respond_never_speaks_error_marker(monkeypatch):
    import asyncio
    from jarvis.llm.client import StreamEvent
    from jarvis.orchestrator import LLM_FAIL_LINE

    class BadLLM:
        async def stream(self, messages, tools=None, client=None):
            yield StreamEvent("text", text="[PARSING ERROR]")
            yield StreamEvent("done", finish_reason="stop")

    spoken: list = []
    o = _stub_audio(monkeypatch, spoken)
    o.llm = BadLLM()
    m = asyncio.run(o.respond("Buka berita US."))
    assert all("PARSING" not in s for s in spoken), spoken
    assert m.answer != "[PARSING ERROR]"
    assert m.answer == LLM_FAIL_LINE


def test_respond_speaks_fail_line_when_llm_raises(monkeypatch):
    import asyncio
    from jarvis.orchestrator import LLM_FAIL_LINE

    class DeadLLM:
        async def stream(self, messages, tools=None, client=None):
            raise RuntimeError("LLM 429: rate limited")
            yield  # pragma: no cover

    spoken: list = []
    o = _stub_audio(monkeypatch, spoken)
    o.llm = DeadLLM()
    m = asyncio.run(o.respond("Berapa IHSG?"))
    assert LLM_FAIL_LINE in spoken, spoken
    assert "429" not in " ".join(spoken)
