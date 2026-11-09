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
