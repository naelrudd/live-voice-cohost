"""Global hotkey handling (they do push-to-talk press/release matching)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from pynput import keyboard

# Windows sends the side-specific keys — Key.ctrl_l / Key.alt_l (VK 162/164) —
# not the generic Key.ctrl / Key.alt (VK 17/18) that specs are written with.
# Comparing raw keys silently never matches, so every binding is canonicalised.
_MOD_GROUPS = {
    "ctrl": (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r),
    "alt": (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr),
    "shift": (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r),
    "win": (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r),
}
_MOD_OF_KEY = {k: name for name, keys in _MOD_GROUPS.items() for k in keys}
_MOD_ALIASES = {"ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
                "win": "win", "cmd": "win", "super": "win", "meta": "win"}


def _parse_char(name: str):
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    key = getattr(keyboard.Key, name, None)
    if key is None:
        raise ValueError(f"unknown key in hotkey spec: {name!r}")
    return key


def _same(a, b) -> bool:
    """Key equality across the shapes pynput reports for the same physical key.

    A letter pressed with Ctrl held comes back as ``KeyCode(vk=74, char=None)``
    because ToUnicode returns a control char, while the binding holds
    ``KeyCode(char='j', vk=None)``. Matching on vk == ord(char) covers that.
    """
    if a == b:
        return True
    ca, cb = _MOD_OF_KEY.get(a), _MOD_OF_KEY.get(b)
    if ca and cb:
        return ca == cb
    cha, chb = getattr(a, "char", None), getattr(b, "char", None)
    if cha and chb:
        return cha.lower() == chb.lower()
    vka, vkb = getattr(a, "vk", None), getattr(b, "vk", None)
    if cha and vkb is not None and cha.isalnum():
        return vkb == ord(cha.upper())
    if chb and vka is not None and chb.isalnum():
        return vka == ord(chb.upper())
    return vka is not None and vka == vkb


@dataclass
class Binding:
    spec: str
    on_down: Callable[[], None]
    on_up: Callable[[], None] | None = None


class HotkeyListener:
    """Matches multi-modifier specs like ``ctrl+alt+j`` and single keys like ``f13``."""

    def __init__(self, bindings: list[Binding]):
        self.bindings = bindings
        self._parsed = [self._parse(b.spec) for b in bindings]
        self._active: set[int] = set()
        self._pressed: set = set()       # every key currently down
        self._mods: set[str] = set()     # canonical modifiers currently down
        self._listener: keyboard.Listener | None = None
        self.debug = False

    @staticmethod
    def _parse(spec: str):
        parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
        mods = {_MOD_ALIASES[p] for p in parts if p in _MOD_ALIASES}
        target = None
        for p in parts:
            if p not in _MOD_ALIASES:
                target = _parse_char(p)
        if target is None:
            raise ValueError(f"hotkey spec needs a non-modifier key: {spec!r}")
        if not mods and len(parts) > 1:
            raise ValueError(f"hotkey spec has no modifier: {spec!r}")
        return mods, target

    def _match(self, key, mods_down: set) -> list[int]:
        hits = []
        for i, (mods, target) in enumerate(self._parsed):
            if _same(key, target) and mods <= mods_down:
                hits.append(i)
        return hits

    def _current_mods(self) -> set:
        return set(self._mods)

    def _on_press(self, key) -> None:
        mod = _MOD_OF_KEY.get(key)
        if mod:
            self._mods.add(mod)
        self._pressed.add(key)
        if self.debug:
            print(f"[key] down {key} mods={sorted(self._mods)}")
        for i in self._match(key, self._current_mods()):
            if i in self._active:
                continue
            self._active.add(i)
            try:
                self.bindings[i].on_down()
            except Exception as exc:  # never kill the listener
                print(f"[hotkey] on_down error: {exc}")

    def _on_release(self, key) -> None:
        # Release on the target key *or* on any of its modifiers, so letting go
        # of Ctrl before J cannot leave push-to-talk stuck in "recording".
        for i in list(self._active):
            mods, target = self._parsed[i]
            if _same(key, target) or _MOD_OF_KEY.get(key) in mods:
                self._active.discard(i)
                cb = self.bindings[i].on_up
                if cb:
                    try:
                        cb()
                    except Exception as exc:
                        print(f"[hotkey] on_up error: {exc}")
        mod = _MOD_OF_KEY.get(key)
        if mod:
            self._mods.discard(mod)
        self._pressed.discard(key)
        if self.debug:
            print(f"[key] up   {key} mods={sorted(self._mods)}")

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()


def _main() -> None:
    """Diagnostic: run this and press the combo to see the raw keys + the match."""
    hl = HotkeyListener([Binding("ctrl+alt+j",
                                 lambda: print("  >>> MATCH ctrl+alt+j DOWN"),
                                 lambda: print("  <<< MATCH ctrl+alt+j UP"))])
    hl.debug = True
    print("listening — press ctrl+alt+j, then Ctrl+C to quit")
    hl.start()
    threading.Event().wait()


if __name__ == "__main__":
    _main()
