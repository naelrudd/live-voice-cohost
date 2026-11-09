"""Audio device resolution, mic capture (PTT + energy VAD trim), streaming playback."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import sounddevice as sd

CAPTURE_SR = 16000  # whisper-friendly


def find_device(substr: str, kind: str) -> int | None:
    """Return device index whose name contains `substr` (case-insensitive)."""
    if not substr:
        return None
    want_in = kind == "input"
    for idx, dev in enumerate(sd.query_devices()):
        name = dev["name"].lower()
        if substr.lower() not in name:
            continue
        if want_in and dev["max_input_channels"] > 0:
            return idx
        if not want_in and dev["max_output_channels"] > 0:
            return idx
    return None


@dataclass
class DeviceChoice:
    index: int | None
    name: str
    fallback: bool


def resolve_output(substr: str) -> DeviceChoice:
    idx = find_device(substr, "output")
    if idx is not None:
        return DeviceChoice(idx, sd.query_devices(idx)["name"], False)
    default = sd.default.device[1]
    name = sd.query_devices(default)["name"] if default is not None and default >= 0 else "system default"
    return DeviceChoice(None, name, bool(substr))


def resolve_input(substr: str) -> DeviceChoice:
    idx = find_device(substr, "input")
    if idx is not None:
        return DeviceChoice(idx, sd.query_devices(idx)["name"], False)
    default = sd.default.device[0]
    name = sd.query_devices(default)["name"] if default is not None and default >= 0 else "system default"
    return DeviceChoice(None, name, bool(substr))


class Recorder:
    """Push-to-talk recorder.

    `start()` opens a stream and buffers frames; `stop()` closes it and returns
    mono float32 PCM. Leading/trailing silence is trimmed with a cheap energy VAD
    so whisper doesn't hallucinate on room noise.
    """

    def __init__(self, device: int | None = None, sr: int = CAPTURE_SR, block: int = 1024):
        self.device = device
        self.sr = sr
        self.block = block
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        with self._lock:
            self._frames.append(indata[:, 0].copy())

    def start(self) -> None:
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      blocksize=self.block, device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self, trim: bool = True) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            frames = self._frames
            self._frames = []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        pcm = np.concatenate(frames)
        return trim_silence(pcm, self.sr) if trim else pcm

    @property
    def recording(self) -> bool:
        return self._stream is not None


def trim_silence(pcm: np.ndarray, sr: int, frame_ms: int = 30, pad_ms: int = 150) -> np.ndarray:
    """Energy VAD: drop leading/trailing quiet frames, keep `pad_ms` margin."""
    if pcm.size == 0:
        return pcm
    flen = max(1, int(sr * frame_ms / 1000))
    n = pcm.size // flen
    if n == 0:
        return pcm
    rms = np.sqrt(np.mean(pcm[: n * flen].reshape(n, flen) ** 2, axis=1))
    thresh = max(float(rms.max()) * 0.12, 0.004)
    loud = np.flatnonzero(rms > thresh)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    pad = int(pad_ms / frame_ms)
    start = max(0, loud[0] - pad) * flen
    end = min(n, loud[-1] + 1 + pad) * flen
    return pcm[start:end]


class StreamHandle:
    """One utterance's worth of PCM, drained by the playback thread."""

    def __init__(self, sr: int, stop_event: threading.Event):
        self.sr = sr
        self.q: queue.Queue = queue.Queue()
        self.done = threading.Event()
        self._stop = stop_event
        self.ended = False

    def feed(self, pcm: np.ndarray) -> None:
        if not self.ended and pcm.size:
            self.q.put(pcm)

    def end(self) -> None:
        if not self.ended:
            self.ended = True
            self.q.put(None)


class PlaybackQueue:
    """Single device-owning thread.

    Keeps the output stream open between utterances (reopening a WASAPI/MME
    stream costs 40-150 ms, which would land right on the first syllable), and
    drains handles in FIFO order so answer sentences never interleave.
    """

    def __init__(self, device: int | None = None, blocksize: int = 1024):
        self.device = device
        self.blocksize = blocksize
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._st: sd.OutputStream | None = None
        self._st_sr: int | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="player", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- public
    def open_stream(self, sr: int) -> StreamHandle:
        handle = StreamHandle(sr, self._stop)
        self._q.put(handle)
        return handle

    def play_clip(self, pcm: np.ndarray, sr: int) -> threading.Event:
        handle = self.open_stream(sr)
        handle.feed(pcm)
        handle.end()
        return handle.done

    def interrupt(self) -> None:
        """Barge-in: stop the current utterance and drop everything queued."""
        self._stop.set()
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            else:
                item.done.set()

    def resume(self) -> None:
        self._stop.clear()

    def warm(self, sr: int) -> None:
        """Pre-open the device so the first real utterance doesn't pay ~100-600 ms
        of WASAPI/MME stream setup on its first syllable."""
        self._stream_for(sr)

    def close(self) -> None:
        self._q.put(None)

    @property
    def busy(self) -> bool:
        return not self._q.empty()

    # ---------------------------------------------------------------- worker
    def _stream_for(self, sr: int) -> sd.OutputStream:
        if self._st is not None and self._st_sr == sr:
            return self._st
        if self._st is not None:
            try:
                self._st.close()
            except Exception:
                pass
            self._st = None
        st = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                             blocksize=self.blocksize, device=self.device)
        st.start()
        self._st, self._st_sr = st, sr
        return st

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            self._play(item)
        with self._lock:
            if self._st is not None:
                try:
                    self._st.close()
                except Exception:
                    pass
                self._st = None

    def _play(self, handle: StreamHandle) -> None:
        self._stop.clear()
        try:
            st = self._stream_for(handle.sr)
            while True:
                chunk = handle.q.get()
                if chunk is None or self._stop.is_set():
                    # drain the rest of this utterance without playing it
                    while self._stop.is_set() and chunk is not None:
                        chunk = handle.q.get()
                    break
                st.write(chunk.reshape(-1, 1))
        except Exception as exc:
            print(f"[audio] playback error: {type(exc).__name__}: {exc}")
        finally:
            handle.done.set()


def wait_all(handles: Iterable[threading.Event], timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    for ev in handles:
        remain = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not ev.wait(remain):
            return False
    return True
