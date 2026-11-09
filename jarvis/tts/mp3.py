"""Incremental MP3 frame decoder.

Edge TTS streams MP3 in ~700-byte chunks that don't align to frame boundaries.
Decoding each chunk on its own silently drops the partial trailing frame (~26 ms
per chunk → audible stutter). This module buffers bytes, cuts *complete* MPEG
frames and only hands whole frames to miniaudio, so the audio is gapless.
"""

from __future__ import annotations

import numpy as np

# Layer III bitrates, kbps
_BITRATES = {
    "V1L3": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
    "V2L3": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
}
_SR = {3: [44100, 48000, 32000, 0], 2: [22050, 24000, 16000, 0], 0: [11025, 12000, 8000, 0]}


def parse_header(header: bytes) -> tuple[int, int] | None:
    """Return (frame_length_bytes, sample_rate) for an MPEG Layer III frame."""
    if len(header) < 4:
        return None
    b1, b2 = header[1], header[2]
    if header[0] != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    version = (b1 >> 3) & 0x03          # 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
    layer = (b1 >> 1) & 0x03            # 1 = Layer III
    if layer != 1 or version == 1:
        return None
    bitrate_idx = (b2 >> 4) & 0x0F
    sr_idx = (b2 >> 2) & 0x03
    padding = (b2 >> 1) & 0x01
    table = _BITRATES["V1L3"] if version == 3 else _BITRATES["V2L3"]
    bitrate = table[bitrate_idx] * 1000
    samplerate = _SR[version][sr_idx]
    if not bitrate or not samplerate:
        return None
    coeff = 144 if version == 3 else 72
    return coeff * bitrate // samplerate + padding, samplerate


def frame_length(header: bytes) -> int | None:
    parsed = parse_header(header)
    return parsed[0] if parsed else None


class Mp3FrameDecoder:
    """Streaming MP3 → float32 mono decoder with monotonic output.

    Decoding each ~700-byte chunk standalone is wrong: chunk boundaries don't
    align to MPEG frames and Layer III uses a bit reservoir, so per-chunk decode
    loses samples (measured: 2.04 s of audio from a 3.84 s clip). Re-decoding the
    buffer-so-far and emitting only samples beyond what we already returned is
    exact and cheap — a 3.8 s clip costs ~1-2 ms per chunk, ~32 ms total.
    """

    MAX_BUFFER = 4 * 1024 * 1024  # reset if a single utterance gets absurd

    def __init__(self, sample_rate: int = 24000):
        self._buf = bytearray()
        self._emitted = 0
        self.sample_rate = sample_rate

    def feed(self, data: bytes) -> np.ndarray | None:
        self._buf += data
        return self._decode()

    def flush(self) -> np.ndarray | None:
        return self._decode()

    def _decode(self) -> np.ndarray | None:
        import miniaudio

        if len(self._buf) > self.MAX_BUFFER:
            return None
        try:
            dec = miniaudio.decode(
                bytes(self._buf),
                nchannels=1,
                sample_rate=self.sample_rate,
                output_format=miniaudio.SampleFormat.SIGNED16,
            )
        except Exception:
            return None  # not enough data yet for a full frame
        samples = np.frombuffer(dec.samples.tobytes(), dtype="<i2").astype(np.float32) / 32768.0
        if samples.size <= self._emitted:
            return None
        new = samples[self._emitted:]
        self._emitted = samples.size
        return new
