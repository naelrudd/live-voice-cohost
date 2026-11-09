"""Diagnose TTS + playback: synthesize one line, dump WAV, report PCM RMS + any error."""
import sys, time, asyncio
sys.path.insert(0, ".")
import numpy as np

from jarvis.tts.backends import EdgeTTS
from jarvis.audio.io import PlaybackQueue

LINE = "Halo, ini suara Jarvis. Kalau kamu dengar ini, audio jalan normal."

async def main():
    tts = EdgeTTS()
    print("backend:", type(tts).__name__)
    frames = []
    t0 = time.perf_counter()
    async for chunk in tts.stream(LINE):
        if chunk is not None and chunk.size:
            frames.append(chunk)
    pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    dt = time.perf_counter() - t0
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0
    print(f"pcm: {pcm.size} samples, {pcm.size/24000:.2f}s, rms={rms:.4f}, synth={dt:.2f}s")
    await tts.close()

    # dump wav
    import wave
    w = wave.open("logs/jarvis_test.wav", "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    w.close()
    print("wav -> logs/jarvis_test.wav")

    # play to default device
    if pcm.size:
        q = PlaybackQueue()
        try:
            ev = q.play_clip(pcm, 24000)
            ev.wait(timeout=6)
            print("play done, played:", ev.is_set())
        except Exception as e:
            print("PLAY ERROR:", type(e).__name__, e)
        finally:
            q.close()

asyncio.run(main())