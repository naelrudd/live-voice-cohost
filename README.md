# live-voice-cohost

A low-latency **speech-to-speech** AI co-host for live finance streams. Hold a key
(or tap a web button), talk, and a synthetic voice answers in natural Indonesian in
under a second.

```
mic (push-to-talk) → Groq STT → LLM (tools) → Edge TTS (WebSocket) → OBS / speakers
                     ~0.9 s          ~0.7–1.4 s     ~0.2 s first audio
```

Everything the voice says goes through TTS, so the model gets a speech-specific
system prompt: short, no markdown, no emoji, digits written as numbers. A
read-only tool layer (`get_price`, `web_search`) keeps it capable of answering
"harga emas sekarang berapa?" with real quotes — never with hallucinated numbers
and never executing a trade.

## What it is

- **Voice pipeline, not a text bot.** PTT → STT → streamed LLM → sentence-level
  TTS → audio device. The LLM token stream is fed to TTS sentence-by-sentence so
  the first audio lands while the model is still generating.
- **Web control deck** at `/control` — hold-to-talk button, text fallback, and
  prompt chips — so it works over OBS Custom Browser Dock and doesn't need a
  global hotkey.
- **Read-only by construction.** The tool layer maps free speech to Yahoo
  Finance symbols and a Tavily web search. No tool can buy, sell, or place an
  order; a deny-list guards the optional browser/desktop tools.
- **Provider-agnostic.** STT, TTS and LLM sit behind interfaces with fallbacks:
  no API key, no crash — it degrades to local faster-whisper and free Edge TTS.

## Latency (measured)

| Stage | Backend | Time |
|-------|---------|------|
| Speech → transcript | Groq `whisper-large-v3-turbo` | ~0.9 s (0.78–1.09 s) |
| Transcript → first LLM token | OpenAI-compatible, reasoning off | ~0.7–1.4 s |
| First token → first audio | Edge TTS WebSocket (persistent) | ~0.2 s |
| Chat path, end-to-end | — | ~2.4 s |
| Tool path (get_price), end-to-end | — | ~5–10 s |

Local faster-whisper fallbacks (12-core CPU, Indonesian clip): tiny ~0.48 s but
bad accuracy, base ~0.78 s, small ~3.57 s but correct on XAUUSD/IHSG/kurs.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or `pip`)
- A microphone and an audio output (optional: VB-Audio Virtual Cable to route
  into OBS)
- Windows for the `.bat` launchers and hotkeys; the core pipeline is OS-agnostic

## Quick start

```bash
git clone https://github.com/<you>/live-voice-cohost.git
cd live-voice-cohost
uv sync                       # installs runtime deps

# 1. Setup wizard (GUI or CLI)
./setup.bat                   # opens setup window (or: python -m jarvis.setup --cli)

# 2. Run
./start.bat                   # or: uv run python -m jarvis.main
```

Open **http://127.0.0.1:8848/control** in a browser, hold **Talk**, and speak.

Optional extras:

```bash
uv sync --extra local-stt     # offline STT fallback
uv sync --extra browser       # browser/desktop tools
uv sync --extra dev           # tests
uv run pytest -q
```

### No API keys

Delete any keys and it still runs: STT falls back to local faster-whisper and TTS
uses free Microsoft Edge voices. Only the LLM needs a key — by default it points
at **OpenRouter**, so a single free API key (no credit card) is all you need. To
use any other model, see *Bring your own model* below.

## Bring your own model (LLM / STT / TTS)

The whole brain is three env vars — swap them to any OpenAI-compatible provider,
no code changes:

```bash
JARVIS_LLM_BASE_URL=https://openrouter.ai/api/v1
JARVIS_LLM_API_KEY=REPLACE_ME
JARVIS_LLM_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

The key falls back through `JARVIS_LLM_API_KEY` → `OPENROUTER_API_KEY` →
`OPENAI_API_KEY`, so a stock `OPENAI_API_KEY`/`OPENROUTER_API_KEY` in the
environment works without touching `.env`. Free-by-default options (all
OpenAI-compatible, one base URL + key swap):

| Provider | Base URL | Model (free tier) | Key var |
|----------|----------|-------------------|---------|
| **OpenRouter** (default) | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (or any `:free`) | `OPENROUTER_API_KEY` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **NVIDIA NIM** | `https://integrate.api.nvidia.com/v1` | DeepSeek / GLM / Gemma (Free endpoint) | `NVIDIA_API_KEY` |
| **GitHub Models** | `https://models.inference.ai.azure.com` | GPT-4.1 / Llama / o4-mini | GitHub token |
| **Together** | `https://api.together.xyz/v1` | 40+ open models ($ credit) | `TOGETHER_API_KEY` |

Same pattern for STT and TTS: point `JARVIS_STT_BACKEND` / `JARVIS_TTS_BACKEND`
at your provider of choice (interface fallbacks mean a missing key never crashes).

## OBS setup

Two OBS sources talk to the same server:

1. **Audio into OBS** — install
   [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (or use the "Audio
   Output Capture" of your speakers), then set `JARVIS_OUTPUT_DEVICE=CABLE Input`
   in `.env` and point your stream's audio source at the cable.
2. **Control deck** — add a **Custom Browser Dock** in OBS pointed at
   `http://127.0.0.1:8848/control` to get a hold-to-talk button inside OBS.
3. **Overlay** (optional) — add a **Browser Source** at
   `http://127.0.0.1:8848/overlay` to show live state (listening / transcribing /
   thinking / speaking) on stream.

## Configuration

Everything is read from `.env` (`JARVIS_*` prefix), nothing host-specific is
hardcoded. The `.env.example` is commented end-to-end. Highlights:

| Variable | Default | Purpose |
|----------|---------|---------|
| `JARVIS_STT_BACKEND` | `groq` | `groq` / `local` / `none` |
| `JARVIS_GROQ_API_KEY` | — | fast cloud STT |
| `JARVIS_TTS_BACKEND` | `edge` | `edge` / `cartesia` / `sapi` |
| `JARVIS_TTS_VOICE` | `id-ID-ArdiNeural` | Indonesian Edge voice |
| `JARVIS_LLM_BASE_URL` | `https://openrouter.ai/api/v1` | any OpenAI-compatible endpoint |
| `JARVIS_LLM_MODEL` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | OpenRouter free model |
| `JARVIS_STREAMER_NAME` | `Pak Febri` | name the voice addresses on air |
| `JARVIS_PTT_KEY` | `f13` | push-to-talk key (`f13` suits Stream Deck) |
| `JARVIS_OUTPUT_DEVICE` | — | substring of output device (e.g. `CABLE Input`) |

## Project layout

```
jarvis/
  audio/io.py        # mic recorder + streaming playback queue
  stt/backends.py    # Groq / local faster-whisper / none
  llm/client.py      # streaming OpenAI-compatible client + tool-call assembly
  tools/finance.py   # get_price (Yahoo) + web_search (Tavily), read-only
  tools/desktop.py   # optional browser/desktop control with deny-list
  tts/backends.py    # Edge / Cartesia / SAPI behind one interface
  tts/edge_ws.py     # persistent Edge TTS WebSocket (recycle per 4 utterances)
  tts/mp3.py         # incremental MP3 decode + monotonic delta emit
  orchestrator.py    # the pipeline: stream → sentences → audio
  server.py          # FastAPI: /state /overlay /control /ptt /say
  main.py            # wiring + hotkeys + serve
scripts/             # smoke_turn, test_stt, bench_llm, probe_tools
tests/               # pytest suite
```

## Safety

The persona prompt forbids the voice from recommending that the streamer execute
a buy/sell order. `get_price` and `web_search` are read-only; the optional
browser/desktop tools refuse any action matching `JARVIS_DENY_ACTIONS`. If a
tool fails, the voice says so plainly instead of inventing a number.

## License

MIT — see [LICENSE](LICENSE).