"""Central config. Everything comes from .env — nothing host-specific in code."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Fallbacks to standard OpenAI-compatible vars if the dedicated var is empty.
_FALLBACK_LLM_KEY = (
    os.environ.get("JARVIS_LLM_API_KEY")
    or os.environ.get("JARVIS_GROQ_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
_FALLBACK_TAVILY = os.environ.get("JARVIS_TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY") or ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_", env_file=str(ROOT / ".env"), extra="ignore")

    # LLM (Groq free tier by default — reliable tool calling for the voice loop).
    # `llm_model` accepts a comma-separated fallback chain: primary,backup1,backup2.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = Field(default=_FALLBACK_LLM_KEY)
    llm_model: str = "openai/gpt-oss-20b,qwen/qwen3.8-27b"
    llm_reasoning_effort: str = "none"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 220

    # STT
    stt_backend: str = "groq"
    groq_api_key: str = ""
    groq_stt_model: str = "whisper-large-v3-turbo"
    local_whisper_model: str = "small"
    local_whisper_device: str = "cpu"
    local_whisper_compute: str = "int8"
    local_whisper_threads: int = 4
    stt_language: str = "id"

    # TTS
    tts_backend: str = "edge"
    tts_voice: str = "id-ID-ArdiNeural"
    tts_rate: str = "+8%"
    cartesia_api_key: str = ""
    cartesia_voice_id: str = ""
    cartesia_model: str = "sonic-2"

    # audio
    output_device: str = ""
    input_device: str = ""
    sample_rate: int = 24000

    # hotkeys
    ptt_key: str = "f13"
    ptt_fallback_key: str = "ctrl+alt+j"
    kill_key: str = "ctrl+alt+q"

    # tools
    tools_enabled: bool = True
    tavily_api_key: str = Field(default=_FALLBACK_TAVILY)

    # desktop control
    chrome_path: str = ""
    chrome_profile_dir: str = ""

    # server
    host: str = "127.0.0.1"
    port: int = 8848

    # persona / streaming
    streamer_name: str = "Pak Febri"
    max_history: int = 30
    enable_fillers: bool = False  # False: hilangkan "Oke", "Hmm" otomatis saat jeda LLM

    # guardrails
    deny_actions: str = "buy,sell,order,withdraw,transfer,confirm payment,place order"

    @property
    def deny_action_list(self) -> list[str]:
        return [a.strip().lower() for a in self.deny_actions.split(",") if a.strip()]

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
