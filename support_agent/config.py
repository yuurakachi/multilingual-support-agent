"""Runtime settings, read from environment variables (usually via the .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAX_ITERATIONS = 8
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(Exception):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    model: str
    # Optional thinking effort. None means "do not send it" (the API default applies),
    # which also keeps the request valid for models without effort support.
    effort: str | None = None
    # Upper bound on model calls per customer message, so the loop can never spin forever.
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        model = env.get("ANTHROPIC_MODEL", "").strip()
        if not model:
            raise ConfigError(
                "ANTHROPIC_MODEL is not set. Copy .env.example to .env and set it there."
            )

        effort = env.get("ANTHROPIC_EFFORT", "").strip().lower() or None
        if effort and effort not in EFFORT_LEVELS:
            raise ConfigError(
                f"ANTHROPIC_EFFORT must be one of {', '.join(EFFORT_LEVELS)} (got {effort!r})."
            )

        raw_limit = env.get("AGENT_MAX_ITERATIONS", "").strip()
        try:
            max_iterations = int(raw_limit) if raw_limit else DEFAULT_MAX_ITERATIONS
        except ValueError:
            raise ConfigError(
                f"AGENT_MAX_ITERATIONS must be a whole number (got {raw_limit!r})."
            ) from None
        if max_iterations < 1:
            raise ConfigError("AGENT_MAX_ITERATIONS must be at least 1.")

        return cls(model=model, effort=effort, max_iterations=max_iterations)


# Loudness below which a recording is treated as silence. A quiet room measures
# well under 100 and speech into a laptop microphone several hundred or more.
DEFAULT_SILENCE_LEVEL = 200


@dataclass(frozen=True)
class VoiceSettings:
    """What the voice channel needs on top of Settings."""

    groq_api_key: str
    # Speech-to-text model. Like the Claude model, it is named only in .env.
    stt_model: str
    # Optional text-to-speech voice per language code, replacing the default one.
    tts_voices: Mapping[str, str]
    # Recordings quieter than this are treated as silence (see Audio.loudness).
    silence_level: int = DEFAULT_SILENCE_LEVEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> VoiceSettings:
        groq_api_key = env.get("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Create a free key at https://console.groq.com "
                "and add it to .env."
            )
        stt_model = env.get("STT_MODEL", "").strip()
        if not stt_model:
            raise ConfigError("STT_MODEL is not set. Copy its line from .env.example to .env.")

        tts_voices = {
            language: voice
            for language in ("es", "ja", "en")
            if (voice := env.get(f"TTS_VOICE_{language.upper()}", "").strip())
        }

        raw_level = env.get("VOICE_SILENCE_LEVEL", "").strip()
        try:
            silence_level = int(raw_level) if raw_level else DEFAULT_SILENCE_LEVEL
        except ValueError:
            raise ConfigError(
                f"VOICE_SILENCE_LEVEL must be a whole number (got {raw_level!r})."
            ) from None
        if silence_level < 0:
            raise ConfigError("VOICE_SILENCE_LEVEL cannot be negative.")

        return cls(
            groq_api_key=groq_api_key,
            stt_model=stt_model,
            tts_voices=tts_voices,
            silence_level=silence_level,
        )


def load_settings() -> Settings:
    """Load .env from the project root (if present) and read the settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings.from_env()


def load_voice_settings() -> VoiceSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    return VoiceSettings.from_env()
