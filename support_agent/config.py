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

# Which service hears and which one speaks, chosen in .env. The first one is the default.
STT_PROVIDERS = ("groq", "elevenlabs")
TTS_PROVIDERS = ("edge", "elevenlabs")


@dataclass(frozen=True)
class VoiceSettings:
    """What the voice channel needs on top of Settings."""

    groq_api_key: str
    # Speech-to-text model on Groq. Like the Claude model, it is named only in .env.
    stt_model: str
    # Optional edge-tts voice per language code, replacing the default one.
    tts_voices: Mapping[str, str]
    # Recordings quieter than this are treated as silence (see Audio.loudness).
    silence_level: int = DEFAULT_SILENCE_LEVEL
    stt_provider: str = STT_PROVIDERS[0]
    tts_provider: str = TTS_PROVIDERS[0]
    # Only needed when one of the providers is ElevenLabs. Its free plan has a
    # monthly credit allowance, so edge-tts stays the default voice.
    elevenlabs_api_key: str = ""
    elevenlabs_stt_model: str = ""
    elevenlabs_tts_model: str = ""
    # One ElevenLabs voice speaks all three languages.
    elevenlabs_voice_id: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> VoiceSettings:
        stt_provider = _choice(env, "STT_PROVIDER", STT_PROVIDERS)
        tts_provider = _choice(env, "TTS_PROVIDER", TTS_PROVIDERS)
        uses_elevenlabs = "elevenlabs" in (stt_provider, tts_provider)

        groq_api_key = env.get("GROQ_API_KEY", "").strip()
        stt_model = env.get("STT_MODEL", "").strip()
        if stt_provider == "groq":
            if not groq_api_key:
                raise ConfigError(
                    "GROQ_API_KEY is not set. Create a free key at https://console.groq.com "
                    "and add it to .env."
                )
            if not stt_model:
                raise ConfigError(
                    "STT_MODEL is not set. Copy its line from .env.example to .env."
                )

        elevenlabs_api_key = env.get("ELEVENLABS_API_KEY", "").strip()
        if uses_elevenlabs and not elevenlabs_api_key:
            raise ConfigError(
                "ELEVENLABS_API_KEY is not set. Create a free key at https://elevenlabs.io "
                "and add it to .env."
            )
        elevenlabs_stt_model = _required_if(
            env, "ELEVENLABS_STT_MODEL", stt_provider == "elevenlabs"
        )
        elevenlabs_tts_model = _required_if(
            env, "ELEVENLABS_TTS_MODEL", tts_provider == "elevenlabs"
        )
        elevenlabs_voice_id = _required_if(
            env, "ELEVENLABS_VOICE_ID", tts_provider == "elevenlabs"
        )

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
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_stt_model=elevenlabs_stt_model,
            elevenlabs_tts_model=elevenlabs_tts_model,
            elevenlabs_voice_id=elevenlabs_voice_id,
        )


def _choice(env: Mapping[str, str], name: str, allowed: tuple[str, ...]) -> str:
    """An optional setting that must be one of `allowed`; the first one is the default."""
    value = env.get(name, "").strip().lower() or allowed[0]
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)} (got {value!r}).")
    return value


def _required_if(env: Mapping[str, str], name: str, needed: bool) -> str:
    value = env.get(name, "").strip()
    if needed and not value:
        raise ConfigError(f"{name} is not set. Copy its line from .env.example to .env.")
    return value


def load_settings() -> Settings:
    """Load .env from the project root (if present) and read the settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings.from_env()


def load_voice_settings() -> VoiceSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    return VoiceSettings.from_env()
