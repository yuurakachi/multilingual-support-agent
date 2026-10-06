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


def load_settings() -> Settings:
    """Load .env from the project root (if present) and read the settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings.from_env()
