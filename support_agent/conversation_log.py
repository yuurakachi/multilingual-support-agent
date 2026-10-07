"""Structured JSON log of one conversation.

One file per conversation, rewritten after every turn so nothing is lost if
the program stops. The layout is meant to be read by code (the future evals)
as much as by people:

    turns       what the customer said, what the agent answered, and what
                happened in between: tools, arguments, results, tokens, timings
    totals      the same numbers added up for the whole conversation
    transcript  the raw message history exactly as it was sent to the API

Bump SCHEMA_VERSION when a field is renamed or removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from support_agent.agent import USAGE_FIELDS, AgentReply
from support_agent.channels import TEXT_CHANNEL, Channel
from support_agent.config import PROJECT_ROOT, Settings

SCHEMA_VERSION = 1
# Keys of a spoken turn's voice.timings_ms, in the order the stages happen.
VOICE_STAGES = ("stt", "llm", "tts", "total")
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_jsonable(value):
    """Fallback for json.dumps: SDK content blocks are pydantic models, not dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return vars(value)


class ConversationLog:
    def __init__(
        self,
        settings: Settings,
        log_dir: str | Path = DEFAULT_LOG_DIR,
        metadata: dict | None = None,
        now: Callable[[], datetime] = _utc_now,
        channel: Channel = TEXT_CHANNEL,
    ):
        self._now = now
        # Must be the channel the agent was created with.
        self.channel = channel
        self.started_at = now()
        # Sortable by time, with a random suffix so two conversations never collide.
        self.conversation_id = f"{self.started_at:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        self.path = Path(log_dir) / f"{self.conversation_id}.json"
        self.settings = settings
        # Free-form labels, for example which test scenario produced this conversation.
        self.metadata = dict(metadata or {})
        self.turns: list[dict] = []
        # Things that happened outside a turn, for example audio nobody could understand.
        self.events: list[dict] = []

    def record_turn(
        self, user_message: str, reply: AgentReply, voice: dict | None = None
    ) -> None:
        """Add one customer message and everything the agent did to answer it.

        `voice` is what only a spoken turn has: the language, the length of the
        audio and the time each stage took (see voice/chat.py). The dict is
        stored as given, so the caller can fill in what it learns after the
        reply, such as the text-to-speech time, and save again.
        """
        self.turns.append(
            {
                "index": len(self.turns) + 1,
                "ended_at": self._now().isoformat(),
                "user_message": user_message,
                "assistant_message": reply.text,
                "stop_reason": reply.stop_reason,
                "completed": reply.completed,
                "iterations": reply.iterations,
                "latency_ms": reply.latency_ms,
                "usage": dict(reply.usage),
                "error": reply.error,
                "model_calls": [asdict(call) for call in reply.model_calls],
                "tool_calls": [asdict(call) for call in reply.tool_calls],
                # None for a typed turn.
                "voice": voice,
            }
        )

    def record_event(self, event_type: str, **details) -> None:
        """Add something worth knowing that is not a customer message and its answer."""
        self.events.append(
            {
                "type": event_type,
                "at": self._now().isoformat(),
                # How many turns had been completed when it happened.
                "after_turn": len(self.turns),
                **details,
            }
        )

    def to_dict(self, transcript: list | None = None) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": self.conversation_id,
            "started_at": self.started_at.isoformat(),
            "updated_at": self._now().isoformat(),
            "channel": self.channel.name,
            "model": self.settings.model,
            "effort": self.settings.effort,
            "max_iterations": self.settings.max_iterations,
            # Tells apart conversations produced by different versions of the prompt.
            "system_prompt_sha256": hashlib.sha256(
                self.channel.system_prompt.encode("utf-8")
            ).hexdigest(),
            "metadata": self.metadata,
            "totals": self._totals(),
            "turns": self.turns,
            "events": self.events,
            "transcript": transcript or [],
        }

    def save(self, transcript: list | None = None) -> Path:
        """Write the log file. `transcript` is the agent's raw message history."""
        text = json.dumps(
            self.to_dict(transcript), ensure_ascii=False, indent=2, default=_to_jsonable
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and swap it in, so a crash mid-write
        # cannot leave a half-written log behind.
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path

    def _totals(self) -> dict:
        return {
            "turns": len(self.turns),
            "model_calls": sum(len(turn["model_calls"]) for turn in self.turns),
            "tool_calls": sum(len(turn["tool_calls"]) for turn in self.turns),
            "latency_ms": sum(turn["latency_ms"] for turn in self.turns),
            "usage": {
                name: sum(turn["usage"].get(name, 0) for turn in self.turns)
                for name in USAGE_FIELDS
            },
        }
