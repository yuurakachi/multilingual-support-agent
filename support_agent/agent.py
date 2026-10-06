"""The agent loop, written by hand on top of the Claude Messages API.

One customer message can take several model calls:

    1. Send the whole conversation to the model.
    2. If the model asks for tools, run them, add the results to the
       conversation and go back to step 1.
    3. If the model answers in text, that text is the reply for the customer.

The API is stateless, so `self.messages` is the agent's only memory: every
call sends the full history again. The loop is bounded by `max_iterations`
so it can never spin forever.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field

import anthropic

from support_agent.config import Settings, load_settings
from support_agent.prompts import SYSTEM_PROMPT
from support_agent.tool_registry import TOOL_DEFINITIONS, execute_tool

# Ceiling for one model response (thinking included), not a target length.
MAX_TOKENS = 16000

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# Shown when the loop has to stop without an answer from the model. It cannot be
# written in the customer's language by the model, so it carries all three.
FALLBACK_MESSAGE = (
    "Sorry, I could not complete your request. Please try again, or ask to speak with a "
    "human agent.\n"
    "Lo siento, no pude completar tu solicitud. Inténtalo de nuevo o pide hablar con un "
    "agente humano.\n"
    "申し訳ございません。ご依頼を完了できませんでした。もう一度お試しいただくか、"
    "担当者との会話をご希望の旨をお伝えください。"
)


@dataclass
class ToolCall:
    """One tool the model asked for, with what it sent and what came back."""

    name: str
    input: dict
    result: dict
    is_error: bool


@dataclass
class AgentReply:
    """Everything that happened while answering one customer message."""

    text: str
    # "end_turn" for a normal answer. Anything else means the loop stopped early:
    # "max_iterations", "refusal", "max_tokens", "empty_response", ...
    stop_reason: str
    iterations: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


class SupportAgent:
    def __init__(self, client, conn: sqlite3.Connection, settings: Settings):
        self.client = client
        self.conn = conn
        self.settings = settings
        self.messages: list[dict] = []

    def reply(self, user_text: str) -> AgentReply:
        """Answer one customer message, calling tools as many times as needed."""
        self.messages.append({"role": "user", "content": user_text})
        tool_calls: list[ToolCall] = []
        usage = dict.fromkeys(USAGE_FIELDS, 0)

        for iteration in range(1, self.settings.max_iterations + 1):
            response = self._ask_model()
            _add_usage(usage, response.usage)

            if response.stop_reason == "tool_use":
                # The model wants tools. Its turn goes into the history unchanged
                # (tool requests and thinking blocks included), followed by one
                # user message carrying every result. Then we ask again.
                self.messages.append({"role": "assistant", "content": response.content})
                results = self._run_tools(response.content, tool_calls)
                self.messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "end_turn":
                text = _text_of(response.content)
                if text:
                    self.messages.append({"role": "assistant", "content": response.content})
                    return AgentReply(text, "end_turn", iteration, tool_calls, usage)
                return self._give_up("empty_response", iteration, tool_calls, usage)

            # "refusal", "max_tokens", ...: the turn is incomplete, so any tool
            # request inside it may be cut off. Never run tools from such a turn.
            return self._give_up(
                response.stop_reason or "unknown", iteration, tool_calls, usage
            )

        return self._give_up("max_iterations", self.settings.max_iterations, tool_calls, usage)

    def _ask_model(self):
        request = {
            "model": self.settings.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": TOOL_DEFINITIONS,
            "messages": self.messages,
            # Prompt caching: tools, system prompt and earlier turns are identical
            # on every call, so the API can reuse them at a fraction of the price.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.effort:
            request["output_config"] = {"effort": self.settings.effort}
        return self.client.messages.create(**request)

    def _run_tools(self, content, tool_calls: list[ToolCall]) -> list[dict]:
        """Run every tool requested in one model turn and build the result blocks."""
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            result = execute_tool(self.conn, block.name, block.input)
            is_error = not result.get("ok", False)
            tool_calls.append(ToolCall(block.name, block.input, result, is_error))
            results.append(
                {
                    "type": "tool_result",
                    # Ties this result to the request it answers.
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                    "is_error": is_error,
                }
            )
        return results

    def _give_up(
        self, stop_reason: str, iterations: int, tool_calls: list[ToolCall], usage: dict
    ) -> AgentReply:
        # Recorded as the assistant's turn so the history stays well-formed
        # and the customer can keep chatting.
        self.messages.append({"role": "assistant", "content": FALLBACK_MESSAGE})
        return AgentReply(FALLBACK_MESSAGE, stop_reason, iterations, tool_calls, usage)


def _text_of(content) -> str:
    """Join the text blocks of a model turn, skipping thinking and tool blocks."""
    return "\n".join(block.text for block in content if block.type == "text").strip()


def _add_usage(total: dict[str, int], usage) -> None:
    for name in USAGE_FIELDS:
        total[name] += getattr(usage, name, None) or 0


def build_agent(conn: sqlite3.Connection) -> SupportAgent:
    """Create an agent that talks to the live API, configured from .env."""
    # Settings first: loading .env is what puts ANTHROPIC_API_KEY in the
    # environment, and the client reads it at the moment it is created.
    settings = load_settings()
    return SupportAgent(anthropic.Anthropic(), conn, settings)


def main() -> None:
    """Send a single message to the live API: python -m support_agent.agent "message"."""
    from support_agent.db import DEFAULT_DB_PATH, connect
    from support_agent.seed import build_database

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m support_agent.agent "your message"')
    if not DEFAULT_DB_PATH.exists():
        build_database()

    conn = connect()
    try:
        reply = build_agent(conn).reply(sys.argv[1])
    finally:
        conn.close()

    for call in reply.tool_calls:
        print(f"[tool] {call.name}({json.dumps(call.input, ensure_ascii=False)})")
        print(f"       -> {json.dumps(call.result, ensure_ascii=False)}")
    print(f"[stop_reason={reply.stop_reason} iterations={reply.iterations} usage={reply.usage}]")
    print()
    print(reply.text)


if __name__ == "__main__":
    main()
