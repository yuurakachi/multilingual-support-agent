"""The agent loop, written by hand on top of the Claude Messages API.

One customer message can take several model calls:

    1. Send the whole conversation to the model.
    2. If the model asks for tools, run them, add the results to the
       conversation and go back to step 1.
    3. If the model answers in text, that text is the reply for the customer.

The API is stateless, so `self.messages` is the agent's only memory: every
call sends the full history again. The loop is bounded by `max_iterations`
so it can never spin forever.

Everything that happens during a turn (model calls, tool calls, tokens,
timings) is collected in the returned AgentReply, which is what the
conversation log and the evals are built from.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from support_agent.channels import TEXT_CHANNEL, Channel
from support_agent.config import Settings, load_settings
from support_agent.tool_registry import TOOL_DEFINITIONS, execute_tool

# Ceiling for one model response (thinking included), not a target length.
MAX_TOKENS = 16000

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _empty_usage() -> dict[str, int]:
    return dict.fromkeys(USAGE_FIELDS, 0)


@dataclass
class ModelCall:
    """One request to the model inside a turn."""

    iteration: int
    stop_reason: str | None
    latency_ms: int
    usage: dict[str, int]


@dataclass
class ToolCall:
    """One tool the model asked for, with what it sent and what came back."""

    iteration: int
    id: str
    name: str
    input: dict
    result: dict
    is_error: bool
    latency_ms: int


@dataclass
class AgentReply:
    """Everything that happened while answering one customer message."""

    text: str = ""
    # "end_turn" for a normal answer. Anything else means the loop stopped early:
    # "max_iterations", "refusal", "max_tokens", "empty_response", "api_error", ...
    stop_reason: str = ""
    iterations: int = 0
    latency_ms: int = 0
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Token usage summed over every model call of the turn.
    usage: dict[str, int] = field(default_factory=_empty_usage)
    # Set when the API itself failed: {"type", "message", "status_code"}.
    error: dict | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


class SupportAgent:
    def __init__(
        self,
        client,
        conn: sqlite3.Connection,
        settings: Settings,
        clock: Callable[[], float] = time.perf_counter,
        channel: Channel = TEXT_CHANNEL,
    ):
        self.client = client
        self.conn = conn
        self.settings = settings
        # The only thing the agent knows about how the customer reaches it.
        self.channel = channel
        self.messages: list[dict] = []
        self._clock = clock

    def reply(self, user_text: str, language: str | None = None) -> AgentReply:
        """Answer one customer message, calling tools as many times as needed.

        `language` ("es", "ja" or "en") is optional and only chooses the fallback
        message, for channels that know which language the customer is using.
        The model works out the reply language from the conversation by itself.
        """
        started = self._clock()
        self.messages.append({"role": "user", "content": user_text})

        reply = AgentReply()
        reply.stop_reason = self._run_loop(reply)
        if not reply.completed:
            # Recorded as the assistant's turn so the history stays well-formed
            # and the customer can keep chatting.
            reply.text = self.channel.fallback_for(language)
            self.messages.append({"role": "assistant", "content": reply.text})

        reply.latency_ms = self._elapsed_ms(started)
        return reply

    def _run_loop(self, reply: AgentReply) -> str:
        """The loop itself. Fills `reply` as it goes and returns why it stopped."""
        for iteration in range(1, self.settings.max_iterations + 1):
            reply.iterations = iteration
            try:
                response = self._ask_model(reply)
            except anthropic.APIError as error:
                # Network, authentication, rate limit... The SDK has already
                # retried what can be retried. Tools that ran earlier in this
                # turn stay recorded in `reply`.
                reply.error = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "status_code": getattr(error, "status_code", None),
                }
                return "api_error"

            if response.stop_reason == "tool_use":
                # The model wants tools. Its turn goes into the history unchanged
                # (tool requests and thinking blocks included), followed by one
                # user message carrying every result. Then we ask again.
                self.messages.append({"role": "assistant", "content": response.content})
                results = self._run_tools(response.content, reply)
                self.messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "end_turn":
                reply.text = _text_of(response.content)
                if not reply.text:
                    return "empty_response"
                self.messages.append({"role": "assistant", "content": response.content})
                return "end_turn"

            # "refusal", "max_tokens", ...: the turn is incomplete, so any tool
            # request inside it may be cut off. Never run tools from such a turn.
            return response.stop_reason or "unknown"

        return "max_iterations"

    def _ask_model(self, reply: AgentReply):
        request = {
            "model": self.settings.model,
            "max_tokens": MAX_TOKENS,
            "system": self.channel.system_prompt,
            "tools": TOOL_DEFINITIONS,
            "messages": self.messages,
            # Prompt caching: tools, system prompt and earlier turns are identical
            # on every call, so the API can reuse them at a fraction of the price.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.effort:
            request["output_config"] = {"effort": self.settings.effort}

        started = self._clock()
        response = self.client.messages.create(**request)
        latency_ms = self._elapsed_ms(started)

        usage = {name: getattr(response.usage, name, None) or 0 for name in USAGE_FIELDS}
        for name, tokens in usage.items():
            reply.usage[name] += tokens
        reply.model_calls.append(
            ModelCall(reply.iterations, response.stop_reason, latency_ms, usage)
        )
        return response

    def _run_tools(self, content, reply: AgentReply) -> list[dict]:
        """Run every tool requested in one model turn and build the result blocks."""
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            started = self._clock()
            result = execute_tool(self.conn, block.name, block.input)
            latency_ms = self._elapsed_ms(started)

            is_error = not result.get("ok", False)
            reply.tool_calls.append(
                ToolCall(
                    iteration=reply.iterations,
                    id=block.id,
                    name=block.name,
                    input=block.input,
                    result=result,
                    is_error=is_error,
                    latency_ms=latency_ms,
                )
            )
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

    def _elapsed_ms(self, started: float) -> int:
        return round((self._clock() - started) * 1000)


def _text_of(content) -> str:
    """Join the text blocks of a model turn, skipping thinking and tool blocks."""
    return "\n".join(block.text for block in content if block.type == "text").strip()


def build_agent(
    conn: sqlite3.Connection,
    settings: Settings | None = None,
    channel: Channel = TEXT_CHANNEL,
) -> SupportAgent:
    """Create an agent that talks to the live API, configured from .env."""
    # Settings first: loading .env is what puts ANTHROPIC_API_KEY in the
    # environment, and the client reads it at the moment it is created.
    settings = settings or load_settings()
    return SupportAgent(anthropic.Anthropic(), conn, settings, channel=channel)
