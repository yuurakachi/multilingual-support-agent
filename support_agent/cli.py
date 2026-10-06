"""Interactive chat in the terminal.

Usage:
    python -m support_agent              start chatting
    python -m support_agent --quiet      hide the tool and timing details
    python -m support_agent --reset-db   rebuild the simulated store first

Every conversation is saved as a JSON file in logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from support_agent.agent import AgentReply, SupportAgent, build_agent
from support_agent.config import ConfigError, load_settings
from support_agent.conversation_log import DEFAULT_LOG_DIR, ConversationLog
from support_agent.db import DEFAULT_DB_PATH, connect
from support_agent.prompts import STORE_NAME
from support_agent.seed import build_database

USER_PROMPT = "You> "
EXIT_COMMANDS = ("/exit", "/quit")
HELP = "Commands: /new starts a new conversation, /exit leaves the chat."

# Creates a fresh agent together with the log that will record it.
NewConversation = Callable[[], "tuple[SupportAgent, ConversationLog]"]


def format_trace(reply: AgentReply) -> list[str]:
    """Lines that show what the agent did behind the scenes during one turn."""
    lines = []
    for call in reply.tool_calls:
        arguments = json.dumps(call.input, ensure_ascii=False)
        outcome = f"error: {call.result.get('error_code')}" if call.is_error else "ok"
        lines.append(f"  [tool] {call.name} {arguments} -> {outcome}")
    if reply.error:
        lines.append(f"  [error] {reply.error['type']}: {reply.error['message']}")

    tokens_in = sum(
        reply.usage[name]
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    lines.append(
        f"  [{len(reply.model_calls)} model call(s), {reply.latency_ms / 1000:.1f} s, "
        f"{tokens_in} tokens in / {reply.usage['output_tokens']} out, "
        f"stop: {reply.stop_reason}]"
    )
    return lines


def run_chat(
    new_conversation: NewConversation,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    show_trace: bool = True,
) -> None:
    """The chat loop: read a line, let the agent answer, log the turn, repeat."""
    agent, log = new_conversation()
    write(f"{STORE_NAME} support chat. Write in Spanish, Japanese or English.")
    write(HELP)

    while True:
        try:
            text = read(USER_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            break

        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break
        if text.lower() == "/new":
            _announce_log(log, write)
            agent, log = new_conversation()
            write("Started a new conversation.")
            continue
        if text.startswith("/"):
            write(f"Unknown command. {HELP}")
            continue

        reply = agent.reply(text)
        # Saved after every turn, so the log survives if the chat is interrupted.
        log.record_turn(text, reply)
        log.save(agent.messages)

        if show_trace:
            for line in format_trace(reply):
                write(line)
        write(f"Agent> {reply.text}")
        write("")

    _announce_log(log, write)


def _announce_log(log: ConversationLog, write: Callable[[str], None]) -> None:
    if log.turns:
        write(f"Conversation saved to {log.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m support_agent", description=f"Chat with the {STORE_NAME} support agent."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="hide tool calls, timings and token counts"
    )
    parser.add_argument(
        "--reset-db", action="store_true", help="rebuild the simulated store before starting"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="where conversation logs are saved"
    )
    args = parser.parse_args(argv)

    # Windows consoles and pipes do not always default to UTF-8.
    for stream in (sys.stdin, sys.stdout):
        stream.reconfigure(encoding="utf-8")

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}", file=sys.stderr)
        return 1
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "Warning: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )

    if args.reset_db or not DEFAULT_DB_PATH.exists():
        build_database()
        print(f"Store database rebuilt at {DEFAULT_DB_PATH}")

    conn = connect()

    def new_conversation() -> tuple[SupportAgent, ConversationLog]:
        return build_agent(conn, settings), ConversationLog(settings, args.log_dir)

    try:
        print(f"Model: {settings.model}")
        run_chat(new_conversation, show_trace=not args.quiet)
    finally:
        conn.close()
    return 0
