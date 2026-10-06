"""Tests for the chat loop, driven by scripted input instead of a keyboard."""

import json

from conftest import ALICE
from fakes import FakeClient, response, text, tool_use

from support_agent.agent import AgentReply, SupportAgent, ToolCall
from support_agent.cli import format_trace, run_chat
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog

SETTINGS = Settings(model="test-model")


class Chat:
    """Runs run_chat with typed lines and collects everything it prints."""

    def __init__(self, conn, tmp_path, *responses):
        self.conn = conn
        self.tmp_path = tmp_path
        self.client = FakeClient(*responses)
        self.output: list[str] = []
        self.logs: list[ConversationLog] = []

    def new_conversation(self):
        log = ConversationLog(SETTINGS, log_dir=self.tmp_path)
        self.logs.append(log)
        return SupportAgent(self.client, self.conn, SETTINGS), log

    def run(self, *typed, **kwargs):
        lines = iter(typed)

        def read(prompt):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

        run_chat(self.new_conversation, read=read, write=self.output.append, **kwargs)
        return self

    def saved_logs(self):
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.tmp_path.glob("*.json"))
        ]


def test_prints_the_reply_and_saves_the_turn(conn, tmp_path):
    chat = Chat(conn, tmp_path, response(text("¡Hola! ¿En qué puedo ayudarte?"))).run(
        "hola", "/exit"
    )

    assert "Agent> ¡Hola! ¿En qué puedo ayudarte?" in chat.output
    (saved,) = chat.saved_logs()
    assert saved["turns"][0]["user_message"] == "hola"
    assert saved["turns"][0]["assistant_message"] == "¡Hola! ¿En qué puedo ayudarte?"
    assert saved["transcript"][0] == {"role": "user", "content": "hola"}
    assert any(str(chat.logs[0].path) in line for line in chat.output)


def test_log_contains_the_tool_call(conn, tmp_path):
    chat = Chat(
        conn,
        tmp_path,
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-2", email=ALICE),
            stop_reason="tool_use",
        ),
        response(text("It has shipped.")),
    ).run(f"where is ORD-2? {ALICE}")

    (saved,) = chat.saved_logs()
    (tool_call,) = saved["turns"][0]["tool_calls"]
    assert tool_call["name"] == "get_order_status"
    assert tool_call["input"] == {"order_id": "ORD-2", "email": ALICE}
    assert tool_call["result"]["status"] == "shipped"
    assert saved["totals"]["model_calls"] == 2
    # The raw transcript keeps the tool request and its result too.
    assert [message["role"] for message in saved["transcript"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert any("[tool] get_order_status" in line and "-> ok" in line for line in chat.output)


def test_log_is_saved_after_every_turn(conn, tmp_path):
    chat = Chat(conn, tmp_path, response(text("one")), response(text("two"))).run(
        "first", "second"
    )

    (saved,) = chat.saved_logs()
    assert [turn["user_message"] for turn in saved["turns"]] == ["first", "second"]
    assert saved["totals"]["turns"] == 2


def test_end_of_input_leaves_the_chat(conn, tmp_path):
    chat = Chat(conn, tmp_path).run()

    assert chat.client.requests == []
    assert chat.saved_logs() == []
    assert not any("saved" in line for line in chat.output)


def test_blank_lines_are_ignored(conn, tmp_path):
    chat = Chat(conn, tmp_path, response(text("hi"))).run("", "   ", "hello", "/quit")

    assert len(chat.client.requests) == 1


def test_new_starts_a_separate_conversation_and_log(conn, tmp_path):
    chat = Chat(conn, tmp_path, response(text("one")), response(text("two"))).run(
        "first", "/new", "second", "/exit"
    )

    assert len(chat.saved_logs()) == 2
    # The second conversation does not remember the first.
    assert chat.client.requests[1]["messages"] == [{"role": "user", "content": "second"}]


def test_unknown_command_is_not_sent_to_the_model(conn, tmp_path):
    chat = Chat(conn, tmp_path).run("/refund-everything", "/exit")

    assert chat.client.requests == []
    assert any("Unknown command" in line for line in chat.output)


def test_quiet_mode_hides_the_trace(conn, tmp_path):
    chat = Chat(conn, tmp_path, response(text("hi"))).run("hello", show_trace=False)

    assert not any(line.startswith("  [") for line in chat.output)
    assert "Agent> hi" in chat.output


def test_trace_shows_tool_errors_and_api_errors():
    reply = AgentReply(
        text="x",
        stop_reason="api_error",
        latency_ms=2500,
        tool_calls=[
            ToolCall(
                iteration=1,
                id="toolu_1",
                name="get_order_status",
                input={"order_id": "ORD-1", "email": "注文@example.jp"},
                result={"ok": False, "error_code": "email_mismatch"},
                is_error=True,
                latency_ms=1,
            )
        ],
        error={"type": "RateLimitError", "message": "slow down", "status_code": 429},
    )
    reply.usage.update(input_tokens=10, cache_read_input_tokens=90, output_tokens=7)

    lines = format_trace(reply)

    assert "error: email_mismatch" in lines[0]
    assert "注文@example.jp" in lines[0]
    assert lines[1] == "  [error] RateLimitError: slow down"
    assert "2.5 s" in lines[2]
    assert "100 tokens in / 7 out" in lines[2]
    assert "stop: api_error" in lines[2]
