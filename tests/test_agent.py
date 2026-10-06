"""Tests for the agent loop, using a scripted fake instead of the real API."""

import json

from conftest import ALICE, YUKI
from fakes import FakeClient, response, text, thinking, tool_use

from support_agent.agent import FALLBACK_MESSAGE, SupportAgent
from support_agent.config import Settings
from support_agent.prompts import SYSTEM_PROMPT
from support_agent.tool_registry import TOOL_DEFINITIONS

SETTINGS = Settings(model="test-model", max_iterations=4)


def test_plain_answer_needs_one_model_call(conn):
    client = FakeClient(response(text("Hello! How can I help?")))
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("hi")

    assert reply.text == "Hello! How can I help?"
    assert reply.completed
    assert reply.iterations == 1
    assert reply.tool_calls == []
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["system"] == SYSTEM_PROMPT
    assert request["tools"] == TOOL_DEFINITIONS
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert "output_config" not in request


def test_tool_call_is_executed_and_its_result_sent_back(conn):
    first = response(
        thinking(),
        tool_use("toolu_1", "get_order_status", order_id="ORD-2", email=ALICE),
        stop_reason="tool_use",
    )
    client = FakeClient(first, response(text("Your order has shipped.")))
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply(f"Where is ORD-2? My email is {ALICE}")

    assert reply.text == "Your order has shipped."
    assert reply.iterations == 2
    (call,) = reply.tool_calls
    assert call.name == "get_order_status"
    assert call.input == {"order_id": "ORD-2", "email": ALICE}
    assert call.result["status"] == "shipped"
    assert call.is_error is False

    # Second request = original question + the model's turn, untouched + the tool result.
    sent = client.requests[1]["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"] is first.content
    (result_block,) = sent[2]["content"]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_1"
    assert result_block["is_error"] is False
    assert json.loads(result_block["content"]) == call.result


def test_tool_changes_the_database(conn):
    client = FakeClient(
        response(
            tool_use(
                "toolu_1",
                "update_shipping_address",
                order_id="ORD-1",
                email=ALICE,
                new_address="5 New Road, Dublin, Ireland",
            ),
            stop_reason="tool_use",
        ),
        response(text("Done.")),
    )

    SupportAgent(client, conn, SETTINGS).reply("change my address")

    stored = conn.execute("SELECT shipping_address FROM orders WHERE id = 'ORD-1'").fetchone()
    assert stored["shipping_address"] == "5 New Road, Dublin, Ireland"


def test_several_tool_calls_in_one_turn_share_one_result_message(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-1", email=ALICE),
            tool_use("toolu_2", "get_order_status", order_id="ORD-2", email=ALICE),
            stop_reason="tool_use",
        ),
        response(text("Both found.")),
    )
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("status of ORD-1 and ORD-2")

    assert [call.result["order_id"] for call in reply.tool_calls] == ["ORD-1", "ORD-2"]
    results = client.requests[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in results] == ["toolu_1", "toolu_2"]


def test_failed_tool_is_reported_to_the_model_and_the_loop_continues(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-1", email=YUKI),
            stop_reason="tool_use",
        ),
        response(text("I could not verify those details.")),
    )
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("show me ORD-1")

    assert reply.completed
    assert reply.tool_calls[0].is_error is True
    assert reply.tool_calls[0].result["error_code"] == "email_mismatch"
    result_block = client.requests[1]["messages"][-1]["content"][0]
    assert result_block["is_error"] is True
    assert "Alice" not in result_block["content"]


def test_unknown_tool_does_not_crash_the_loop(conn):
    client = FakeClient(
        response(tool_use("toolu_1", "cancel_order", order_id="ORD-1"), stop_reason="tool_use"),
        response(text("I cannot cancel orders, but I can open a ticket.")),
    )

    reply = SupportAgent(client, conn, SETTINGS).reply("cancel ORD-1")

    assert reply.completed
    assert reply.tool_calls[0].result["error_code"] == "unknown_tool"


def test_loop_stops_at_the_iteration_limit(conn):
    always_tool = response(
        tool_use("toolu_1", "get_order_status", order_id="ORD-1", email=ALICE),
        stop_reason="tool_use",
    )
    client = FakeClient(always_tool, repeat_last=True)
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("loop forever")

    assert len(client.requests) == SETTINGS.max_iterations
    assert reply.stop_reason == "max_iterations"
    assert reply.completed is False
    assert reply.iterations == SETTINGS.max_iterations
    assert reply.text == FALLBACK_MESSAGE
    assert len(reply.tool_calls) == SETTINGS.max_iterations
    assert agent.messages[-1] == {"role": "assistant", "content": FALLBACK_MESSAGE}


def test_refusal_never_runs_tools(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "escalate_to_human", reason="cut off"), stop_reason="refusal"
        )
    )
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("something the model declines")

    assert reply.stop_reason == "refusal"
    assert reply.text == FALLBACK_MESSAGE
    assert reply.tool_calls == []
    assert conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0] == 0


def test_truncated_turn_never_runs_tools(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "request_refund", order_id="ORD-3", email=ALICE, reason="x"),
            stop_reason="max_tokens",
        )
    )

    reply = SupportAgent(client, conn, SETTINGS).reply("refund please")

    assert reply.stop_reason == "max_tokens"
    assert reply.tool_calls == []
    assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0


def test_answer_without_text_falls_back(conn):
    client = FakeClient(response(thinking()))

    reply = SupportAgent(client, conn, SETTINGS).reply("hi")

    assert reply.stop_reason == "empty_response"
    assert reply.text == FALLBACK_MESSAGE


def test_conversation_history_carries_over_between_messages(conn):
    client = FakeClient(response(text("Hola, ¿en qué puedo ayudarte?")), response(text("Claro.")))
    agent = SupportAgent(client, conn, SETTINGS)

    agent.reply("hola")
    agent.reply("quiero saber de mi pedido")

    sent = client.requests[1]["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "hola"
    assert sent[2]["content"] == "quiero saber de mi pedido"


def test_usage_is_summed_across_model_calls(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-1", email=ALICE),
            stop_reason="tool_use",
            input_tokens=100,
            output_tokens=20,
        ),
        response(text("ok"), input_tokens=150, output_tokens=30),
    )

    reply = SupportAgent(client, conn, SETTINGS).reply("status?")

    assert reply.usage == {
        "input_tokens": 250,
        "output_tokens": 50,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 6,
    }


def test_effort_is_sent_only_when_configured(conn):
    client = FakeClient(response(text("ok")))
    settings = Settings(model="test-model", effort="low")

    SupportAgent(client, conn, settings).reply("hi")

    assert client.requests[0]["output_config"] == {"effort": "low"}


def test_build_agent_loads_env_before_creating_the_client(conn, monkeypatch):
    """Regression: the client was once created before .env was loaded, so it had no key."""
    from support_agent import agent as agent_module

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    key_seen_by_client = []

    def fake_load_settings():
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-dotenv")
        return SETTINGS

    def fake_client():
        import os

        key_seen_by_client.append(os.environ.get("ANTHROPIC_API_KEY"))
        return FakeClient()

    monkeypatch.setattr(agent_module, "load_settings", fake_load_settings)
    monkeypatch.setattr(agent_module.anthropic, "Anthropic", fake_client)

    built = agent_module.build_agent(conn)

    assert key_seen_by_client == ["test-key-from-dotenv"]
    assert built.settings is SETTINGS


class FakeClock:
    """Advances half a second every time it is read."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.5
        return self.now


def test_records_model_calls_tool_calls_and_latency(conn):
    client = FakeClient(
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-1", email=ALICE),
            stop_reason="tool_use",
            input_tokens=100,
            output_tokens=20,
        ),
        response(text("ok"), input_tokens=150, output_tokens=30),
    )
    agent = SupportAgent(client, conn, SETTINGS, clock=FakeClock())

    reply = agent.reply("status?")

    assert [(call.iteration, call.stop_reason) for call in reply.model_calls] == [
        (1, "tool_use"),
        (2, "end_turn"),
    ]
    assert [call.latency_ms for call in reply.model_calls] == [500, 500]
    assert reply.model_calls[0].usage["input_tokens"] == 100
    assert reply.model_calls[1].usage["output_tokens"] == 30

    (tool_call,) = reply.tool_calls
    assert (tool_call.iteration, tool_call.id) == (1, "toolu_1")
    assert tool_call.latency_ms == 500

    # The whole turn takes at least as long as its parts.
    assert reply.latency_ms >= 1500


def api_connection_error():
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="Connection error.", request=request)


class FailingClient(FakeClient):
    """Returns its scripted responses, then fails like a dropped connection."""

    def _create(self, **request):
        if not self._responses:
            self.requests.append(request)
            raise api_connection_error()
        return super()._create(**request)


def test_api_failure_becomes_a_reply_instead_of_an_exception(conn):
    agent = SupportAgent(FailingClient(), conn, SETTINGS)

    reply = agent.reply("hello?")

    assert reply.stop_reason == "api_error"
    assert reply.completed is False
    assert reply.text == FALLBACK_MESSAGE
    assert reply.error == {
        "type": "APIConnectionError",
        "message": "Connection error.",
        "status_code": None,
    }
    assert reply.model_calls == []
    assert [message["role"] for message in agent.messages] == ["user", "assistant"]


def test_api_failure_keeps_the_tools_that_already_ran(conn):
    client = FailingClient(
        response(
            tool_use("toolu_1", "escalate_to_human", reason="Customer asked for a person."),
            stop_reason="tool_use",
        )
    )
    agent = SupportAgent(client, conn, SETTINGS)

    reply = agent.reply("I want a human")

    assert reply.stop_reason == "api_error"
    assert [call.name for call in reply.tool_calls] == ["escalate_to_human"]
    assert conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0] == 1
    # History stays well-formed: question, tool request, tool result, fallback.
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
