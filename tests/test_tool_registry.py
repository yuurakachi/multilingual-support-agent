import inspect

import pytest
from conftest import ALICE

from support_agent import tool_registry
from support_agent.tool_registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS, execute_tool


def test_every_definition_has_a_function_and_vice_versa():
    assert [definition["name"] for definition in TOOL_DEFINITIONS] == list(TOOL_FUNCTIONS)


@pytest.mark.parametrize("definition", TOOL_DEFINITIONS, ids=lambda d: d["name"])
def test_schema_matches_the_python_signature(definition):
    """The arguments the model is told about are exactly the function's public arguments."""
    signature = inspect.signature(TOOL_FUNCTIONS[definition["name"]])
    public_arguments = [
        name
        for name, parameter in signature.parameters.items()
        if name != "conn" and parameter.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    schema = definition["input_schema"]

    assert list(schema["properties"]) == public_arguments
    assert schema["required"] == public_arguments
    assert schema["additionalProperties"] is False
    assert definition["strict"] is True
    assert definition["description"]


def test_runs_the_requested_tool(conn):
    result = execute_tool(conn, "get_order_status", {"order_id": "ORD-1", "email": ALICE})

    assert result["ok"] is True
    assert result["order_id"] == "ORD-1"


def test_passes_through_tool_error_results(conn):
    result = execute_tool(conn, "get_order_status", {"order_id": "ORD-999", "email": ALICE})

    assert result["error_code"] == "order_not_found"


def test_unknown_tool_is_an_error_result(conn):
    result = execute_tool(conn, "delete_everything", {})

    assert result["ok"] is False
    assert result["error_code"] == "unknown_tool"


@pytest.mark.parametrize(
    "tool_input",
    [
        {"order_id": "ORD-1"},  # missing argument
        {"order_id": "ORD-1", "email": ALICE, "extra": "x"},  # unexpected argument
        "ORD-1",  # not an object
        None,
    ],
)
def test_wrong_arguments_are_an_error_result(conn, tool_input):
    result = execute_tool(conn, "get_order_status", tool_input)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"


def test_model_cannot_set_the_internal_clock(conn):
    """request_refund accepts a hidden `now` for tests; the model must not reach it."""
    result = execute_tool(
        conn,
        "request_refund",
        {"order_id": "ORD-5", "email": "yuki@example.jp", "reason": "late", "now": "2020-01-01"},
    )

    assert result["error_code"] == "invalid_input"
    assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0


def test_tool_bug_becomes_an_error_result(conn, monkeypatch):
    def broken_tool(conn, reason):
        raise RuntimeError("boom")

    monkeypatch.setitem(tool_registry.TOOL_FUNCTIONS, "escalate_to_human", broken_tool)

    result = execute_tool(conn, "escalate_to_human", {"reason": "anything"})

    assert result["ok"] is False
    assert result["error_code"] == "internal_error"
    assert "boom" not in str(result)
