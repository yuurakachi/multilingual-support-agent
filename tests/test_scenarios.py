"""Tests for the scenario definitions, checks and runner (no real API calls)."""

import json
import re
from datetime import date

import pytest
from fakes import FailingClient, FakeClient, response, text, tool_use

from support_agent.agent import AgentReply, ToolCall
from support_agent.config import Settings
from support_agent.db import connect, create_schema
from support_agent.run_scenarios import main, run_all
from support_agent.scenarios import (
    LANGUAGES,
    Expectations,
    Scenario,
    ScenarioError,
    check_expectations,
    load_scenarios,
    run_scenario,
)
from support_agent.seed import seed

SETTINGS = Settings(model="test-model")
TODAY = date(2026, 10, 1)


# --- the scenarios file -------------------------------------------------------


def test_there_are_ten_scenarios_with_unique_ids():
    scenarios = load_scenarios()

    assert len(scenarios) == 10
    assert len({scenario.id for scenario in scenarios}) == 10


def test_scenarios_cover_every_language_and_required_case():
    scenarios = load_scenarios()

    assert {scenario.language for scenario in scenarios} == set(LANGUAGES)
    categories = {scenario.category for scenario in scenarios}
    assert {"happy_path", "out_of_policy", "privacy", "escalation"} <= categories


def test_every_scenario_expects_something():
    for scenario in load_scenarios():
        expect = scenario.expect
        assert (
            expect.must_succeed
            or expect.must_succeed_one_of
            or expect.must_not_succeed
            or expect.must_not_say
        ), scenario.id


def test_orders_and_emails_used_by_scenarios_match_the_seed_data():
    """Guards against the seed changing under the scenarios."""
    conn = connect(":memory:")
    create_schema(conn)
    seed(conn, TODAY)
    emails = {row["email"] for row in conn.execute("SELECT email FROM customers")}
    orders = {row["id"] for row in conn.execute("SELECT id FROM orders")}
    conn.close()

    intentionally_missing = {"ORD-1080"}
    for scenario in load_scenarios():
        conversation = " ".join(scenario.turns)
        # ASCII only: in Japanese text the address is glued to the words around it.
        for email in re.findall(r"[a-z.]+@example\.[a-z]+", conversation):
            assert email in emails, (scenario.id, email)
        for order_id in set(re.findall(r"ORD-\d+", conversation)) - intentionally_missing:
            assert order_id in orders, (scenario.id, order_id)


def write_scenarios(tmp_path, *scenarios):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(list(scenarios), ensure_ascii=False), encoding="utf-8")
    return path


def valid_scenario(**overrides):
    return {
        "id": "demo",
        "language": "en",
        "category": "happy_path",
        "description": "A demo.",
        "turns": ["hello"],
        "expect": {"must_succeed": ["get_order_status"]},
        **overrides,
    }


@pytest.mark.parametrize(
    "broken",
    [
        valid_scenario(language="fr"),
        valid_scenario(turns=[]),
        valid_scenario(expect={"must_succeed": ["teleport_package"]}),
        valid_scenario(expect={"must_do_magic": ["x"]}),
        {key: value for key, value in valid_scenario().items() if key != "turns"},
    ],
)
def test_malformed_scenarios_are_rejected(tmp_path, broken):
    with pytest.raises(ScenarioError, match="demo"):
        load_scenarios(write_scenarios(tmp_path, broken))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ScenarioError, match="Duplicate"):
        load_scenarios(write_scenarios(tmp_path, valid_scenario(), valid_scenario()))


# --- checks -------------------------------------------------------------------


def reply(*tool_outcomes, said="ok", stop_reason="end_turn"):
    """A finished turn. tool_outcomes are (tool name, succeeded) pairs."""
    return AgentReply(
        text=said,
        stop_reason=stop_reason,
        tool_calls=[
            ToolCall(1, f"toolu_{i}", name, {}, {"ok": succeeded}, not succeeded, 1)
            for i, (name, succeeded) in enumerate(tool_outcomes)
        ],
    )


def outcome(expect, *replies):
    return {check.description: check.passed for check in check_expectations(expect, list(replies))}


def test_must_succeed_needs_a_successful_call():
    expect = Expectations(must_succeed=("request_refund",))

    assert outcome(expect, reply(("request_refund", True)))["request_refund succeeded"] is True
    assert outcome(expect, reply(("request_refund", False)))["request_refund succeeded"] is False
    assert outcome(expect, reply())["request_refund succeeded"] is False


def test_success_in_any_turn_counts():
    expect = Expectations(must_succeed=("request_refund",))

    result = outcome(expect, reply(("request_refund", False)), reply(("request_refund", True)))

    assert result["request_refund succeeded"] is True


def test_must_succeed_one_of_accepts_either_tool():
    expect = Expectations(must_succeed_one_of=("create_support_ticket", "escalate_to_human"))
    label = "create_support_ticket or escalate_to_human succeeded"

    assert outcome(expect, reply(("escalate_to_human", True)))[label] is True
    assert outcome(expect, reply(("get_order_status", True)))[label] is False


def test_must_not_succeed_allows_failed_attempts_only():
    expect = Expectations(must_not_succeed=("request_refund",))
    label = "request_refund never succeeded"

    assert outcome(expect, reply(("request_refund", False)))[label] is True
    assert outcome(expect, reply())[label] is True
    assert outcome(expect, reply(("request_refund", True)))[label] is False


def test_must_not_say_looks_at_every_reply_ignoring_case():
    expect = Expectations(must_not_say=("Mechanical Keyboard", "神宮前"))

    clean = outcome(expect, reply(said="I could not verify those details."))
    leaked = outcome(expect, reply(said="fine"), reply(said="It has a mechanical keyboard."))

    assert all(clean.values())
    assert leaked['replies never mention "Mechanical Keyboard"'] is False
    assert leaked['replies never mention "神宮前"'] is True


def test_a_turn_that_stopped_early_fails_the_scenario():
    checks = check_expectations(Expectations(), [reply(stop_reason="max_iterations")])

    assert checks[0].passed is False
    assert "max_iterations" in checks[0].description


# --- running ------------------------------------------------------------------

STATUS_SCENARIO = Scenario(
    id="status",
    language="es",
    category="happy_path",
    description="Order status.",
    turns=("¿Cómo va mi pedido ORD-1002? maria.garcia@example.com",),
    expect=Expectations(must_succeed=("get_order_status",)),
)


def status_responses():
    return (
        response(
            tool_use(
                "toolu_1",
                "get_order_status",
                order_id="ORD-1002",
                email="maria.garcia@example.com",
            ),
            stop_reason="tool_use",
        ),
        response(text("Tu pedido ya fue enviado.")),
    )


def test_run_scenario_uses_seed_data_and_saves_a_labelled_log(tmp_path):
    client = FakeClient(*status_responses())

    result = run_scenario(STATUS_SCENARIO, client, SETTINGS, tmp_path, run_id="run-1", today=TODAY)

    assert result.passed
    assert result.replies[0].tool_calls[0].result["status"] == "shipped"
    saved = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert saved["metadata"]["scenario_id"] == "status"
    assert saved["metadata"]["run_id"] == "run-1"
    assert saved["metadata"]["category"] == "happy_path"
    assert saved["metadata"]["passed"] is True
    assert {"description": "get_order_status succeeded", "passed": True} in saved["metadata"][
        "checks"
    ]
    assert saved["turns"][0]["user_message"] == STATUS_SCENARIO.turns[0]


def test_each_scenario_starts_from_a_fresh_store(tmp_path):
    refund = Scenario(
        id="refund",
        language="en",
        category="happy_path",
        description="Refund.",
        turns=("refund ORD-1003 please",),
        expect=Expectations(must_succeed=("request_refund",)),
    )

    def responses():
        return (
            response(
                tool_use(
                    "toolu_1",
                    "request_refund",
                    order_id="ORD-1003",
                    email="emily.johnson@example.com",
                    reason="broken",
                ),
                stop_reason="tool_use",
            ),
            response(text("Done.")),
        )

    # A second refund for the same order would fail if the first one had persisted.
    first = run_scenario(refund, FakeClient(*responses()), SETTINGS, tmp_path, today=TODAY)
    second = run_scenario(refund, FakeClient(*responses()), SETTINGS, tmp_path, today=TODAY)

    assert first.passed and second.passed


def test_run_all_prints_what_happened_and_writes_a_summary(tmp_path):
    failing = Scenario(
        id="never_escalates",
        language="en",
        category="escalation",
        description="Wants a human.",
        turns=("human please",),
        expect=Expectations(must_succeed=("escalate_to_human",)),
    )
    client = FakeClient(*status_responses(), response(text("How can I help?")))
    output = []

    summary = run_all(
        [STATUS_SCENARIO, failing], client, SETTINGS, tmp_path, "run-1", output.append
    )

    printed = "\n".join(output)
    assert "[1/2] status" in printed
    assert "Customer> ¿Cómo va mi pedido ORD-1002?" in printed
    assert "[tool] get_order_status" in printed
    assert "Agent> Tu pedido ya fue enviado." in printed
    assert "PASS  get_order_status succeeded" in printed
    assert "FAIL  escalate_to_human succeeded" in printed
    assert "1 of 2 scenarios passed" in printed

    assert (summary["passed"], summary["failed"]) == (1, 1)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "run-1"
    assert saved["model"] == "test-model"
    assert saved["results"][0]["tools"] == ["get_order_status:ok"]
    assert saved["results"][1]["failed_checks"] == ["escalate_to_human succeeded"]
    assert len(list(tmp_path.glob("*.json"))) == 3  # two conversation logs + summary


def test_api_failure_stops_the_scenario_and_fails_it(tmp_path):
    two_turns = Scenario(
        id="two_turns",
        language="en",
        category="happy_path",
        description="Two turns.",
        turns=("first", "second"),
        expect=Expectations(),
    )

    result = run_scenario(two_turns, FailingClient(), SETTINGS, tmp_path, today=TODAY)

    assert len(result.replies) == 1
    assert result.passed is False


def test_list_option_needs_no_api_key(capsys):
    assert main(["--list"]) == 0

    listed = capsys.readouterr().out
    assert "es_order_status" in listed
    assert len(listed.strip().splitlines()) == 10


def test_unknown_scenario_id_is_reported(capsys):
    assert main(["no_such_scenario"]) == 2
    assert "no_such_scenario" in capsys.readouterr().err
