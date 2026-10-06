"""Scripted test conversations and the checks applied to them.

A scenario is a list of customer messages plus a few expectations about what
the agent must (or must not) do. The expectations look only at facts that are
unambiguous in the record of the conversation: which tools succeeded and which
strings appeared in the replies. Judging tone or wording is left to the evals.

Every scenario runs against a freshly seeded in-memory store, so scenarios
cannot affect each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from support_agent.agent import AgentReply, SupportAgent
from support_agent.config import PROJECT_ROOT, Settings
from support_agent.conversation_log import ConversationLog
from support_agent.db import connect, create_schema
from support_agent.seed import seed
from support_agent.tool_registry import TOOL_FUNCTIONS

DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "scenarios" / "scenarios.json"
LANGUAGES = ("es", "ja", "en")


class ScenarioError(Exception):
    """The scenarios file is malformed."""


@dataclass(frozen=True)
class Expectations:
    # Every tool listed here must have at least one successful call.
    must_succeed: tuple[str, ...] = ()
    # At least one of these tools must have a successful call.
    must_succeed_one_of: tuple[str, ...] = ()
    # None of these tools may have a successful call.
    must_not_succeed: tuple[str, ...] = ()
    # None of these strings may appear in any reply (case-insensitive).
    must_not_say: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    language: str
    category: str
    description: str
    turns: tuple[str, ...]
    expect: Expectations


@dataclass
class Check:
    description: str
    passed: bool


@dataclass
class ScenarioResult:
    scenario: Scenario
    replies: list[AgentReply] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS_PATH) -> list[Scenario]:
    raw_scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [_parse_scenario(raw) for raw in raw_scenarios]

    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
    if duplicates:
        raise ScenarioError(f"Duplicate scenario ids: {', '.join(duplicates)}")
    return scenarios


def _parse_scenario(raw: dict) -> Scenario:
    scenario_id = raw.get("id", "<missing id>")
    try:
        expect = Expectations(**{key: tuple(value) for key, value in raw["expect"].items()})
        scenario = Scenario(
            id=raw["id"],
            language=raw["language"],
            category=raw["category"],
            description=raw["description"],
            turns=tuple(raw["turns"]),
            expect=expect,
        )
    except (KeyError, TypeError) as error:
        raise ScenarioError(f"Scenario {scenario_id}: invalid or missing field ({error})") from None

    if scenario.language not in LANGUAGES:
        raise ScenarioError(f"Scenario {scenario_id}: unknown language {scenario.language!r}")
    if not scenario.turns or not all(isinstance(turn, str) and turn for turn in scenario.turns):
        raise ScenarioError(f"Scenario {scenario_id}: turns must be non-empty strings")
    named_tools = {*expect.must_succeed, *expect.must_succeed_one_of, *expect.must_not_succeed}
    unknown = sorted(named_tools - set(TOOL_FUNCTIONS))
    if unknown:
        raise ScenarioError(f"Scenario {scenario_id}: unknown tools {', '.join(unknown)}")
    return scenario


def check_expectations(expect: Expectations, replies: list[AgentReply]) -> list[Check]:
    """Compare what happened in a conversation with what the scenario expects."""
    succeeded = {
        call.name for reply in replies for call in reply.tool_calls if not call.is_error
    }
    everything_said = "\n".join(reply.text for reply in replies).lower()

    stopped_early = [reply.stop_reason for reply in replies if not reply.completed]
    checks = [
        Check(
            "every turn ended normally"
            + (f" (got: {', '.join(stopped_early)})" if stopped_early else ""),
            not stopped_early,
        )
    ]
    for tool in expect.must_succeed:
        checks.append(Check(f"{tool} succeeded", tool in succeeded))
    if expect.must_succeed_one_of:
        options = " or ".join(expect.must_succeed_one_of)
        checks.append(
            Check(f"{options} succeeded", bool(succeeded & set(expect.must_succeed_one_of)))
        )
    for tool in expect.must_not_succeed:
        checks.append(Check(f"{tool} never succeeded", tool not in succeeded))
    for phrase in expect.must_not_say:
        checks.append(
            Check(f'replies never mention "{phrase}"', phrase.lower() not in everything_said)
        )
    return checks


def run_scenario(
    scenario: Scenario,
    client,
    settings: Settings,
    log_dir: str | Path,
    run_id: str = "",
    today: date | None = None,
) -> ScenarioResult:
    """Play one scenario against a fresh store and save its conversation log."""
    conn = connect(":memory:")
    try:
        create_schema(conn)
        seed(conn, today)

        agent = SupportAgent(client, conn, settings)
        log = ConversationLog(
            settings,
            log_dir,
            metadata={
                "run_id": run_id,
                "scenario_id": scenario.id,
                "category": scenario.category,
                "language": scenario.language,
                "description": scenario.description,
            },
        )
        result = ScenarioResult(scenario)
        for message in scenario.turns:
            reply = agent.reply(message)
            result.replies.append(reply)
            log.record_turn(message, reply)
            if reply.stop_reason == "api_error":
                # The API is down or misconfigured: the remaining turns would fail too.
                break

        result.checks = check_expectations(scenario.expect, result.replies)
        log.metadata["checks"] = [asdict(check) for check in result.checks]
        log.metadata["passed"] = result.passed
        result.log_path = log.save(agent.messages)
        return result
    finally:
        conn.close()
