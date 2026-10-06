"""Run the scripted scenarios against the live API and show what happened.

Usage:
    python -m support_agent.run_scenarios                 run every scenario
    python -m support_agent.run_scenarios ID [ID ...]     run only these
    python -m support_agent.run_scenarios --list          list the scenarios

Each run saves one conversation log per scenario plus a summary.json under
logs/scenarios/<run id>/. The exit code is 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import anthropic

from support_agent.cli import format_trace
from support_agent.config import ConfigError, Settings, load_settings
from support_agent.conversation_log import DEFAULT_LOG_DIR
from support_agent.scenarios import (
    DEFAULT_SCENARIOS_PATH,
    Scenario,
    ScenarioError,
    ScenarioResult,
    load_scenarios,
    run_scenario,
)

Write = Callable[[str], None]


def print_result(result: ScenarioResult, position: str, write: Write = print) -> None:
    scenario = result.scenario
    write(f"[{position}] {scenario.id}  ({scenario.category}, {scenario.language})")
    write(f"  {scenario.description}")
    write("")
    for message, reply in zip(scenario.turns, result.replies):
        write(f"  Customer> {message}")
        for line in format_trace(reply):
            write(f"  {line}")
        write(_indent(f"Agent> {reply.text}"))
        write("")
    skipped = len(scenario.turns) - len(result.replies)
    if skipped:
        write(f"  ({skipped} customer message(s) not sent because the API failed)")
    for check in result.checks:
        write(f"  {'PASS' if check.passed else 'FAIL'}  {check.description}")
    write(f"  Log: {result.log_path}")
    write("")


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def summarize(results: list[ScenarioResult]) -> dict:
    """Totals for the whole run, also saved as summary.json."""
    replies = [reply for result in results for reply in result.replies]
    return {
        "scenarios": len(results),
        "passed": sum(result.passed for result in results),
        "failed": sum(not result.passed for result in results),
        "model_calls": sum(len(reply.model_calls) for reply in replies),
        "tool_calls": sum(len(reply.tool_calls) for reply in replies),
        "latency_ms": sum(reply.latency_ms for reply in replies),
        "output_tokens": sum(reply.usage["output_tokens"] for reply in replies),
        "input_tokens": sum(
            reply.usage[name]
            for reply in replies
            for name in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        ),
        "results": [
            {
                "scenario_id": result.scenario.id,
                "category": result.scenario.category,
                "language": result.scenario.language,
                "passed": result.passed,
                "failed_checks": [c.description for c in result.checks if not c.passed],
                "tools": [
                    f"{call.name}:{'error' if call.is_error else 'ok'}"
                    for reply in result.replies
                    for call in reply.tool_calls
                ],
                "log_file": result.log_path.name if result.log_path else None,
            }
            for result in results
        ],
    }


def print_summary(summary: dict, write: Write = print) -> None:
    write("=" * 78)
    write(f"{'RESULT':<7}{'SCENARIO':<40}{'TOOLS (in order)'}")
    for row in summary["results"]:
        tools = ", ".join(row["tools"]) or "-"
        write(f"{'PASS' if row['passed'] else 'FAIL':<7}{row['scenario_id']:<40}{tools}")
    write("-" * 78)
    write(
        f"{summary['passed']} of {summary['scenarios']} scenarios passed. "
        f"{summary['model_calls']} model calls, {summary['tool_calls']} tool calls, "
        f"{summary['latency_ms'] / 1000:.0f} s, "
        f"{summary['input_tokens']} tokens in / {summary['output_tokens']} out."
    )


def run_all(
    scenarios: list[Scenario],
    client,
    settings: Settings,
    log_dir: Path,
    run_id: str,
    write: Write = print,
) -> dict:
    """Run the scenarios one by one, printing each as it finishes."""
    results = []
    for number, scenario in enumerate(scenarios, start=1):
        result = run_scenario(scenario, client, settings, log_dir, run_id)
        results.append(result)
        print_result(result, f"{number}/{len(scenarios)}", write)

    summary = {"run_id": run_id, "model": settings.model, **summarize(results)}
    print_summary(summary, write)
    summary_path = log_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write(f"Logs and summary saved in {log_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m support_agent.run_scenarios",
        description="Run the scripted test conversations against the live API.",
    )
    parser.add_argument("ids", nargs="*", help="scenario ids to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument(
        "--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH, help="scenarios JSON file"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR / "scenarios",
        help="where run folders are created",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        scenarios = load_scenarios(args.scenarios)
    except ScenarioError as error:
        print(f"Scenario file problem: {error}", file=sys.stderr)
        return 2

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id:<40}{scenario.category:<19}{scenario.language}")
        return 0

    if args.ids:
        known = {scenario.id for scenario in scenarios}
        unknown = [scenario_id for scenario_id in args.ids if scenario_id not in known]
        if unknown:
            print(f"Unknown scenario id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        scenarios = [scenario for scenario in scenarios if scenario.id in args.ids]

    try:
        # Settings first: loading .env is what gives the client its API key.
        settings = load_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}", file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    print(f"Run {run_id}: {len(scenarios)} scenario(s) with model {settings.model}\n")
    summary = run_all(scenarios, client, settings, args.log_dir / run_id, run_id)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
