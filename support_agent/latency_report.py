"""Summarise the latency of the voice channel from the conversation logs.

Usage:
    python -m support_agent.latency_report                 every log under logs/
    python -m support_agent.latency_report PATH [PATH ...]  these log files or folders

For each stage of a spoken turn (speech-to-text, the agent, text-to-speech, and
the total the customer waited) it prints the average and the worst case. Typed
turns have no stages and are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from support_agent.conversation_log import DEFAULT_LOG_DIR, VOICE_STAGES

STAGE_LABELS = {
    "stt": "speech-to-text",
    "llm": "agent (model + tools)",
    "tts": "text-to-speech",
    "total": "total wait",
}


def find_logs(paths: list[Path]) -> list[Path]:
    """Expand folders into the JSON files inside them, at any depth."""
    files = []
    for path in paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    return files


def load_voice_turns(files: list[Path]) -> list[dict]:
    """Every spoken turn found in the files, labelled with where it came from.

    Files that are not conversation logs (a scenario run's summary.json, for
    example) are ignored.
    """
    turns = []
    for file in files:
        try:
            log = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(log, dict) or not isinstance(log.get("turns"), list):
            continue
        for turn in log["turns"]:
            voice = turn.get("voice")
            if not voice or "timings_ms" not in voice:
                continue
            turns.append(
                {
                    "file": file.name,
                    "index": turn.get("index"),
                    "language": voice.get("language"),
                    "recording_seconds": voice.get("recording_seconds"),
                    "speech_seconds": voice.get("speech_seconds"),
                    "timings_ms": voice["timings_ms"],
                }
            )
    return turns


def stage_stats(turns: list[dict]) -> dict:
    """Count, average and worst case of each stage, in milliseconds.

    A stage can be missing from a turn (text-to-speech failed, so it has no
    tts or total): each stage is averaged over the turns that have it.
    """
    stats = {}
    for stage in VOICE_STAGES:
        values = [
            turn["timings_ms"][stage]
            for turn in turns
            if turn["timings_ms"].get(stage) is not None
        ]
        stats[stage] = {
            "turns": len(values),
            "average_ms": round(sum(values) / len(values)) if values else None,
            "worst_ms": max(values) if values else None,
        }
    return stats


def summarize(turns: list[dict]) -> dict:
    languages = sorted({turn["language"] for turn in turns if turn["language"]})
    complete = [turn for turn in turns if turn["timings_ms"].get("total") is not None]
    slowest = max(complete, key=lambda turn: turn["timings_ms"]["total"], default=None)
    return {
        "turns": len(turns),
        "conversations": len({turn["file"] for turn in turns}),
        "overall": stage_stats(turns),
        "by_language": {
            language: stage_stats([turn for turn in turns if turn["language"] == language])
            for language in languages
        },
        "average_recording_seconds": _average(turn["recording_seconds"] for turn in turns),
        "average_speech_seconds": _average(turn["speech_seconds"] for turn in turns),
        "slowest_turn": slowest and {key: slowest[key] for key in ("file", "index", "timings_ms")},
    }


def _average(values) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 1) if values else None


def format_report(summary: dict) -> list[str]:
    if not summary["turns"]:
        return ["No spoken turns with timings were found."]

    lines = [
        f"Voice latency: {summary['turns']} spoken turn(s) "
        f"in {summary['conversations']} conversation(s)",
        "",
    ]
    lines += _table(summary["overall"])
    for language, stats in summary["by_language"].items():
        lines += ["", f"Language: {language}"]
        lines += _table(stats)

    lines.append("")
    if summary["average_recording_seconds"] is not None:
        lines.append(
            f"The customer spoke for {summary['average_recording_seconds']} s on average"
            + (
                f", and the reply lasted {summary['average_speech_seconds']} s."
                if summary["average_speech_seconds"] is not None
                else "."
            )
        )
    slowest = summary["slowest_turn"]
    if slowest:
        lines.append(
            f"Slowest turn: {_seconds(slowest['timings_ms']['total'])} "
            f"(turn {slowest['index']} of {slowest['file']})"
        )
    return lines


def _table(stats: dict) -> list[str]:
    lines = [f"{'STAGE':<24}{'TURNS':>6}{'AVERAGE':>10}{'WORST':>10}"]
    for stage in VOICE_STAGES:
        row = stats[stage]
        lines.append(
            f"{STAGE_LABELS[stage]:<24}{row['turns']:>6}"
            f"{_seconds(row['average_ms']):>10}{_seconds(row['worst_ms']):>10}"
        )
    return lines


def _seconds(milliseconds: int | None) -> str:
    return "-" if milliseconds is None else f"{milliseconds / 1000:.2f} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m support_agent.latency_report",
        description="Average and worst-case latency of each stage of the voice channel.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_LOG_DIR],
        help="log files or folders (default: logs/)",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        print(f"Not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    for line in format_report(summarize(load_voice_turns(find_logs(args.paths)))):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
