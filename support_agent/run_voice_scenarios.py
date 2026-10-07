"""Run the scripted scenarios as spoken conversations, without a microphone.

Usage:
    python -m support_agent.run_voice_scenarios                run every scenario
    python -m support_agent.run_voice_scenarios ID [ID ...]    run only these
    python -m support_agent.run_voice_scenarios --play         also play both sides aloud
    python -m support_agent.run_voice_scenarios --list         list the spoken lines
    python -m support_agent.run_voice_scenarios --make-audio   regenerate the sound files

The customer's lines are sound files in scenarios/audio/, made once with
text-to-speech. Each one goes through the real pipeline (speech-to-text, the
agent, text-to-speech) against the live APIs. Every run saves one conversation
log per scenario plus a summary.json under logs/voice_scenarios/<run id>/, and
ends with the latency of each stage. The exit code is 0 only when every check
passed.
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
from support_agent.config import (
    DEFAULT_SILENCE_LEVEL,
    ConfigError,
    Settings,
    load_settings,
    load_voice_settings,
)
from support_agent.conversation_log import DEFAULT_LOG_DIR
from support_agent.latency_report import find_logs, format_report, load_voice_turns
from support_agent.latency_report import summarize as summarize_latency
from support_agent.run_scenarios import print_summary, summarize
from support_agent.scenarios import DEFAULT_SCENARIOS_PATH, Scenario, ScenarioError, load_scenarios
from support_agent.voice.chat import describe_problem, format_timings, speech_metadata
from support_agent.voice.scenarios import (
    CUSTOMER_VOICES,
    DEFAULT_AUDIO_DIR,
    VoiceScenarioResult,
    audio_problems,
    make_audio,
    run_voice_scenario,
    spoken_lines,
)
from support_agent.voice.speech import SpeechToText, TextToSpeech

Write = Callable[[str], None]


def print_result(
    result: VoiceScenarioResult, position: str, silence_level: int, write: Write = print
) -> None:
    scenario = result.scenario
    write(f"[{position}] {scenario.id}  ({scenario.category}, {scenario.language})")
    write(f"  {scenario.description}")
    write("")
    for exchange in result.exchanges:
        write(f"  Said>  {exchange.line.text}")
        if exchange.answer is None:
            write(f"  {describe_problem(exchange.heard, silence_level)}")
            write("")
            continue
        write(f"  Heard> {exchange.heard.transcript.text}")
        for line in format_trace(exchange.answer.reply):
            write(f"  {line}")
        write(_indent(f"Agent> {exchange.answer.reply.text}"))
        if exchange.answer.speech:
            write(f"  {format_timings(exchange.answer.timings_ms)}")
        else:
            write(f"    [text-to-speech error] {exchange.answer.tts_error}")
        write("")
    skipped = len(spoken_lines(scenario)) - len(result.exchanges)
    if skipped:
        write(f"  ({skipped} customer line(s) not sent because the API failed)")
    for check in result.checks:
        write(f"  {'PASS' if check.passed else 'FAIL'}  {check.description}")
    write(f"  Log: {result.log_path}")
    write("")


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def run_all(
    scenarios: list[Scenario],
    client,
    settings: Settings,
    stt: SpeechToText,
    tts: TextToSpeech,
    log_dir: Path,
    run_id: str,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    metadata: dict | None = None,
    silence_level: int = DEFAULT_SILENCE_LEVEL,
    speaker=None,
    write: Write = print,
) -> dict:
    """Run the scenarios one by one, printing each as it finishes."""
    results = []
    repeat_clips: dict = {}
    for number, scenario in enumerate(scenarios, start=1):
        result = run_voice_scenario(
            scenario,
            client,
            settings,
            stt,
            tts,
            log_dir,
            audio_dir,
            run_id,
            metadata,
            speaker=speaker,
            repeat_clips=repeat_clips,
            silence_level=silence_level,
        )
        results.append(result)
        print_result(result, f"{number}/{len(scenarios)}", silence_level, write)

    latency = summarize_latency(load_voice_turns(find_logs([log_dir])))
    summary = {
        "run_id": run_id,
        "model": settings.model,
        **(metadata or {}),
        **summarize(results),
        "unheard_lines": sum(
            not exchange.heard.understood for result in results for exchange in result.exchanges
        ),
        "latency": latency,
    }
    print_summary(summary, write)
    write("")
    for line in format_report(latency):
        write(line)
    write("")

    summary_path = log_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write(f"Logs and summary saved in {log_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m support_agent.run_voice_scenarios",
        description="Run the scripted conversations as speech, through the whole voice pipeline.",
    )
    parser.add_argument("ids", nargs="*", help="scenario ids to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the spoken lines and exit")
    parser.add_argument(
        "--make-audio",
        action="store_true",
        help="generate the customer's sound files with text-to-speech and exit",
    )
    parser.add_argument(
        "--play", action="store_true", help="play the customer and the agent aloud while running"
    )
    parser.add_argument(
        "--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH, help="scenarios JSON file"
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR, help="where the sound files are"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR / "voice_scenarios",
        help="where run folders are created",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        scenarios = load_scenarios(args.scenarios)
    except ScenarioError as error:
        print(f"Scenario file problem: {error}", file=sys.stderr)
        return 2

    if args.ids:
        known = {scenario.id for scenario in scenarios}
        unknown = [scenario_id for scenario_id in args.ids if scenario_id not in known]
        if unknown:
            print(f"Unknown scenario id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        scenarios = [scenario for scenario in scenarios if scenario.id in args.ids]

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id}  ({scenario.category}, {scenario.language})")
            for line in spoken_lines(scenario):
                print(f"  {line.file:<46}{line.text}")
        return 0

    if args.make_audio:
        # Imported here: it needs the network, which --list and the checks do not.
        from support_agent.voice.tts_edge import EdgeTextToSpeech

        # Always the whole set, so the manifest describes every file in the folder.
        make_audio(load_scenarios(args.scenarios), EdgeTextToSpeech(CUSTOMER_VOICES), args.audio_dir)
        print(f"Sound files saved in {args.audio_dir}")
        return 0

    problems = audio_problems(scenarios, args.audio_dir)
    if problems:
        print("The sound files do not match the scenarios:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Regenerate them with --make-audio.", file=sys.stderr)
        return 2

    try:
        # Settings first: loading .env is what gives the clients their API keys.
        settings = load_settings()
        voice_settings = load_voice_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}", file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    from support_agent.voice.stt_groq import GroqSpeechToText
    from support_agent.voice.tts_edge import DEFAULT_VOICES, EdgeTextToSpeech

    voices = {**DEFAULT_VOICES, **voice_settings.tts_voices}
    speaker = None
    if args.play:
        from support_agent.voice.audio import Speaker

        speaker = Speaker()

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    print(f"Run {run_id}: {len(scenarios)} spoken scenario(s) with model {settings.model}\n")
    summary = run_all(
        scenarios,
        client,
        settings,
        GroqSpeechToText(voice_settings.groq_api_key, voice_settings.stt_model),
        EdgeTextToSpeech(voices),
        args.log_dir / run_id,
        run_id,
        args.audio_dir,
        metadata=speech_metadata(voice_settings, voices),
        silence_level=voice_settings.silence_level,
        speaker=speaker,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
