"""The scripted scenarios as spoken conversations, for testing without a microphone.

Each customer message of scenarios.json is read aloud once by a text-to-speech
voice and kept as a sound file. Running a scenario then sends those files
through the real pipeline: speech-to-text, the agent, text-to-speech.

On a call the agent reads order numbers and emails back before using them, so
a spoken scenario has more turns than the written one: after the turns listed
in `voice_confirm_after`, the customer answers "yes, that's right".

The scripted customer cannot listen, so it says yes whatever the agent read
back. If the recogniser garbled an order number, the scenario fails on the
tool call that follows, which is the signal we want.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from support_agent.agent import AgentReply, SupportAgent
from support_agent.channels import VOICE_CHANNEL
from support_agent.config import DEFAULT_SILENCE_LEVEL, PROJECT_ROOT, Settings
from support_agent.conversation_log import ConversationLog
from support_agent.db import connect, create_schema
from support_agent.scenarios import Check, Scenario, check_expectations
from support_agent.seed import seed
from support_agent.tool_registry import TOOL_DEFINITIONS
from support_agent.voice.audio import Audio
from support_agent.voice.pipeline import Answer, Heard, VoiceConversation
from support_agent.voice.speech import SpeechToText, TextToSpeech

DEFAULT_AUDIO_DIR = PROJECT_ROOT / "scenarios" / "audio"
MANIFEST_NAME = "manifest.json"

CONFIRMATIONS = {
    "es": "Sí, es correcto.",
    "ja": "はい、合っています。",
    "en": "Yes, that's right.",
}

# The customer's voices, different from the agent's so a played-back
# conversation is easy to follow.
CUSTOMER_VOICES = {
    "es": "es-MX-JorgeNeural",
    "ja": "ja-JP-KeitaNeural",
    "en": "en-US-GuyNeural",
}

# Tools that take an order number or an email: on a call they must wait until
# the customer has confirmed what the agent understood.
IDENTITY_TOOLS = frozenset(
    definition["name"]
    for definition in TOOL_DEFINITIONS
    if {"order_id", "email"} & set(definition["input_schema"]["properties"])
)


@dataclass(frozen=True)
class SpokenLine:
    text: str
    language: str
    # Name of its sound file inside the audio folder.
    file: str
    # True for the "yes, that's right" that answers a read-back.
    is_confirmation: bool = False


def spoken_lines(scenario: Scenario) -> list[SpokenLine]:
    """What the customer says in the spoken version of a scenario, in order."""
    lines = []
    for number, text in enumerate(scenario.turns, start=1):
        lines.append(SpokenLine(text, scenario.language, f"{scenario.id}_{number}.mp3"))
        if number in scenario.voice_confirm_after:
            lines.append(
                SpokenLine(
                    CONFIRMATIONS[scenario.language],
                    scenario.language,
                    # The same clip serves every scenario of that language.
                    f"confirm_{scenario.language}.mp3",
                    is_confirmation=True,
                )
            )
    return lines


# --- the sound files -----------------------------------------------------------


def expected_clips(scenarios: list[Scenario]) -> dict[str, dict]:
    """Every sound file the scenarios need: file name -> its language and text."""
    return {
        line.file: {"language": line.language, "text": line.text}
        for scenario in scenarios
        for line in spoken_lines(scenario)
    }


def make_audio(
    scenarios: list[Scenario],
    tts: TextToSpeech,
    audio_dir: str | Path = DEFAULT_AUDIO_DIR,
    write: Callable[[str], None] = print,
) -> None:
    """Synthesize every customer line and record what was made in a manifest."""
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    clips = expected_clips(scenarios)
    for name, clip in clips.items():
        audio = tts.synthesize(clip["text"], clip["language"])
        audio.save(audio_dir / name)
        write(f"{name:<48}{audio.seconds:5.1f} s  {clip['text']}")
    manifest = {"voices": CUSTOMER_VOICES, "clips": clips}
    (audio_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def audio_problems(
    scenarios: list[Scenario], audio_dir: str | Path = DEFAULT_AUDIO_DIR
) -> list[str]:
    """What is wrong with the sound files for these scenarios; empty when they are usable.

    The manifest says which text each file was made from, so a scenario edited
    after its audio was generated is noticed instead of silently testing old words.
    """
    audio_dir = Path(audio_dir)
    try:
        recorded = json.loads((audio_dir / MANIFEST_NAME).read_text(encoding="utf-8"))["clips"]
    except (OSError, ValueError, KeyError):
        return [f"no usable {MANIFEST_NAME} in {audio_dir}"]

    problems = []
    for name, clip in expected_clips(scenarios).items():
        if not (audio_dir / name).is_file():
            problems.append(f"{name} is missing")
        elif recorded.get(name) != clip:
            problems.append(f"{name} was made from a different text")
    return problems


# --- running a scenario ----------------------------------------------------------


@dataclass
class Exchange:
    """One customer line, what was heard, and what the agent answered (if anything)."""

    line: SpokenLine
    heard: Heard
    # None when the customer was not understood and was asked to repeat.
    answer: Answer | None = None


@dataclass
class VoiceScenarioResult:
    scenario: Scenario
    exchanges: list[Exchange] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def replies(self) -> list[AgentReply]:
        return [exchange.answer.reply for exchange in self.exchanges if exchange.answer]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def check_voice(exchanges: list[Exchange]) -> list[Check]:
    """The checks that only make sense for a spoken conversation."""
    unheard = [exchange.heard.problem for exchange in exchanges if not exchange.heard.understood]
    unvoiced = sum(1 for exchange in exchanges if exchange.answer and not exchange.answer.speech)
    checks = [
        Check(
            "every customer line was understood"
            + (f" (not understood: {', '.join(unheard)})" if unheard else ""),
            not unheard,
        ),
        Check("every reply was turned into speech", unvoiced == 0),
    ]

    # The reply to a line that is followed by "yes, that's right" is the one in
    # which the agent should read the details back instead of using them.
    premature = [
        call.name
        for exchange, following in zip(exchanges, exchanges[1:])
        if following.line.is_confirmation and exchange.answer
        for call in exchange.answer.reply.tool_calls
        if call.name in IDENTITY_TOOLS
    ]
    if any(exchange.line.is_confirmation for exchange in exchanges):
        checks.append(
            Check(
                "no order number or email was used before the customer confirmed it"
                + (f" (used early by: {', '.join(premature)})" if premature else ""),
                not premature,
            )
        )
    return checks


def run_voice_scenario(
    scenario: Scenario,
    client,
    settings: Settings,
    stt: SpeechToText,
    tts: TextToSpeech,
    log_dir: str | Path,
    audio_dir: str | Path = DEFAULT_AUDIO_DIR,
    run_id: str = "",
    metadata: dict | None = None,
    today: date | None = None,
    speaker=None,
    repeat_clips: dict[str, Audio] | None = None,
    silence_level: int = DEFAULT_SILENCE_LEVEL,
) -> VoiceScenarioResult:
    """Play one scenario's sound files through the whole pipeline on a fresh store.

    With a `speaker`, both sides of the conversation are also played aloud.
    """
    conn = connect(":memory:")
    try:
        create_schema(conn)
        seed(conn, today)

        log = ConversationLog(
            settings,
            log_dir,
            metadata={
                "run_id": run_id,
                "scenario_id": scenario.id,
                "category": scenario.category,
                "language": scenario.language,
                "description": scenario.description,
                **(metadata or {}),
            },
            channel=VOICE_CHANNEL,
        )
        agent = SupportAgent(client, conn, settings, channel=VOICE_CHANNEL)
        conversation = VoiceConversation(
            agent, log, stt, tts, silence_level=silence_level, repeat_clips=repeat_clips
        )

        result = VoiceScenarioResult(scenario)
        for line in spoken_lines(scenario):
            audio = Audio.load(Path(audio_dir) / line.file)
            if speaker:
                speaker.play(audio)
            heard = conversation.listen(audio)
            if not heard.understood:
                # A scripted customer cannot say it again: note it and move on.
                request = conversation.ask_to_repeat(heard)
                result.exchanges.append(Exchange(line, heard))
                for clip in request.clips if speaker else ():
                    speaker.play(clip)
                continue

            answer = conversation.answer(heard)
            result.exchanges.append(Exchange(line, heard, answer))
            if speaker and answer.speech:
                speaker.play(answer.speech)
            if answer.reply.stop_reason == "api_error":
                # The API is down or misconfigured: the remaining lines would fail too.
                break

        result.checks = check_expectations(scenario.expect, result.replies) + check_voice(
            result.exchanges
        )
        log.metadata["checks"] = [asdict(check) for check in result.checks]
        log.metadata["passed"] = result.passed
        result.log_path = log.save(agent.messages)
        return result
    finally:
        conn.close()
