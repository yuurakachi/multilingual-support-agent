"""Push-to-talk voice chat in the terminal: the interface of the voice channel.

Started with `python -m support_agent --voice`. One turn is:

    hold SPACE and speak -> release -> speech-to-text -> agent -> text-to-speech -> speaker

This module only deals with the key, the microphone, the speaker and what is
printed. What happens to a recording is in pipeline.py.

The transcript and the reply are also printed, so it is easy to see what the
recogniser understood and what the voice is reading.

Every turn is timed stage by stage and the times go into the conversation log:

    stt    speech-to-text
    llm    the agent: every model call and tool call of the turn
    tts    text-to-speech
    total  from the end of the recording until the reply is ready to play,
           which is how long the customer waits in silence
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from support_agent.agent import SupportAgent
from support_agent.cli import format_trace
from support_agent.config import DEFAULT_SILENCE_LEVEL, ConfigError, load_voice_settings
from support_agent.conversation_log import VOICE_STAGES, ConversationLog
from support_agent.prompts import STORE_NAME
from support_agent.voice.audio import Audio
from support_agent.voice.keys import NEW, QUIT
from support_agent.voice.pipeline import SILENCE, STT_ERROR, Heard, VoiceConversation
from support_agent.voice.speech import SpeechToText, TextToSpeech

HELP = (
    "Hold SPACE while you talk and release it to send. "
    "N starts a new conversation, Q or Esc leaves."
)
# A tap on the key is not speech: do not spend a request on it.
MIN_SECONDS = 0.4

# Creates a fresh agent together with the log that will record it.
NewConversation = Callable[[], "tuple[SupportAgent, ConversationLog]"]


def run_voice_chat(
    new_conversation: NewConversation,
    key,
    microphone,
    stt: SpeechToText,
    tts: TextToSpeech,
    speaker,
    write: Callable[[str], None] = print,
    show_trace: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    metadata: dict | None = None,
    silence_level: int = DEFAULT_SILENCE_LEVEL,
) -> None:
    """The voice loop: record, transcribe, let the agent answer, speak, repeat.

    `metadata` is added to every conversation log: it says which speech
    providers were used, so timings from different setups can be told apart.
    """
    # Shared by every conversation of this chat, so the request is synthesized once.
    repeat_clips: dict[str, Audio] = {}

    def start_conversation() -> VoiceConversation:
        agent, log = new_conversation()
        log.metadata.update(metadata or {})
        return VoiceConversation(agent, log, stt, tts, clock, silence_level, repeat_clips)

    conversation = start_conversation()
    write(f"{STORE_NAME} voice support. Speak Spanish, Japanese or English.")
    write(HELP)

    while True:
        try:
            action = key.wait()
        except (EOFError, KeyboardInterrupt):
            write("")
            break
        if action == QUIT:
            break
        if action == NEW:
            _announce_log(conversation.log, write)
            conversation = start_conversation()
            write("Started a new conversation.")
            continue

        write("Listening... release SPACE to send.")
        audio = microphone.record_while(key.is_held)
        # The customer has stopped talking: their wait starts now.
        recorded_at = clock()
        key.drain()
        if audio.seconds < MIN_SECONDS:
            write("That was too short. Keep SPACE held while you speak.")
            continue

        try:
            heard = conversation.listen(audio, recorded_at)
            if not heard.understood:
                request = conversation.ask_to_repeat(heard)
                if show_trace:
                    write(describe_problem(heard, silence_level))
                write(f"Agent> {request.text}")
                write("")
                for clip in request.clips:
                    speaker.play(clip)
                continue

            write(f"You> {heard.transcript.text}")
            answer = conversation.answer(heard)
            if show_trace:
                for line in format_trace(answer.reply):
                    write(line)
            write(f"Agent> {answer.reply.text}")
            write("")
            if answer.speech is None:
                # The reply is already on screen, so the conversation can go on.
                write(f"  [text-to-speech error] {answer.tts_error}")
                continue
            if show_trace:
                write(format_timings(answer.timings_ms))
            speaker.play(answer.speech)
        except KeyboardInterrupt:
            # Ctrl+C while waiting for the reply or while it is being spoken.
            write("")
            break

    _announce_log(conversation.log, write)


def format_timings(timings: dict) -> str:
    """One line with how long each stage of a spoken turn took."""
    stt, llm, tts, total = (timings[stage] / 1000 for stage in VOICE_STAGES)
    return (
        f"  [waited {total:.1f} s: speech-to-text {stt:.1f} s, agent {llm:.1f} s, "
        f"text-to-speech {tts:.1f} s]"
    )


def describe_problem(heard: Heard, silence_level: int) -> str:
    """One line saying why a recording was not passed on to the agent."""
    if heard.problem == SILENCE:
        return (
            f"  [not understood: too quiet, level {heard.level} is under {silence_level}. "
            "If you did speak, lower VOICE_SILENCE_LEVEL in .env]"
        )
    if heard.problem == STT_ERROR:
        return f"  [speech-to-text error] {heard.error}"
    return f"  [not understood: {heard.problem}, heard {heard.transcript.text!r}]"


def _announce_log(log: ConversationLog, write: Callable[[str], None]) -> None:
    if log.turns or log.events:
        write(f"Conversation saved to {log.path}")


def start_voice_chat(new_conversation: NewConversation, show_trace: bool = True) -> int:
    """Connect the real microphone, speaker and speech providers, then run the chat."""
    try:
        settings = load_voice_settings()
    except ConfigError as error:
        print(f"Configuration problem: {error}", file=sys.stderr)
        return 1

    # Imported here so that everything above can be tested without audio
    # hardware or a network connection.
    import sounddevice

    from support_agent.voice.audio import Microphone, Speaker
    from support_agent.voice.keys import PushToTalkKey
    from support_agent.voice.stt_groq import GroqSpeechToText
    from support_agent.voice.tts_edge import DEFAULT_VOICES, EdgeTextToSpeech

    # The first time after a restart, Windows can take several seconds to wake the microphone.
    print("Preparing the microphone...")
    try:
        key = PushToTalkKey()
        speaker = Speaker()
        microphone = Microphone()
    except (OSError, sounddevice.PortAudioError) as error:
        print(f"Audio problem: {error}", file=sys.stderr)
        return 1

    stt = GroqSpeechToText(settings.groq_api_key, settings.stt_model)
    voices = {**DEFAULT_VOICES, **settings.tts_voices}
    tts = EdgeTextToSpeech(voices)
    print(f"Speech-to-text: {settings.stt_model} (Groq). Text-to-speech: edge-tts.")
    with microphone:
        run_voice_chat(
            new_conversation,
            key,
            microphone,
            stt,
            tts,
            speaker,
            show_trace=show_trace,
            metadata=speech_metadata(settings, voices),
            silence_level=settings.silence_level,
        )
    return 0


def speech_metadata(settings, voices: dict) -> dict:
    """Which speech providers produced a conversation, for its log."""
    return {
        "stt_provider": "groq",
        "stt_model": settings.stt_model,
        "tts_provider": "edge-tts",
        "tts_voices": dict(voices),
    }
