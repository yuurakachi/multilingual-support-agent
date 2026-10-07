"""Push-to-talk voice chat in the terminal: the interface of the voice channel.

Started with `python -m support_agent --voice`. One turn is:

    hold SPACE and speak -> release -> speech-to-text -> agent -> text-to-speech -> speaker

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
from support_agent.config import ConfigError, load_voice_settings
from support_agent.conversation_log import VOICE_STAGES, ConversationLog
from support_agent.prompts import STORE_NAME
from support_agent.voice.keys import NEW, QUIT
from support_agent.voice.speech import DEFAULT_LANGUAGE, SpeechError, SpeechToText, TextToSpeech

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
) -> None:
    """The voice loop: record, transcribe, let the agent answer, speak, repeat.

    `metadata` is added to every conversation log: it says which speech
    providers were used, so timings from different setups can be told apart.
    """

    def elapsed_ms(started: float) -> int:
        return round((clock() - started) * 1000)

    def start_conversation() -> tuple[SupportAgent, ConversationLog]:
        agent, log = new_conversation()
        log.metadata.update(metadata or {})
        return agent, log

    agent, log = start_conversation()
    # The voice that reads the reply follows the language the customer speaks.
    language = DEFAULT_LANGUAGE
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
            _announce_log(log, write)
            agent, log = start_conversation()
            language = DEFAULT_LANGUAGE
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

        started = clock()
        try:
            transcript = stt.transcribe(audio)
        except SpeechError as error:
            write(f"  [speech-to-text error] {error}")
            continue
        stt_ms = elapsed_ms(started)
        if not transcript.text:
            write("I could not hear anything. Please try again.")
            continue
        # An unrecognised language keeps the voice of the previous turn.
        language = transcript.language or language
        write(f"You> {transcript.text}")

        reply = agent.reply(transcript.text, language=language)

        # tts and total are not known yet: they are filled in below.
        timings = {"stt": stt_ms, "llm": reply.latency_ms, "tts": None, "total": None}
        voice = {
            # The language the reply is spoken in, and what the recogniser reported
            # (None when it heard a language the agent does not support).
            "language": language,
            "detected_language": transcript.language,
            "recording_seconds": round(audio.seconds, 2),
            "speech_seconds": None,
            "timings_ms": timings,
        }
        # Saved as soon as the agent has answered, so the log survives if the chat
        # is interrupted while the reply is being spoken.
        log.record_turn(transcript.text, reply, voice=voice)
        log.save(agent.messages)

        if show_trace:
            for line in format_trace(reply):
                write(line)
        write(f"Agent> {reply.text}")
        write("")

        started = clock()
        try:
            speech = tts.synthesize(reply.text, language)
        except SpeechError as error:
            # The reply is already on screen, so the conversation can go on.
            write(f"  [text-to-speech error] {error}")
            continue
        timings["tts"] = elapsed_ms(started)
        timings["total"] = elapsed_ms(recorded_at)
        voice["speech_seconds"] = round(speech.seconds, 2)
        log.save(agent.messages)
        if show_trace:
            write(format_timings(timings))

        try:
            speaker.play(speech)
        except KeyboardInterrupt:
            write("")
            break

    _announce_log(log, write)


def format_timings(timings: dict) -> str:
    """One line with how long each stage of a spoken turn took."""
    stt, llm, tts, total = (timings[stage] / 1000 for stage in VOICE_STAGES)
    return (
        f"  [waited {total:.1f} s: speech-to-text {stt:.1f} s, agent {llm:.1f} s, "
        f"text-to-speech {tts:.1f} s]"
    )


def _announce_log(log: ConversationLog, write: Callable[[str], None]) -> None:
    if log.turns:
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
    metadata = {
        "stt_provider": "groq",
        "stt_model": settings.stt_model,
        "tts_provider": "edge-tts",
        "tts_voices": voices,
    }
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
            metadata=metadata,
        )
    return 0
