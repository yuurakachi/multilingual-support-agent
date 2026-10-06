"""Push-to-talk voice chat in the terminal: the interface of the voice channel.

Started with `python -m support_agent --voice`. One turn is:

    hold SPACE and speak -> release -> speech-to-text -> agent -> text-to-speech -> speaker

The transcript and the reply are also printed, so it is easy to see what the
recogniser understood and what the voice is reading.
"""

from __future__ import annotations

import sys
from typing import Callable

from support_agent.agent import SupportAgent
from support_agent.cli import format_trace
from support_agent.config import ConfigError, load_voice_settings
from support_agent.conversation_log import ConversationLog
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
) -> None:
    """The voice loop: record, transcribe, let the agent answer, speak, repeat."""
    agent, log = new_conversation()
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
            agent, log = new_conversation()
            language = DEFAULT_LANGUAGE
            write("Started a new conversation.")
            continue

        write("Listening... release SPACE to send.")
        audio = microphone.record_while(key.is_held)
        key.drain()
        if audio.seconds < MIN_SECONDS:
            write("That was too short. Keep SPACE held while you speak.")
            continue

        try:
            transcript = stt.transcribe(audio)
        except SpeechError as error:
            write(f"  [speech-to-text error] {error}")
            continue
        if not transcript.text:
            write("I could not hear anything. Please try again.")
            continue
        # An unrecognised language keeps the voice of the previous turn.
        language = transcript.language or language
        write(f"You> {transcript.text}")

        reply = agent.reply(transcript.text, language=language)
        # Saved after every turn, so the log survives if the chat is interrupted.
        log.record_turn(transcript.text, reply)
        log.save(agent.messages)

        if show_trace:
            for line in format_trace(reply):
                write(line)
        write(f"Agent> {reply.text}")
        write("")

        try:
            speaker.play(tts.synthesize(reply.text, language))
        except SpeechError as error:
            # The reply is already on screen, so the conversation can go on.
            write(f"  [text-to-speech error] {error}")
        except KeyboardInterrupt:
            write("")
            break

    _announce_log(log, write)


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

    try:
        key = PushToTalkKey()
        speaker = Speaker()
        microphone = Microphone()
    except (OSError, sounddevice.PortAudioError) as error:
        print(f"Audio problem: {error}", file=sys.stderr)
        return 1

    stt = GroqSpeechToText(settings.groq_api_key, settings.stt_model)
    tts = EdgeTextToSpeech({**DEFAULT_VOICES, **settings.tts_voices})
    print(f"Speech-to-text: {settings.stt_model} (Groq). Text-to-speech: edge-tts.")
    with microphone:
        run_voice_chat(new_conversation, key, microphone, stt, tts, speaker, show_trace=show_trace)
    return 0
