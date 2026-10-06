"""Tests for the voice loop, with a scripted key, microphone and speech providers."""

import json

import numpy as np
from conftest import ALICE
from fakes import FakeClient, response, text, thinking, tool_use

from support_agent.agent import SupportAgent
from support_agent.channels import VOICE_CHANNEL, VOICE_FALLBACK_MESSAGES
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog
from support_agent.voice.audio import Audio
from support_agent.voice.chat import run_voice_chat
from support_agent.voice.keys import NEW, QUIT, TALK
from support_agent.voice.speech import SpeechError, Transcript

SETTINGS = Settings(model="test-model")


def recording(seconds=2.0):
    return Audio(np.zeros(int(seconds * 16_000), dtype=np.int16), 16_000)


class FakeKey:
    def __init__(self, actions):
        self.actions = list(actions)
        self.drained = 0

    def wait(self):
        if not self.actions:
            raise EOFError
        return self.actions.pop(0)

    def is_held(self):
        return False

    def drain(self):
        self.drained += 1


class FakeMicrophone:
    def __init__(self, recordings):
        self.recordings = list(recordings)

    def record_while(self, is_held):
        return self.recordings.pop(0)


class FakeSpeechToText:
    """Returns its scripted results in order; an exception in the list is raised."""

    def __init__(self, results):
        self.results = list(results)
        self.heard = []

    def transcribe(self, audio):
        self.heard.append(audio)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTextToSpeech:
    def __init__(self, error=None):
        self.error = error
        self.spoken = []

    def synthesize(self, text, language):
        self.spoken.append((text, language))
        if self.error:
            raise self.error
        return Audio(np.zeros(100, dtype=np.int16), 24_000)


class FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, audio):
        self.played.append(audio)


class VoiceChat:
    """Runs run_voice_chat with scripted parts and collects everything it prints."""

    def __init__(self, conn, tmp_path, *responses):
        self.conn = conn
        self.tmp_path = tmp_path
        self.client = FakeClient(*responses)
        self.output: list[str] = []
        self.tts = FakeTextToSpeech()
        self.speaker = FakeSpeaker()

    def new_conversation(self):
        return (
            SupportAgent(self.client, self.conn, SETTINGS, channel=VOICE_CHANNEL),
            ConversationLog(SETTINGS, log_dir=self.tmp_path, channel=VOICE_CHANNEL),
        )

    def run(self, actions, transcripts=(), recordings=None, **kwargs):
        talks = actions.count(TALK)
        self.key = FakeKey(actions)
        self.stt = FakeSpeechToText(transcripts)
        microphone = FakeMicrophone(recordings or [recording()] * talks)
        run_voice_chat(
            self.new_conversation,
            self.key,
            microphone,
            self.stt,
            self.tts,
            self.speaker,
            write=self.output.append,
            **kwargs,
        )
        return self

    def saved_logs(self):
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.tmp_path.glob("*.json"))
        ]


def test_a_spoken_turn_is_transcribed_answered_and_spoken(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("Hola, ¿en qué puedo ayudarte?"))).run(
        [TALK, QUIT], [Transcript("hola", "es")]
    )

    assert chat.client.requests[0]["system"] == VOICE_CHANNEL.system_prompt
    assert chat.client.requests[0]["messages"] == [{"role": "user", "content": "hola"}]
    assert "You> hola" in chat.output
    assert "Agent> Hola, ¿en qué puedo ayudarte?" in chat.output
    # The reply is read with the voice of the language the customer spoke.
    assert chat.tts.spoken == [("Hola, ¿en qué puedo ayudarte?", "es")]
    assert len(chat.speaker.played) == 1

    (saved,) = chat.saved_logs()
    assert saved["channel"] == "voice"
    assert saved["turns"][0]["user_message"] == "hola"
    assert saved["turns"][0]["assistant_message"] == "Hola, ¿en qué puedo ayudarte?"


def test_tools_work_the_same_as_in_the_text_chat(conn, tmp_path):
    chat = VoiceChat(
        conn,
        tmp_path,
        response(
            tool_use("toolu_1", "get_order_status", order_id="ORD-2", email=ALICE),
            stop_reason="tool_use",
        ),
        response(text("It has shipped.")),
    ).run([TALK], [Transcript(f"where is ord 2, {ALICE}", "en")])

    (saved,) = chat.saved_logs()
    (tool_call,) = saved["turns"][0]["tool_calls"]
    assert tool_call["name"] == "get_order_status"
    assert tool_call["result"]["status"] == "shipped"
    assert chat.tts.spoken == [("It has shipped.", "en")]
    assert any("[tool] get_order_status" in line for line in chat.output)


def test_a_tap_on_the_key_is_not_transcribed(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([TALK, QUIT], recordings=[recording(seconds=0.1)])

    assert chat.stt.heard == []
    assert chat.client.requests == []
    assert any("too short" in line for line in chat.output)


def test_silence_is_not_sent_to_the_model(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([TALK, QUIT], [Transcript("", None)])

    assert chat.client.requests == []
    assert chat.tts.spoken == []
    assert chat.saved_logs() == []
    assert any("could not hear" in line for line in chat.output)


def test_speech_to_text_failure_does_not_end_the_chat(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("Hi!"))).run(
        [TALK, TALK, QUIT], [SpeechError("Groq answered 429: slow down"), Transcript("hi", "en")]
    )

    assert any("[speech-to-text error] Groq answered 429" in line for line in chat.output)
    assert len(chat.client.requests) == 1
    assert chat.tts.spoken == [("Hi!", "en")]


def test_text_to_speech_failure_still_shows_and_logs_the_reply(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("one")), response(text("two")))
    chat.tts = FakeTextToSpeech(error=SpeechError("edge-tts failed: TimeoutError"))

    chat.run([TALK, TALK, QUIT], [Transcript("first", "en"), Transcript("second", "en")])

    assert "Agent> one" in chat.output and "Agent> two" in chat.output
    assert any("[text-to-speech error]" in line for line in chat.output)
    assert chat.speaker.played == []
    (saved,) = chat.saved_logs()
    assert saved["totals"]["turns"] == 2


def test_the_voice_follows_the_customer_and_survives_an_unknown_language(conn, tmp_path):
    chat = VoiceChat(
        conn, tmp_path, response(text("はい。")), response(text("かしこまりました。")), response(text("Sure."))
    ).run(
        [TALK, TALK, TALK],
        [Transcript("こんにちは", "ja"), Transcript("ええと", None), Transcript("hello", "en")],
    )

    assert [language for _, language in chat.tts.spoken] == ["ja", "ja", "en"]


def test_fallback_is_spoken_in_the_language_of_the_customer(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(thinking())).run([TALK], [Transcript("hola", "es")])

    assert chat.tts.spoken == [(VOICE_FALLBACK_MESSAGES["es"], "es")]


def test_new_starts_a_separate_conversation_and_log(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("one")), response(text("two"))).run(
        [TALK, NEW, TALK, QUIT], [Transcript("first", "ja"), Transcript("second", None)]
    )

    assert len(chat.saved_logs()) == 2
    assert chat.client.requests[1]["messages"] == [{"role": "user", "content": "second"}]
    # The new conversation does not inherit the previous customer's language.
    assert [language for _, language in chat.tts.spoken] == ["ja", "en"]


def test_held_key_repeats_are_discarded_after_each_recording(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("hi"))).run([TALK], [Transcript("hi", "en")])

    assert chat.key.drained == 1


def test_quiet_mode_hides_the_trace(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("hi"))).run(
        [TALK], [Transcript("hello", "en")], show_trace=False
    )

    assert not any(line.startswith("  [") for line in chat.output)
    assert "Agent> hi" in chat.output


def test_quitting_without_speaking_saves_nothing(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([QUIT])

    assert chat.saved_logs() == []
    assert not any("saved" in line for line in chat.output)
