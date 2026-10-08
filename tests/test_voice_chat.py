"""Tests for the push-to-talk loop, with a scripted key, microphone and speech providers."""

import json

from conftest import ALICE
from fakes import FakeClient, response, text, tool_use
from voice_fakes import (
    FakeKey,
    FakeMicrophone,
    FakeSpeaker,
    FakeSpeechToText,
    FakeTextToSpeech,
    ManualClock,
    quiet,
    speech,
)

from support_agent.agent import SupportAgent
from support_agent.channels import VOICE_CHANNEL
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog
from support_agent.voice.chat import run_voice_chat
from support_agent.voice.keys import NEW, QUIT, TALK
from support_agent.voice.pipeline import REPEAT_MESSAGES
from support_agent.voice.speech import LANGUAGES, SpeechError, Transcript

SETTINGS = Settings(model="test-model")


class VoiceChat:
    """Runs run_voice_chat with scripted parts and collects everything it prints."""

    def __init__(self, conn, tmp_path, *responses):
        self.conn = conn
        self.tmp_path = tmp_path
        self.client = FakeClient(*responses)
        self.output: list[str] = []
        # Speech-to-text takes 1.5 s and text-to-speech 2.25 s on this clock.
        self.clock = ManualClock()
        self.tts = FakeTextToSpeech(clock=self.clock, seconds=2.25)
        self.speaker = FakeSpeaker()

    def new_conversation(self):
        return (
            SupportAgent(self.client, self.conn, SETTINGS, channel=VOICE_CHANNEL),
            ConversationLog(SETTINGS, log_dir=self.tmp_path, channel=VOICE_CHANNEL),
        )

    def run(self, actions, transcripts=(), recordings=None, **kwargs):
        talks = actions.count(TALK)
        self.key = FakeKey(actions)
        self.stt = FakeSpeechToText(transcripts, clock=self.clock, seconds=1.5)
        microphone = FakeMicrophone(recordings or [speech()] * talks)
        run_voice_chat(
            self.new_conversation,
            self.key,
            microphone,
            self.stt,
            self.tts,
            self.speaker,
            write=self.output.append,
            clock=self.clock,
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
    assert saved["turns"][0]["voice"]["timings_ms"]["total"] == 3750
    assert any(str(tmp_path) in line and "saved" in line for line in chat.output)


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


def test_the_stages_of_a_turn_are_shown(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("Hola."))).run([TALK], [Transcript("hola", "es")])

    (timing_line,) = [line for line in chat.output if line.startswith("  [waited")]
    assert timing_line.startswith("  [waited 3.8 s: speech-to-text 1.5 s, agent ")
    assert timing_line.endswith("text-to-speech 2.2 s]")


def test_a_tap_on_the_key_is_not_transcribed(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([TALK, QUIT], recordings=[speech(seconds=0.1)])

    assert chat.stt.hints == []
    assert chat.client.requests == []
    assert chat.saved_logs() == []
    assert any("too short" in line for line in chat.output)


def test_silence_makes_the_agent_ask_to_repeat_without_a_model_call(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([TALK, QUIT], recordings=[quiet()])

    assert chat.stt.hints == []
    assert chat.client.requests == []
    # Nobody knows the customer's language yet: the request comes in all three.
    in_all_three = " ".join(REPEAT_MESSAGES[language] for language in LANGUAGES)
    assert f"Agent> {in_all_three}" in chat.output
    assert len(chat.speaker.played) == 3
    assert any("too quiet" in line and "VOICE_SILENCE_LEVEL" in line for line in chat.output)

    (saved,) = chat.saved_logs()
    assert saved["turns"] == []
    assert saved["events"][0]["reason"] == "silence"
    assert any("saved" in line for line in chat.output)


def test_an_unintelligible_turn_is_followed_by_a_normal_one(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("Claro.")), response(text("Dime."))).run(
        [TALK, TALK, TALK, QUIT],
        [
            Transcript("Hola, tengo una pregunta.", "es"),
            Transcript(" .", "en", -0.8),
            Transcript("Sobre mi pedido.", "es"),
        ],
    )

    assert any("not understood: no_words" in line for line in chat.output)
    assert f"Agent> {REPEAT_MESSAGES['es']}" in chat.output
    # Only the two understood turns reached the model.
    assert [request["messages"][-1]["content"] for request in chat.client.requests] == [
        "Hola, tengo una pregunta.",
        "Sobre mi pedido.",
    ]
    (saved,) = chat.saved_logs()
    assert saved["totals"]["turns"] == 2
    assert [event["after_turn"] for event in saved["events"]] == [1]


def test_speech_to_text_failure_does_not_end_the_chat(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("Hi!"))).run(
        [TALK, TALK, QUIT], [SpeechError("Groq answered 429: slow down"), Transcript("hi", "en")]
    )

    assert "  [speech-to-text error] Groq answered 429: slow down" in chat.output
    assert len(chat.client.requests) == 1
    assert ("Hi!", "en") in chat.tts.spoken


def test_text_to_speech_failure_still_shows_and_logs_the_reply(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("one")), response(text("two")))
    chat.tts = FakeTextToSpeech(error=SpeechError("edge-tts failed: TimeoutError"))

    chat.run([TALK, TALK, QUIT], [Transcript("first", "en"), Transcript("second", "en")])

    assert "Agent> one" in chat.output and "Agent> two" in chat.output
    assert "  [text-to-speech error] edge-tts failed: TimeoutError" in chat.output
    assert chat.speaker.played == []
    (saved,) = chat.saved_logs()
    assert saved["totals"]["turns"] == 2


def test_new_starts_a_separate_conversation_and_log(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("one")), response(text("two"))).run(
        [TALK, NEW, TALK, QUIT],
        [Transcript("最初の質問です。", "ja"), Transcript("Yes.", "en")],
        recordings=[speech(seconds=5), speech(seconds=1.5)],
    )

    assert len(chat.saved_logs()) == 2
    assert chat.client.requests[1]["messages"] == [{"role": "user", "content": "Yes."}]
    # The new conversation does not inherit the previous customer's language:
    # its short first clip is transcribed without expecting Japanese.
    assert chat.stt.hints == [None, None]
    assert [language for _, language in chat.tts.spoken] == ["ja", "en"]


def test_speech_providers_are_recorded_in_every_log(conn, tmp_path):
    providers = {"stt_provider": "groq", "stt_model": "some-stt-model"}
    chat = VoiceChat(conn, tmp_path, response(text("one")), response(text("two"))).run(
        [TALK, NEW, TALK],
        [Transcript("first", "en"), Transcript("second", "en")],
        metadata=providers,
    )

    assert [saved["metadata"] for saved in chat.saved_logs()] == [providers, providers]


def test_held_key_repeats_are_discarded_after_each_recording(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("hi"))).run([TALK], [Transcript("hi", "en")])

    assert chat.key.drained == 1


def test_quiet_mode_hides_the_trace(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("hi"))).run(
        [TALK, TALK], [Transcript("hello", "en")], recordings=[speech(), quiet()], show_trace=False
    )

    assert not any(line.startswith("  [") for line in chat.output)
    assert "Agent> hi" in chat.output
    assert f"Agent> {REPEAT_MESSAGES['en']}" in chat.output


def test_ctrl_c_while_the_reply_is_spoken_leaves_and_keeps_the_log(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path, response(text("A long answer.")))
    chat.speaker = FakeSpeaker(error=KeyboardInterrupt())

    chat.run([TALK, TALK], [Transcript("hello", "en")])

    (saved,) = chat.saved_logs()
    assert saved["turns"][0]["assistant_message"] == "A long answer."
    assert len(chat.client.requests) == 1


def test_quitting_without_speaking_saves_nothing(conn, tmp_path):
    chat = VoiceChat(conn, tmp_path).run([QUIT])

    assert chat.saved_logs() == []
    assert not any("saved" in line for line in chat.output)
