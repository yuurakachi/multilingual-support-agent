"""Tests for one spoken conversation: listening, asking to repeat and answering."""

import json

import pytest
from fakes import FakeClient, response, text, thinking
from voice_fakes import FakeSpeechToText, FakeTextToSpeech, ManualClock, quiet, speech

from support_agent.agent import SupportAgent
from support_agent.channels import VOICE_CHANNEL, VOICE_FALLBACK_MESSAGES
from support_agent.config import Settings
from support_agent.conversation_log import ConversationLog
from support_agent.voice.pipeline import (
    LOW_CONFIDENCE,
    NO_WORDS,
    REPEAT_MESSAGES,
    SILENCE,
    STT_ERROR,
    UNSUPPORTED_LANGUAGE,
    VoiceConversation,
)
from support_agent.voice.speech import SpeechError, Transcript

SETTINGS = Settings(model="test-model")


class Conversation:
    """A VoiceConversation wired to fakes, with everything a test wants to inspect."""

    def __init__(self, conn, tmp_path, transcripts=(), responses=(), **kwargs):
        self.clock = ManualClock()
        self.client = FakeClient(*responses)
        self.stt = FakeSpeechToText(transcripts, clock=self.clock, seconds=1.5)
        self.tts = FakeTextToSpeech(clock=self.clock, seconds=2.25)
        self.agent = SupportAgent(self.client, conn, SETTINGS, channel=VOICE_CHANNEL)
        self.log = ConversationLog(SETTINGS, log_dir=tmp_path, channel=VOICE_CHANNEL)
        self.voice = VoiceConversation(
            self.agent, self.log, self.stt, self.tts, clock=self.clock, **kwargs
        )

    def saved(self):
        return json.loads(self.log.path.read_text(encoding="utf-8"))


# --- listen ---


def test_clear_speech_is_understood(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("¿Dónde está mi pedido?", "es", -0.2)])

    heard = c.voice.listen(speech(seconds=5))

    assert heard.understood
    assert heard.transcript.text == "¿Dónde está mi pedido?"
    assert heard.stt_ms == 1500
    assert heard.level > 1000
    assert c.voice.language == "es"
    # Nothing is known about the customer yet, so the recogniser had to guess.
    assert c.stt.hints == [None]


def test_a_quiet_recording_is_silence_and_is_never_transcribed(conn, tmp_path):
    c = Conversation(conn, tmp_path)

    heard = c.voice.listen(quiet())

    assert heard.problem == SILENCE
    assert heard.level < 200
    assert heard.transcript is None
    assert c.stt.hints == []


def test_the_silence_level_can_be_changed(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hello", "en")], silence_level=10)

    assert c.voice.listen(quiet()).understood


@pytest.mark.parametrize("written", ["", "   ", " .", "…", "?!"])
def test_a_transcript_without_words_is_not_understood(conn, tmp_path, written):
    c = Conversation(conn, tmp_path, [Transcript(written, "en", -0.5)])

    assert c.voice.listen(speech()).problem == NO_WORDS


def test_a_guessed_transcript_is_not_understood(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("¿Qué es el servicio?", "es", -1.09)])

    heard = c.voice.listen(speech())

    assert heard.problem == LOW_CONFIDENCE
    # A turn that was not understood does not set the conversation's language.
    assert c.voice.language is None


def test_a_provider_without_confidence_is_trusted(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hello", "en", None)])

    assert c.voice.listen(speech()).understood


def test_another_language_at_the_start_is_not_understood(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("Onde está o meu pedido?", None, -0.2)])

    heard = c.voice.listen(speech())

    assert heard.problem == UNSUPPORTED_LANGUAGE
    assert c.stt.hints == [None]


def test_another_language_mid_conversation_is_heard_again_as_the_known_one(conn, tmp_path):
    c = Conversation(
        conn,
        tmp_path,
        [
            Transcript("Hola, quiero saber de mi pedido.", "es", -0.2),
            Transcript("Sim, o número é mil e dois.", None, -0.4),
            Transcript("Sí, el número es mil dos.", "es", -0.3),
        ],
    )
    c.voice.listen(speech(seconds=5))

    heard = c.voice.listen(speech(seconds=5))

    assert heard.understood
    assert heard.transcript.text == "Sí, el número es mil dos."
    assert heard.language_hint == "es"
    assert c.stt.hints == [None, None, "es"]
    # Both attempts count as time spent on speech-to-text.
    assert heard.stt_ms == 3000


def test_short_answers_are_transcribed_in_the_language_of_the_conversation(conn, tmp_path):
    c = Conversation(
        conn,
        tmp_path,
        [Transcript("Hola, mi pedido es el mil dos.", "es"), Transcript("Sí.", "es")],
    )
    c.voice.listen(speech(seconds=6))

    heard = c.voice.listen(speech(seconds=1.5))

    assert heard.language_hint == "es"
    assert c.stt.hints == [None, "es"]


def test_a_short_first_message_has_no_language_to_expect(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("Hello.", "en")])

    assert c.voice.listen(speech(seconds=1.5)).language_hint is None


def test_a_long_message_in_another_supported_language_switches(conn, tmp_path):
    c = Conversation(
        conn,
        tmp_path,
        [Transcript("Hola, buenas tardes.", "es"), Transcript("Could we continue in English?", "en")],
    )
    c.voice.listen(speech(seconds=5))

    c.voice.listen(speech(seconds=5))

    assert c.voice.language == "en"
    assert c.stt.hints == [None, None]


def test_a_speech_to_text_failure_is_reported_not_raised(conn, tmp_path):
    c = Conversation(conn, tmp_path, [SpeechError("Groq answered 429: slow down")])

    heard = c.voice.listen(speech())

    assert heard.problem == STT_ERROR
    assert heard.error == "Groq answered 429: slow down"
    assert heard.stt_ms == 1500


# --- ask_to_repeat ---


def test_asking_to_repeat_uses_no_model_call_and_is_logged(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript(".", "en", -0.83)])
    heard = c.voice.listen(speech(seconds=3.5))

    c.voice.ask_to_repeat(heard)

    assert c.client.requests == []
    assert c.agent.messages == []
    saved = c.saved()
    assert saved["turns"] == []
    (event,) = saved["events"]
    assert event["type"] == "unheard_audio"
    assert event["reason"] == NO_WORDS
    assert event["after_turn"] == 0
    assert event["recording_seconds"] == 3.5
    assert event["level"] == heard.level
    assert event["stt_ms"] == 1500
    assert event["transcript"] == "."
    assert event["detected_language"] == "en"
    assert event["avg_logprob"] == -0.83
    assert event["error"] is None


def test_before_the_language_is_known_the_request_is_in_all_three(conn, tmp_path):
    c = Conversation(conn, tmp_path)

    request = c.voice.ask_to_repeat(c.voice.listen(quiet()))

    for message in REPEAT_MESSAGES.values():
        assert message in request.text
    # Each language is read by its own voice.
    assert sorted(language for _, language in c.tts.spoken) == ["en", "es", "ja"]
    assert len(request.clips) == 3


def test_once_the_language_is_known_the_request_uses_it(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("注文について聞きたいです。", "ja")])
    c.voice.listen(speech(seconds=5))

    request = c.voice.ask_to_repeat(c.voice.listen(quiet()))

    assert request.text == REPEAT_MESSAGES["ja"]
    assert c.tts.spoken == [(REPEAT_MESSAGES["ja"], "ja")]
    assert c.saved()["events"][0]["reason"] == SILENCE


def test_the_request_is_synthesized_once_and_can_be_shared(conn, tmp_path):
    shared = {}
    first = Conversation(conn, tmp_path, repeat_clips=shared)
    second = Conversation(conn, tmp_path, repeat_clips=shared)

    first.voice.ask_to_repeat(first.voice.listen(quiet()))
    first.voice.ask_to_repeat(first.voice.listen(quiet()))
    request = second.voice.ask_to_repeat(second.voice.listen(quiet()))

    assert len(first.tts.spoken) == 3
    assert second.tts.spoken == []
    assert len(request.clips) == 3


def test_the_request_is_still_shown_when_it_cannot_be_voiced(conn, tmp_path):
    c = Conversation(conn, tmp_path)
    c.voice.tts = FakeTextToSpeech(error=SpeechError("edge-tts failed"))

    request = c.voice.ask_to_repeat(c.voice.listen(quiet()))

    assert REPEAT_MESSAGES["en"] in request.text
    assert request.clips == []


def test_a_failed_recogniser_is_logged_with_its_error(conn, tmp_path):
    c = Conversation(conn, tmp_path, [SpeechError("Could not reach Groq: ConnectError")])

    c.voice.ask_to_repeat(c.voice.listen(speech()))

    (event,) = c.saved()["events"]
    assert event["reason"] == STT_ERROR
    assert event["error"] == "Could not reach Groq: ConnectError"
    assert event["transcript"] is None


# --- answer ---


def test_the_answer_is_logged_timed_and_voiced(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hola", "es", -0.2)], [response(text("Hola."))])
    audio = speech(seconds=5)
    heard = c.voice.listen(audio)

    answer = c.voice.answer(heard)

    assert answer.reply.text == "Hola."
    assert c.client.requests[0]["system"] == VOICE_CHANNEL.system_prompt
    assert c.tts.spoken == [("Hola.", "es")]
    assert answer.speech is not None

    saved = c.saved()
    assert saved["channel"] == "voice"
    turn = saved["turns"][0]
    assert turn["user_message"] == "hola"
    assert turn["assistant_message"] == "Hola."
    assert turn["voice"] == {
        "language": "es",
        "detected_language": "es",
        "language_hint": None,
        "recording_seconds": 5.0,
        "level": round(audio.loudness),
        "avg_logprob": -0.2,
        # Length of the reply the fake text-to-speech returned.
        "speech_seconds": 0.1,
        "timings_ms": {
            "stt": 1500,
            # The agent times itself: the same number as the turn's latency.
            "llm": turn["latency_ms"],
            "tts": 2250,
            # The agent ran on the real clock, so only the two fakes add up here.
            "total": 3750,
        },
    }
    assert answer.timings_ms == turn["voice"]["timings_ms"]


def test_the_wait_is_counted_from_the_end_of_the_recording(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hi", "en")], [response(text("Hi."))])

    # The recording ended half a second before listening started.
    heard = c.voice.listen(speech(), recorded_at=c.clock.now - 0.5)
    answer = c.voice.answer(heard)

    assert answer.timings_ms["total"] == 500 + 1500 + 2250


def test_a_reply_that_cannot_be_voiced_is_still_logged(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hi", "en")], [response(text("Hi."))])
    c.voice.tts = FakeTextToSpeech(error=SpeechError("edge-tts failed: TimeoutError"))

    answer = c.voice.answer(c.voice.listen(speech()))

    assert answer.speech is None
    assert answer.tts_error == "edge-tts failed: TimeoutError"
    assert answer.reply.text == "Hi."
    voice = c.saved()["turns"][0]["voice"]
    assert voice["timings_ms"]["stt"] == 1500
    assert voice["timings_ms"]["tts"] is None
    assert voice["timings_ms"]["total"] is None
    assert voice["speech_seconds"] is None


def test_the_fallback_is_spoken_in_the_language_of_the_customer(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hola", "es")], [response(thinking())])

    answer = c.voice.answer(c.voice.listen(speech()))

    assert answer.reply.text == VOICE_FALLBACK_MESSAGES["es"]
    assert c.tts.spoken == [(VOICE_FALLBACK_MESSAGES["es"], "es")]


def test_events_remember_how_far_the_conversation_was(conn, tmp_path):
    c = Conversation(conn, tmp_path, [Transcript("hello there", "en")], [response(text("Hi."))])
    c.voice.answer(c.voice.listen(speech()))

    c.voice.ask_to_repeat(c.voice.listen(quiet()))

    saved = c.saved()
    assert len(saved["turns"]) == 1
    assert saved["events"][0]["after_turn"] == 1
