"""Tests for ElevenLabs speech, the fallback voice and choosing providers. No network is used."""

import httpx2
import numpy as np
import pytest

from support_agent.config import ConfigError, VoiceSettings
from support_agent.voice.audio import Audio
from support_agent.voice.elevenlabs import (
    API_URL,
    OUTPUT_FORMAT,
    ElevenLabsSpeechToText,
    ElevenLabsTextToSpeech,
)
from support_agent.voice.providers import (
    build_speech_to_text,
    build_text_to_speech,
    speech_metadata,
)
from support_agent.voice.speech import FallbackTextToSpeech, SpeechError, Transcript, provider_name
from support_agent.voice.stt_groq import GroqSpeechToText
from support_agent.voice.tts_edge import EdgeTextToSpeech


def tone(seconds=1.0, sample_rate=16_000):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return Audio((np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16), sample_rate)


class FakePost:
    """Stands in for httpx2.post: records the request and returns a canned response."""

    def __init__(self, status_code=200, json=None, content=None, error=None):
        if json is not None:
            self.response = httpx2.Response(status_code, json=json)
        else:
            self.response = httpx2.Response(status_code, content=content or b"")
        self.error = error
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        return self.response


# --- text-to-speech ---


def test_elevenlabs_voice_turns_raw_samples_into_audio():
    samples = np.array([0, 1000, -1000, 32767], dtype="<i2")
    post = FakePost(content=samples.tobytes())
    tts = ElevenLabsTextToSpeech("secret-key", "some-tts-model", "voice-123", post=post)

    audio = tts.synthesize("ご注文は発送済みです。", "ja")

    assert audio.sample_rate == 24_000
    assert np.array_equal(audio.samples, samples)
    (call,) = post.calls
    assert call["url"] == f"{API_URL}/text-to-speech/voice-123"
    assert call["headers"] == {"xi-api-key": "secret-key"}
    assert call["params"] == {"output_format": OUTPUT_FORMAT}
    assert call["json"] == {
        "text": "ご注文は発送済みです。",
        "model_id": "some-tts-model",
        "language_code": "ja",
    }


def test_elevenlabs_voice_errors_become_speech_errors_without_the_key():
    detail = {"detail": {"status": "quota_exceeded", "message": "You have 12 credits left."}}
    tts = ElevenLabsTextToSpeech("secret-key", "m", "v", post=FakePost(401, json=detail))

    with pytest.raises(SpeechError) as raised:
        tts.synthesize("hello", "en")

    assert "401" in str(raised.value)
    assert "You have 12 credits left." in str(raised.value)
    assert "secret-key" not in str(raised.value)


@pytest.mark.parametrize(
    "post",
    [
        FakePost(error=httpx2.ConnectError("no route")),
        FakePost(content=b""),
        FakePost(422, json={"detail": [{"msg": "field required"}]}),
    ],
)
def test_elevenlabs_voice_failures_are_speech_errors(post):
    with pytest.raises(SpeechError):
        ElevenLabsTextToSpeech("k", "m", "v", post=post).synthesize("hello", "en")


# --- speech-to-text (Scribe) ---


def test_scribe_sends_the_recording_and_reads_text_language_and_confidence():
    body = {
        "text": " ¿Dónde está mi pedido? ",
        "language_code": "spa",
        "language_probability": 0.98,
        "words": [
            {"text": "¿Dónde", "type": "word", "logprob": -0.2},
            {"text": " ", "type": "spacing", "logprob": 0.0},
            {"text": "pedido?", "type": "word", "logprob": -0.4},
        ],
    }
    post = FakePost(json=body)
    stt = ElevenLabsSpeechToText("secret-key", "some-stt-model", post=post)
    audio = tone()

    transcript = stt.transcribe(audio)

    assert transcript == Transcript("¿Dónde está mi pedido?", "es", pytest.approx(-0.3))
    (call,) = post.calls
    assert call["url"] == f"{API_URL}/speech-to-text"
    assert call["headers"] == {"xi-api-key": "secret-key"}
    assert call["data"] == {"model_id": "some-stt-model", "tag_audio_events": "false"}
    assert call["files"]["file"][1] == audio.to_wav()


def test_scribe_is_told_the_language_only_when_one_is_given():
    post = FakePost(json={"text": "Sí.", "language_code": "es", "words": []})
    stt = ElevenLabsSpeechToText("k", "m", post=post)

    stt.transcribe(tone())
    transcript = stt.transcribe(tone(), language="es")

    assert "language_code" not in post.calls[0]["data"]
    assert post.calls[1]["data"]["language_code"] == "es"
    # No words, so no confidence to report.
    assert transcript.avg_logprob is None


@pytest.mark.parametrize(("reported", "expected"), [("jpn", "ja"), ("eng", "en"), ("por", None)])
def test_scribe_language_is_one_of_ours_or_none(reported, expected):
    post = FakePost(json={"text": "x", "language_code": reported, "words": []})

    assert ElevenLabsSpeechToText("k", "m", post=post).transcribe(tone()).language == expected


def test_scribe_failure_is_a_speech_error():
    stt = ElevenLabsSpeechToText("k", "m", post=FakePost(429, json={"detail": "Too many requests"}))

    with pytest.raises(SpeechError, match="429: Too many requests"):
        stt.transcribe(tone())


# --- the fallback voice ---


class Voice:
    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.spoken = []

    def synthesize(self, text, language):
        self.spoken.append((text, language))
        if self.error:
            raise self.error
        return tone(seconds=0.1, sample_rate=24_000)


def test_the_primary_voice_speaks_while_it_works():
    primary, backup = Voice("elevenlabs"), Voice("edge-tts")
    tts = FallbackTextToSpeech(primary, backup)

    tts.synthesize("hola", "es")

    assert primary.spoken == [("hola", "es")]
    assert backup.spoken == []
    assert provider_name(tts) == "elevenlabs"


def test_the_backup_speaks_when_the_primary_fails_and_says_why():
    primary = Voice("elevenlabs", SpeechError("ElevenLabs answered 401: quota exceeded"))
    backup = Voice("edge-tts")
    warnings = []
    tts = FallbackTextToSpeech(primary, backup, warn=warnings.append)

    audio = tts.synthesize("hola", "es")

    assert audio.seconds == 0.1
    assert backup.spoken == [("hola", "es")]
    assert provider_name(tts) == "edge-tts"
    assert warnings == [
        "elevenlabs failed (ElevenLabs answered 401: quota exceeded), using edge-tts"
    ]


def test_when_both_voices_fail_the_error_reaches_the_caller():
    tts = FallbackTextToSpeech(
        Voice("elevenlabs", SpeechError("down")), Voice("edge-tts", SpeechError("also down"))
    )

    with pytest.raises(SpeechError, match="also down"):
        tts.synthesize("hola", "es")


def test_a_single_provider_reports_its_own_name():
    assert provider_name(Voice("edge-tts")) == "edge-tts"


# --- settings and choosing providers ---

GROQ_ENV = {"GROQ_API_KEY": "gsk_test", "STT_MODEL": "some-stt-model"}
ELEVENLABS_ENV = {
    "ELEVENLABS_API_KEY": "el_test",
    "ELEVENLABS_TTS_MODEL": "some-tts-model",
    "ELEVENLABS_VOICE_ID": "voice-123",
    "ELEVENLABS_STT_MODEL": "some-scribe-model",
}


def test_groq_and_edge_tts_are_the_default():
    settings = VoiceSettings.from_env(GROQ_ENV)

    assert (settings.stt_provider, settings.tts_provider) == ("groq", "edge")
    assert isinstance(build_speech_to_text(settings), GroqSpeechToText)
    assert isinstance(build_text_to_speech(settings), EdgeTextToSpeech)
    assert speech_metadata(settings)["tts_provider"] == "edge"


def test_elevenlabs_voice_comes_with_edge_tts_as_its_backup():
    settings = VoiceSettings.from_env({**GROQ_ENV, **ELEVENLABS_ENV, "TTS_PROVIDER": "elevenlabs"})

    tts = build_text_to_speech(settings)

    assert isinstance(tts, FallbackTextToSpeech)
    assert isinstance(tts.primary, ElevenLabsTextToSpeech)
    assert (tts.primary.model, tts.primary.voice_id) == ("some-tts-model", "voice-123")
    assert isinstance(tts.backup, EdgeTextToSpeech)
    metadata = speech_metadata(settings)
    assert metadata["tts_provider"] == "elevenlabs"
    assert metadata["tts_model"] == "some-tts-model"


def test_scribe_does_not_need_groq():
    settings = VoiceSettings.from_env({**ELEVENLABS_ENV, "STT_PROVIDER": "ElevenLabs"})

    stt = build_speech_to_text(settings)

    assert isinstance(stt, ElevenLabsSpeechToText)
    assert stt.model == "some-scribe-model"
    assert speech_metadata(settings)["stt_model"] == "some-scribe-model"


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({**GROQ_ENV, "TTS_PROVIDER": "elevenlabs"}, "ELEVENLABS_API_KEY"),
        (
            {**GROQ_ENV, "TTS_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k", "ELEVENLABS_TTS_MODEL": "m"},
            "ELEVENLABS_VOICE_ID",
        ),
        ({"STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k"}, "ELEVENLABS_STT_MODEL"),
        ({**GROQ_ENV, "TTS_PROVIDER": "openai"}, "TTS_PROVIDER"),
        ({**GROQ_ENV, "STT_PROVIDER": "local"}, "STT_PROVIDER"),
    ],
)
def test_elevenlabs_settings_are_checked_only_when_it_is_chosen(env, missing):
    with pytest.raises(ConfigError, match=missing):
        VoiceSettings.from_env(env)
