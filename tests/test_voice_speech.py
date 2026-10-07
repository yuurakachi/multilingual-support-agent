"""Tests for the voice building blocks. No microphone, speaker or network is used."""

import io
import wave

import httpx2
import numpy as np
import pytest

from support_agent.config import ConfigError, VoiceSettings
from support_agent.voice.audio import Audio
from support_agent.voice.keys import NEW, QUIT, TALK, PushToTalkKey
from support_agent.voice.speech import SpeechError, Transcript
from support_agent.voice.stt_groq import TRANSCRIPTIONS_URL, GroqSpeechToText
from support_agent.voice.tts_edge import DEFAULT_VOICES, EdgeTextToSpeech


def tone(seconds=1.0, sample_rate=16_000):
    """A quiet beep, standing in for recorded speech."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return Audio((np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16), sample_rate)


# --- Audio ---


def test_audio_knows_its_duration():
    assert tone(seconds=1.5).seconds == 1.5
    assert Audio(np.zeros(0, dtype=np.int16), 16_000).seconds == 0


def test_wav_is_mono_16_bit_at_the_recorded_rate():
    audio = tone(seconds=0.5)

    with wave.open(io.BytesIO(audio.to_wav()), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 8_000


def test_encoded_sound_is_decoded_back_to_the_same_samples():
    audio = tone(seconds=0.25, sample_rate=24_000)

    decoded = Audio.from_encoded(audio.to_wav())

    assert decoded.sample_rate == 24_000
    assert decoded.samples.dtype == np.int16
    assert np.array_equal(decoded.samples, audio.samples)


def test_loudness_tells_speech_from_an_empty_room():
    room = np.random.default_rng(0).normal(0, 40, 48_000).astype(np.int16)

    assert Audio(np.zeros(16_000, dtype=np.int16), 16_000).loudness == 0
    assert Audio(room, 16_000).loudness < 100
    assert tone().loudness > 5000
    # Too short to hold a single frame.
    assert Audio(np.zeros(10, dtype=np.int16), 16_000).loudness == 0


def test_one_click_in_a_silent_recording_is_not_speech():
    samples = np.zeros(48_000, dtype=np.int16)
    samples[20_000:20_200] = 20_000

    assert Audio(samples, 16_000).loudness == 0


def test_audio_can_be_saved_and_loaded(tmp_path):
    audio = tone(seconds=0.5, sample_rate=24_000)

    audio.save(tmp_path / "clip.wav")
    audio.save(tmp_path / "clip.mp3")

    assert np.array_equal(Audio.load(tmp_path / "clip.wav").samples, audio.samples)
    # MP3 is lossy and pads the ends: same sound, not the same samples.
    compressed = Audio.load(tmp_path / "clip.mp3")
    assert compressed.sample_rate == 24_000
    assert abs(compressed.seconds - 0.5) < 0.1
    assert compressed.loudness > 5000


# --- Speech-to-text (Groq) ---


class FakePost:
    """Stands in for httpx2.post: records the request and returns a canned response."""

    def __init__(self, status_code=200, body=None, error=None):
        self.response = httpx2.Response(status_code, json=body or {})
        self.error = error
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        return self.response


def test_groq_sends_the_recording_and_reads_the_transcript():
    post = FakePost(body={"text": " ¿Dónde está mi pedido? ", "language": "Spanish"})
    stt = GroqSpeechToText("secret-key", "some-stt-model", post=post)
    audio = tone()

    transcript = stt.transcribe(audio)

    assert transcript == Transcript("¿Dónde está mi pedido?", "es")
    (call,) = post.calls
    assert call["url"] == TRANSCRIPTIONS_URL
    assert call["headers"] == {"Authorization": "Bearer secret-key"}
    assert call["data"]["model"] == "some-stt-model"
    assert call["data"]["response_format"] == "verbose_json"
    filename, content, content_type = call["files"]["file"]
    assert content == audio.to_wav()
    assert content_type == "audio/wav"


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("Japanese", "ja"), ("english", "en"), ("Portuguese", None), ("", None)],
)
def test_groq_language_is_one_of_ours_or_none(reported, expected):
    stt = GroqSpeechToText("k", "m", post=FakePost(body={"text": "x", "language": reported}))

    assert stt.transcribe(tone()).language == expected


def test_groq_is_told_the_language_only_when_one_is_given():
    post = FakePost(body={"text": "Sí.", "language": "Spanish"})
    stt = GroqSpeechToText("k", "m", post=post)

    stt.transcribe(tone())
    stt.transcribe(tone(), language="es")

    assert "language" not in post.calls[0]["data"]
    assert post.calls[1]["data"]["language"] == "es"


def test_groq_confidence_is_the_average_over_the_segments():
    body = {
        "text": "hello there",
        "language": "English",
        "segments": [{"avg_logprob": -0.2}, {"avg_logprob": -0.4}],
    }

    assert GroqSpeechToText("k", "m", post=FakePost(body=body)).transcribe(
        tone()
    ).avg_logprob == pytest.approx(-0.3)
    without = GroqSpeechToText("k", "m", post=FakePost(body={"text": "hello"}))
    assert without.transcribe(tone()).avg_logprob is None


def test_groq_silence_gives_an_empty_transcript():
    stt = GroqSpeechToText("k", "m", post=FakePost(body={"text": "  "}))

    assert stt.transcribe(tone()) == Transcript("", None)


def test_groq_http_error_becomes_a_speech_error_without_the_key():
    post = FakePost(429, {"error": {"message": "Rate limit reached"}})
    stt = GroqSpeechToText("secret-key", "m", post=post)

    with pytest.raises(SpeechError) as raised:
        stt.transcribe(tone())

    assert "429" in str(raised.value)
    assert "Rate limit reached" in str(raised.value)
    assert "secret-key" not in str(raised.value)


def test_groq_network_failure_becomes_a_speech_error():
    stt = GroqSpeechToText("k", "m", post=FakePost(error=httpx2.ConnectError("no route")))

    with pytest.raises(SpeechError, match="Could not reach Groq"):
        stt.transcribe(tone())


# --- Text-to-speech (edge-tts) ---


class FakeFetch:
    """Stands in for the call to the Edge service: returns a WAV instead of an MP3."""

    def __init__(self, data=None, error=None):
        self.data = tone(seconds=0.5, sample_rate=24_000).to_wav() if data is None else data
        self.error = error
        self.calls = []

    def __call__(self, text, voice, timeout_seconds):
        self.calls.append((text, voice))
        if self.error:
            raise self.error
        return self.data


def test_edge_tts_picks_the_voice_of_the_language():
    fetch = FakeFetch()
    tts = EdgeTextToSpeech(fetch=fetch)

    audio = tts.synthesize("ご注文は発送済みです。", "ja")

    assert fetch.calls == [("ご注文は発送済みです。", DEFAULT_VOICES["ja"])]
    assert audio.sample_rate == 24_000
    assert audio.seconds == 0.5


def test_edge_tts_voices_can_be_replaced_and_unknown_languages_use_english():
    fetch = FakeFetch()
    tts = EdgeTextToSpeech({**DEFAULT_VOICES, "es": "es-ES-ElviraNeural"}, fetch=fetch)

    tts.synthesize("hola", "es")
    tts.synthesize("olá", "pt")

    assert [voice for _, voice in fetch.calls] == ["es-ES-ElviraNeural", DEFAULT_VOICES["en"]]


@pytest.mark.parametrize(
    "fetch",
    [
        FakeFetch(error=TimeoutError()),
        FakeFetch(error=OSError("network down")),
        FakeFetch(data=b""),
        FakeFetch(data=b"this is not a sound file"),
    ],
)
def test_edge_tts_failures_become_speech_errors(fetch):
    with pytest.raises(SpeechError):
        EdgeTextToSpeech(fetch=fetch).synthesize("hello", "en")


# --- Push-to-talk key ---


def scripted_key(*presses, pending=0, space_down=False):
    presses = list(presses)
    remaining = [pending]

    def key_pending():
        return remaining[0] > 0

    def read_key():
        if remaining[0] > 0:
            remaining[0] -= 1
        return presses.pop(0)

    key = PushToTalkKey(read_key=read_key, key_pending=key_pending, space_is_down=lambda: space_down)
    return key, presses


@pytest.mark.parametrize(
    ("pressed", "action"),
    [(" ", TALK), ("n", NEW), ("N", NEW), ("q", QUIT), ("Q", QUIT), ("\x1b", QUIT), ("\x03", QUIT)],
)
def test_key_presses_map_to_actions(pressed, action):
    key, _ = scripted_key(pressed)

    assert key.wait() == action


def test_other_keys_and_arrow_keys_are_ignored():
    # "\xe0" + "H" is the up arrow: its second code must not be read as a key.
    key, _ = scripted_key("x", "\xe0", "q", " ")

    assert key.wait() == TALK


def test_is_held_reports_the_space_bar():
    assert scripted_key(space_down=True)[0].is_held() is True
    assert scripted_key(space_down=False)[0].is_held() is False


def test_drain_discards_the_repeats_of_a_held_key():
    key, presses = scripted_key(" ", " ", " ", "n", pending=3)

    key.drain()

    assert presses == ["n"]


# --- Settings ---

VOICE_ENV = {"GROQ_API_KEY": "gsk_test", "STT_MODEL": "some-stt-model"}


def test_voice_settings_are_read_from_the_environment():
    settings = VoiceSettings.from_env({**VOICE_ENV, "TTS_VOICE_ES": " es-ES-ElviraNeural "})

    assert settings.groq_api_key == "gsk_test"
    assert settings.stt_model == "some-stt-model"
    assert settings.tts_voices == {"es": "es-ES-ElviraNeural"}
    assert settings.silence_level == 200


def test_silence_level_can_be_set_and_must_be_a_number():
    assert VoiceSettings.from_env({**VOICE_ENV, "VOICE_SILENCE_LEVEL": "80"}).silence_level == 80
    for wrong in ("loud", "-5"):
        with pytest.raises(ConfigError, match="VOICE_SILENCE_LEVEL"):
            VoiceSettings.from_env({**VOICE_ENV, "VOICE_SILENCE_LEVEL": wrong})


@pytest.mark.parametrize("missing", ["GROQ_API_KEY", "STT_MODEL"])
def test_voice_settings_require_the_key_and_the_model(missing):
    env = {name: value for name, value in VOICE_ENV.items() if name != missing}

    with pytest.raises(ConfigError, match=missing):
        VoiceSettings.from_env(env)
