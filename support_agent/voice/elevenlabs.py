"""Speech-to-text (Scribe) and text-to-speech with ElevenLabs.

Both are plain HTTP calls with the client the Anthropic SDK already brings, like
the Groq one, so no ElevenLabs SDK is needed. The key goes in the `xi-api-key`
header.

The free plan has a monthly allowance of credits shared by both services
(about one credit per spoken character, and several hundred per minute heard),
which is why ElevenLabs is optional and edge-tts stays the default voice.
"""

from __future__ import annotations

from typing import Callable

import httpx2
import numpy as np

from support_agent.voice.audio import Audio
from support_agent.voice.speech import SpeechError, Transcript

API_URL = "https://api.elevenlabs.io/v1"

# Raw 16-bit mono samples at 24 kHz: nothing to decode, and allowed on the free plan.
OUTPUT_FORMAT = "pcm_24000"
OUTPUT_SAMPLE_RATE = 24_000

# Scribe reports three-letter codes ("spa"); two-letter ones are accepted too.
LANGUAGE_CODES = {
    "spa": "es",
    "jpn": "ja",
    "eng": "en",
    "es": "es",
    "ja": "ja",
    "en": "en",
}


class ElevenLabsTextToSpeech:
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice_id: str,
        post: Callable[..., httpx2.Response] = httpx2.post,
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key
        self.model = model
        # ElevenLabs voices are multilingual: the same one reads all three languages.
        self.voice_id = voice_id
        self._post = post
        self._timeout_seconds = timeout_seconds

    def synthesize(self, text: str, language: str) -> Audio:
        try:
            response = self._post(
                f"{API_URL}/text-to-speech/{self.voice_id}",
                headers={"xi-api-key": self._api_key},
                params={"output_format": OUTPUT_FORMAT},
                # language_code tells the model how to read numbers and dates.
                json={"text": text, "model_id": self.model, "language_code": language},
                timeout=self._timeout_seconds,
            )
        except httpx2.HTTPError as error:
            raise SpeechError(f"Could not reach ElevenLabs: {type(error).__name__}") from error
        if response.status_code != 200:
            raise SpeechError(
                f"ElevenLabs answered {response.status_code}: {_error_message(response)}"
            )
        if not response.content:
            raise SpeechError("ElevenLabs returned no audio")
        samples = np.frombuffer(response.content, dtype="<i2").copy()
        return Audio(samples, OUTPUT_SAMPLE_RATE)


class ElevenLabsSpeechToText:
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        model: str,
        post: Callable[..., httpx2.Response] = httpx2.post,
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key
        self.model = model
        self._post = post
        self._timeout_seconds = timeout_seconds

    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        data = {
            "model_id": self.model,
            # Do not write "(laughter)" or "(noise)" into the customer's message.
            "tag_audio_events": "false",
        }
        if language:
            data["language_code"] = language
        try:
            response = self._post(
                f"{API_URL}/speech-to-text",
                headers={"xi-api-key": self._api_key},
                files={"file": ("speech.wav", audio.to_wav(), "audio/wav")},
                data=data,
                timeout=self._timeout_seconds,
            )
        except httpx2.HTTPError as error:
            raise SpeechError(f"Could not reach ElevenLabs: {type(error).__name__}") from error
        if response.status_code != 200:
            raise SpeechError(
                f"ElevenLabs answered {response.status_code}: {_error_message(response)}"
            )

        body = response.json()
        return Transcript(
            text=str(body.get("text", "")).strip(),
            language=LANGUAGE_CODES.get(str(body.get("language_code", "")).lower()),
            avg_logprob=_average_word_logprob(body.get("words") or []),
        )


def _average_word_logprob(words: list[dict]) -> float | None:
    # Scribe gives one log probability per word (Whisper gives one per segment),
    # so the two averages are similar in meaning but not on exactly the same scale.
    values = [
        word["logprob"]
        for word in words
        if word.get("type") == "word" and isinstance(word.get("logprob"), (int, float))
    ]
    return round(sum(values) / len(values), 3) if values else None


def _error_message(response: httpx2.Response) -> str:
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]
    if isinstance(detail, dict):
        # For example {"status": "quota_exceeded", "message": "..."}.
        return str(detail.get("message") or detail.get("status") or detail)
    return str(detail)[:200]
