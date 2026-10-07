"""Speech-to-text with Whisper hosted by Groq (free tier).

The endpoint follows the OpenAI audio API: one multipart POST with the sound
file, answered with JSON. It is called directly with the HTTP client the
Anthropic SDK already brings, so no extra SDK is needed.
"""

from __future__ import annotations

from typing import Callable

import httpx2

from support_agent.voice.audio import Audio
from support_agent.voice.speech import SpeechError, Transcript

TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Whisper names the language in English. The codes map to themselves in case
# a provider answers with those instead.
LANGUAGE_CODES = {
    "spanish": "es",
    "japanese": "ja",
    "english": "en",
    "es": "es",
    "ja": "ja",
    "en": "en",
}


class GroqSpeechToText:
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
            "model": self.model,
            # verbose_json also reports the language and how confident the model is.
            "response_format": "verbose_json",
            # 0 = always the most likely transcription, no creativity.
            "temperature": "0",
        }
        if language:
            # Without it Whisper guesses the language from the sound.
            data["language"] = language
        try:
            response = self._post(
                TRANSCRIPTIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("speech.wav", audio.to_wav(), "audio/wav")},
                data=data,
                timeout=self._timeout_seconds,
            )
        except httpx2.HTTPError as error:
            raise SpeechError(f"Could not reach Groq: {type(error).__name__}") from error

        if response.status_code != 200:
            raise SpeechError(f"Groq answered {response.status_code}: {_error_message(response)}")

        body = response.json()
        return Transcript(
            text=str(body.get("text", "")).strip(),
            language=LANGUAGE_CODES.get(str(body.get("language", "")).lower()),
            avg_logprob=_average_logprob(body.get("segments") or []),
        )


def _average_logprob(segments: list[dict]) -> float | None:
    values = [segment["avg_logprob"] for segment in segments if "avg_logprob" in segment]
    return round(sum(values) / len(values), 3) if values else None


def _error_message(response: httpx2.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return response.text[:200]
