"""The two sockets of the voice channel: speech-to-text and text-to-speech.

The voice chat only knows these interfaces. A provider (Groq, edge-tts, a local
model...) is a class with the right method, so one can be replaced without
touching the chat loop, and the tests plug in fakes that need no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from support_agent.voice.audio import Audio

# The languages the agent supports. Providers report and accept these codes.
LANGUAGES = ("es", "ja", "en")
DEFAULT_LANGUAGE = "en"


class SpeechError(Exception):
    """A speech provider failed: network problem, quota, rejected audio..."""


@dataclass(frozen=True)
class Transcript:
    text: str
    # One of LANGUAGES, or None when the recogniser heard another language or
    # could not tell.
    language: str | None = None
    # How sure the recogniser is, as Whisper reports it: the average log
    # probability of the words, 0 being certain and about -1 a guess. None when
    # the provider gives no such number.
    avg_logprob: float | None = None


class SpeechToText(Protocol):
    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        """Turn recorded speech into text. Raises SpeechError on failure.

        `language` (one of LANGUAGES) tells the recogniser what to expect
        instead of letting it guess the language from the sound.
        """
        ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, language: str) -> Audio:
        """Turn text into speech in one of LANGUAGES. Raises SpeechError on failure."""
        ...
