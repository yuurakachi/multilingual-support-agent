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


class SpeechToText(Protocol):
    def transcribe(self, audio: Audio) -> Transcript:
        """Turn recorded speech into text. Raises SpeechError on failure."""
        ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, language: str) -> Audio:
        """Turn text into speech in one of LANGUAGES. Raises SpeechError on failure."""
        ...
