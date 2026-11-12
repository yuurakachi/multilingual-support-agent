"""The two sockets of the voice channel: speech-to-text and text-to-speech.

The voice chat only knows these interfaces. A provider (Groq, edge-tts, a local
model...) is a class with the right method, so one can be replaced without
touching the chat loop, and the tests plug in fakes that need no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

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


class FallbackTextToSpeech:
    """Speaks with `primary`, and with `backup` whenever `primary` fails.

    ElevenLabs' free credits run out, and a reply that cannot be spoken is worse
    than one spoken by a plainer voice. `last_used` says which one spoke last,
    so the log can tell the two apart when latencies are compared.
    """

    def __init__(
        self,
        primary: TextToSpeech,
        backup: TextToSpeech,
        warn: Callable[[str], None] = lambda message: None,
    ):
        self.primary = primary
        self.backup = backup
        self._warn = warn
        self.last_used: str | None = None

    def synthesize(self, text: str, language: str) -> Audio:
        try:
            audio = self.primary.synthesize(text, language)
            self.last_used = _name(self.primary)
        except SpeechError as error:
            self._warn(f"{_name(self.primary)} failed ({error}), using {_name(self.backup)}")
            audio = self.backup.synthesize(text, language)
            self.last_used = _name(self.backup)
        return audio


def provider_name(tts: TextToSpeech) -> str | None:
    """Which service produced the last audio, for the log."""
    return getattr(tts, "last_used", None) or getattr(tts, "name", None)


def _name(provider) -> str:
    return getattr(provider, "name", type(provider).__name__)
