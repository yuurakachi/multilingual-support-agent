"""Text-to-speech with the neural voices of Microsoft Edge's "Read aloud".

Free and without an account, but unofficial: `edge-tts` talks to the service
the Edge browser uses, which Microsoft can change at any time. That is the
reason speech sits behind the TextToSpeech interface.

One voice per language, because each of these voices is native to one language.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Mapping

import truststore

from support_agent.voice.audio import Audio
from support_agent.voice.speech import DEFAULT_LANGUAGE, SpeechError

# edge-tts checks HTTPS certificates against a list bundled with Python, which
# fails on machines where an antivirus or a company proxy re-signs HTTPS. This
# makes Python trust what the operating system trusts instead. It has to run
# before edge_tts is imported, because edge_tts prepares its TLS setup on import.
truststore.inject_into_ssl()

import aiohttp  # noqa: E402
import edge_tts  # noqa: E402
from edge_tts.exceptions import EdgeTTSException  # noqa: E402

DEFAULT_VOICES = {
    "es": "es-MX-DaliaNeural",
    "ja": "ja-JP-NanamiNeural",
    "en": "en-US-AriaNeural",
}


def _fetch_mp3(text: str, voice: str, timeout_seconds: float) -> bytes:
    """Ask the service to speak `text` and collect the MP3 it streams back."""

    async def collect() -> bytes:
        chunks = []
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    # The library is asynchronous. The time limit covers the whole exchange, so a
    # stalled connection can never freeze the chat.
    return asyncio.run(asyncio.wait_for(collect(), timeout_seconds))


class EdgeTextToSpeech:
    def __init__(
        self,
        voices: Mapping[str, str] = DEFAULT_VOICES,
        fetch: Callable[[str, str, float], bytes] = _fetch_mp3,
        timeout_seconds: float = 30.0,
    ):
        self.voices = dict(voices)
        self._fetch = fetch
        self._timeout_seconds = timeout_seconds

    def synthesize(self, text: str, language: str) -> Audio:
        voice = self.voices.get(language) or self.voices[DEFAULT_LANGUAGE]
        try:
            mp3 = self._fetch(text, voice, self._timeout_seconds)
        except (EdgeTTSException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            raise SpeechError(f"edge-tts failed: {type(error).__name__}") from error
        if not mp3:
            raise SpeechError("edge-tts returned no audio")
        try:
            return Audio.from_encoded(mp3)
        except RuntimeError as error:  # soundfile could not decode what came back
            raise SpeechError("edge-tts returned audio that could not be decoded") from error
