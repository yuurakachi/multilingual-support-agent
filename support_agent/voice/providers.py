"""Build the speech-to-text and text-to-speech services chosen in .env.

    STT_PROVIDER  groq (default) or elevenlabs
    TTS_PROVIDER  edge (default) or elevenlabs; ElevenLabs falls back to edge-tts
                  when it fails, for example once the free credits run out
"""

from __future__ import annotations

from typing import Callable

from support_agent.config import VoiceSettings
from support_agent.voice.speech import FallbackTextToSpeech, SpeechToText, TextToSpeech


def build_speech_to_text(settings: VoiceSettings) -> SpeechToText:
    if settings.stt_provider == "elevenlabs":
        from support_agent.voice.elevenlabs import ElevenLabsSpeechToText

        return ElevenLabsSpeechToText(settings.elevenlabs_api_key, settings.elevenlabs_stt_model)
    from support_agent.voice.stt_groq import GroqSpeechToText

    return GroqSpeechToText(settings.groq_api_key, settings.stt_model)


def build_text_to_speech(
    settings: VoiceSettings, warn: Callable[[str], None] = lambda message: None
) -> TextToSpeech:
    from support_agent.voice.tts_edge import DEFAULT_VOICES, EdgeTextToSpeech

    edge = EdgeTextToSpeech({**DEFAULT_VOICES, **settings.tts_voices})
    if settings.tts_provider != "elevenlabs":
        return edge
    from support_agent.voice.elevenlabs import ElevenLabsTextToSpeech

    elevenlabs = ElevenLabsTextToSpeech(
        settings.elevenlabs_api_key, settings.elevenlabs_tts_model, settings.elevenlabs_voice_id
    )
    return FallbackTextToSpeech(elevenlabs, edge, warn)


def speech_metadata(settings: VoiceSettings) -> dict:
    """Which speech services produced a conversation, for its log."""
    from support_agent.voice.tts_edge import DEFAULT_VOICES

    metadata = {"stt_provider": settings.stt_provider, "tts_provider": settings.tts_provider}
    if settings.stt_provider == "elevenlabs":
        metadata["stt_model"] = settings.elevenlabs_stt_model
    else:
        metadata["stt_model"] = settings.stt_model
    if settings.tts_provider == "elevenlabs":
        metadata["tts_model"] = settings.elevenlabs_tts_model
        metadata["tts_voice"] = settings.elevenlabs_voice_id
    # edge-tts speaks every reply, or stands by as the fallback.
    metadata["tts_voices"] = {**DEFAULT_VOICES, **settings.tts_voices}
    return metadata
