"""One spoken conversation, without any microphone, key or speaker.

    listen(audio)        speech-to-text, plus the checks that decide whether the
                         customer was understood
    ask_to_repeat(heard) when they were not: a fixed spoken request, no model call
    answer(heard)        the agent's reply, logged, timed and turned into speech

The push-to-talk chat and the scenario runner both drive a conversation through
these three calls: one takes its audio from the microphone, the other from files.

Speech recognition fails in ways typing never does, and the model cannot be
trusted to notice: given silence, Whisper writes a confident "Thank you." So
the checks happen here, before anything reaches the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from support_agent.agent import AgentReply, SupportAgent
from support_agent.config import DEFAULT_SILENCE_LEVEL
from support_agent.conversation_log import ConversationLog
from support_agent.voice.audio import Audio
from support_agent.voice.speech import (
    LANGUAGES,
    SpeechError,
    SpeechToText,
    TextToSpeech,
    Transcript,
)

# Why a recording was not passed on to the agent.
SILENCE = "silence"  # too quiet to be someone talking into the microphone
NO_WORDS = "no_words"  # the transcript is empty or only punctuation
LOW_CONFIDENCE = "low_confidence"  # the recogniser was guessing
UNSUPPORTED_LANGUAGE = "unsupported_language"  # not Spanish, Japanese or English
STT_ERROR = "stt_error"  # the speech-to-text service failed

# Whisper's own rule of thumb: below this average log probability, the words are a guess.
LOW_CONFIDENCE_LOGPROB = -1.0
# A clip this short ("yes", "no") gives the recogniser too little to tell the
# language: a Spanish "Sí." came back as an English "C." So short clips are
# transcribed in the language the conversation is already in.
SHORT_CLIP_SECONDS = 3.0

REPEAT_MESSAGES = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "es": "Perdón, no te escuché bien. ¿Puedes repetirlo?",
    "ja": "申し訳ございません、うまく聞き取れませんでした。もう一度お願いできますか。",
}


@dataclass
class Heard:
    """What came out of listening to one recording."""

    audio: Audio
    # Clock reading when the customer stopped talking: their wait starts here.
    recorded_at: float
    level: int
    transcript: Transcript | None = None
    stt_ms: int | None = None
    # The language the recogniser was told to expect, if any.
    language_hint: str | None = None
    # One of the constants above, or None when the customer was understood.
    problem: str | None = None
    error: str | None = None

    @property
    def understood(self) -> bool:
        return self.problem is None


@dataclass
class RepeatRequest:
    text: str
    # One clip per language spoken; empty if text-to-speech failed.
    clips: list[Audio] = field(default_factory=list)


@dataclass
class Answer:
    reply: AgentReply
    # None if text-to-speech failed; the reply text is still there to show.
    speech: Audio | None
    timings_ms: dict
    tts_error: str | None = None


class VoiceConversation:
    def __init__(
        self,
        agent: SupportAgent,
        log: ConversationLog,
        stt: SpeechToText,
        tts: TextToSpeech,
        clock: Callable[[], float] = time.perf_counter,
        silence_level: int = DEFAULT_SILENCE_LEVEL,
        repeat_clips: dict[str, Audio] | None = None,
    ):
        self.agent = agent
        self.log = log
        self.stt = stt
        self.tts = tts
        self.silence_level = silence_level
        # The language the customer is speaking, once it is known. It picks the
        # voice of the reply and is the hint for short clips.
        self.language: str | None = None
        self._clock = clock
        # The repeat request never changes, so its audio is synthesized once per
        # language. Pass the same dict to several conversations to share it.
        self._repeat_clips = {} if repeat_clips is None else repeat_clips

    def listen(self, audio: Audio, recorded_at: float | None = None) -> Heard:
        """Transcribe a recording and decide whether the customer was understood."""
        heard = Heard(
            audio=audio,
            recorded_at=self._clock() if recorded_at is None else recorded_at,
            level=round(audio.loudness),
        )
        # Checked before anything else, because Whisper invents text for silent audio.
        if heard.level < self.silence_level:
            heard.problem = SILENCE
            return heard

        if audio.seconds < SHORT_CLIP_SECONDS:
            heard.language_hint = self.language
        started = self._clock()
        try:
            transcript = self.stt.transcribe(audio, language=heard.language_hint)
            if transcript.language is None and self.language and _has_words(transcript.text):
                # It heard a language the agent does not speak, in the middle of a
                # conversation held in one it does. Far more likely a wrong guess
                # than a customer switching to Portuguese: listen again, expecting
                # the language of the conversation.
                heard.language_hint = self.language
                transcript = self.stt.transcribe(audio, language=self.language)
        except SpeechError as error:
            heard.stt_ms = self._elapsed_ms(started)
            heard.problem = STT_ERROR
            heard.error = str(error)
            return heard
        heard.stt_ms = self._elapsed_ms(started)
        heard.transcript = transcript

        if not _has_words(transcript.text):
            heard.problem = NO_WORDS
        elif transcript.avg_logprob is not None and transcript.avg_logprob < LOW_CONFIDENCE_LOGPROB:
            heard.problem = LOW_CONFIDENCE
        elif transcript.language is None:
            heard.problem = UNSUPPORTED_LANGUAGE
        else:
            self.language = transcript.language
        return heard

    def ask_to_repeat(self, heard: Heard) -> RepeatRequest:
        """Ask the customer to say it again, and note in the log why it was needed."""
        transcript = heard.transcript
        self.log.record_event(
            "unheard_audio",
            reason=heard.problem,
            recording_seconds=round(heard.audio.seconds, 2),
            level=heard.level,
            stt_ms=heard.stt_ms,
            # What the recogniser wrote, if anything: useful to tune the checks.
            transcript=transcript.text if transcript else None,
            detected_language=transcript.language if transcript else None,
            avg_logprob=transcript.avg_logprob if transcript else None,
            language_hint=heard.language_hint,
            error=heard.error,
        )
        self.log.save(self.agent.messages)

        # Before the customer's language is known, ask in all three, each read by
        # its own voice.
        languages = [self.language] if self.language else list(LANGUAGES)
        request = RepeatRequest(" ".join(REPEAT_MESSAGES[language] for language in languages))
        try:
            for language in languages:
                if language not in self._repeat_clips:
                    self._repeat_clips[language] = self.tts.synthesize(
                        REPEAT_MESSAGES[language], language
                    )
                request.clips.append(self._repeat_clips[language])
        except SpeechError:
            request.clips = []
        return request

    def answer(self, heard: Heard) -> Answer:
        """Let the agent answer what was heard, log the turn and voice the reply."""
        transcript = heard.transcript
        reply = self.agent.reply(transcript.text, language=self.language)

        # tts and total are not known yet: they are filled in below.
        timings = {"stt": heard.stt_ms, "llm": reply.latency_ms, "tts": None, "total": None}
        voice = {
            # The language the reply is spoken in, what the recogniser reported and
            # whether it had been told what to expect.
            "language": self.language,
            "detected_language": transcript.language,
            "language_hint": heard.language_hint,
            "recording_seconds": round(heard.audio.seconds, 2),
            "level": heard.level,
            "avg_logprob": transcript.avg_logprob,
            "speech_seconds": None,
            "timings_ms": timings,
        }
        # Saved as soon as the agent has answered, so the log survives if the
        # conversation is interrupted while the reply is being voiced.
        self.log.record_turn(transcript.text, reply, voice=voice)
        self.log.save(self.agent.messages)

        started = self._clock()
        try:
            speech = self.tts.synthesize(reply.text, self.language)
        except SpeechError as error:
            return Answer(reply, None, timings, tts_error=str(error))
        timings["tts"] = self._elapsed_ms(started)
        # The reply is ready to play: the customer's wait ends here.
        timings["total"] = self._elapsed_ms(heard.recorded_at)
        voice["speech_seconds"] = round(speech.seconds, 2)
        self.log.save(self.agent.messages)
        return Answer(reply, speech, timings)

    def _elapsed_ms(self, started: float) -> int:
        return round((self._clock() - started) * 1000)


def _has_words(text: str) -> bool:
    return any(character.isalnum() for character in text)
