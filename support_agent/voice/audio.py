"""Sound as data, plus the microphone and the speaker.

Audio is always mono 16-bit PCM: the format speech recognisers expect and the
simplest one to record, save and play.

`sounddevice` (the microphone and speaker driver) is imported only when a
Microphone or Speaker is created, so the rest of the package, and the tests,
work on a machine without audio hardware.
"""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# 16 kHz is what speech recognition models are trained on; more is wasted upload.
RECORDING_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # one dimension, dtype int16
    sample_rate: int  # samples per second

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def loudness(self) -> float:
        """How loud the loudest moments are, from 0 (digital silence) to 32768.

        The sound is cut into 30 ms frames and the level of the loudest ones is
        returned (the 95th percentile, so a single click does not count as
        speech). A quiet room gives a few dozen; someone talking into the
        microphone gives hundreds or thousands.
        """
        frame = int(self.sample_rate * 0.03)
        count = len(self.samples) // frame
        if count == 0:
            return 0.0
        frames = self.samples[: count * frame].astype(np.float64).reshape(count, frame)
        levels = np.sqrt((frames**2).mean(axis=1))
        return float(np.percentile(levels, 95))

    def to_wav(self) -> bytes:
        """Encode as a WAV file, the format sent to the speech recogniser."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # bytes per sample: 16 bits
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.samples.astype("<i2").tobytes())
        return buffer.getvalue()

    @classmethod
    def from_encoded(cls, data: bytes) -> Audio:
        """Decode a sound file held in memory (MP3, WAV, FLAC, OGG)."""
        import soundfile

        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype="int16", always_2d=True)
        # Keep the first channel: speech from these providers is mono anyway.
        return cls(samples[:, 0].copy(), sample_rate)

    @classmethod
    def load(cls, path: str | Path) -> Audio:
        return cls.from_encoded(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        """Write a sound file; the extension chooses the format (.mp3, .wav, .flac)."""
        import soundfile

        soundfile.write(str(path), self.samples, self.sample_rate)


class Microphone:
    """Records while a key is held.

    The input stream stays open for the whole chat and is only *kept* while
    recording. Opening it on every key press would take a moment and cut off
    the first syllable.
    """

    def __init__(
        self,
        sample_rate: int = RECORDING_SAMPLE_RATE,
        max_seconds: float = 60.0,
        tail_seconds: float = 0.2,
    ):
        import sounddevice

        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        # People release the key on their last syllable: keep listening briefly.
        self.tail_seconds = tail_seconds
        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._stream = sounddevice.InputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=self._on_audio
        )

    def __enter__(self) -> Microphone:
        self._stream.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stream.close()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        # Called by the audio driver, on its own thread, with each new block of sound.
        if self._recording:
            self._chunks.append(indata[:, 0].copy())

    def record_while(self, is_held: Callable[[], bool], poll_seconds: float = 0.01) -> Audio:
        """Record until `is_held()` turns false (or max_seconds pass)."""
        self._chunks = []
        self._recording = True
        deadline = time.monotonic() + self.max_seconds
        try:
            while is_held() and time.monotonic() < deadline:
                time.sleep(poll_seconds)
            time.sleep(self.tail_seconds)
        finally:
            self._recording = False

        chunks = self._chunks
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return Audio(samples, self.sample_rate)


class Speaker:
    def __init__(self):
        import sounddevice

        self._sounddevice = sounddevice

    def play(self, audio: Audio) -> None:
        """Play to the default output device and return when it has finished."""
        try:
            self._sounddevice.play(audio.samples, audio.sample_rate)
            self._sounddevice.wait()
        except KeyboardInterrupt:
            # Ctrl+C while the agent is talking cuts the audio, then leaves.
            self._sounddevice.stop()
            raise
