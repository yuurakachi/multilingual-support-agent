"""Scripted stand-ins for the key, the microphone, the speaker and the speech providers."""

import numpy as np

from support_agent.voice.audio import Audio


def speech(seconds=4.0, sample_rate=16_000):
    """A recording loud enough to count as someone talking (it is a beep)."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return Audio((np.sin(2 * np.pi * 220 * t) * 6000).astype(np.int16), sample_rate)


def quiet(seconds=4.0):
    """A recording of an empty room: faint noise, nobody talking."""
    noise = np.random.default_rng(0).normal(0, 40, int(seconds * 16_000))
    return Audio(noise.astype(np.int16), 16_000)


class ManualClock:
    """Time that only moves when a fake says so, to make timings exact."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeKey:
    def __init__(self, actions):
        self.actions = list(actions)
        self.drained = 0

    def wait(self):
        if not self.actions:
            raise EOFError
        return self.actions.pop(0)

    def is_held(self):
        return False

    def drain(self):
        self.drained += 1


class FakeMicrophone:
    def __init__(self, recordings):
        self.recordings = list(recordings)

    def record_while(self, is_held):
        return self.recordings.pop(0)


class FakeSpeechToText:
    """Returns its scripted results in order; an exception in the list is raised."""

    def __init__(self, results=(), clock=None, seconds=0.0):
        self.results = list(results)
        # The language hint of every call, None when the recogniser was left to guess.
        self.hints = []
        self.clock = clock
        self.seconds = seconds

    def transcribe(self, audio, language=None):
        self.hints.append(language)
        if self.clock:
            self.clock.now += self.seconds
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTextToSpeech:
    def __init__(self, error=None, clock=None, seconds=0.0):
        self.error = error
        self.spoken = []
        self.clock = clock
        self.seconds = seconds

    def synthesize(self, text, language):
        self.spoken.append((text, language))
        if self.clock:
            self.clock.now += self.seconds
        if self.error:
            raise self.error
        return Audio(np.zeros(2400, dtype=np.int16), 24_000)


class FakeSpeaker:
    def __init__(self, error=None):
        self.played = []
        self.error = error

    def play(self, audio):
        if self.error:
            raise self.error
        self.played.append(audio)
