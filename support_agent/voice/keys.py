"""The push-to-talk key, for a Windows console.

A terminal only reports that a key was *pressed*, never that it was released.
So the key is read in two ways:

    wait()     reads key presses from the console, which only arrive when the
               terminal window has the focus
    is_held()  asks Windows whether the space bar is physically down right now

Recording starts on a press seen by wait() and lasts while is_held() is true.
"""

from __future__ import annotations

import sys
from typing import Callable

TALK = "talk"
NEW = "new"
QUIT = "quit"

VK_SPACE = 0x20
_ESCAPE = "\x1b"
_CTRL_C = "\x03"
# Arrow and function keys arrive as two codes: one of these, then the key itself.
_TWO_CODE_PREFIXES = ("\x00", "\xe0")


class PushToTalkKey:
    def __init__(
        self,
        read_key: Callable[[], str] | None = None,
        key_pending: Callable[[], bool] | None = None,
        space_is_down: Callable[[], bool] | None = None,
    ):
        if read_key is None or key_pending is None or space_is_down is None:
            if sys.platform != "win32":
                raise OSError("Push-to-talk is only implemented for Windows consoles.")
            import ctypes
            import msvcrt

            read_key = read_key or msvcrt.getwch
            key_pending = key_pending or msvcrt.kbhit
            # The top bit of the result is set while the key is down.
            space_is_down = space_is_down or (
                lambda: bool(ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000)
            )
        self._read_key = read_key
        self._key_pending = key_pending
        self._space_is_down = space_is_down

    def wait(self) -> str:
        """Block until the customer does something: TALK, NEW or QUIT."""
        while True:
            key = self._read_key()
            if key in _TWO_CODE_PREFIXES:
                self._read_key()
                continue
            if key == " ":
                return TALK
            if key.lower() == "n":
                return NEW
            if key.lower() == "q" or key in (_ESCAPE, _CTRL_C):
                return QUIT

    def is_held(self) -> bool:
        return self._space_is_down()

    def drain(self) -> None:
        """Throw away the repeats a held key produced, so they do not start a new recording."""
        while self._key_pending():
            self._read_key()
