"""
Generic Action enumerations for MotionVerse.
Provides a unified action vocabulary across all input modalities (body, hand, etc.).
"""

from enum import Enum


class Action(Enum):
    """
    Standard generic actions emitted by gesture engines.
    Strictly game-independent and input-independent.
    """
    NONE = "NONE"
    JUMP = "JUMP"
    CROUCH = "CROUCH"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    PAUSE = "PAUSE"
    SELECT = "SELECT"

    def __str__(self) -> str:
        return self.value
