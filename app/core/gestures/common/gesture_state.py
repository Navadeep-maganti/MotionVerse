"""
Gesture State Machine module for MotionVerse.
Manages discrete event lifecycles:
- Horizontal: READY -> TRIGGERED (1 event) -> SETTLING -> Update Reference -> READY
- Vertical:   READY -> TRIGGERED (1 event) -> RECOVERING (Wait for recovery) -> READY
"""

from enum import Enum
from typing import Optional


class GestureState(Enum):
    """Lifecycle states of a gesture channel."""
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    SETTLING = "SETTLING"
    RECOVERING = "RECOVERING"

    def __str__(self) -> str:
        return self.value


class HorizontalState(Enum):
    """Lifecycle states for dynamic-reference horizontal movement."""
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    SETTLING = "SETTLING"

    def __str__(self) -> str:
        return self.value


class VerticalState(Enum):
    """Lifecycle states for discrete vertical gestures (Jump & Crouch)."""
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    RECOVERING = "RECOVERING"

    def __str__(self) -> str:
        return self.value
