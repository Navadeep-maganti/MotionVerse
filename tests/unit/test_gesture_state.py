"""
Unit tests for Gesture States and Lifecycle Enums.
"""

import pytest
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture_state import GestureState, HorizontalState, VerticalState
from app.core.gestures.filters.cooldown import CooldownTracker


def test_cooldown_tracker():
    tracker = CooldownTracker(default_cooldown_sec=0.40)
    assert tracker.is_in_cooldown(Action.JUMP, timestamp=1.0) is False

    tracker.trigger(Action.JUMP, duration_sec=0.40, timestamp=1.0)
    assert tracker.is_in_cooldown(Action.JUMP, timestamp=1.2) is True
    assert pytest.approx(tracker.remaining_cooldown_sec(Action.JUMP, timestamp=1.2), 0.01) == 0.20

    # Past cooldown
    assert tracker.is_in_cooldown(Action.JUMP, timestamp=1.45) is False
    assert tracker.remaining_cooldown_sec(Action.JUMP, timestamp=1.45) == 0.0


def test_gesture_state_enums():
    assert GestureState.READY.value == "READY"
    assert GestureState.TRIGGERED.value == "TRIGGERED"
    assert GestureState.SETTLING.value == "SETTLING"
    assert GestureState.RECOVERING.value == "RECOVERING"

    assert HorizontalState.READY.value == "READY"
    assert HorizontalState.TRIGGERED.value == "TRIGGERED"
    assert HorizontalState.SETTLING.value == "SETTLING"

    assert VerticalState.READY.value == "READY"
    assert VerticalState.TRIGGERED.value == "TRIGGERED"
    assert VerticalState.RECOVERING.value == "RECOVERING"
