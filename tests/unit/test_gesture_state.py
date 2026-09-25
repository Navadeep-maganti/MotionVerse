"""
Unit tests for Gesture State Machine and Cooldown Tracker (Step 10).
"""

import pytest
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture_state import GestureState, GestureStateMachine
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


def test_state_machine_discrete_jump_debouncing():
    """Test jump triggers on leading edge and is suppressed during airborne cooldown."""
    sm = GestureStateMachine(jump_cooldown_sec=0.45)
    assert sm.current_state == GestureState.IDLE

    # 1. First frame of jump (t = 1.0) -> Dispatches JUMP, State = TRIGGERED
    act1, state1 = sm.update(Action.JUMP, timestamp=1.0)
    assert act1 == Action.JUMP
    assert state1 == GestureState.TRIGGERED

    # 2. Subsequent airborne frames (t = 1.1, 1.2, 1.3) -> Suppressed (NONE), State = COOLDOWN
    act2, state2 = sm.update(Action.JUMP, timestamp=1.1)
    assert act2 == Action.NONE
    assert state2 == GestureState.COOLDOWN

    act3, state3 = sm.update(Action.JUMP, timestamp=1.3)
    assert act3 == Action.NONE
    assert state3 == GestureState.COOLDOWN

    # 3. User lands (t = 1.4, candidate = NONE) -> Cooldown finishes around 1.45
    act4, state4 = sm.update(Action.NONE, timestamp=1.4)
    assert act4 == Action.NONE
    assert state4 == GestureState.COOLDOWN

    # 4. At t = 1.5 (> 1.45s) -> Returns to IDLE
    act5, state5 = sm.update(Action.NONE, timestamp=1.5)
    assert act5 == Action.NONE
    assert state5 == GestureState.IDLE

    # 5. Next jump at t = 1.6 -> Can trigger again cleanly!
    act6, state6 = sm.update(Action.JUMP, timestamp=1.6)
    assert act6 == Action.JUMP
    assert state6 == GestureState.TRIGGERED


def test_state_machine_continuous_holding():
    """Test continuous gestures (CROUCH, MOVE_LEFT) maintain HOLDING state."""
    sm = GestureStateMachine()

    # Sustained MOVE_LEFT
    act1, s1 = sm.update(Action.MOVE_LEFT, timestamp=1.0)
    assert act1 == Action.MOVE_LEFT
    assert s1 == GestureState.HOLDING

    act2, s2 = sm.update(Action.MOVE_LEFT, timestamp=1.1)
    assert act2 == Action.MOVE_LEFT
    assert s2 == GestureState.HOLDING

    # Release to NONE
    act3, s3 = sm.update(Action.NONE, timestamp=1.2)
    assert act3 == Action.NONE
    assert s3 == GestureState.IDLE
