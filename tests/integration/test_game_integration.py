"""
Integration tests for Game Integration Layer with Safety Arming and Discrete Dispatch.
"""

import pytest
import time
import numpy as np

from app.core.gestures.common.actions import Action
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.calibration.baseline import CalibrationBaseline
from app.core.vision.pose.pose_features import BodyFeatures
from app.games.game_profile import GameProfile
from app.games.game_mapper import GameMapper
from app.input.keyboard_controller import KeyboardController
from app.modes.body_mode import BodyModeController, ControllerMode


@pytest.fixture
def test_setup():
    profile = GameProfile(
        game_name="Subway Surfers Test",
        bindings={
            "JUMP": "up",
            "CROUCH": "down",
            "MOVE_LEFT": "left",
            "MOVE_RIGHT": "right"
        },
        key_tap_duration_ms=10
    )
    recorded_events = []

    def mock_backend(event_type, key):
        recorded_events.append((event_type, key))

    keyboard = KeyboardController(mock_mode=True, backend_callback=mock_backend)
    mapper = GameMapper(profile)
    engine = BodyGestureEngine(enable_temporal_filtering=False)

    baseline = CalibrationBaseline(
        hip_center_x=0.50,
        hip_center_y=0.50,
        shoulder_center_x=0.50,
        shoulder_center_y=0.20,
        body_scale=0.50
    )

    return {
        "profile": profile,
        "keyboard": keyboard,
        "mapper": mapper,
        "engine": engine,
        "baseline": baseline,
        "recorded_events": recorded_events
    }


def test_game_integration_disarmed_safety(test_setup):
    """When DISARMED, actions trigger in engine but 0 keyboard events are dispatched."""
    keyboard = test_setup["keyboard"]
    mapper = test_setup["mapper"]
    engine = test_setup["engine"]
    baseline = test_setup["baseline"]
    recorded = test_setup["recorded_events"]

    # Trigger MOVE_RIGHT
    f_right = BodyFeatures(hip_center_x=0.65, is_valid=True)
    res = engine.process(f_right, baseline=baseline, timestamp=1.0)
    assert res.action == Action.MOVE_RIGHT

    # Simulate disarmed mode
    is_armed = False
    mapped_key = mapper.map_action(res.action)
    assert mapped_key == "right"

    if is_armed and mapped_key:
        keyboard.tap(mapped_key)

    time.sleep(0.02)
    assert len(recorded) == 0
    assert keyboard.total_dispatches == 0


def test_game_integration_armed_discrete_dispatch(test_setup):
    """When ARMED, one Action event triggers exactly ONE keyboard tap."""
    keyboard = test_setup["keyboard"]
    mapper = test_setup["mapper"]
    engine = test_setup["engine"]
    baseline = test_setup["baseline"]
    recorded = test_setup["recorded_events"]

    is_armed = True

    # 1. First Frame: MOVE_RIGHT triggers -> 1 key tap dispatched
    f_right = BodyFeatures(hip_center_x=0.65, is_valid=True)
    res1 = engine.process(f_right, baseline=baseline, timestamp=1.0)
    assert res1.action == Action.MOVE_RIGHT

    key1 = mapper.map_action(res1.action)
    if is_armed and key1:
        keyboard.tap(key1)

    time.sleep(0.03)
    assert keyboard.total_dispatches == 1
    assert ("press", "right") in recorded

    # 2. Subsequent Frames: User stays at x=0.65 -> Action is NONE -> NO extra key taps!
    for t_step in [1.05, 1.10, 1.15]:
        res_idle = engine.process(f_right, baseline=baseline, timestamp=t_step)
        assert res_idle.action == Action.NONE
        key_idle = mapper.map_action(res_idle.action)
        assert key_idle is None
        if is_armed and key_idle:
            keyboard.tap(key_idle)

    time.sleep(0.03)
    # Total dispatches must STILL be exactly 1!
    assert keyboard.total_dispatches == 1


def test_game_integration_discrete_jump_and_crouch(test_setup):
    """JUMP and CROUCH emit single discrete key events without frame-polling repeats."""
    keyboard = test_setup["keyboard"]
    mapper = test_setup["mapper"]
    engine = test_setup["engine"]
    baseline = test_setup["baseline"]
    recorded = test_setup["recorded_events"]

    is_armed = True

    # 1. JUMP
    f_jump = BodyFeatures(normalized_y_displacement=0.15, vertical_velocity=0.90, is_valid=True)
    res_jump = engine.process(f_jump, baseline=baseline, timestamp=1.0)
    assert res_jump.action == Action.JUMP
    key_jump = mapper.map_action(res_jump.action)
    assert key_jump == "up"
    keyboard.tap(key_jump)

    # Multi-frame airborne -> Action is NONE
    for t_step in [1.05, 1.10]:
        res_air = engine.process(f_jump, baseline=baseline, timestamp=t_step)
        assert res_air.action == Action.NONE
        key_air = mapper.map_action(res_air.action)
        if is_armed and key_air:
            keyboard.tap(key_air)

    time.sleep(0.03)
    assert keyboard.total_dispatches == 1
    assert ("press", "up") in recorded
