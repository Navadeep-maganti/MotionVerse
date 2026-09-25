"""
Unit tests for BodyGestureEngine and Dynamic-Reference Discrete Gesture Recognition.
Covers all required discrete test cases from the specification.
"""

import pytest
from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.body.body_rules import BodyGestureThresholds
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture_state import HorizontalState, VerticalState
from app.core.vision.pose.pose_features import BodyFeatures


@pytest.fixture
def baseline():
    """Calibrated baseline with hip at 0.50, scale at 0.50."""
    return CalibrationBaseline(
        hip_center_x=0.50,
        hip_center_y=0.50,
        shoulder_center_x=0.50,
        shoulder_center_y=0.20,
        body_scale=0.50
    )


@pytest.fixture
def engine():
    """Engine with filtering disabled for predictable step-by-step unit testing."""
    thresholds = BodyGestureThresholds(
        move_left_trigger=0.18,
        move_right_trigger=0.18,
        jump_trigger_displacement=0.10,
        jump_trigger_velocity=0.60,
        crouch_trigger_ratio=0.78,
        crouch_release_ratio=0.84,
        crouch_trigger_disp=-0.08,
        crouch_release_disp=-0.04,
        horizontal_velocity_settle=0.12
    )
    return BodyGestureEngine(
        thresholds=thresholds,
        enable_temporal_filtering=False,
        horizontal_settle_duration_sec=0.10,
        jump_cooldown_sec=0.30
    )


# ==========================================
# 10 EXPLICIT SPECIFICATION TEST CASES
# ==========================================

def test_1_single_right(engine, baseline):
    """Test 1: neutral -> right emits MOVE_RIGHT exactly once."""
    # 1. Neutral frame (hip at 0.50, Dx = 0.0)
    f_neutral = BodyFeatures(hip_center_x=0.50, horizontal_velocity=0.0, is_valid=True)
    r1 = engine.process(f_neutral, baseline=baseline, timestamp=1.0)
    assert r1.action == Action.NONE

    # 2. User moves right (hip at 0.60, Dx = (0.60-0.50)/0.50 = +0.20 >= 0.18)
    f_right = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.5, is_valid=True)
    r2 = engine.process(f_right, baseline=baseline, timestamp=1.05)
    assert r2.action == Action.MOVE_RIGHT
    assert engine.horizontal_state == HorizontalState.SETTLING


def test_2_stay_right(engine, baseline):
    """Test 2: neutral -> right -> stay right emits MOVE_RIGHT exactly once, then NONE."""
    # Trigger right
    f_right = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.5, is_valid=True)
    r1 = engine.process(f_right, baseline=baseline, timestamp=1.0)
    assert r1.action == Action.MOVE_RIGHT

    # Still at x=0.60, moving / settling -> Action must be NONE!
    f_stay1 = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.05, is_valid=True)
    r2 = engine.process(f_stay1, baseline=baseline, timestamp=1.05)
    assert r2.action == Action.NONE

    # Settled past 0.10s -> Action remains NONE, reference updates to 0.60
    r3 = engine.process(f_stay1, baseline=baseline, timestamp=1.20)
    assert r3.action == Action.NONE
    assert engine.horizontal_state == HorizontalState.READY
    assert pytest.approx(engine.reference_x, 0.01) == 0.60

    # User remains at x=0.60 -> Delta is now (0.60 - 0.60) = 0.0 -> Action remains NONE
    r4 = engine.process(f_stay1, baseline=baseline, timestamp=1.30)
    assert r4.action == Action.NONE


def test_3_two_consecutive_rights(engine, baseline):
    """Test 3: neutral -> right -> settle -> right again emits exactly two MOVE_RIGHT events."""
    # 1. First move right: 0.50 -> 0.60
    f1 = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.5, is_valid=True)
    assert engine.process(f1, baseline=baseline, timestamp=1.0).action == Action.MOVE_RIGHT

    # Settle at 0.60
    f1_stable = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.0, is_valid=True)
    engine.process(f1_stable, baseline=baseline, timestamp=1.05)
    engine.process(f1_stable, baseline=baseline, timestamp=1.20)
    assert engine.horizontal_state == HorizontalState.READY
    assert pytest.approx(engine.reference_x, 0.01) == 0.60

    # 2. Second move right: 0.60 -> 0.70 (Dx = (0.70 - 0.60) / 0.50 = +0.20 >= 0.18)
    f2 = BodyFeatures(hip_center_x=0.70, horizontal_velocity=0.5, is_valid=True)
    r_second = engine.process(f2, baseline=baseline, timestamp=1.30)
    assert r_second.action == Action.MOVE_RIGHT


def test_4_right_then_left(engine, baseline):
    """Test 4: neutral -> right -> settle -> left (no return to original calibration position)."""
    # 1. Move right from 0.50 -> 0.60
    f1 = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.5, is_valid=True)
    assert engine.process(f1, baseline=baseline, timestamp=1.0).action == Action.MOVE_RIGHT

    # Settle at 0.60
    f1_stable = BodyFeatures(hip_center_x=0.60, horizontal_velocity=0.0, is_valid=True)
    engine.process(f1_stable, baseline=baseline, timestamp=1.05)
    engine.process(f1_stable, baseline=baseline, timestamp=1.20)
    assert pytest.approx(engine.reference_x, 0.01) == 0.60

    # 2. Move left from 0.60 -> 0.50 (Dx = (0.50 - 0.60)/0.50 = -0.20 <= -0.18)
    f2 = BodyFeatures(hip_center_x=0.50, horizontal_velocity=-0.5, is_valid=True)
    r_left = engine.process(f2, baseline=baseline, timestamp=1.30)
    assert r_left.action == Action.MOVE_LEFT


def test_5_multiple_right_movements(engine, baseline):
    """Test 5: right -> right -> right -> right without artificial boundary."""
    positions = [0.60, 0.70, 0.80, 0.90]
    t = 1.0
    for pos in positions:
        # Move
        f_move = BodyFeatures(hip_center_x=pos, horizontal_velocity=0.5, is_valid=True)
        res = engine.process(f_move, baseline=baseline, timestamp=t)
        assert res.action == Action.MOVE_RIGHT

        # Settle
        t += 0.05
        f_stable = BodyFeatures(hip_center_x=pos, horizontal_velocity=0.0, is_valid=True)
        engine.process(f_stable, baseline=baseline, timestamp=t)
        t += 0.15
        engine.process(f_stable, baseline=baseline, timestamp=t)
        assert engine.horizontal_state == HorizontalState.READY
        assert pytest.approx(engine.reference_x, 0.01) == pos
        t += 0.10


def test_6_multiple_left_movements(engine, baseline):
    """Test 6: left -> left -> left -> left without artificial boundary."""
    positions = [0.40, 0.30, 0.20, 0.10]
    t = 1.0
    for pos in positions:
        f_move = BodyFeatures(hip_center_x=pos, horizontal_velocity=-0.5, is_valid=True)
        res = engine.process(f_move, baseline=baseline, timestamp=t)
        assert res.action == Action.MOVE_LEFT

        t += 0.05
        f_stable = BodyFeatures(hip_center_x=pos, horizontal_velocity=0.0, is_valid=True)
        engine.process(f_stable, baseline=baseline, timestamp=t)
        t += 0.15
        engine.process(f_stable, baseline=baseline, timestamp=t)
        assert engine.horizontal_state == HorizontalState.READY
        assert pytest.approx(engine.reference_x, 0.01) == pos
        t += 0.10


def test_7_hold_crouch(engine, baseline):
    """Test 7: standing -> crouch -> remain crouched emits CROUCH exactly once."""
    # 1. Trigger crouch (height ratio 0.70 <= 0.78, disp -0.10 <= -0.08)
    f_crouch = BodyFeatures(body_height_ratio=0.70, normalized_y_displacement=-0.10, is_valid=True)
    r1 = engine.process(f_crouch, baseline=baseline, timestamp=1.0)
    assert r1.action == Action.CROUCH
    assert engine.vertical_state == VerticalState.RECOVERING

    # 2. Stay crouched across multiple frames -> NONE
    for t_offset in [0.05, 0.10, 0.20, 0.50]:
        r = engine.process(f_crouch, baseline=baseline, timestamp=1.0 + t_offset)
        assert r.action == Action.NONE
        assert engine.vertical_state == VerticalState.RECOVERING


def test_8_repeated_crouch(engine, baseline):
    """Test 8: standing -> crouch -> stand -> crouch -> stand emits exactly two CROUCH events."""
    f_crouch = BodyFeatures(body_height_ratio=0.70, normalized_y_displacement=-0.10, is_valid=True)
    f_standing = BodyFeatures(body_height_ratio=0.95, normalized_y_displacement=0.0, is_valid=True)

    # 1. First crouch
    assert engine.process(f_crouch, baseline=baseline, timestamp=1.0).action == Action.CROUCH

    # 2. Stand up (ratio >= 0.84 and disp >= -0.04)
    assert engine.process(f_standing, baseline=baseline, timestamp=1.2).action == Action.NONE
    assert engine.vertical_state == VerticalState.READY

    # 3. Second crouch
    assert engine.process(f_crouch, baseline=baseline, timestamp=1.4).action == Action.CROUCH


def test_9_hold_jump_airborne_frames(engine, baseline):
    """Test 9: Single jump across multiple airborne frames emits JUMP exactly once."""
    f_jump_trigger = BodyFeatures(normalized_y_displacement=0.15, vertical_velocity=0.85, is_valid=True)
    f_airborne = BodyFeatures(normalized_y_displacement=0.18, vertical_velocity=-0.20, is_valid=True)
    f_landed = BodyFeatures(normalized_y_displacement=0.02, vertical_velocity=0.0, is_valid=True)

    # 1. Jump launch -> JUMP
    r1 = engine.process(f_jump_trigger, baseline=baseline, timestamp=1.0)
    assert r1.action == Action.JUMP
    assert engine.vertical_state == VerticalState.RECOVERING

    # 2. Airborne peak frames -> NONE
    assert engine.process(f_airborne, baseline=baseline, timestamp=1.10).action == Action.NONE
    assert engine.process(f_airborne, baseline=baseline, timestamp=1.20).action == Action.NONE

    # 3. Landing after cooldown -> Returns to READY
    assert engine.process(f_landed, baseline=baseline, timestamp=1.40).action == Action.NONE
    assert engine.vertical_state == VerticalState.READY


def test_10_noise_around_threshold(engine, baseline):
    """Test 10: Small body/landmark fluctuations should not generate actions."""
    noise_samples = [
        BodyFeatures(hip_center_x=0.52, horizontal_velocity=0.02, is_valid=True),   # Dx = 0.04 (< 0.18)
        BodyFeatures(hip_center_x=0.48, horizontal_velocity=-0.03, is_valid=True),  # Dx = -0.04 (> -0.18)
        BodyFeatures(normalized_y_displacement=0.03, vertical_velocity=0.15, is_valid=True),  # Low jump
        BodyFeatures(body_height_ratio=0.88, normalized_y_displacement=-0.02, is_valid=True), # Minor posture shift
    ]
    for i, sample in enumerate(noise_samples):
        res = engine.process(sample, baseline=baseline, timestamp=1.0 + i * 0.1)
        assert res.action == Action.NONE


# ==========================================
# ADDITIONAL VALIDATION TESTS
# ==========================================

def test_uncalibrated_returns_none(engine):
    features = BodyFeatures(hip_center_x=0.70, is_valid=True)
    result = engine.process(features, baseline=None)
    assert result.action == Action.NONE
    assert result.debug_info.get("reason") == "uncalibrated_or_invalid"


def test_invalid_features_returns_none(engine, baseline):
    features = BodyFeatures(hip_center_x=0.70, is_valid=False)
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE


def test_jump_priority_over_horizontal(engine, baseline):
    """Jump takes priority over horizontal displacement."""
    features = BodyFeatures(
        hip_center_x=0.70,               # Exceeds horizontal trigger
        normalized_y_displacement=0.15,  # Exceeds jump displacement
        vertical_velocity=0.90,          # Exceeds jump velocity
        is_valid=True
    )
    result = engine.process(features, baseline=baseline, timestamp=1.0)
    assert result.action == Action.JUMP
