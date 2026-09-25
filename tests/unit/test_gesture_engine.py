"""
Unit tests for BodyGestureEngine and horizontal movement rules (Step 5).
"""

import pytest
from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.body.body_rules import BodyGestureThresholds, BodyMovementRules
from app.core.gestures.common.actions import Action
from app.core.vision.pose.pose_features import BodyFeatures


@pytest.fixture
def baseline():
    return CalibrationBaseline(
        hip_center_x=0.50,
        hip_center_y=0.50,
        shoulder_center_x=0.50,
        shoulder_center_y=0.20,
        body_scale=0.60
    )


@pytest.fixture
def engine():
    thresholds = BodyGestureThresholds(
        move_left_trigger=0.18,
        move_left_release=0.10,
        move_right_trigger=0.18,
        move_right_release=0.10
    )
    return BodyGestureEngine(thresholds)


def test_horizontal_movement_neutral(engine, baseline):
    features = BodyFeatures(
        normalized_x=0.05,
        horizontal_velocity=0.0,
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE
    assert result.gesture_name == "NONE"


def test_horizontal_movement_move_left(engine, baseline):
    features = BodyFeatures(
        normalized_x=-0.25,  # Exceeds -0.18 threshold
        horizontal_velocity=-1.2,
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.MOVE_LEFT
    assert result.gesture_name == "MOVE_LEFT"
    assert result.confidence == 1.0


def test_horizontal_movement_move_right(engine, baseline):
    features = BodyFeatures(
        normalized_x=0.22,  # Exceeds +0.18 threshold
        horizontal_velocity=1.0,
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.MOVE_RIGHT
    assert result.gesture_name == "MOVE_RIGHT"
    assert result.confidence > 0.9


def test_uncalibrated_returns_none(engine):
    features = BodyFeatures(
        normalized_x=0.30,
        is_valid=True
    )
    # Pass baseline=None
    result = engine.process(features, baseline=None)
    assert result.action == Action.NONE
    assert result.debug_info.get("reason") == "uncalibrated_or_invalid"


def test_invalid_features_returns_none(engine, baseline):
    features = BodyFeatures(
        normalized_x=0.30,
        is_valid=False
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE


def test_jump_detection_success(engine, baseline):
    """Test jump triggers when BOTH displacement and upward velocity meet thresholds."""
    features = BodyFeatures(
        normalized_y_displacement=0.15,  # >= 0.10
        vertical_velocity=0.85,          # >= 0.60
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.JUMP
    assert result.gesture_name == "JUMP"
    assert result.confidence == 1.0


def test_jump_fails_low_displacement(engine, baseline):
    """Test jump does not trigger if upward displacement is too small even with high velocity."""
    features = BodyFeatures(
        normalized_y_displacement=0.04,  # < 0.10
        vertical_velocity=1.20,          # high velocity
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE


def test_jump_fails_low_velocity(engine, baseline):
    """Test jump does not trigger if movement is slow (e.g. slowly raising heels or standing on toes)."""
    features = BodyFeatures(
        normalized_y_displacement=0.14,  # >= 0.10
        vertical_velocity=0.20,          # < 0.60 (too slow)
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE


def test_jump_priority_over_horizontal(engine, baseline):
    """Test jump takes precedence when jumping slightly angled."""
    features = BodyFeatures(
        normalized_x=0.25,               # Exceeds horizontal right threshold
        normalized_y_displacement=0.15,  # Exceeds jump displacement
        vertical_velocity=0.90,          # Exceeds jump velocity
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.JUMP


def test_crouch_detection_height_compression(engine, baseline):
    """Test crouch triggers when body height ratio compresses <= 0.78."""
    features = BodyFeatures(
        body_height_ratio=0.72,          # <= 0.78
        normalized_y_displacement=-0.05,
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.CROUCH
    assert result.gesture_name == "CROUCH"


def test_crouch_detection_downward_displacement(engine, baseline):
    """Test crouch triggers when hips drop downwards and height compresses."""
    features = BodyFeatures(
        body_height_ratio=0.82,          # moderate compression
        normalized_y_displacement=-0.12, # <= -0.08 downward hip drop
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.CROUCH


def test_crouch_fails_standing_neutral(engine, baseline):
    """Test crouch does not trigger when user is standing normally."""
    features = BodyFeatures(
        body_height_ratio=0.98,
        normalized_y_displacement=0.01,
        is_valid=True
    )
    result = engine.process(features, baseline=baseline)
    assert result.action == Action.NONE


def test_hysteresis_horizontal_trigger_and_release(baseline):
    """Test horizontal dual-threshold Schmitt trigger hysteresis."""
    thresholds = BodyGestureThresholds(
        move_right_trigger=0.20,
        move_right_release=0.12
    )
    engine = BodyGestureEngine(thresholds, enable_temporal_filtering=False)

    # 1. Start in neutral: X = 0.15 (below trigger 0.20) -> NONE
    f1 = BodyFeatures(normalized_x=0.15, is_valid=True)
    assert engine.process(f1, baseline=baseline).action == Action.NONE

    # 2. Cross trigger threshold: X = 0.22 -> MOVE_RIGHT
    f2 = BodyFeatures(normalized_x=0.22, is_valid=True)
    assert engine.process(f2, baseline=baseline).action == Action.MOVE_RIGHT

    # 3. Drift back into hysteresis band: X = 0.16 (above release 0.12) -> STAYS MOVE_RIGHT!
    f3 = BodyFeatures(normalized_x=0.16, is_valid=True)
    assert engine.process(f3, baseline=baseline).action == Action.MOVE_RIGHT

    # 4. Fall below release threshold: X = 0.10 (< 0.12) -> Deactivates to NONE
    f4 = BodyFeatures(normalized_x=0.10, is_valid=True)
    assert engine.process(f4, baseline=baseline).action == Action.NONE


def test_hysteresis_crouch_trigger_and_release(baseline):
    """Test crouch dual-threshold hysteresis."""
    thresholds = BodyGestureThresholds(
        crouch_trigger_ratio=0.76,
        crouch_release_ratio=0.84,
        crouch_trigger_disp=-0.09,
        crouch_release_disp=-0.04
    )
    engine = BodyGestureEngine(thresholds, enable_temporal_filtering=False)

    # 1. Neutral standing: ratio = 0.80 (above trigger 0.76) -> NONE
    f1 = BodyFeatures(body_height_ratio=0.80, normalized_y_displacement=-0.02, is_valid=True)
    assert engine.process(f1, baseline=baseline).action == Action.NONE

    # 2. Deep crouch: ratio = 0.74 (<= 0.76) -> CROUCH
    f2 = BodyFeatures(body_height_ratio=0.74, normalized_y_displacement=-0.10, is_valid=True)
    assert engine.process(f2, baseline=baseline).action == Action.CROUCH

    # 3. Rising slightly: ratio = 0.80, disp = -0.06 -> STAYS CROUCH!
    f3 = BodyFeatures(body_height_ratio=0.80, normalized_y_displacement=-0.06, is_valid=True)
    assert engine.process(f3, baseline=baseline).action == Action.CROUCH

    # 4. Fully standing up: ratio = 0.88 (>= 0.84), disp = 0.00 -> Deactivates to NONE
    f4 = BodyFeatures(body_height_ratio=0.88, normalized_y_displacement=0.00, is_valid=True)
    assert engine.process(f4, baseline=baseline).action == Action.NONE



