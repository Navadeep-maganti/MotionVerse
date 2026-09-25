"""
Unit tests for Generic Action Abstraction (Step 11).
Verifies strict game-decoupling and standard Action emissions.
"""

import pytest
import numpy as np

from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.calibration.baseline import CalibrationBaseline
from app.core.vision.pose.pose_features import BodyFeatures
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine


def test_action_enum_values():
    """Verify standard generic actions set."""
    expected_actions = {"NONE", "JUMP", "CROUCH", "MOVE_LEFT", "MOVE_RIGHT", "PAUSE", "SELECT"}
    actual_actions = {a.value for a in Action}
    assert expected_actions.issubset(actual_actions)


def test_gesture_result_action_type():
    """Verify GestureResult strictly contains Action instances."""
    res = GestureResult(action=Action.JUMP)
    assert isinstance(res.action, Action)
    assert res.is_active is True

    res_none = GestureResult(action=Action.NONE)
    assert res_none.is_active is False


def test_engine_emits_only_generic_actions():
    """Verify BodyGestureEngine strictly returns valid Action enums for all states."""
    engine = BodyGestureEngine(enable_temporal_filtering=False)
    baseline = CalibrationBaseline(
        hip_center_x=0.5,
        hip_center_y=0.5,
        shoulder_center_x=0.5,
        shoulder_center_y=0.2,
        body_scale=0.6
    )

    # Test cases: (features, expected Action)
    test_cases = [
        (BodyFeatures(normalized_x=0.0, is_valid=True), Action.NONE),
        (BodyFeatures(normalized_x=-0.25, is_valid=True), Action.MOVE_LEFT),
        (BodyFeatures(normalized_x=0.25, is_valid=True), Action.MOVE_RIGHT),
        (BodyFeatures(normalized_y_displacement=0.15, vertical_velocity=0.9, is_valid=True), Action.JUMP),
        (BodyFeatures(body_height_ratio=0.70, is_valid=True), Action.CROUCH),
    ]

    for feat, expected_action in test_cases:
        res = engine.process(feat, baseline=baseline, timestamp=1.0)
        assert isinstance(res.action, Action)
        assert res.action == expected_action
        assert res.action.value in {"NONE", "JUMP", "CROUCH", "MOVE_LEFT", "MOVE_RIGHT", "PAUSE", "SELECT"}
