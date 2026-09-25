"""
Integration tests for the complete Body Gesture Pipeline (Phase 1).
Tests end-to-end flow from landmarks to generic Action dispatch.
"""

import pytest
import numpy as np

from app.core.vision.pose.pose_landmarks import Point3D, PoseLandmarks
from app.core.vision.pose.pose_features import BodyFeatureExtractor
from app.core.calibration.calibration_manager import CalibrationManager, CalibrationStatus, CalibrationConfig
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture_state import GestureState


def make_landmarks(
    hip_x: float = 0.5,
    hip_y: float = 0.5,
    shoulder_y: float = 0.2,
    ankle_y: float = 0.9
) -> PoseLandmarks:
    """Helper to generate standardized mock landmarks."""
    return PoseLandmarks(
        left_shoulder=Point3D(hip_x - 0.1, shoulder_y, visibility=0.9),
        right_shoulder=Point3D(hip_x + 0.1, shoulder_y, visibility=0.9),
        left_hip=Point3D(hip_x - 0.08, hip_y, visibility=0.9),
        right_hip=Point3D(hip_x + 0.08, hip_y, visibility=0.9),
        left_knee=Point3D(hip_x - 0.08, (hip_y + ankle_y) / 2, visibility=0.9),
        right_knee=Point3D(hip_x + 0.08, (hip_y + ankle_y) / 2, visibility=0.9),
        left_ankle=Point3D(hip_x - 0.08, ankle_y, visibility=0.9),
        right_ankle=Point3D(hip_x + 0.08, ankle_y, visibility=0.9)
    )


def test_full_pipeline_calibration_and_actions():
    """Verify full pipeline: Calibrates neutral standing, then recognizes LEFT, RIGHT, JUMP, CROUCH."""
    extractor = BodyFeatureExtractor()
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=1.0, min_samples=5))
    engine = BodyGestureEngine()

    # 1. Calibration phase
    calibrator.start()
    for i in range(10):
        t = i * 0.15
        lm = make_landmarks(hip_x=0.50, hip_y=0.50, shoulder_y=0.20, ankle_y=0.90)
        feat = extractor.extract(lm, timestamp=t)
        calibrator.update(lm, feat, current_time=t)

    assert calibrator.status == CalibrationStatus.SUCCESS
    baseline = calibrator.baseline
    assert baseline is not None
    assert pytest.approx(baseline.hip_center_x, 0.01) == 0.50
    assert pytest.approx(baseline.body_scale, 0.01) == 0.70

    # 2. Test Neutral (Action.NONE)
    t = 2.0
    lm_neutral = make_landmarks(hip_x=0.50, hip_y=0.50, shoulder_y=0.20, ankle_y=0.90)
    feat_neutral = extractor.extract(lm_neutral, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    res_neutral = engine.process(feat_neutral, baseline=baseline, timestamp=t)
    assert res_neutral.action == Action.NONE

    # 3. Test Shift Right (Action.MOVE_RIGHT)
    # hip_x moves to 0.70 -> (0.70 - 0.50) / 0.70 = +0.2857 (after 0.75 EMA -> 0.214 > trigger 0.20)
    t = 2.1
    lm_right = make_landmarks(hip_x=0.70, hip_y=0.50, shoulder_y=0.20, ankle_y=0.90)
    feat_right = extractor.extract(lm_right, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    res_right = engine.process(feat_right, baseline=baseline, timestamp=t)
    assert res_right.action == Action.MOVE_RIGHT

    # 4. Return to center, then Shift Left (Action.MOVE_LEFT)
    t = 2.2
    feat_neutral2 = extractor.extract(lm_neutral, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    engine.process(feat_neutral2, baseline=baseline, timestamp=t)

    t = 2.25
    lm_left = make_landmarks(hip_x=0.30, hip_y=0.50, shoulder_y=0.20, ankle_y=0.90)
    feat_left = extractor.extract(lm_left, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    res_left = engine.process(feat_left, baseline=baseline, timestamp=t)
    assert res_left.action == Action.MOVE_LEFT

    # 5. Test Jump (Action.JUMP)
    # Return to center first
    t = 2.3
    extractor.extract(lm_neutral, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    engine.process(feat_neutral, baseline=baseline, timestamp=t)

    # Explosive upward movement: hip_y 0.50 -> 0.40 (D_t = 0.14) in 0.05s (V_y = 2.8/s)
    t = 2.35
    lm_jump = make_landmarks(hip_x=0.50, hip_y=0.40, shoulder_y=0.10, ankle_y=0.80)
    feat_jump = extractor.extract(lm_jump, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    res_jump = engine.process(feat_jump, baseline=baseline, timestamp=t)
    assert res_jump.action == Action.JUMP
    assert res_jump.debug_info.get("state") == GestureState.TRIGGERED.value

    # 6. Test Crouch (Action.CROUCH)
    # User squats down: body scale compresses to 0.50 (ratio = 0.50 / 0.70 = 0.71)
    t = 3.0  # (after jump cooldown has expired)
    lm_crouch = make_landmarks(hip_x=0.50, hip_y=0.62, shoulder_y=0.35, ankle_y=0.85)
    feat_crouch = extractor.extract(lm_crouch, baseline_hip=(baseline.hip_center_x, baseline.hip_center_y), baseline_scale=baseline.body_scale, timestamp=t)
    res_crouch = engine.process(feat_crouch, baseline=baseline, timestamp=t)
    assert res_crouch.action == Action.CROUCH
    assert res_crouch.debug_info.get("state") == GestureState.HOLDING.value
