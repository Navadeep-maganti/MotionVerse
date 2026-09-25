"""
Unit tests for Calibration Manager and User Profiles (Step 4).
"""

import os
import tempfile
import pytest
import numpy as np

from app.core.vision.pose.pose_landmarks import Point3D, PoseLandmarks
from app.core.vision.pose.pose_features import BodyFeatures
from app.core.calibration.baseline import CalibrationBaseline
from app.core.calibration.calibration_manager import CalibrationManager, CalibrationStatus, CalibrationConfig
from app.core.calibration.user_profile import UserProfile


def create_valid_landmarks() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Point3D(0.4, 0.2, visibility=0.9),
        right_shoulder=Point3D(0.6, 0.2, visibility=0.9),
        left_hip=Point3D(0.42, 0.5, visibility=0.9),
        right_hip=Point3D(0.58, 0.5, visibility=0.9),
    )


def test_calibration_median_computation():
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=1.0, min_samples=5))
    calibrator.start()
    assert calibrator.status == CalibrationStatus.COLLECTING

    lm = create_valid_landmarks()

    # Simulate 10 frames with slight jitter around hip_x=0.50, hip_y=0.52, scale=0.60
    jitter_hx = [0.49, 0.51, 0.50, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50]
    for i, hx in enumerate(jitter_hx):
        feat = BodyFeatures(
            hip_center_x=hx,
            hip_center_y=0.52,
            shoulder_center_x=0.50,
            shoulder_center_y=0.20,
            body_scale=0.60,
            is_valid=True
        )
        status = calibrator.update(lm, feat, current_time=i * 0.15)

    assert status == CalibrationStatus.SUCCESS
    baseline = calibrator.baseline
    assert baseline is not None
    assert baseline.is_valid is True
    assert pytest.approx(baseline.hip_center_x, 0.01) == 0.50
    assert pytest.approx(baseline.hip_center_y, 0.01) == 0.52
    assert pytest.approx(baseline.body_scale, 0.01) == 0.60
    assert baseline.sample_count >= 5


def test_calibration_rejects_excessive_movement():
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=1.0, min_samples=5, max_position_std_dev=0.04))
    calibrator.start()

    lm = create_valid_landmarks()

    # Large sudden movement during calibration
    wild_hy = [0.30, 0.70, 0.20, 0.80, 0.40, 0.60, 0.20, 0.90]
    for i, hy in enumerate(wild_hy):
        feat = BodyFeatures(
            hip_center_x=0.50,
            hip_center_y=hy,
            shoulder_center_x=0.50,
            shoulder_center_y=0.20,
            body_scale=0.60,
            is_valid=True
        )
        status = calibrator.update(lm, feat, current_time=i * 0.2)

    assert status == CalibrationStatus.FAILED
    assert "Too much movement" in calibrator.error_message
    assert calibrator.baseline is None


def test_calibration_baseline_serialization():
    baseline = CalibrationBaseline(
        hip_center_x=0.512,
        hip_center_y=0.485,
        shoulder_center_x=0.500,
        shoulder_center_y=0.210,
        body_scale=0.650,
        sample_count=45,
        std_dev_hip_y=0.012,
        timestamp=12345.6
    )
    d = baseline.to_dict()
    restored = CalibrationBaseline.from_dict(d)
    assert restored.hip_center_x == baseline.hip_center_x
    assert restored.body_scale == baseline.body_scale
    assert restored.sample_count == 45


def test_user_profile_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile_path = os.path.join(tmp_dir, "test_profile.json")

        baseline = CalibrationBaseline(
            hip_center_x=0.5,
            hip_center_y=0.5,
            shoulder_center_x=0.5,
            shoulder_center_y=0.2,
            body_scale=0.7
        )
        profile = UserProfile(user_id="player1", baseline=baseline)

        assert profile.save_to_file(profile_path) is True
        loaded = UserProfile.load_from_file(profile_path)
        assert loaded is not None
        assert loaded.user_id == "player1"
        assert loaded.baseline is not None
        assert loaded.baseline.body_scale == 0.7
