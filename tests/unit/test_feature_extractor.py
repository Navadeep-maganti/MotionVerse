"""
Unit tests for BodyFeatureExtractor (Step 3).
"""

import pytest
from app.core.vision.pose.pose_landmarks import Point3D, PoseLandmarks
from app.core.vision.pose.pose_features import BodyFeatureExtractor, BodyFeatures


def create_mock_landmarks(
    shoulder_y: float = 0.2,
    hip_y: float = 0.5,
    hip_x: float = 0.5,
    ankle_y: float = 0.9,
    include_ankles: bool = True
) -> PoseLandmarks:
    """Helper to create standardized mock pose landmarks."""
    lm = PoseLandmarks(
        left_shoulder=Point3D(x=hip_x - 0.1, y=shoulder_y, visibility=0.9),
        right_shoulder=Point3D(x=hip_x + 0.1, y=shoulder_y, visibility=0.9),
        left_hip=Point3D(x=hip_x - 0.08, y=hip_y, visibility=0.9),
        right_hip=Point3D(x=hip_x + 0.08, y=hip_y, visibility=0.9),
        left_knee=Point3D(x=hip_x - 0.08, y=(hip_y + 0.2), visibility=0.9),
        right_knee=Point3D(x=hip_x + 0.08, y=(hip_y + 0.2), visibility=0.9),
    )
    if include_ankles:
        lm.left_ankle = Point3D(x=hip_x - 0.08, y=ankle_y, visibility=0.9)
        lm.right_ankle = Point3D(x=hip_x + 0.08, y=ankle_y, visibility=0.9)
    return lm


def test_hip_and_shoulder_center():
    extractor = BodyFeatureExtractor()
    lm = create_mock_landmarks(shoulder_y=0.2, hip_y=0.5, hip_x=0.5)

    hip_center = extractor.calculate_hip_center(lm)
    shoulder_center = extractor.calculate_shoulder_center(lm)

    assert hip_center is not None
    assert pytest.approx(hip_center[0], 0.001) == 0.5
    assert pytest.approx(hip_center[1], 0.001) == 0.5

    assert shoulder_center is not None
    assert pytest.approx(shoulder_center[0], 0.001) == 0.5
    assert pytest.approx(shoulder_center[1], 0.001) == 0.2


def test_body_scale_calculation_and_fallback():
    extractor = BodyFeatureExtractor()

    # Full body with ankles: scale = |0.2 - 0.9| = 0.7
    lm_full = create_mock_landmarks(shoulder_y=0.2, hip_y=0.5, ankle_y=0.9, include_ankles=True)
    sc = extractor.calculate_shoulder_center(lm_full)
    hc = extractor.calculate_hip_center(lm_full)
    scale_full = extractor.calculate_body_scale(lm_full, sc, hc)
    assert pytest.approx(scale_full, 0.01) == 0.7

    # Upper body only (no ankles, fallback to knees: knee_y = 0.7, |0.7-0.2|*1.45 = 0.725)
    lm_no_ankles = create_mock_landmarks(shoulder_y=0.2, hip_y=0.5, include_ankles=False)
    scale_no_ankles = extractor.calculate_body_scale(lm_no_ankles, sc, hc)
    assert scale_no_ankles > 0.5


def test_normalized_coordinates_and_displacement():
    extractor = BodyFeatureExtractor()
    lm = create_mock_landmarks(shoulder_y=0.2, hip_y=0.5, hip_x=0.6, ankle_y=0.9)

    baseline_hip = (0.5, 0.5)
    baseline_scale = 0.5

    # Neutral frame with baseline:
    # norm_x = (0.6 - 0.5) / 0.5 = +0.2
    # norm_y_disp = (0.5 - 0.5) / 0.5 = 0.0
    features = extractor.extract(lm, baseline_hip=baseline_hip, baseline_scale=baseline_scale, timestamp=1.0)

    assert features.is_valid is True
    assert pytest.approx(features.normalized_x, 0.01) == 0.20
    assert pytest.approx(features.normalized_y_displacement, 0.01) == 0.0

    # Person jumps (hip_y moves UP from 0.5 to 0.4 in image coordinates)
    lm_jump = create_mock_landmarks(shoulder_y=0.1, hip_y=0.4, hip_x=0.5, ankle_y=0.8)
    features_jump = extractor.extract(lm_jump, baseline_hip=baseline_hip, baseline_scale=baseline_scale, timestamp=1.1)

    # Displacement = (0.5 - 0.4) / 0.5 = +0.20 (positive for upward jump)
    assert pytest.approx(features_jump.normalized_y_displacement, 0.01) == 0.20


def test_velocity_calculation():
    extractor = BodyFeatureExtractor()
    baseline_hip = (0.5, 0.5)
    baseline_scale = 0.5

    # Frame 1 at t = 1.0 (neutral)
    lm1 = create_mock_landmarks(hip_x=0.5, hip_y=0.5)
    f1 = extractor.extract(lm1, baseline_hip=baseline_hip, baseline_scale=baseline_scale, timestamp=1.0)
    assert f1.horizontal_velocity == 0.0
    assert f1.vertical_velocity == 0.0

    # Frame 2 at t = 1.1 (moved right: hip_x 0.5 -> 0.6, norm_x 0.0 -> 0.2 in dt = 0.1s)
    # Vx = (0.2 - 0.0) / 0.1 = +2.0/s
    lm2 = create_mock_landmarks(hip_x=0.6, hip_y=0.5)
    f2 = extractor.extract(lm2, baseline_hip=baseline_hip, baseline_scale=baseline_scale, timestamp=1.1)
    assert pytest.approx(f2.horizontal_velocity, 0.05) == 2.0


def test_body_height_ratio_crouch():
    extractor = BodyFeatureExtractor()
    baseline_hip = (0.5, 0.5)
    baseline_scale = 0.70  # Baseline standing scale

    # Person crouches: scale shrinks from 0.70 to 0.49
    lm_crouch = create_mock_landmarks(shoulder_y=0.35, hip_y=0.6, ankle_y=0.84)
    f_crouch = extractor.extract(lm_crouch, baseline_hip=baseline_hip, baseline_scale=baseline_scale, timestamp=1.0)

    # Height ratio = 0.49 / 0.70 = 0.70
    assert pytest.approx(f_crouch.body_height_ratio, 0.02) == 0.70


def test_invalid_landmarks_handling():
    extractor = BodyFeatureExtractor()
    f_none = extractor.extract(None)
    assert f_none.is_valid is False
