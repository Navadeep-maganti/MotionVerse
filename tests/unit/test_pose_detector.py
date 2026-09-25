"""
Unit tests for Pose Detection & Landmarks (Step 2).
"""

import numpy as np
import pytest

from app.core.vision.pose.pose_landmarks import Point3D, PoseLandmarks
from app.core.vision.pose.pose_detector import PoseDetector, PoseDetectorConfig


def test_point3d_to_pixel():
    pt = Point3D(x=0.5, y=0.25, z=0.0, visibility=0.9)
    px, py = pt.to_pixel(640, 480)
    assert px == 320
    assert py == 120
    assert pt.is_visible(0.8) is True
    assert pt.is_visible(0.95) is False


def test_pose_landmarks_core_check():
    # Incomplete landmarks
    lm = PoseLandmarks(
        left_shoulder=Point3D(0.4, 0.3, visibility=0.9),
        right_shoulder=Point3D(0.6, 0.3, visibility=0.9),
        left_hip=None,
        right_hip=None
    )
    assert lm.has_core_landmarks() is False

    # Complete core landmarks
    lm.left_hip = Point3D(0.45, 0.6, visibility=0.85)
    lm.right_hip = Point3D(0.55, 0.6, visibility=0.85)
    assert lm.has_core_landmarks() is True


def test_pose_detector_blank_frame():
    """Verify detector processes empty/blank frame without crashing and returns None."""
    detector = PoseDetector(PoseDetectorConfig(model_complexity=0))
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(blank_frame)
    assert result is None
    assert detector.is_person_detected is False
    assert detector.inference_latency_ms >= 0.0

    # Test drawing on empty result
    drawn = detector.draw_skeleton(blank_frame.copy(), result)
    assert drawn.shape == blank_frame.shape

    detector.close()


def test_pose_detector_draw_skeleton_with_landmarks():
    detector = PoseDetector(PoseDetectorConfig(model_complexity=0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = PoseLandmarks(
        left_shoulder=Point3D(0.4, 0.3, visibility=0.9),
        right_shoulder=Point3D(0.6, 0.3, visibility=0.9),
        left_hip=Point3D(0.4, 0.6, visibility=0.9),
        right_hip=Point3D(0.6, 0.6, visibility=0.9)
    )

    drawn = detector.draw_skeleton(frame, landmarks)
    # Ensure drawing produced non-zero pixels
    assert np.count_nonzero(drawn) > 0
    detector.close()
