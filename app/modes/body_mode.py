"""
Body Mode Controller module for MotionVerse.
Encapsulates the end-to-end full-body recognition pipeline and emits clean generic Action outputs.
"""

from typing import Optional, Tuple
import numpy as np

from app.core.calibration.baseline import CalibrationBaseline
from app.core.calibration.calibration_manager import CalibrationManager
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.vision.pose.pose_detector import PoseDetector
from app.core.vision.pose.pose_features import BodyFeatureExtractor, BodyFeatures
from app.core.vision.pose.pose_landmarks import PoseLandmarks


class BodyModeController:
    """
    High-level full-body motion controller.
    Coordinates vision inference, feature extraction, calibration, and gesture recognition.
    Strictly game-independent and emits only standard generic Action values.
    """

    def __init__(
        self,
        pose_detector: Optional[PoseDetector] = None,
        feature_extractor: Optional[BodyFeatureExtractor] = None,
        calibration_manager: Optional[CalibrationManager] = None,
        gesture_engine: Optional[BodyGestureEngine] = None
    ):
        self.detector = pose_detector or PoseDetector()
        self.extractor = feature_extractor or BodyFeatureExtractor()
        self.calibrator = calibration_manager or CalibrationManager()
        self.engine = gesture_engine or BodyGestureEngine()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Tuple[GestureResult, Optional[PoseLandmarks], BodyFeatures]:
        """
        Process a single video frame end-to-end.
        
        Args:
            frame: BGR image from webcam.
            timestamp: High-resolution frame timestamp.
            
        Returns:
            Tuple of (GestureResult, PoseLandmarks or None, BodyFeatures)
        """
        # 1. Pose Detection
        landmarks = self.detector.detect(frame, timestamp=timestamp)

        # 2. Get active calibration baseline
        baseline = self.calibrator.baseline
        base_hip = (baseline.hip_center_x, baseline.hip_center_y) if baseline else None
        base_scale = baseline.body_scale if baseline else None

        # 3. Feature Extraction
        features = self.extractor.extract(
            landmarks,
            baseline_hip=base_hip,
            baseline_scale=base_scale,
            timestamp=timestamp
        )

        # 4. Calibration update if in progress
        self.calibrator.update(landmarks, features, current_time=timestamp)

        # 5. Gesture Recognition & State Machine
        gesture_result = self.engine.process(
            features,
            baseline=baseline,
            timestamp=timestamp
        )

        return gesture_result, landmarks, features

    @property
    def baseline(self) -> Optional[CalibrationBaseline]:
        return self.calibrator.baseline

    def start_calibration(self) -> None:
        """Begin standing calibration sequence."""
        self.calibrator.start()

    def reset(self) -> None:
        """Reset all pipeline internal states."""
        self.calibrator.reset()
        self.extractor.reset()
        self.engine.reset()
