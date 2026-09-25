"""
Body Mode Controller module for MotionVerse.
Encapsulates the end-to-end full-body recognition pipeline and optional game input dispatch.
Strictly game-independent at the recognition core, with decoupled game mapping and safety controls.
"""

from enum import Enum
from typing import Optional, Tuple, Dict, Any
import numpy as np

from app.core.calibration.baseline import CalibrationBaseline
from app.core.calibration.calibration_manager import CalibrationManager
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.vision.pose.pose_detector import PoseDetector
from app.core.vision.pose.pose_features import BodyFeatureExtractor, BodyFeatures
from app.core.vision.pose.pose_landmarks import PoseLandmarks
from app.games.game_mapper import GameMapper
from app.games.game_profile import GameProfile
from app.input.keyboard_controller import KeyboardController


class ControllerMode(Enum):
    """Execution modes for BodyModeController."""
    BODY_DEBUG = "BODY_DEBUG"  # Recognition & Telemetry only (no game inputs)
    BODY_GAME = "BODY_GAME"    # Game input dispatch enabled when ARMED


class BodyModeController:
    """
    High-level full-body motion controller.
    Coordinates vision inference, feature extraction, calibration, gesture recognition,
    and optional game input mapping with explicit safety arming.
    """

    def __init__(
        self,
        pose_detector: Optional[PoseDetector] = None,
        feature_extractor: Optional[BodyFeatureExtractor] = None,
        calibration_manager: Optional[CalibrationManager] = None,
        gesture_engine: Optional[BodyGestureEngine] = None,
        game_mapper: Optional[GameMapper] = None,
        keyboard_controller: Optional[KeyboardController] = None,
        mode: ControllerMode = ControllerMode.BODY_GAME
    ):
        self.detector = pose_detector or PoseDetector()
        self.extractor = feature_extractor or BodyFeatureExtractor()
        self.calibrator = calibration_manager or CalibrationManager()
        self.engine = gesture_engine or BodyGestureEngine()

        # Game & Input Integration Layer
        self.mapper = game_mapper or GameMapper()
        self.keyboard = keyboard_controller or KeyboardController()
        self.mode = mode

        # Safety Arming State
        self._is_armed: bool = False
        self._last_mapped_key: Optional[str] = None
        self._last_dispatch_status: str = "IDLE"

    @property
    def is_armed(self) -> bool:
        """True if keyboard input dispatch is currently armed and active."""
        return self._is_armed

    def arm(self) -> None:
        """Arm keyboard input dispatch for active game control."""
        self._is_armed = True
        self._last_dispatch_status = "ARMED"

    def disarm(self) -> None:
        """Disarm keyboard input dispatch. Recognition continues safely."""
        self._is_armed = False
        self._last_dispatch_status = "DISARMED"
        self.keyboard.release_all()

    def toggle_arm(self) -> bool:
        """Toggle between ARMED and DISARMED states."""
        if self._is_armed:
            self.disarm()
        else:
            self.arm()
        return self._is_armed

    def emergency_stop(self) -> None:
        """Immediate safety stop: disarm and release all active keys."""
        self.disarm()
        self.keyboard.release_all()

    def set_game_profile(self, profile: GameProfile) -> None:
        """Set or update the active game profile."""
        self.mapper.set_profile(profile)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Tuple[GestureResult, Optional[PoseLandmarks], BodyFeatures, Optional[str], str]:
        """
        Process a single video frame end-to-end.
        
        Args:
            frame: BGR image from webcam.
            timestamp: High-resolution frame timestamp.
            
        Returns:
            Tuple of (GestureResult, PoseLandmarks or None, BodyFeatures, MappedKey or None, DispatchStatus)
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

        # 5. Gesture Recognition & Discrete State Machine
        gesture_result = self.engine.process(
            features,
            baseline=baseline,
            timestamp=timestamp
        )

        mapped_key = None
        dispatch_status = "IDLE" if not self._is_armed else "ARMED"

        # 6. Game Mapping & Input Dispatch Layer (Decoupled)
        if self.mode == ControllerMode.BODY_GAME and self.mapper.profile is not None:
            if gesture_result.action != Action.NONE:
                mapped_key = self.mapper.map_action(gesture_result.action)
                self._last_mapped_key = mapped_key

                if self._is_armed and mapped_key:
                    # Dispatch discrete key tap
                    success = self.keyboard.tap(mapped_key)
                    dispatch_status = f"SENT: {mapped_key.upper()}" if success else "FAILED"
                elif not self._is_armed:
                    dispatch_status = f"BLOCKED (DISARMED): {mapped_key.upper()}"
            else:
                if self._is_armed:
                    dispatch_status = "ARMED (Listening)"

        self._last_dispatch_status = dispatch_status

        return gesture_result, landmarks, features, mapped_key, dispatch_status

    @property
    def baseline(self) -> Optional[CalibrationBaseline]:
        return self.calibrator.baseline

    def start_calibration(self) -> None:
        """Begin standing calibration sequence."""
        self.calibrator.start()

    def reset(self) -> None:
        """Reset all pipeline internal states and safety release keys."""
        self.emergency_stop()
        self.calibrator.reset()
        self.extractor.reset()
        self.engine.reset()
