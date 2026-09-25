"""
Body Gesture Engine for MotionVerse.
Processes calibrated body features and emits generic Action signals.
"""

import time
from typing import Optional

from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.gestures.body.body_rules import BodyGestureThresholds, BodyMovementRules
from app.core.vision.pose.pose_features import BodyFeatures


class BodyGestureEngine:
    """
    Core rule-based engine translating body features into generic game-independent actions.
    """

    def __init__(self, thresholds: Optional[BodyGestureThresholds] = None):
        self.thresholds = thresholds or BodyGestureThresholds()
        self.rules = BodyMovementRules(self.thresholds)
        self._processing_latency_ms: float = 0.0

    def process(
        self,
        features: BodyFeatures,
        baseline: Optional[CalibrationBaseline] = None,
        timestamp: Optional[float] = None
    ) -> GestureResult:
        """
        Process the current body features against baseline calibration.
        
        Args:
            features: Current frame's extracted BodyFeatures.
            baseline: Calibrated standing baseline (if None, actions cannot be recognized).
            timestamp: Timestamp in seconds.
            
        Returns:
            GestureResult with the recognized generic Action.
        """
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else features.timestamp

        # Safety Check: If no baseline calibration exists or features are invalid, return NONE
        if baseline is None or not baseline.is_valid or not features.is_valid:
            self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return GestureResult(
                action=Action.NONE,
                confidence=0.0,
                gesture_name="NONE",
                normalized_value=0.0,
                timestamp=now,
                debug_info={"reason": "uncalibrated_or_invalid"}
            )

        # 1. Jump evaluation (Priority 1: vertical upward motion)
        if self.rules.evaluate_jump(features):
            result = GestureResult(
                action=Action.JUMP,
                confidence=min(1.0, features.normalized_y_displacement / self.thresholds.jump_displacement_threshold),
                gesture_name="JUMP",
                normalized_value=features.normalized_y_displacement,
                timestamp=now,
                debug_info={
                    "norm_y_disp": features.normalized_y_displacement,
                    "vy": features.vertical_velocity
                }
            )
            self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return result

        # 2. Crouch evaluation (Priority 2: body compression / downward hip)
        if self.rules.evaluate_crouch(features):
            conf = min(1.0, max(0.5, 1.0 - (features.body_height_ratio - self.thresholds.crouch_ratio_threshold)))
            result = GestureResult(
                action=Action.CROUCH,
                confidence=conf,
                gesture_name="CROUCH",
                normalized_value=features.body_height_ratio,
                timestamp=now,
                debug_info={
                    "body_height_ratio": features.body_height_ratio,
                    "norm_y_disp": features.normalized_y_displacement
                }
            )
            self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return result

        # 3. Horizontal movement evaluation (MOVE_LEFT / MOVE_RIGHT)
        action = self.rules.evaluate_horizontal(features)

        if action == Action.MOVE_LEFT:
            result = GestureResult(
                action=Action.MOVE_LEFT,
                confidence=min(1.0, abs(features.normalized_x) / self.thresholds.move_left_threshold),
                gesture_name="MOVE_LEFT",
                normalized_value=features.normalized_x,
                timestamp=now,
                debug_info={"norm_x": features.normalized_x, "vx": features.horizontal_velocity}
            )
        elif action == Action.MOVE_RIGHT:
            result = GestureResult(
                action=Action.MOVE_RIGHT,
                confidence=min(1.0, abs(features.normalized_x) / self.thresholds.move_right_threshold),
                gesture_name="MOVE_RIGHT",
                normalized_value=features.normalized_x,
                timestamp=now,
                debug_info={"norm_x": features.normalized_x, "vx": features.horizontal_velocity}
            )
        else:
            result = GestureResult(
                action=Action.NONE,
                confidence=1.0,
                gesture_name="NONE",
                normalized_value=features.normalized_x,
                timestamp=now,
                debug_info={"norm_x": features.normalized_x, "norm_y_disp": features.normalized_y_displacement}
            )

        self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    @property
    def processing_latency_ms(self) -> float:
        """Gesture rule processing latency in milliseconds."""
        return self._processing_latency_ms
