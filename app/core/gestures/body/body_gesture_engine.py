"""
Body Gesture Engine for MotionVerse.
Processes calibrated body features and emits generic Action signals with temporal filtering,
Schmitt-trigger hysteresis, and discrete/continuous state machine transitions.
"""

import time
from typing import Optional

from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.gestures.common.gesture_state import GestureState, GestureStateMachine
from app.core.gestures.body.body_rules import BodyGestureThresholds, BodyMovementRules
from app.core.gestures.filters.temporal_filter import FeatureTemporalFilter
from app.core.vision.pose.pose_features import BodyFeatures


class BodyGestureEngine:
    """
    Core rule-based engine translating body features into generic game-independent actions.
    Combines lightweight temporal filtering, Schmitt-trigger hysteresis, and state machine debouncing.
    """

    def __init__(
        self,
        thresholds: Optional[BodyGestureThresholds] = None,
        enable_temporal_filtering: bool = True,
        jump_cooldown_sec: float = 0.45
    ):
        self.thresholds = thresholds or BodyGestureThresholds()
        self.rules = BodyMovementRules(self.thresholds)
        self.enable_temporal_filtering = enable_temporal_filtering
        self.temporal_filter = FeatureTemporalFilter()
        self.state_machine = GestureStateMachine(jump_cooldown_sec=jump_cooldown_sec)

        # State tracking for hysteresis
        self._current_horizontal_action: Action = Action.NONE
        self._is_currently_crouched: bool = False

        self._processing_latency_ms: float = 0.0
        self._filtering_latency_ms: float = 0.0

    def reset(self) -> None:
        """Reset internal filter, hysteresis, and state machine states."""
        self.temporal_filter.reset()
        self.state_machine.reset()
        self._current_horizontal_action = Action.NONE
        self._is_currently_crouched = False

    def process(
        self,
        features: BodyFeatures,
        baseline: Optional[CalibrationBaseline] = None,
        timestamp: Optional[float] = None
    ) -> GestureResult:
        """
        Process current body features against baseline calibration.
        """
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else features.timestamp

        # Safety Check: If uncalibrated or tracking lost, reset states and return NONE
        if baseline is None or not baseline.is_valid or not features.is_valid:
            self.reset()
            self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return GestureResult(
                action=Action.NONE,
                confidence=0.0,
                gesture_name="NONE",
                normalized_value=0.0,
                timestamp=now,
                debug_info={"reason": "uncalibrated_or_invalid", "state": GestureState.IDLE.value}
            )

        # 1. Apply Temporal Filtering
        if self.enable_temporal_filtering:
            active_features = self.temporal_filter.filter(features)
            self._filtering_latency_ms = self.temporal_filter.latency_ms
        else:
            active_features = features
            self._filtering_latency_ms = 0.0

        # Determine Candidate Action via Priority Rules
        candidate_action = Action.NONE
        norm_val = 0.0
        confidence = 1.0

        # Priority 1: Jump (vertical upward motion)
        if self.rules.evaluate_jump(active_features):
            candidate_action = Action.JUMP
            norm_val = active_features.normalized_y_displacement
            confidence = min(1.0, active_features.normalized_y_displacement / self.thresholds.jump_displacement_threshold)
            self._is_currently_crouched = False
            self._current_horizontal_action = Action.NONE

        else:
            # Priority 2: Crouch with Hysteresis
            is_crouched = self.rules.evaluate_crouch_with_hysteresis(
                active_features,
                self._is_currently_crouched
            )
            self._is_currently_crouched = is_crouched

            if is_crouched:
                candidate_action = Action.CROUCH
                norm_val = active_features.body_height_ratio
                confidence = min(1.0, max(0.5, 1.0 - (active_features.body_height_ratio - self.thresholds.crouch_trigger_ratio)))
                self._current_horizontal_action = Action.NONE
            else:
                # Priority 3: Horizontal Movement with Hysteresis
                horiz_action = self.rules.evaluate_horizontal_with_hysteresis(
                    active_features,
                    self._current_horizontal_action
                )
                self._current_horizontal_action = horiz_action
                candidate_action = horiz_action
                norm_val = active_features.normalized_x
                if horiz_action == Action.MOVE_LEFT:
                    confidence = min(1.0, abs(active_features.normalized_x) / self.thresholds.move_left_trigger)
                elif horiz_action == Action.MOVE_RIGHT:
                    confidence = min(1.0, abs(active_features.normalized_x) / self.thresholds.move_right_trigger)

        # 2. Pass Candidate Action through Gesture State Machine & Cooldown
        dispatched_action, current_state = self.state_machine.update(candidate_action, timestamp=now)

        self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return GestureResult(
            action=dispatched_action,
            confidence=confidence,
            gesture_name=dispatched_action.value,
            normalized_value=norm_val,
            timestamp=now,
            debug_info={
                "candidate_action": candidate_action.value,
                "state": current_state.value,
                "norm_x": active_features.normalized_x,
                "norm_y_disp": active_features.normalized_y_displacement,
                "height_ratio": active_features.body_height_ratio,
                "cooldown_remaining": self.state_machine.cooldown_tracker.remaining_cooldown_sec(Action.JUMP, timestamp=now)
            }
        )

    @property
    def processing_latency_ms(self) -> float:
        return self._processing_latency_ms

    @property
    def filtering_latency_ms(self) -> float:
        return self._filtering_latency_ms

    @property
    def current_state(self) -> GestureState:
        return self.state_machine.current_state
