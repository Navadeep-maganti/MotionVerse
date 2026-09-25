"""
Body Gesture Engine for MotionVerse.
Implements discrete event semantics:
- Dynamic Horizontal Reference (relative displacement + settling + re-referencing).
- Rapid Consecutive Step Tracking with robust double-trigger prevention.
- Discrete Jump (1 event + landing recovery).
- Discrete Crouch (1 event + standing recovery).
Strictly game-independent and emits only one Action event per physical gesture.
"""

import time
from typing import Optional, Dict, Any

from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture import GestureResult
from app.core.gestures.common.gesture_state import GestureState, HorizontalState, VerticalState
from app.core.gestures.body.body_rules import BodyGestureThresholds, BodyMovementRules
from app.core.gestures.filters.temporal_filter import FeatureTemporalFilter
from app.core.gestures.filters.cooldown import CooldownTracker
from app.core.vision.pose.pose_features import BodyFeatures


class BodyGestureEngine:
    """
    Core rule-based engine translating body kinematics into discrete Action events.
    
    Discrete Semantics:
    - Each physical gesture triggers EXACTLY ONE Action event on the leading edge.
    - Horizontal movement uses a dynamic reference X_ref.
    - Robust double-trigger prevention prevents single-stroke overshoot from triggering twice.
    - Vertical gestures (Jump / Crouch) require landing/standing recovery before re-arming.
    """

    def __init__(
        self,
        thresholds: Optional[BodyGestureThresholds] = None,
        enable_temporal_filtering: bool = True,
        jump_cooldown_sec: float = 0.40,
        horizontal_settle_duration_sec: Optional[float] = None,
        min_consecutive_step_interval_sec: float = 0.22
    ):
        self.thresholds = thresholds or BodyGestureThresholds()
        if horizontal_settle_duration_sec is not None:
            self.thresholds.horizontal_settle_duration_sec = horizontal_settle_duration_sec

        self.rules = BodyMovementRules(self.thresholds)
        self.enable_temporal_filtering = enable_temporal_filtering
        self.temporal_filter = FeatureTemporalFilter()
        self.cooldown_tracker = CooldownTracker(default_cooldown_sec=jump_cooldown_sec)
        self.min_consecutive_step_interval_sec = min_consecutive_step_interval_sec

        # Dynamic Horizontal State
        self._horizontal_reference_x: Optional[float] = None
        self._horizontal_state: HorizontalState = HorizontalState.READY
        self._settle_start_time: Optional[float] = None
        self._pending_direction: Action = Action.NONE
        self._last_trigger_x: Optional[float] = None
        self._last_trigger_time: float = 0.0

        # Vertical State (Jump / Crouch)
        self._vertical_state: VerticalState = VerticalState.READY
        self._active_vertical_action: Action = Action.NONE

        # Performance Telemetry
        self._processing_latency_ms: float = 0.0
        self._filtering_latency_ms: float = 0.0

    def reset(self) -> None:
        """Reset internal filter, dynamic reference, and state machines."""
        self.temporal_filter.reset()
        self.cooldown_tracker.reset()
        self._horizontal_reference_x = None
        self._horizontal_state = HorizontalState.READY
        self._settle_start_time = None
        self._pending_direction = Action.NONE
        self._last_trigger_x = None
        self._last_trigger_time = 0.0

        self._vertical_state = VerticalState.READY
        self._active_vertical_action = Action.NONE

    def set_reference_x(self, ref_x: float) -> None:
        """Manually set or reset the horizontal reference point."""
        self._horizontal_reference_x = ref_x
        self._horizontal_state = HorizontalState.READY
        self._settle_start_time = None
        self._last_trigger_x = ref_x

    def process(
        self,
        features: BodyFeatures,
        baseline: Optional[CalibrationBaseline] = None,
        timestamp: Optional[float] = None
    ) -> GestureResult:
        """
        Process calibrated body features and emit discrete generic Action events.
        
        Args:
            features: Current frame BodyFeatures.
            baseline: Active standing calibration baseline.
            timestamp: High-resolution timestamp.
            
        Returns:
            GestureResult with exactly one Action event on gesture trigger, else Action.NONE.
        """
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else features.timestamp

        # Safety Check: Uncalibrated or tracking lost -> Return NONE
        if baseline is None or not baseline.is_valid or not features.is_valid:
            self.reset()
            self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return GestureResult(
                action=Action.NONE,
                confidence=0.0,
                gesture_name="NONE",
                normalized_value=0.0,
                timestamp=now,
                debug_info={"reason": "uncalibrated_or_invalid", "state": GestureState.READY.value}
            )

        # Initialize horizontal reference to calibrated neutral hip position if not set
        if self._horizontal_reference_x is None:
            self._horizontal_reference_x = baseline.hip_center_x
            self._last_trigger_x = baseline.hip_center_x

        # 1. Apply Temporal Filtering
        if self.enable_temporal_filtering:
            active_features = self.temporal_filter.filter(features)
            self._filtering_latency_ms = self.temporal_filter.latency_ms
        else:
            active_features = features
            self._filtering_latency_ms = 0.0

        dispatched_action = Action.NONE
        dispatched_state = GestureState.READY
        normalized_val = 0.0
        confidence = 1.0

        # =========================================================================
        # 2. VERTICAL EVALUATION (Priority 1: JUMP & CROUCH)
        # =========================================================================
        # A. JUMP & CROUCH TRIGGER
        if self._vertical_state == VerticalState.READY:
            if self.rules.evaluate_jump_trigger(active_features) and not self.cooldown_tracker.is_in_cooldown(Action.JUMP, timestamp=now):
                # Trigger JUMP once
                dispatched_action = Action.JUMP
                dispatched_state = GestureState.TRIGGERED
                self._vertical_state = VerticalState.RECOVERING
                self._active_vertical_action = Action.JUMP
                self.cooldown_tracker.trigger(Action.JUMP, timestamp=now)
                normalized_val = active_features.normalized_y_displacement
                confidence = min(1.0, active_features.normalized_y_displacement / max(0.01, self.thresholds.jump_displacement_threshold))

            elif self.rules.evaluate_crouch_trigger(active_features):
                # Trigger CROUCH once
                dispatched_action = Action.CROUCH
                dispatched_state = GestureState.TRIGGERED
                self._vertical_state = VerticalState.RECOVERING
                self._active_vertical_action = Action.CROUCH
                normalized_val = active_features.body_height_ratio
                confidence = min(1.0, max(0.5, 1.0 - (active_features.body_height_ratio - self.thresholds.crouch_trigger_ratio)))

        elif self._vertical_state == VerticalState.RECOVERING:
            dispatched_action = Action.NONE
            dispatched_state = GestureState.RECOVERING

            # Check recovery conditions
            if self._active_vertical_action == Action.JUMP:
                # Landed and cooldown expired
                is_landed = self.rules.evaluate_jump_recovered(active_features)
                cooldown_done = not self.cooldown_tracker.is_in_cooldown(Action.JUMP, timestamp=now)
                if is_landed and cooldown_done:
                    self._vertical_state = VerticalState.READY
                    self._active_vertical_action = Action.NONE

            elif self._active_vertical_action == Action.CROUCH:
                # Stood back up to standing posture
                is_standing = self.rules.evaluate_crouch_recovered(active_features)
                if is_standing:
                    self._vertical_state = VerticalState.READY
                    self._active_vertical_action = Action.NONE

        # =========================================================================
        # 3. HORIZONTAL EVALUATION (Priority 2: Dynamic Reference LEFT / RIGHT)
        # =========================================================================
        # Determine actual hip x or reconstruct from normalized_x for synthetic features
        effective_hip_x = active_features.hip_center_x
        if effective_hip_x == 0.0 and active_features.normalized_x != 0.0:
            effective_hip_x = baseline.hip_center_x + active_features.normalized_x * baseline.body_scale

        delta_x = self.rules.calculate_horizontal_delta(
            active_features,
            reference_x=self._horizontal_reference_x,
            body_scale=baseline.body_scale
        )

        # Only evaluate horizontal movement if vertical gesture didn't fire this frame
        if dispatched_action == Action.NONE and self._vertical_state == VerticalState.READY:
            if self._horizontal_state == HorizontalState.READY:
                horiz_trig = self.rules.evaluate_horizontal_trigger(delta_x)
                if horiz_trig != Action.NONE:
                    # Trigger MOVE_LEFT or MOVE_RIGHT once
                    dispatched_action = horiz_trig
                    dispatched_state = GestureState.TRIGGERED
                    self._horizontal_state = HorizontalState.SETTLING
                    self._pending_direction = horiz_trig
                    self._settle_start_time = None
                    self._last_trigger_x = effective_hip_x
                    self._last_trigger_time = now
                    normalized_val = delta_x
                    confidence = min(1.0, abs(delta_x) / max(0.01, self.thresholds.move_right_trigger))

            elif self._horizontal_state == HorizontalState.SETTLING:
                dispatched_action = Action.NONE
                dispatched_state = GestureState.SETTLING

                # Check incremental displacement from last triggered step
                step_ref_x = self._last_trigger_x if self._last_trigger_x is not None else self._horizontal_reference_x
                incremental_delta_x = (effective_hip_x - step_ref_x) / baseline.body_scale
                time_since_last_trigger = now - self._last_trigger_time

                # A. Rapid Consecutive Step Detection (requires >= min_consecutive_step_interval_sec to avoid single-push double firing)
                if time_since_last_trigger >= self.min_consecutive_step_interval_sec:
                    if self._pending_direction == Action.MOVE_RIGHT and incremental_delta_x >= (self.thresholds.move_right_trigger + 0.02):
                        dispatched_action = Action.MOVE_RIGHT
                        dispatched_state = GestureState.TRIGGERED
                        self._last_trigger_x = effective_hip_x
                        self._last_trigger_time = now
                        self._horizontal_reference_x = effective_hip_x
                        self._settle_start_time = None
                        normalized_val = incremental_delta_x
                    elif self._pending_direction == Action.MOVE_LEFT and incremental_delta_x <= -(self.thresholds.move_left_trigger + 0.02):
                        dispatched_action = Action.MOVE_LEFT
                        dispatched_state = GestureState.TRIGGERED
                        self._last_trigger_x = effective_hip_x
                        self._last_trigger_time = now
                        self._horizontal_reference_x = effective_hip_x
                        self._settle_start_time = None
                        normalized_val = incremental_delta_x

                if dispatched_action == Action.NONE:
                    # B. Snappy Reversal Detection (user actively moves in the opposite direction)
                    is_reversal = False
                    if self._pending_direction == Action.MOVE_RIGHT and active_features.horizontal_velocity < -0.15:
                        is_reversal = True
                    elif self._pending_direction == Action.MOVE_LEFT and active_features.horizontal_velocity > 0.15:
                        is_reversal = True

                    # C. Velocity settling detection
                    is_settled = self.rules.is_horizontal_settled(active_features)

                    if is_reversal:
                        # Immediately update reference to current position so counter-gesture triggers without lag
                        self._horizontal_reference_x = effective_hip_x
                        self._last_trigger_x = effective_hip_x
                        self._horizontal_state = HorizontalState.READY
                        self._pending_direction = Action.NONE
                        self._settle_start_time = None
                    elif is_settled:
                        if self._settle_start_time is None:
                            self._settle_start_time = now
                        elif (now - self._settle_start_time) >= self.thresholds.horizontal_settle_duration_sec:
                            # Movement settled -> Update reference to new stabilized position!
                            self._horizontal_reference_x = effective_hip_x
                            self._last_trigger_x = effective_hip_x
                            self._horizontal_state = HorizontalState.READY
                            self._pending_direction = Action.NONE
                            self._settle_start_time = None
                    else:
                        # Still moving in same direction -> Reset settle timer
                        self._settle_start_time = None

        self._processing_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return GestureResult(
            action=dispatched_action,
            confidence=confidence,
            gesture_name=dispatched_action.value,
            normalized_value=normalized_val if dispatched_action != Action.NONE else delta_x,
            timestamp=now,
            debug_info={
                "horizontal_state": self._horizontal_state.value,
                "vertical_state": self._vertical_state.value,
                "current_x": effective_hip_x,
                "reference_x": self._horizontal_reference_x,
                "delta_x": delta_x,
                "horizontal_velocity": active_features.horizontal_velocity,
                "vertical_velocity": active_features.vertical_velocity,
                "body_height_ratio": active_features.body_height_ratio,
                "y_displacement": active_features.normalized_y_displacement,
                "jump_cooldown": self.cooldown_tracker.remaining_cooldown_sec(Action.JUMP, timestamp=now)
            }
        )

    @property
    def horizontal_reference_x(self) -> Optional[float]:
        return self._horizontal_reference_x

    @property
    def reference_x(self) -> Optional[float]:
        """Convenience alias for horizontal_reference_x."""
        return self._horizontal_reference_x

    @property
    def horizontal_state(self) -> HorizontalState:
        return self._horizontal_state

    @property
    def vertical_state(self) -> VerticalState:
        return self._vertical_state

    @property
    def processing_latency_ms(self) -> float:
        return self._processing_latency_ms

    @property
    def filtering_latency_ms(self) -> float:
        return self._filtering_latency_ms
