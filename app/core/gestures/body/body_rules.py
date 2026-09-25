"""
Configurable rule definitions for body movement detection with dual-threshold hysteresis.
"""

from dataclasses import dataclass
from typing import Optional

from app.core.gestures.common.actions import Action
from app.core.vision.pose.pose_features import BodyFeatures


@dataclass
class BodyGestureThresholds:
    """
    Configurable threshold parameters with separate trigger (activation)
    and release (deactivation) boundaries for Schmitt-trigger hysteresis.
    """
    # Horizontal Movement Hysteresis
    move_left_trigger: float = 0.20   # Must move left beyond -0.20 to activate
    move_left_release: float = 0.12   # Must return past -0.12 toward center to deactivate
    move_right_trigger: float = 0.20  # Must move right beyond +0.20 to activate
    move_right_release: float = 0.12  # Must return past +0.12 toward center to deactivate
    min_horizontal_velocity: float = 0.05

    # Jump Detection (discrete trigger)
    jump_displacement_threshold: float = 0.10  # 10% of body scale upward displacement
    jump_velocity_threshold: float = 0.60      # Normalized upward velocity (scales/sec)

    # Crouch Detection Hysteresis
    crouch_trigger_ratio: float = 0.76         # Must compress height to <= 76% to activate crouch
    crouch_release_ratio: float = 0.84         # Must stand back up past 84% to release crouch
    crouch_trigger_disp: float = -0.09         # Downward hip drop to activate
    crouch_release_disp: float = -0.04         # Hip rise to release

    @property
    def move_left_threshold(self) -> float:
        return self.move_left_trigger

    @property
    def move_right_threshold(self) -> float:
        return self.move_right_trigger

    @property
    def crouch_ratio_threshold(self) -> float:
        return self.crouch_trigger_ratio


class BodyMovementRules:
    """
    Evaluates kinematic body features against dual-threshold hysteresis bands.
    """

    def __init__(self, thresholds: Optional[BodyGestureThresholds] = None):
        self.thresholds = thresholds or BodyGestureThresholds()

    def evaluate_jump(self, features: BodyFeatures) -> bool:
        """
        Evaluate jump condition based on hip upward displacement and velocity.
        """
        if not features.is_valid:
            return False

        disp_y = features.normalized_y_displacement
        vy = features.vertical_velocity

        return (disp_y >= self.thresholds.jump_displacement_threshold and
                vy >= self.thresholds.jump_velocity_threshold)

    def evaluate_crouch_with_hysteresis(
        self,
        features: BodyFeatures,
        is_currently_crouched: bool
    ) -> bool:
        """
        Evaluate crouch state with hysteresis to avoid threshold bouncing.
        
        - If currently CROUCHED: remains crouched until body expands above crouch_release_ratio
          AND hip rises above crouch_release_disp.
        - If NOT crouched: activates only when height compresses <= crouch_trigger_ratio
          OR downward displacement <= crouch_trigger_disp.
        """
        if not features.is_valid:
            return False

        ratio = features.body_height_ratio
        disp_y = features.normalized_y_displacement

        if is_currently_crouched:
            # Release condition: user has stood back up
            has_released_ratio = ratio >= self.thresholds.crouch_release_ratio
            has_released_disp = disp_y >= self.thresholds.crouch_release_disp
            if has_released_ratio and has_released_disp:
                return False  # Crouch ended
            return True  # Retain crouch state
        else:
            # Trigger condition: user has crouched down
            is_height_compressed = ratio <= self.thresholds.crouch_trigger_ratio
            is_hip_dropped = disp_y <= self.thresholds.crouch_trigger_disp
            return is_height_compressed or (is_hip_dropped and ratio <= 0.86)

    def evaluate_horizontal_with_hysteresis(
        self,
        features: BodyFeatures,
        current_horizontal_action: Action
    ) -> Action:
        """
        Evaluate horizontal position with dual-threshold hysteresis.
        
        - If currently MOVE_LEFT: stays MOVE_LEFT until X_t > -move_left_release (e.g. > -0.12).
        - If currently MOVE_RIGHT: stays MOVE_RIGHT until X_t < +move_right_release (e.g. < 0.12).
        - If currently NONE: requires |X_t| >= trigger (0.20) to engage.
        """
        if not features.is_valid:
            return Action.NONE

        x = features.normalized_x

        if current_horizontal_action == Action.MOVE_LEFT:
            # Check if user returned past the release threshold
            if x > -self.thresholds.move_left_release:
                # Check if user went all the way to the right side
                if x > self.thresholds.move_right_trigger:
                    return Action.MOVE_RIGHT
                return Action.NONE
            return Action.MOVE_LEFT

        elif current_horizontal_action == Action.MOVE_RIGHT:
            # Check if user returned past the release threshold
            if x < self.thresholds.move_right_release:
                # Check if user went all the way to the left side
                if x < -self.thresholds.move_left_trigger:
                    return Action.MOVE_LEFT
                return Action.NONE
            return Action.MOVE_RIGHT

        else:
            # Currently NONE: requires trigger threshold to engage
            if x <= -self.thresholds.move_left_trigger:
                return Action.MOVE_LEFT
            elif x >= self.thresholds.move_right_trigger:
                return Action.MOVE_RIGHT
            return Action.NONE
