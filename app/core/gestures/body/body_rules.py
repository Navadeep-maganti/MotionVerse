"""
Configurable rule definitions for body movement detection with dynamic reference and discrete triggers.
"""

from dataclasses import dataclass
from typing import Optional

from app.core.gestures.common.actions import Action
from app.core.vision.pose.pose_features import BodyFeatures


class BodyGestureThresholds:
    """
    Configurable threshold parameters for discrete gesture triggers and dynamic reference settling.
    Supports standard naming and common parameter aliases.
    """
    def __init__(
        self,
        move_left_trigger: float = 0.18,
        move_left_release: float = 0.10,
        move_right_trigger: float = 0.18,
        move_right_release: float = 0.10,
        horizontal_settle_velocity: float = 0.18,
        horizontal_velocity_settle: Optional[float] = None,
        horizontal_settle_duration_sec: float = 0.08,
        jump_displacement_threshold: float = 0.10,
        jump_trigger_displacement: Optional[float] = None,
        jump_velocity_threshold: float = 0.60,
        jump_trigger_velocity: Optional[float] = None,
        jump_recovery_displacement: float = 0.04,
        crouch_trigger_ratio: float = 0.76,
        crouch_trigger_disp: float = -0.09,
        crouch_recovery_ratio: float = 0.84,
        crouch_release_ratio: Optional[float] = None,
        crouch_recovery_disp: float = -0.04,
        crouch_release_disp: Optional[float] = None,
    ):
        self.move_left_trigger = move_left_trigger
        self.move_left_release = move_left_release
        self.move_right_trigger = move_right_trigger
        self.move_right_release = move_right_release

        self.horizontal_settle_velocity = horizontal_velocity_settle if horizontal_velocity_settle is not None else horizontal_settle_velocity
        self.horizontal_settle_duration_sec = horizontal_settle_duration_sec

        self.jump_displacement_threshold = jump_trigger_displacement if jump_trigger_displacement is not None else jump_displacement_threshold
        self.jump_velocity_threshold = jump_trigger_velocity if jump_trigger_velocity is not None else jump_velocity_threshold
        self.jump_recovery_displacement = jump_recovery_displacement

        self.crouch_trigger_ratio = crouch_trigger_ratio
        self.crouch_trigger_disp = crouch_trigger_disp
        self.crouch_recovery_ratio = crouch_release_ratio if crouch_release_ratio is not None else crouch_recovery_ratio
        self.crouch_recovery_disp = crouch_release_disp if crouch_release_disp is not None else crouch_recovery_disp


class BodyMovementRules:
    """
    Evaluates kinematic body features against discrete trigger and recovery thresholds.
    """

    def __init__(self, thresholds: Optional[BodyGestureThresholds] = None):
        self.thresholds = thresholds or BodyGestureThresholds()

    def calculate_horizontal_delta(
        self,
        features: BodyFeatures,
        reference_x: float,
        body_scale: float
    ) -> float:
        """
        Calculate relative horizontal displacement from dynamic reference:
        D_x = (H_x,t - X_ref) / B_0
        """
        if not features.is_valid or body_scale <= 0.01:
            return 0.0
        if features.hip_center_x != 0.0:
            return (features.hip_center_x - reference_x) / body_scale
        # Fallback for mock objects where only normalized_x was populated
        return features.normalized_x

    def evaluate_jump_trigger(self, features: BodyFeatures) -> bool:
        """
        Evaluate jump trigger condition (requires displacement + velocity).
        """
        if not features.is_valid:
            return False

        disp_y = features.normalized_y_displacement
        vy = features.vertical_velocity

        return (disp_y >= self.thresholds.jump_displacement_threshold and
                vy >= self.thresholds.jump_velocity_threshold)

    def evaluate_jump_recovered(self, features: BodyFeatures) -> bool:
        """
        Check if user has landed and returned close to neutral height.
        """
        if not features.is_valid:
            return True
        return features.normalized_y_displacement <= self.thresholds.jump_recovery_displacement

    def evaluate_crouch_trigger(self, features: BodyFeatures) -> bool:
        """
        Evaluate crouch trigger condition (height compression or downward hip drop).
        """
        if not features.is_valid:
            return False

        ratio = features.body_height_ratio
        disp_y = features.normalized_y_displacement

        is_height_compressed = ratio <= self.thresholds.crouch_trigger_ratio
        is_hip_dropped = disp_y <= self.thresholds.crouch_trigger_disp

        return is_height_compressed or (is_hip_dropped and ratio <= 0.86)

    def evaluate_crouch_recovered(self, features: BodyFeatures) -> bool:
        """
        Check if user has stood back up to neutral standing posture.
        """
        if not features.is_valid:
            return True

        has_height_expanded = features.body_height_ratio >= self.thresholds.crouch_recovery_ratio
        has_hip_risen = features.normalized_y_displacement >= self.thresholds.crouch_recovery_disp

        return has_height_expanded and has_hip_risen

    def evaluate_horizontal_trigger(self, delta_x: float) -> Action:
        """
        Evaluate horizontal trigger relative to dynamic reference.
        
        Returns:
            Action.MOVE_LEFT if delta_x <= -move_left_trigger
            Action.MOVE_RIGHT if delta_x >= +move_right_trigger
            Action.NONE otherwise
        """
        if delta_x <= -self.thresholds.move_left_trigger:
            return Action.MOVE_LEFT
        elif delta_x >= self.thresholds.move_right_trigger:
            return Action.MOVE_RIGHT
        return Action.NONE

    def is_horizontal_settled(self, features: BodyFeatures) -> bool:
        """
        Check if horizontal movement has stabilized (velocity below settle threshold).
        """
        if not features.is_valid:
            return True
        return abs(features.horizontal_velocity) <= self.thresholds.horizontal_settle_velocity
