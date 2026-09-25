"""
Configurable rule definitions for body movement detection.
"""

from dataclasses import dataclass
from typing import Optional

from app.core.gestures.common.actions import Action
from app.core.vision.pose.pose_features import BodyFeatures


@dataclass
class BodyGestureThresholds:
    """Configurable threshold parameters for body gesture recognition."""
    # Horizontal Movement (normalized displacement X_t = (H_x,t - H_x,0) / B_0)
    move_left_threshold: float = 0.18
    move_right_threshold: float = 0.18
    min_horizontal_velocity: float = 0.05

    # Jump Detection (normalized displacement D_t = (H_y,0 - H_y,t) / B_0 and upward velocity V_y)
    jump_displacement_threshold: float = 0.10  # 10% of body scale upward displacement
    jump_velocity_threshold: float = 0.60      # Normalized upward velocity (scales/sec)

    # Crouch Detection (height ratio R_t = B_t / B_0 and downward displacement D_t)
    crouch_ratio_threshold: float = 0.78       # Body height compresses to <= 78% of baseline
    crouch_displacement_threshold: float = -0.08 # Hip drops downward (D_t is negative for downward)


class BodyMovementRules:
    """
    Evaluates kinematic body features against configurable thresholds.
    """

    def __init__(self, thresholds: Optional[BodyGestureThresholds] = None):
        self.thresholds = thresholds or BodyGestureThresholds()

    def evaluate_jump(self, features: BodyFeatures) -> bool:
        """
        Evaluate jump condition based on hip upward displacement and velocity.
        
        Requires BOTH:
        1. Upward normalized displacement: D_t >= jump_displacement_threshold
        2. Upward vertical velocity: V_y >= jump_velocity_threshold
        """
        if not features.is_valid:
            return False

        disp_y = features.normalized_y_displacement
        vy = features.vertical_velocity

        return (disp_y >= self.thresholds.jump_displacement_threshold and
                vy >= self.thresholds.jump_velocity_threshold)

    def evaluate_crouch(self, features: BodyFeatures) -> bool:
        """
        Evaluate crouch condition based on body height compression or downward hip displacement.
        
        Triggers if:
        - Body scale ratio R_t <= crouch_ratio_threshold, OR
        - Downward displacement D_t <= crouch_displacement_threshold (and not jumping)
        """
        if not features.is_valid:
            return False

        ratio = features.body_height_ratio
        disp_y = features.normalized_y_displacement

        # Crouch occurs when user bends knees, lowering hips and compressing visible height
        is_height_compressed = ratio <= self.thresholds.crouch_ratio_threshold
        is_hip_lowered = disp_y <= self.thresholds.crouch_displacement_threshold

        return is_height_compressed or (is_hip_lowered and ratio <= 0.88)

    def evaluate_horizontal(self, features: BodyFeatures) -> Action:
        """
        Evaluate horizontal position and velocity.
        
        Normalized X definition:
        X_t = (H_x,t - H_x,0) / B_0
        
        Because camera view is mirrored:
        - Shifting left decreases X_t (negative values).
        - Shifting right increases X_t (positive values).
        
        Returns:
            Action.MOVE_LEFT, Action.MOVE_RIGHT, or Action.NONE.
        """
        if not features.is_valid:
            return Action.NONE

        norm_x = features.normalized_x

        # Check Left: norm_x < -move_left_threshold
        if norm_x < -self.thresholds.move_left_threshold:
            return Action.MOVE_LEFT

        # Check Right: norm_x > move_right_threshold
        if norm_x > self.thresholds.move_right_threshold:
            return Action.MOVE_RIGHT

        return Action.NONE
