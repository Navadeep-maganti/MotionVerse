"""
Body Feature Extractor module for MotionVerse.
Calculates normalized body coordinates, scale, displacements, and velocities from PoseLandmarks.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from app.core.vision.pose.pose_landmarks import PoseLandmarks, Point3D


@dataclass
class BodyFeatures:
    """
    Mathematical and normalized body feature representation.
    Decoupled from raw pixels and camera resolutions.
    """
    hip_center_x: float = 0.0
    hip_center_y: float = 0.0
    shoulder_center_x: float = 0.0
    shoulder_center_y: float = 0.0
    body_scale: float = 0.0

    # Normalized relative to calibration baseline
    normalized_x: float = 0.0               # X_t = (H_x,t - H_x,0) / B_0
    normalized_y_displacement: float = 0.0  # D_t = (H_y,0 - H_y,t) / B_0 (positive is UP)
    vertical_velocity: float = 0.0          # V_y: normalized displacement change per sec
    horizontal_velocity: float = 0.0        # V_x: normalized X change per sec
    body_height_ratio: float = 1.0          # R_t = B_t / B_0

    timestamp: float = 0.0
    is_valid: bool = False

    def to_summary_dict(self) -> dict:
        """Return formatted summary dictionary of key features for display and logging."""
        return {
            "Hip Center": f"({self.hip_center_x:.3f}, {self.hip_center_y:.3f})",
            "Shoulder Center": f"({self.shoulder_center_x:.3f}, {self.shoulder_center_y:.3f})",
            "Body Scale": f"{self.body_scale:.3f}",
            "Norm X": f"{self.normalized_x:+.3f}",
            "Norm Y Disp": f"{self.normalized_y_displacement:+.3f}",
            "V_x (Horiz Vel)": f"{self.horizontal_velocity:+.3f}/s",
            "V_y (Vert Vel)": f"{self.vertical_velocity:+.3f}/s",
            "Body Height Ratio": f"{self.body_height_ratio:.3f}"
        }


class BodyFeatureExtractor:
    """
    Extracts geometric and kinematic body features from PoseLandmarks across frames.
    """

    # Scale estimation expansion ratios when lower body parts are occluded
    SHOULDER_TO_KNEE_FACTOR = 1.45
    SHOULDER_TO_HIP_FACTOR = 2.20

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility

        # Previous frame state for velocity calculation
        self._prev_normalized_x: Optional[float] = None
        self._prev_normalized_y_disp: Optional[float] = None
        self._prev_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset historical state (e.g. after tracking loss or mode switch)."""
        self._prev_normalized_x = None
        self._prev_normalized_y_disp = None
        self._prev_timestamp = None

    def calculate_hip_center(self, landmarks: PoseLandmarks) -> Optional[Tuple[float, float]]:
        """
        Calculate hip center midpoint:
        H_x = (x_LH + x_RH) / 2
        H_y = (y_LH + y_RH) / 2
        """
        lh, rh = landmarks.left_hip, landmarks.right_hip
        if lh and rh and lh.is_visible(self.min_visibility) and rh.is_visible(self.min_visibility):
            return (lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0
        elif lh and lh.is_visible(self.min_visibility):
            return lh.x, lh.y
        elif rh and rh.is_visible(self.min_visibility):
            return rh.x, rh.y
        return None

    def calculate_shoulder_center(self, landmarks: PoseLandmarks) -> Optional[Tuple[float, float]]:
        """
        Calculate shoulder center midpoint:
        S_x = (x_LS + x_RS) / 2
        S_y = (y_LS + y_RS) / 2
        """
        ls, rs = landmarks.left_shoulder, landmarks.right_shoulder
        if ls and rs and ls.is_visible(self.min_visibility) and rs.is_visible(self.min_visibility):
            return (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0
        elif ls and ls.is_visible(self.min_visibility):
            return ls.x, ls.y
        elif rs and rs.is_visible(self.min_visibility):
            return rs.x, rs.y
        return None

    def calculate_body_scale(
        self,
        landmarks: PoseLandmarks,
        shoulder_center: Tuple[float, float],
        hip_center: Tuple[float, float]
    ) -> float:
        """
        Compute robust visible body scale B.
        Priority 1: Shoulder center to ankle center: B = |y_shoulder - y_ankle|
        Fallback 2: Shoulder center to knee center: B = |y_shoulder - y_knee| * 1.45
        Fallback 3: Shoulder center to hip center:  B = |y_shoulder - y_hip| * 2.20
        """
        sy = shoulder_center[1]

        # 1. Try Ankles
        la, ra = landmarks.left_ankle, landmarks.right_ankle
        ankles = [a for a in (la, ra) if a and a.is_visible(self.min_visibility)]
        if ankles:
            ankle_y = sum(a.y for a in ankles) / len(ankles)
            scale = abs(ankle_y - sy)
            if scale > 0.05:
                return scale

        # 2. Fallback: Knees
        lk, rk = landmarks.left_knee, landmarks.right_knee
        knees = [k for k in (lk, rk) if k and k.is_visible(self.min_visibility)]
        if knees:
            knee_y = sum(k.y for k in knees) / len(knees)
            scale = abs(knee_y - sy) * self.SHOULDER_TO_KNEE_FACTOR
            if scale > 0.05:
                return scale

        # 3. Fallback: Torso (Shoulder to Hip)
        torso_height = abs(hip_center[1] - sy)
        return max(0.05, torso_height * self.SHOULDER_TO_HIP_FACTOR)

    def extract(
        self,
        landmarks: Optional[PoseLandmarks],
        baseline_hip: Optional[Tuple[float, float]] = None,
        baseline_scale: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> BodyFeatures:
        """
        Extract normalized features and velocities from current pose landmarks.
        
        Args:
            landmarks: Current frame PoseLandmarks or None.
            baseline_hip: Calibrated neutral hip center (H_x0, H_y0), if calibrated.
            baseline_scale: Calibrated neutral body scale (B_0), if calibrated.
            timestamp: Current timestamp (seconds).
        """
        curr_ts = timestamp if timestamp is not None else (landmarks.timestamp if landmarks else time.perf_counter())

        if landmarks is None or not landmarks.has_core_landmarks(self.min_visibility):
            self.reset()
            return BodyFeatures(timestamp=curr_ts, is_valid=False)

        hip_center = self.calculate_hip_center(landmarks)
        shoulder_center = self.calculate_shoulder_center(landmarks)

        if hip_center is None or shoulder_center is None:
            self.reset()
            return BodyFeatures(timestamp=curr_ts, is_valid=False)

        hx, hy = hip_center
        sx, sy = shoulder_center

        # Calculate current body scale
        current_scale = self.calculate_body_scale(landmarks, shoulder_center, hip_center)

        # Use calibrated reference values if available, else current scale and current hip
        ref_scale = baseline_scale if (baseline_scale is not None and baseline_scale > 0.01) else current_scale
        ref_hx = baseline_hip[0] if baseline_hip is not None else hx
        ref_hy = baseline_hip[1] if baseline_hip is not None else hy

        # Normalized horizontal position: X_t = (H_x,t - H_x,0) / B_0
        norm_x = (hx - ref_hx) / ref_scale

        # Normalized vertical displacement: D_t = (H_y,0 - H_y,t) / B_0 (positive UP)
        norm_y_disp = (ref_hy - hy) / ref_scale

        # Body height ratio: R_t = B_t / B_0
        height_ratio = current_scale / ref_scale

        # Finite difference velocities (per second)
        vx = 0.0
        vy = 0.0

        if self._prev_timestamp is not None:
            dt = curr_ts - self._prev_timestamp
            # Avoid division by zero or large spikes if frames were delayed (>0.5s)
            if 0.001 < dt < 0.5:
                if self._prev_normalized_x is not None:
                    vx = (norm_x - self._prev_normalized_x) / dt
                if self._prev_normalized_y_disp is not None:
                    vy = (norm_y_disp - self._prev_normalized_y_disp) / dt

        # Update history
        self._prev_normalized_x = norm_x
        self._prev_normalized_y_disp = norm_y_disp
        self._prev_timestamp = curr_ts

        return BodyFeatures(
            hip_center_x=hx,
            hip_center_y=hy,
            shoulder_center_x=sx,
            shoulder_center_y=sy,
            body_scale=current_scale,
            normalized_x=norm_x,
            normalized_y_displacement=norm_y_disp,
            vertical_velocity=vy,
            horizontal_velocity=vx,
            body_height_ratio=height_ratio,
            timestamp=curr_ts,
            is_valid=True
        )
