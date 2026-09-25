"""
Pose landmark definitions and clean data structures for MotionVerse.
Decouples application logic from raw MediaPipe structures.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Point3D:
    """Represents a normalized 3D landmark coordinate with visibility."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_pixel(self, image_width: int, image_height: int) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to integer pixel coordinates."""
        px = int(self.x * image_width)
        py = int(self.y * image_height)
        return px, py

    def is_visible(self, threshold: float = 0.5) -> bool:
        """Check if landmark visibility exceeds the given threshold."""
        return self.visibility >= threshold


@dataclass
class PoseLandmarks:
    """
    Clean, decoupled container for key human body landmarks.
    All coordinates are normalized [0.0, 1.0] relative to image dimensions.
    """
    nose: Optional[Point3D] = None
    left_shoulder: Optional[Point3D] = None
    right_shoulder: Optional[Point3D] = None
    left_elbow: Optional[Point3D] = None
    right_elbow: Optional[Point3D] = None
    left_wrist: Optional[Point3D] = None
    right_wrist: Optional[Point3D] = None
    left_hip: Optional[Point3D] = None
    right_hip: Optional[Point3D] = None
    left_knee: Optional[Point3D] = None
    right_knee: Optional[Point3D] = None
    left_ankle: Optional[Point3D] = None
    right_ankle: Optional[Point3D] = None

    # Timestamp of when the frame was captured/processed
    timestamp: float = 0.0

    def has_core_landmarks(self, min_visibility: float = 0.5) -> bool:
        """
        Verify that primary body landmarks (shoulders and hips) are present and visible.
        """
        core = [self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip]
        return all(pt is not None and pt.is_visible(min_visibility) for pt in core)

    def has_lower_body_landmarks(self, min_visibility: float = 0.5) -> bool:
        """
        Verify that lower body landmarks (knees and ankles) are present and visible.
        """
        lower = [self.left_knee, self.right_knee, self.left_ankle, self.right_ankle]
        return all(pt is not None and pt.is_visible(min_visibility) for pt in lower)

    def to_dict(self) -> Dict[str, Optional[Point3D]]:
        """Return landmarks as a dictionary."""
        return {
            "nose": self.nose,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_elbow": self.left_elbow,
            "right_elbow": self.right_elbow,
            "left_wrist": self.left_wrist,
            "right_wrist": self.right_wrist,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle,
        }
