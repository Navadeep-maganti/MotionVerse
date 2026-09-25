"""
Calibration baseline data structures for MotionVerse.
Stores calibrated neutral standing reference values.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import time


@dataclass
class CalibrationBaseline:
    """
    Stores the median baseline values measured while the participant was standing in neutral pose.
    """
    hip_center_x: float
    hip_center_y: float
    shoulder_center_x: float
    shoulder_center_y: float
    body_scale: float
    sample_count: int = 0
    std_dev_hip_y: float = 0.0
    timestamp: float = 0.0
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize baseline to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationBaseline":
        """Deserialize baseline from dictionary."""
        return cls(
            hip_center_x=float(data.get("hip_center_x", 0.5)),
            hip_center_y=float(data.get("hip_center_y", 0.5)),
            shoulder_center_x=float(data.get("shoulder_center_x", 0.5)),
            shoulder_center_y=float(data.get("shoulder_center_y", 0.2)),
            body_scale=float(data.get("body_scale", 0.6)),
            sample_count=int(data.get("sample_count", 0)),
            std_dev_hip_y=float(data.get("std_dev_hip_y", 0.0)),
            timestamp=float(data.get("timestamp", time.time())),
            is_valid=bool(data.get("is_valid", True))
        )
