"""
User profile and calibration storage management for MotionVerse.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from app.core.calibration.baseline import CalibrationBaseline


@dataclass
class UserProfile:
    """
    Encapsulates user-specific calibration baseline and threshold tuning.
    """
    user_id: str = "default_user"
    baseline: Optional[CalibrationBaseline] = None
    custom_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "move_left_threshold": 0.20,
        "move_right_threshold": 0.20,
        "jump_displacement_threshold": 0.12,
        "jump_velocity_threshold": 0.80,
        "crouch_ratio_threshold": 0.75,
        "cooldown_seconds": 0.35
    })

    def save_to_file(self, filepath: str) -> bool:
        """Save user profile to JSON file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            data = {
                "user_id": self.user_id,
                "baseline": self.baseline.to_dict() if self.baseline else None,
                "custom_thresholds": self.custom_thresholds
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save user profile: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["UserProfile"]:
        """Load user profile from JSON file."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            baseline = CalibrationBaseline.from_dict(data["baseline"]) if data.get("baseline") else None
            return cls(
                user_id=data.get("user_id", "default_user"),
                baseline=baseline,
                custom_thresholds=data.get("custom_thresholds", {})
            )
        except Exception as e:
            print(f"[ERROR] Failed to load user profile: {e}")
            return None
