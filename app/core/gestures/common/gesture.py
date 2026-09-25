"""
Base Gesture data classes and results container for MotionVerse.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from app.core.gestures.common.actions import Action


@dataclass
class GestureResult:
    """
    Standardized result payload emitted by gesture engines for each frame.
    """
    action: Action = Action.NONE
    confidence: float = 1.0
    gesture_name: str = "NONE"
    normalized_value: float = 0.0
    timestamp: float = 0.0
    debug_info: Optional[Dict[str, Any]] = None

    @property
    def is_active(self) -> bool:
        """Check if a non-NONE action was triggered."""
        return self.action != Action.NONE
