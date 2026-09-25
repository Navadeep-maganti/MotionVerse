"""
Cooldown and debouncing tracker for MotionVerse gesture recognition.
Prevents rapid re-triggering of discrete gestures.
"""

import time
from typing import Dict, Optional
from app.core.gestures.common.actions import Action


class CooldownTracker:
    """
    Tracks action cooldown timers using high-resolution performance counters.
    """

    def __init__(self, default_cooldown_sec: float = 0.40):
        self.default_cooldown_sec = default_cooldown_sec
        self._last_trigger_times: Dict[Action, float] = {}
        self._cooldown_durations: Dict[Action, float] = {}

    def trigger(self, action: Action, duration_sec: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        """
        Record a trigger event for the specified action, starting its cooldown timer.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        dur = duration_sec if duration_sec is not None else self.default_cooldown_sec
        self._last_trigger_times[action] = now
        self._cooldown_durations[action] = dur

    def is_in_cooldown(self, action: Action, timestamp: Optional[float] = None) -> bool:
        """
        Check if an action is currently locked in cooldown.
        """
        if action not in self._last_trigger_times:
            return False

        now = timestamp if timestamp is not None else time.perf_counter()
        last_t = self._last_trigger_times[action]
        dur = self._cooldown_durations.get(action, self.default_cooldown_sec)

        return (now - last_t) < dur

    def remaining_cooldown_sec(self, action: Action, timestamp: Optional[float] = None) -> float:
        """
        Get remaining cooldown duration in seconds (returns 0.0 if not in cooldown).
        """
        if action not in self._last_trigger_times:
            return 0.0

        now = timestamp if timestamp is not None else time.perf_counter()
        last_t = self._last_trigger_times[action]
        dur = self._cooldown_durations.get(action, self.default_cooldown_sec)
        remaining = dur - (now - last_t)
        return max(0.0, remaining)

    def reset(self) -> None:
        """Reset all active cooldown timers."""
        self._last_trigger_times.clear()
        self._cooldown_durations.clear()
