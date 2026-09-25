"""
Game Mapper module for MotionVerse.
Translates generic Action enums into game-specific keyboard keys using GameProfile.
Strictly decoupled from gesture recognition and keyboard dispatch.
"""

import time
from typing import Optional, Dict, Any

from app.core.gestures.common.actions import Action
from app.games.game_profile import GameProfile


class GameMapper:
    """
    Translates generic Action enums to game-specific keyboard keys.
    
    Responsibilities:
    1. Look up Action in the active GameProfile bindings.
    2. Return key string (e.g. 'space', 'down', 'left', 'right') or None.
    3. Safely handle unmapped actions and Action.NONE.
    4. Provide mapping latency telemetry.
    """

    def __init__(self, profile: Optional[GameProfile] = None):
        self.profile: Optional[GameProfile] = profile
        self._last_mapping_latency_ms: float = 0.0

    def set_profile(self, profile: GameProfile) -> None:
        """Switch or set the active game profile."""
        self.profile = profile

    def map_action(self, action: Action) -> Optional[str]:
        """
        Map a generic Action enum to a keyboard key string.
        
        Args:
            action: Generic Action enum instance.
            
        Returns:
            Key string (e.g. 'left', 'right', 'space', 'down') or None if Action.NONE or unmapped.
        """
        t_start = time.perf_counter()

        if action == Action.NONE or self.profile is None:
            self._last_mapping_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return None

        # Look up binding by action name
        key = self.profile.get_key_for_action_name(action.name)
        self._last_mapping_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return key

    @property
    def mapping_latency_ms(self) -> float:
        """Latency of the last action mapping evaluation."""
        return self._last_mapping_latency_ms

    @property
    def current_game_name(self) -> str:
        """Return name of active game or 'None'."""
        return self.profile.game_name if self.profile else "None"
