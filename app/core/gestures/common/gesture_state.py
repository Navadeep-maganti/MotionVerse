"""
Gesture State Machine module for MotionVerse.
Manages transition states (IDLE -> CANDIDATE -> TRIGGERED -> HOLDING -> COOLDOWN -> IDLE)
for both discrete and continuous movement gestures.
"""

from enum import Enum
from typing import Optional, Dict, Tuple
from app.core.gestures.common.actions import Action
from app.core.gestures.filters.cooldown import CooldownTracker


class GestureState(Enum):
    """Lifecycle states of a gesture recognition channel."""
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    TRIGGERED = "TRIGGERED"
    HOLDING = "HOLDING"
    COOLDOWN = "COOLDOWN"

    def __str__(self) -> str:
        return self.value


class GestureStateMachine:
    """
    State machine controlling action dispatch and discrete gesture debouncing.
    
    Discrete Actions (e.g. JUMP):
    - IDLE -> TRIGGERED (emits action on 1st frame) -> COOLDOWN (suppresses repeats) -> IDLE
    
    Continuous Actions (e.g. CROUCH, MOVE_LEFT, MOVE_RIGHT):
    - IDLE -> HOLDING (emits action continuously while held) -> IDLE (when released)
    """

    DISCRETE_ACTIONS = {Action.JUMP, Action.PAUSE, Action.SELECT}
    CONTINUOUS_ACTIONS = {Action.CROUCH, Action.MOVE_LEFT, Action.MOVE_RIGHT}

    def __init__(self, jump_cooldown_sec: float = 0.45):
        self.jump_cooldown_sec = jump_cooldown_sec
        self.cooldown_tracker = CooldownTracker(default_cooldown_sec=jump_cooldown_sec)

        self.current_state = GestureState.IDLE
        self.active_action = Action.NONE
        self._previous_candidate_action = Action.NONE

    def update(
        self,
        candidate_action: Action,
        timestamp: Optional[float] = None
    ) -> Tuple[Action, GestureState]:
        """
        Process the raw proposed action from rule evaluation through state transitions.
        
        Args:
            candidate_action: The raw Action proposed by rule evaluation.
            timestamp: Frame timestamp in seconds.
            
        Returns:
            Tuple of (Dispatched Action, Current GestureState)
        """
        now = timestamp

        # 1. Handle Active Cooldown for Discrete Gestures
        if self.cooldown_tracker.is_in_cooldown(Action.JUMP, timestamp=now):
            self.current_state = GestureState.COOLDOWN
            # Suppress jump while in cooldown
            if candidate_action == Action.JUMP:
                return Action.NONE, GestureState.COOLDOWN

        # 2. Discrete Action Trigger Logic (e.g. JUMP)
        if candidate_action in self.DISCRETE_ACTIONS:
            if not self.cooldown_tracker.is_in_cooldown(candidate_action, timestamp=now):
                # Trigger discrete action on leading edge
                self.current_state = GestureState.TRIGGERED
                self.active_action = candidate_action
                self.cooldown_tracker.trigger(candidate_action, duration_sec=self.jump_cooldown_sec, timestamp=now)
                return candidate_action, GestureState.TRIGGERED
            else:
                self.current_state = GestureState.COOLDOWN
                return Action.NONE, GestureState.COOLDOWN

        # 3. Continuous Action Logic (e.g. CROUCH, MOVE_LEFT, MOVE_RIGHT)
        if candidate_action in self.CONTINUOUS_ACTIONS:
            self.current_state = GestureState.HOLDING
            self.active_action = candidate_action
            return candidate_action, GestureState.HOLDING

        # 4. Return to IDLE
        if self.cooldown_tracker.is_in_cooldown(Action.JUMP, timestamp=now):
            self.current_state = GestureState.COOLDOWN
        else:
            self.current_state = GestureState.IDLE

        self.active_action = Action.NONE
        return Action.NONE, self.current_state

    def reset(self) -> None:
        """Reset state machine and cooldowns."""
        self.current_state = GestureState.IDLE
        self.active_action = Action.NONE
        self._previous_candidate_action = Action.NONE
        self.cooldown_tracker.reset()
