"""
Game Profile module for MotionVerse.
Defines schemas and loaders for game-specific keyboard bindings and configurations.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class GameProfile:
    """
    Metadata and keybindings for a game target.
    Decoupled from body gesture recognition.
    """
    game_name: str
    game_type: str = "browser"  # "browser" or "desktop"
    game_url: Optional[str] = None
    bindings: Dict[str, str] = field(default_factory=dict)
    key_tap_duration_ms: int = 30
    description: str = ""

    def get_key_for_action_name(self, action_name: str) -> Optional[str]:
        """Look up mapped key for a generic action name (e.g. 'MOVE_RIGHT')."""
        return self.bindings.get(action_name.upper())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "game_name": self.game_name,
            "game_type": self.game_type,
            "game_url": self.game_url,
            "bindings": self.bindings,
            "key_tap_duration_ms": self.key_tap_duration_ms,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameProfile":
        """Construct GameProfile from dictionary."""
        return cls(
            game_name=data.get("game_name", "Unknown Game"),
            game_type=data.get("game_type", "browser"),
            game_url=data.get("game_url"),
            bindings=data.get("bindings", {}),
            key_tap_duration_ms=data.get("key_tap_duration_ms", 30),
            description=data.get("description", "")
        )

    @classmethod
    def load_from_file(cls, filepath: str) -> "GameProfile":
        """Load profile from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Game profile not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, filepath: str) -> None:
        """Save profile to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
