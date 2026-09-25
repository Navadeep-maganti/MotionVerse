"""
Unit tests for GameProfile and GameMapper.
"""

import pytest
import os
import tempfile
import json

from app.core.gestures.common.actions import Action
from app.games.game_profile import GameProfile
from app.games.game_mapper import GameMapper


@pytest.fixture
def sample_profile():
    return GameProfile(
        game_name="Test Runner",
        game_type="browser",
        bindings={
            "JUMP": "space",
            "CROUCH": "down",
            "MOVE_LEFT": "left",
            "MOVE_RIGHT": "right"
        },
        key_tap_duration_ms=30
    )


def test_game_profile_serialization(sample_profile):
    d = sample_profile.to_dict()
    assert d["game_name"] == "Test Runner"
    assert d["bindings"]["JUMP"] == "space"

    restored = GameProfile.from_dict(d)
    assert restored.game_name == sample_profile.game_name
    assert restored.bindings == sample_profile.bindings


def test_game_profile_file_io(sample_profile):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sample_profile.save_to_file(tmp_path)
        loaded = GameProfile.load_from_file(tmp_path)
        assert loaded.game_name == sample_profile.game_name
        assert loaded.get_key_for_action_name("MOVE_LEFT") == "left"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_game_mapper_standard_actions(sample_profile):
    mapper = GameMapper(sample_profile)

    assert mapper.map_action(Action.JUMP) == "space"
    assert mapper.map_action(Action.CROUCH) == "down"
    assert mapper.map_action(Action.MOVE_LEFT) == "left"
    assert mapper.map_action(Action.MOVE_RIGHT) == "right"
    assert mapper.map_action(Action.NONE) is None


def test_game_mapper_unmapped_action():
    profile = GameProfile(
        game_name="Partial Game",
        bindings={"JUMP": "up"}
    )
    mapper = GameMapper(profile)

    assert mapper.map_action(Action.JUMP) == "up"
    assert mapper.map_action(Action.MOVE_LEFT) is None
    assert mapper.map_action(Action.NONE) is None


def test_game_mapper_no_profile():
    mapper = GameMapper(profile=None)
    assert mapper.map_action(Action.JUMP) is None
    assert mapper.current_game_name == "None"
