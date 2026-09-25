"""
Unit tests for KeyboardController in Mock Mode.
"""

import pytest
import time
from app.input.keyboard_controller import KeyboardController


def test_keyboard_controller_mock_tap():
    recorded_events = []

    def mock_backend(event_type, key):
        recorded_events.append((event_type, key))

    controller = KeyboardController(key_tap_duration_ms=20, mock_mode=True, backend_callback=mock_backend)

    res = controller.tap("space")
    assert res is True
    time.sleep(0.05)  # Allow async thread to dispatch

    assert controller.total_dispatches == 1
    assert controller.last_dispatched_key == "space"
    assert ("press", "space") in recorded_events


def test_keyboard_controller_key_down_up():
    recorded_events = []

    def mock_backend(event_type, key):
        recorded_events.append((event_type, key))

    controller = KeyboardController(mock_mode=True, backend_callback=mock_backend)

    controller.key_down("left")
    assert "left" in controller.active_keys
    assert ("down", "left") in recorded_events

    controller.key_up("left")
    assert "left" not in controller.active_keys
    assert ("up", "left") in recorded_events


def test_keyboard_controller_emergency_release_all():
    recorded_events = []

    def mock_backend(event_type, key):
        recorded_events.append((event_type, key))

    controller = KeyboardController(mock_mode=True, backend_callback=mock_backend)

    controller.key_down("left")
    controller.key_down("space")
    assert len(controller.active_keys) == 2

    controller.release_all()
    assert len(controller.active_keys) == 0
    assert ("up", "left") in recorded_events
    assert ("up", "space") in recorded_events
