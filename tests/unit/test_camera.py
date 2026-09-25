"""
Unit tests for Camera System (Step 1).
Tests FrameBuffer latest-frame guarantees, CameraConfig, and CameraManager offline behavior.
"""

import time
import numpy as np
import pytest

from app.core.camera.camera_config import CameraConfig
from app.core.camera.frame_buffer import FrameBuffer
from app.core.camera.camera_manager import CameraManager


def test_camera_config_defaults():
    config = CameraConfig()
    assert config.camera_index == 0
    assert config.width == 640
    assert config.height == 480
    assert config.target_fps == 30
    assert config.buffer_size == 1
    assert config.flip_horizontal is True
    assert config.use_threading is True


def test_frame_buffer_single_frame():
    buffer = FrameBuffer(max_size=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = time.perf_counter()

    assert buffer.has_new_frame() is False
    buffer.put(frame, ts)

    assert buffer.has_new_frame() is True
    assert buffer.total_frames == 1
    assert buffer.dropped_frames == 0

    ret_frame, ret_ts = buffer.get_latest()
    assert ret_frame is not None
    assert np.array_equal(ret_frame, frame)
    assert ret_ts == ts
    assert buffer.has_new_frame() is False


def test_frame_buffer_latest_frame_overwrites_stale():
    """Verify that pushing multiple frames without reading only preserves the latest frame."""
    buffer = FrameBuffer(max_size=1)
    frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 10
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 20
    frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 30

    buffer.put(frame1, 1.0)
    buffer.put(frame2, 2.0)
    buffer.put(frame3, 3.0)

    assert buffer.total_frames == 3
    assert buffer.dropped_frames == 2  # 2 stale frames dropped

    ret_frame, ret_ts = buffer.get_latest()
    assert ret_frame is not None
    assert ret_ts == 3.0
    assert np.all(ret_frame == 30)


def test_frame_buffer_clear():
    buffer = FrameBuffer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    buffer.put(frame, time.perf_counter())
    assert buffer.has_new_frame() is True

    buffer.clear()
    assert buffer.has_new_frame() is False
    ret_frame, ret_ts = buffer.get_latest()
    assert ret_frame is None
    assert ret_ts == 0.0


def test_camera_manager_unavailable_graceful_failure():
    """Verify camera manager handles non-existent camera index gracefully without crashing."""
    config = CameraConfig(camera_index=999)
    manager = CameraManager(config)
    success = manager.start()
    assert success is False
    assert manager.is_opened is False

    ret, frame, ts = manager.read_frame()
    assert ret is False
    assert frame is None
    assert ts == 0.0

    manager.stop()  # Should not raise exception
