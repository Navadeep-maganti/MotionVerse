"""
Thread-safe latest-frame buffer for MotionVerse.
Guarantees processing of the newest frame without queue accumulation.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np


class FrameBuffer:
    """
    A bounded, latest-frame buffer.
    Overwrites old frames immediately so consumers always receive the freshest frame.
    """

    def __init__(self, max_size: int = 1):
        self._max_size = max(1, max_size)
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._has_unread: bool = False

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Store a new frame in the buffer.
        Overwrites any unread frame to avoid stale frame accumulation.
        """
        if frame is None:
            return

        ts = timestamp if timestamp is not None else time.perf_counter()
        with self._lock:
            if self._has_unread:
                self._dropped_frames += 1
            self._latest_frame = frame
            self._latest_timestamp = ts
            self._frame_count += 1
            self._has_unread = True
            self._new_frame_event.set()

    def get_latest(self, wait_timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Retrieve the latest frame and its capture timestamp.
        
        Args:
            wait_timeout: If specified and no frame is unread, wait up to wait_timeout seconds.
            
        Returns:
            Tuple of (frame ndarray or None, capture timestamp)
        """
        if wait_timeout is not None and not self._has_unread:
            self._new_frame_event.wait(timeout=wait_timeout)

        with self._lock:
            frame = self._latest_frame
            ts = self._latest_timestamp
            self._has_unread = False
            self._new_frame_event.clear()
            return frame, ts

    def peek(self) -> Tuple[Optional[np.ndarray], float]:
        """View the latest frame without marking it as read."""
        with self._lock:
            return self._latest_frame, self._latest_timestamp

    def has_new_frame(self) -> bool:
        """Check if an unread frame is available."""
        with self._lock:
            return self._has_unread

    def clear(self) -> None:
        """Clear the frame buffer."""
        with self._lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0
            self._has_unread = False
            self._new_frame_event.clear()

    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames
