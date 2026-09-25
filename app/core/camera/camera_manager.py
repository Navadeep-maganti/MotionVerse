"""
Camera Manager module for MotionVerse.
Manages camera capture, background capture worker, FPS tracking, and graceful fallback.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.core.camera.camera_config import CameraConfig
from app.core.camera.frame_buffer import FrameBuffer


class CameraManager:
    """
    Manages webcam capture lifecycle, latest-frame buffering, and FPS measurement.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.buffer = FrameBuffer(max_size=self.config.buffer_size)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Performance & FPS tracking
        self._fps: float = 0.0
        self._smoothed_fps: float = 0.0
        self._frame_count: int = 0
        self._last_fps_calc_time: float = 0.0
        self._fps_history_alpha: float = 0.9  # Exponential smoothing factor
        self._last_frame_time: float = 0.0
        self._capture_latency_ms: float = 0.0

    def start(self) -> bool:
        """
        Initialize the camera and start the frame capture thread.
        
        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._is_running:
            return True

        # Try opening camera with DirectShow on Windows, fallback to default if needed
        self._cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.config.camera_index)

        if not self._cap.isOpened():
            self._cap = None
            return False

        # Apply camera configurations
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self._is_running = True
        self._last_fps_calc_time = time.perf_counter()
        self._last_frame_time = time.perf_counter()

        if self.config.use_threading:
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()

        return True

    def _capture_worker(self) -> None:
        """Background thread continuously pulling latest frames into the buffer."""
        while self._is_running and self._cap is not None:
            t_start = time.perf_counter()
            ret, frame = self._cap.read()
            t_capture = time.perf_counter()

            if not ret or frame is None:
                # Brief sleep if capture momentarily fails or stalls
                time.sleep(0.005)
                continue

            self._capture_latency_ms = (t_capture - t_start) * 1000.0

            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            self._update_fps(t_capture)
            self.buffer.put(frame, timestamp=t_capture)

    def _update_fps(self, current_time: float) -> None:
        """Update instantaneous and smoothed FPS measurements."""
        self._frame_count += 1
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        if dt > 0:
            inst_fps = 1.0 / dt
            if self._smoothed_fps == 0.0:
                self._smoothed_fps = inst_fps
            else:
                self._smoothed_fps = (self._fps_history_alpha * self._smoothed_fps) + ((1.0 - self._fps_history_alpha) * inst_fps)
            self._fps = inst_fps

    def read_frame(self, timeout: Optional[float] = 0.05) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Get the latest captured frame.
        
        Returns:
            Tuple of (success: bool, frame: np.ndarray or None, timestamp: float)
        """
        if not self._is_running:
            return False, None, 0.0

        if not self.config.use_threading:
            # Synchronous read mode
            if self._cap is None:
                return False, None, 0.0
            t_start = time.perf_counter()
            ret, frame = self._cap.read()
            t_capture = time.perf_counter()
            if not ret or frame is None:
                return False, None, 0.0
            self._capture_latency_ms = (t_capture - t_start) * 1000.0
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            self._update_fps(t_capture)
            return True, frame, t_capture

        frame, ts = self.buffer.get_latest(wait_timeout=timeout)
        if frame is None:
            return False, None, 0.0
        return True, frame, ts

    @property
    def is_opened(self) -> bool:
        """Check if camera device is active and running."""
        return self._is_running and self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get smoothed frames per second."""
        return self._smoothed_fps

    @property
    def capture_latency_ms(self) -> float:
        """Get time taken by driver to read a frame in milliseconds."""
        return self._capture_latency_ms

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Get actual width and height of video stream."""
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.config.width, self.config.height

    def stop(self) -> None:
        """Stop camera capture and release resources gracefully."""
        self._is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self.buffer.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
