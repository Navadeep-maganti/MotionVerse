"""
Camera configuration settings for MotionVerse.
"""

from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration parameters for video capture."""
    camera_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    buffer_size: int = 1
    flip_horizontal: bool = True
    use_threading: bool = True
