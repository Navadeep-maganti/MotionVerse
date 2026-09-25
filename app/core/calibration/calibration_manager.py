"""
Calibration Manager module for MotionVerse.
Orchestrates multi-frame calibration data collection, stillness validation, and robust median baseline computation.
"""

import enum
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from app.core.calibration.baseline import CalibrationBaseline
from app.core.vision.pose.pose_features import BodyFeatures
from app.core.vision.pose.pose_landmarks import PoseLandmarks


class CalibrationStatus(enum.Enum):
    """Lifecycle states of the calibration routine."""
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class CalibrationConfig:
    """Settings for calibration duration and quality constraints."""
    target_duration_sec: float = 2.0   # Collection window duration
    min_samples: int = 30              # Minimum valid frames required
    max_position_std_dev: float = 0.06 # Maximum allowed standard deviation in hip position (stillness check)


class CalibrationManager:
    """
    Collects body measurements over time, validates posture stability,
    and computes median baseline metrics.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.status = CalibrationStatus.IDLE
        self.error_message: str = ""

        # Collected sample buffers
        self._hip_x_samples: List[float] = []
        self._hip_y_samples: List[float] = []
        self._shoulder_x_samples: List[float] = []
        self._shoulder_y_samples: List[float] = []
        self._scale_samples: List[float] = []

        self._start_time: float = 0.0
        self._baseline: Optional[CalibrationBaseline] = None

    def start(self, start_time: Optional[float] = None) -> None:
        """Start or restart the calibration sequence."""
        self.status = CalibrationStatus.COLLECTING
        self.error_message = ""
        self._hip_x_samples.clear()
        self._hip_y_samples.clear()
        self._shoulder_x_samples.clear()
        self._shoulder_y_samples.clear()
        self._scale_samples.clear()
        self._start_time = start_time if start_time is not None else 0.0
        self._baseline = None

    def update(
        self,
        landmarks: Optional[PoseLandmarks],
        features: BodyFeatures,
        current_time: Optional[float] = None
    ) -> CalibrationStatus:
        """
        Feed the latest frame's features into the calibration accumulator.
        
        Returns:
            Current CalibrationStatus.
        """
        if self.status != CalibrationStatus.COLLECTING:
            return self.status

        now = current_time if current_time is not None else time.perf_counter()
        if self._start_time == 0.0:
            self._start_time = now

        # Reject frame if body is not properly detected
        if landmarks is None or not features.is_valid or not landmarks.has_core_landmarks():
            # If tracking was lost too long during collection, we report waiting
            return self.status

        # Accumulate valid feature metrics
        self._hip_x_samples.append(features.hip_center_x)
        self._hip_y_samples.append(features.hip_center_y)
        self._shoulder_x_samples.append(features.shoulder_center_x)
        self._shoulder_y_samples.append(features.shoulder_center_y)
        self._scale_samples.append(features.body_scale)

        elapsed = now - self._start_time

        # Check if duration criteria is reached
        if elapsed >= self.config.target_duration_sec:
            if len(self._hip_y_samples) < self.config.min_samples:
                self.status = CalibrationStatus.FAILED
                self.error_message = f"Not enough valid frames ({len(self._hip_y_samples)}/{self.config.min_samples})."
                return self.status

            # Calculate variance / standard deviation of hip position
            std_y = float(np.std(self._hip_y_samples))
            std_x = float(np.std(self._hip_x_samples))

            if std_y > self.config.max_position_std_dev or std_x > self.config.max_position_std_dev:
                self.status = CalibrationStatus.FAILED
                self.error_message = f"Too much movement during calibration (std: {max(std_x, std_y):.3f}). Please stand still."
                return self.status

            # Calculate robust medians
            median_hx = float(np.median(self._hip_x_samples))
            median_hy = float(np.median(self._hip_y_samples))
            median_sx = float(np.median(self._shoulder_x_samples))
            median_sy = float(np.median(self._shoulder_y_samples))
            median_scale = float(np.median(self._scale_samples))

            self._baseline = CalibrationBaseline(
                hip_center_x=median_hx,
                hip_center_y=median_hy,
                shoulder_center_x=median_sx,
                shoulder_center_y=median_sy,
                body_scale=median_scale,
                sample_count=len(self._hip_y_samples),
                std_dev_hip_y=std_y,
                timestamp=now,
                is_valid=True
            )
            self.status = CalibrationStatus.SUCCESS

        return self.status

    @property
    def progress(self) -> float:
        """Returns collection progress ratio between 0.0 and 1.0."""
        if self.status == CalibrationStatus.IDLE:
            return 0.0
        if self.status == CalibrationStatus.SUCCESS:
            return 1.0
        elapsed = time.perf_counter() - self._start_time
        return min(1.0, max(0.0, elapsed / max(0.1, self.config.target_duration_sec)))

    @property
    def baseline(self) -> Optional[CalibrationBaseline]:
        """Get the computed calibration baseline if successful."""
        return self._baseline

    def reset(self) -> None:
        """Reset calibration manager back to IDLE state."""
        self.status = CalibrationStatus.IDLE
        self.error_message = ""
        self._baseline = None
        self._hip_x_samples.clear()
        self._hip_y_samples.clear()
        self._shoulder_x_samples.clear()
        self._shoulder_y_samples.clear()
        self._scale_samples.clear()
