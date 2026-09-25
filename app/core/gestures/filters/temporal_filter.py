"""
Temporal Filtering module for MotionVerse.
Provides lightweight Exponential Moving Average (EMA) filtering to smooth noise with minimal latency.
"""

import time
from typing import Optional, Dict
from app.core.vision.pose.pose_features import BodyFeatures


class ExponentialMovingAverage:
    """
    Lightweight 1D Exponential Moving Average filter.
    S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    """

    def __init__(self, alpha: float = 0.75):
        """
        Args:
            alpha: Smoothing factor in (0.0, 1.0]. Higher values = faster response, lower smoothing.
                   Default 0.75 provides clean noise reduction without perceptual latency.
        """
        self.alpha = max(0.01, min(1.0, alpha))
        self._state: Optional[float] = None

    def filter(self, value: float) -> float:
        """Apply EMA smoothing to a scalar sample."""
        if self._state is None:
            self._state = value
        else:
            self._state = (self.alpha * value) + ((1.0 - self.alpha) * self._state)
        return self._state

    @property
    def value(self) -> Optional[float]:
        return self._state

    def reset(self) -> None:
        """Reset internal filter state."""
        self._state = None


class FeatureTemporalFilter:
    """
    Multi-channel temporal filter smoothing kinematic body features.
    """

    def __init__(
        self,
        position_alpha: float = 0.75,
        velocity_alpha: float = 0.65,
        scale_alpha: float = 0.80
    ):
        self.position_alpha = position_alpha
        self.velocity_alpha = velocity_alpha
        self.scale_alpha = scale_alpha

        self._filter_x = ExponentialMovingAverage(position_alpha)
        self._filter_y_disp = ExponentialMovingAverage(position_alpha)
        self._filter_vx = ExponentialMovingAverage(velocity_alpha)
        self._filter_vy = ExponentialMovingAverage(velocity_alpha)
        self._filter_ratio = ExponentialMovingAverage(scale_alpha)

        self._last_filter_latency_ms: float = 0.0

    def filter(self, features: BodyFeatures) -> BodyFeatures:
        """
        Apply temporal smoothing across normalized body features.
        
        Args:
            features: Raw extracted BodyFeatures.
            
        Returns:
            Smoothed BodyFeatures instance.
        """
        if not features.is_valid:
            self.reset()
            return features

        t_start = time.perf_counter()

        filtered_x = self._filter_x.filter(features.normalized_x)
        filtered_y_disp = self._filter_y_disp.filter(features.normalized_y_displacement)
        filtered_vx = self._filter_vx.filter(features.horizontal_velocity)
        filtered_vy = self._filter_vy.filter(features.vertical_velocity)
        filtered_ratio = self._filter_ratio.filter(features.body_height_ratio)

        self._last_filter_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return BodyFeatures(
            hip_center_x=features.hip_center_x,
            hip_center_y=features.hip_center_y,
            shoulder_center_x=features.shoulder_center_x,
            shoulder_center_y=features.shoulder_center_y,
            body_scale=features.body_scale,
            normalized_x=filtered_x,
            normalized_y_displacement=filtered_y_disp,
            vertical_velocity=filtered_vy,
            horizontal_velocity=filtered_vx,
            body_height_ratio=filtered_ratio,
            timestamp=features.timestamp,
            is_valid=True
        )

    def reset(self) -> None:
        """Reset all feature filter channels."""
        self._filter_x.reset()
        self._filter_y_disp.reset()
        self._filter_vx.reset()
        self._filter_vy.reset()
        self._filter_ratio.reset()

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds of the last filter operation."""
        return self._last_filter_latency_ms
