"""
Unit tests for Temporal Filtering (Step 8).
"""

import pytest
import numpy as np

from app.core.gestures.filters.temporal_filter import ExponentialMovingAverage, FeatureTemporalFilter
from app.core.vision.pose.pose_features import BodyFeatures


def test_ema_initialization():
    ema = ExponentialMovingAverage(alpha=0.5)
    assert ema.value is None

    # First sample becomes state directly
    val = ema.filter(10.0)
    assert val == 10.0
    assert ema.value == 10.0


def test_ema_formula_step_response():
    # S_t = 0.5 * X_t + 0.5 * S_{t-1}
    ema = ExponentialMovingAverage(alpha=0.5)
    ema.filter(0.0)  # S_0 = 0.0

    # Step to 10.0
    s1 = ema.filter(10.0)  # 0.5 * 10.0 + 0.5 * 0.0 = 5.0
    assert pytest.approx(s1, 0.001) == 5.0

    s2 = ema.filter(10.0)  # 0.5 * 10.0 + 0.5 * 5.0 = 7.5
    assert pytest.approx(s2, 0.001) == 7.5

    s3 = ema.filter(10.0)  # 0.5 * 10.0 + 0.5 * 7.5 = 8.75
    assert pytest.approx(s3, 0.001) == 8.75


def test_ema_reset():
    ema = ExponentialMovingAverage(alpha=0.8)
    ema.filter(50.0)
    assert ema.value == 50.0

    ema.reset()
    assert ema.value is None
    # Next sample should re-initialize cleanly
    assert ema.filter(20.0) == 20.0


def test_feature_temporal_filter():
    filter_engine = FeatureTemporalFilter(position_alpha=0.5, velocity_alpha=0.5, scale_alpha=0.5)

    f1 = BodyFeatures(
        normalized_x=0.0,
        normalized_y_displacement=0.0,
        horizontal_velocity=0.0,
        vertical_velocity=0.0,
        body_height_ratio=1.0,
        is_valid=True
    )
    res1 = filter_engine.filter(f1)
    assert res1.normalized_x == 0.0
    assert res1.body_height_ratio == 1.0

    # Sudden noisy step
    f2 = BodyFeatures(
        normalized_x=0.40,
        normalized_y_displacement=0.20,
        horizontal_velocity=2.0,
        vertical_velocity=1.0,
        body_height_ratio=0.80,
        is_valid=True
    )
    res2 = filter_engine.filter(f2)

    # 0.5 * 0.40 + 0.5 * 0.0 = 0.20
    assert pytest.approx(res2.normalized_x, 0.001) == 0.20
    assert pytest.approx(res2.normalized_y_displacement, 0.001) == 0.10
    assert pytest.approx(res2.horizontal_velocity, 0.001) == 1.0
    assert pytest.approx(res2.body_height_ratio, 0.001) == 0.90
    assert filter_engine.latency_ms >= 0.0


def test_filter_invalid_features_resets():
    filter_engine = FeatureTemporalFilter()
    f_valid = BodyFeatures(normalized_x=0.5, is_valid=True)
    filter_engine.filter(f_valid)

    f_invalid = BodyFeatures(normalized_x=0.0, is_valid=False)
    out = filter_engine.filter(f_invalid)
    assert out.is_valid is False
