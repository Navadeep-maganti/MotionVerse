"""
Performance benchmarking script for MotionVerse Phase 1.
Measures latency breakdowns across each stage of the pipeline over N iterations.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.vision.pose.pose_detector import PoseDetector, PoseDetectorConfig
from app.core.vision.pose.pose_features import BodyFeatureExtractor, BodyFeatures
from app.core.calibration.baseline import CalibrationBaseline
from app.core.gestures.filters.temporal_filter import FeatureTemporalFilter
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.body.body_rules import BodyGestureThresholds


def run_benchmark(iterations: int = 100):
    print("=" * 65)
    print("      MotionVerse — Phase 1 Performance Benchmark")
    print("=" * 65)
    print(f"Benchmarking pipeline components over {iterations} iterations...\n")

    # Initialize components
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor()
    temporal_filter = FeatureTemporalFilter()
    gesture_engine = BodyGestureEngine(enable_temporal_filtering=True)

    baseline = CalibrationBaseline(
        hip_center_x=0.5,
        hip_center_y=0.5,
        shoulder_center_x=0.5,
        shoulder_center_y=0.2,
        body_scale=0.6
    )

    # Synthetic 640x480 test image
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    inference_times = []
    feature_times = []
    filter_times = []
    gesture_times = []
    total_pipeline_times = []

    # Warmup (10 runs)
    for _ in range(10):
        detector.detect(test_frame)

    print("Running timed inference & processing passes...")
    for i in range(iterations):
        t0 = time.perf_counter()

        # 1. Pose Inference
        t_inf_start = time.perf_counter()
        landmarks = detector.detect(test_frame, timestamp=t0)
        t_inf_end = time.perf_counter()

        # 2. Feature Extraction
        t_feat_start = time.perf_counter()
        features = extractor.extract(landmarks, baseline_hip=(0.5, 0.5), baseline_scale=0.6, timestamp=t_inf_end)
        t_feat_end = time.perf_counter()

        # 3. Temporal Filtering (using synthetic sample if no person detected in noise)
        mock_feat = BodyFeatures(
            normalized_x=0.15,
            normalized_y_displacement=0.05,
            horizontal_velocity=0.5,
            vertical_velocity=0.2,
            body_height_ratio=0.95,
            is_valid=True,
            timestamp=t_feat_end
        )
        t_filt_start = time.perf_counter()
        filtered = temporal_filter.filter(mock_feat)
        t_filt_end = time.perf_counter()

        # 4. Gesture Engine Processing
        t_gest_start = time.perf_counter()
        result = gesture_engine.process(filtered, baseline=baseline, timestamp=t_filt_end)
        t_gest_end = time.perf_counter()

        t_total = t_gest_end - t0

        inference_times.append((t_inf_end - t_inf_start) * 1000.0)
        feature_times.append((t_feat_end - t_feat_start) * 1000.0)
        filter_times.append((t_filt_end - t_filt_start) * 1000.0)
        gesture_times.append((t_gest_end - t_gest_start) * 1000.0)
        total_pipeline_times.append(t_total * 1000.0)

    detector.close()

    # Results Table
    print("\n" + "-" * 65)
    print(f"{'Stage':<28} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 65)

    def stats(arr):
        return f"{np.mean(arr):.3f}", f"{np.min(arr):.3f}", f"{np.max(arr):.3f}"

    for name, data in [
        ("1. MediaPipe Pose Inference", inference_times),
        ("2. Body Feature Extraction", feature_times),
        ("3. Temporal EMA Filtering", filter_times),
        ("4. Gesture State & Hysteresis", gesture_times),
        ("Total Pipeline Latency", total_pipeline_times)
    ]:
        avg_s, min_s, max_s = stats(data)
        print(f"{name:<28} | {avg_s:<10} | {min_s:<10} | {max_s:<10}")

    print("-" * 65)
    mean_lat = np.mean(total_pipeline_times)
    est_fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    print(f"Estimated Max Pipeline Throughput: {est_fps:.1f} FPS (Mean: {mean_lat:.2f}ms/frame)\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(iterations=n)
