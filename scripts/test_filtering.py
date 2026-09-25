"""
Step 8 Temporal Filtering Test Script for MotionVerse.
Compares raw noisy features against filtered features in real time.
"""

import sys
import os
import cv2
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.camera.camera_manager import CameraManager
from app.core.camera.camera_config import CameraConfig
from app.core.vision.pose.pose_detector import PoseDetector, PoseDetectorConfig
from app.core.vision.pose.pose_features import BodyFeatureExtractor
from app.core.calibration.calibration_manager import CalibrationManager, CalibrationStatus, CalibrationConfig
from app.core.gestures.body.body_gesture_engine import BodyGestureEngine
from app.core.gestures.filters.temporal_filter import FeatureTemporalFilter


def run_filtering_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 8: Temporal Filtering (EMA) Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))
    filter_engine = FeatureTemporalFilter(position_alpha=0.75, velocity_alpha=0.65, scale_alpha=0.80)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}).")
    print("Controls: 'c' to Calibrate | 'r' to Reset | 'q' to Quit\n")

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            landmarks = detector.detect(frame, timestamp=ts)

            baseline = calibrator.baseline
            base_hip = (baseline.hip_center_x, baseline.hip_center_y) if baseline else None
            base_scale = baseline.body_scale if baseline else None

            # Raw features
            raw_features = extractor.extract(landmarks, baseline_hip=base_hip, baseline_scale=base_scale, timestamp=ts)

            # Filtered features
            filtered_features = filter_engine.filter(raw_features)

            status = calibrator.update(landmarks, raw_features, current_time=ts)

            if landmarks is not None:
                frame = detector.draw_skeleton(frame, landmarks)

            # Telemetry Comparison Panel
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (420, 260), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 8: Temporal Filter | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            calib_status_str = "ACTIVE" if status == CalibrationStatus.SUCCESS else ("CALIBRATING..." if status == CalibrationStatus.COLLECTING else "NOT CALIBRATED [c]")
            cv2.putText(frame, f"Calibration: {calib_status_str}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if status == CalibrationStatus.SUCCESS else (0, 255, 255), 1)

            # Header
            cv2.putText(frame, "Metric          Raw Sample       Filtered (EMA)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.line(frame, (20, 92), (400, 92), (100, 100, 100), 1)

            if raw_features.is_valid:
                rows = [
                    ("Norm X:       ", f"{raw_features.normalized_x:+.4f}", f"{filtered_features.normalized_x:+.4f}"),
                    ("Y Disp:       ", f"{raw_features.normalized_y_displacement:+.4f}", f"{filtered_features.normalized_y_displacement:+.4f}"),
                    ("V_x (Horiz):  ", f"{raw_features.horizontal_velocity:+.3f}/s", f"{filtered_features.horizontal_velocity:+.3f}/s"),
                    ("V_y (Vert):   ", f"{raw_features.vertical_velocity:+.3f}/s", f"{filtered_features.vertical_velocity:+.3f}/s"),
                    ("Height Ratio: ", f"{raw_features.body_height_ratio:.4f}", f"{filtered_features.body_height_ratio:.4f}")
                ]
                for i, (label, raw_val, filt_val) in enumerate(rows):
                    y_pos = 115 + i * 22
                    cv2.putText(frame, label, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
                    cv2.putText(frame, raw_val, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1)
                    cv2.putText(frame, filt_val, (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

            cv2.putText(frame, f"Filter Latency: {filter_engine.latency_ms:.3f}ms", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
            cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 8: Temporal Filtering", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == 32:
                calibrator.start()
            elif key == ord('r'):
                calibrator.reset()
                extractor.reset()
                filter_engine.reset()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_filtering_test(idx)
