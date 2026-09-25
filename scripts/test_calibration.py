"""
Step 4 Calibration Test Script for MotionVerse.
Performs interactive multi-frame standing calibration, computes median baseline metrics,
and tests normalized coordinates with the active baseline.
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


def run_calibration_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 4: Calibration Routine Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}).")
    print("Controls: Press 'c' or SPACE to start calibration | 'r' to reset | 'q' to quit\n")

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detect pose
            landmarks = detector.detect(frame, timestamp=ts)

            # Get current baseline if available
            baseline = calibrator.baseline
            base_hip = (baseline.hip_center_x, baseline.hip_center_y) if baseline else None
            base_scale = baseline.body_scale if baseline else None

            # Extract features
            features = extractor.extract(landmarks, baseline_hip=base_hip, baseline_scale=base_scale, timestamp=ts)

            # Update calibration state machine
            status = calibrator.update(landmarks, features, current_time=ts)

            # Draw skeleton
            if landmarks is not None:
                frame = detector.draw_skeleton(frame, landmarks)

            # Calibration HUD Panel
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (340, 260), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 4: Calibration | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Status and Progress Bar
            if status == CalibrationStatus.IDLE:
                status_str = "Status: READY (Press [c] to Calibrate)"
                color = (255, 255, 0)
            elif status == CalibrationStatus.COLLECTING:
                prog = calibrator.progress
                status_str = f"CALIBRATING... {int(prog * 100)}%"
                color = (0, 165, 255)
                # Draw Progress Bar
                cv2.rectangle(frame, (20, 70), (320, 85), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 70), (20 + int(300 * prog), 85), (0, 255, 0), -1)
            elif status == CalibrationStatus.SUCCESS:
                status_str = "Status: CALIBRATED (SUCCESS)"
                color = (0, 255, 0)
            else:
                status_str = f"FAILED: {calibrator.error_message}"
                color = (0, 0, 255)

            cv2.putText(frame, status_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Baseline and Feature metrics
            if baseline is not None:
                lines = [
                    f"Baseline Hip: ({baseline.hip_center_x:.3f}, {baseline.hip_center_y:.3f})",
                    f"Baseline Scale (B0): {baseline.body_scale:.3f}",
                    f"Samples Analyzed: {baseline.sample_count}",
                    "--- Live Normalized Metrics ---",
                    f"Live Norm X: {features.normalized_x:+.3f}",
                    f"Live Norm Y Disp: {features.normalized_y_displacement:+.3f}",
                    f"Live Height Ratio: {features.body_height_ratio:.3f}"
                ]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (20, 105 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

                # Draw Baseline Crosshair on frame
                bx = int(baseline.hip_center_x * actual_w)
                by = int(baseline.hip_center_y * actual_h)
                cv2.drawMarker(frame, (bx, by), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            else:
                cv2.putText(frame, "Stand in neutral position and press [c]", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.putText(frame, "[c]: Start Calibration | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 4: Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == 32:  # 'c' or SPACE
                print("\n[INFO] Starting 2-second standing calibration...")
                calibrator.start()
            elif key == ord('r'):
                calibrator.reset()
                extractor.reset()
                print("\n[INFO] Calibration reset.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_calibration_test(idx)
