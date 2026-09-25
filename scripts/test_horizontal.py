"""
Step 5 Horizontal Movement Test Script for MotionVerse.
Calibrates standing posture and evaluates real-time MOVE_LEFT, MOVE_RIGHT, and NONE actions.
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
from app.core.gestures.body.body_rules import BodyGestureThresholds
from app.core.gestures.common.actions import Action


def run_horizontal_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 5: Horizontal Movement (LEFT / RIGHT) Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))
    gesture_engine = BodyGestureEngine(BodyGestureThresholds(move_left_threshold=0.18, move_right_threshold=0.18))

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}).")
    print("Press 'c' to Calibrate, 'r' to Reset, 'q' to Quit.\n")

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detect pose
            landmarks = detector.detect(frame, timestamp=ts)

            # Get active baseline
            baseline = calibrator.baseline
            base_hip = (baseline.hip_center_x, baseline.hip_center_y) if baseline else None
            base_scale = baseline.body_scale if baseline else None

            # Extract features
            features = extractor.extract(landmarks, baseline_hip=base_hip, baseline_scale=base_scale, timestamp=ts)

            # Update calibration
            status = calibrator.update(landmarks, features, current_time=ts)

            # Evaluate gestures
            gesture_result = gesture_engine.process(features, baseline=baseline, timestamp=ts)

            # Draw skeleton
            if landmarks is not None:
                frame = detector.draw_skeleton(frame, landmarks)

            # HUD Background Panel
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (360, 240), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 5: Horizontal Motion | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Calibration Status Display
            if status == CalibrationStatus.IDLE:
                calib_str = "Calibration: NOT CALIBRATED (Press [c])"
                calib_color = (0, 255, 255)
            elif status == CalibrationStatus.COLLECTING:
                calib_str = f"CALIBRATING... {int(calibrator.progress * 100)}%"
                calib_color = (0, 165, 255)
            elif status == CalibrationStatus.SUCCESS:
                calib_str = "Calibration: ACTIVE"
                calib_color = (0, 255, 0)
            else:
                calib_str = f"Calibration: FAILED"
                calib_color = (0, 0, 255)

            cv2.putText(frame, calib_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_color, 1)

            # Recognized Action Banner
            act = gesture_result.action
            if act == Action.MOVE_LEFT:
                act_color = (255, 100, 0)   # Blue/Cyan
                arrow_text = "◄◄ MOVE_LEFT"
            elif act == Action.MOVE_RIGHT:
                act_color = (0, 140, 255)   # Orange
                arrow_text = "MOVE_RIGHT ►►"
            else:
                act_color = (180, 180, 180) # Gray
                arrow_text = "NONE (Center)"

            # Big Action Box
            cv2.rectangle(frame, (20, 75), (340, 120), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 75), (340, 120), act_color, 2)
            cv2.putText(frame, f"ACTION: {arrow_text}", (30, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.65, act_color, 2)

            # Feature telemetry
            cv2.putText(frame, f"Normalized X: {features.normalized_x:+.3f} (Threshold: ±0.18)", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(frame, f"Horiz Velocity (V_x): {features.horizontal_velocity:+.2f}/s", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(frame, f"Latency: {gesture_engine.processing_latency_ms:.2f}ms", (20, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            # Visual threshold guide zones
            if baseline is not None:
                bx = int(baseline.hip_center_x * actual_w)
                b_scale_px = int(baseline.body_scale * actual_w)
                left_zone_px = int(bx - 0.18 * b_scale_px)
                right_zone_px = int(bx + 0.18 * b_scale_px)

                # Draw neutral center and threshold boundary lines
                cv2.line(frame, (bx, 0), (bx, actual_h), (100, 100, 100), 1, cv2.LINE_DASH if hasattr(cv2, 'LINE_DASH') else cv2.LINE_AA)
                cv2.line(frame, (left_zone_px, 0), (left_zone_px, actual_h), (255, 100, 0), 1)
                cv2.line(frame, (right_zone_px, 0), (right_zone_px, actual_h), (0, 140, 255), 1)

            cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 5: Horizontal Movement", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == 32:
                calibrator.start()
            elif key == ord('r'):
                calibrator.reset()
                extractor.reset()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_horizontal_test(idx)
