"""
Step 7 Crouch Gesture Test Script for MotionVerse.
Calibrates standing posture and evaluates real-time CROUCH action via body-height ratio & downward displacement.
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


def run_crouch_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 7: Crouch Gesture Detection Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))
    gesture_engine = BodyGestureEngine(BodyGestureThresholds(
        crouch_ratio_threshold=0.78,
        crouch_displacement_threshold=-0.08
    ))

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}).")
    print("Controls: Press 'c' to Calibrate | 'r' to Reset | 'q' to Quit\n")

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
            cv2.rectangle(hud, (10, 10), (380, 260), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 7: Crouch Detection | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
                calib_str = "Calibration: FAILED"
                calib_color = (0, 0, 255)

            cv2.putText(frame, calib_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_color, 1)

            # Action Banner
            act = gesture_result.action
            if act == Action.CROUCH:
                act_color = (0, 215, 255)   # Yellow/Gold
                act_text = "▼▼ CROUCH ACTIVE ▼▼"
            elif act == Action.JUMP:
                act_color = (0, 255, 0)
                act_text = "▲▲ JUMP ACTIVE ▲▲"
            elif act == Action.MOVE_LEFT:
                act_color = (255, 100, 0)
                act_text = "◄◄ MOVE_LEFT"
            elif act == Action.MOVE_RIGHT:
                act_color = (0, 140, 255)
                act_text = "MOVE_RIGHT ►►"
            else:
                act_color = (180, 180, 180)
                act_text = "NONE (Standing)"

            # Big Action Box
            cv2.rectangle(frame, (20, 75), (360, 120), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 75), (360, 120), act_color, 2)
            cv2.putText(frame, f"ACTION: {act_text}", (30, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.65, act_color, 2)

            # Crouch Metrics Telemetry
            ratio = features.body_height_ratio
            disp = features.normalized_y_displacement
            ratio_thresh = gesture_engine.thresholds.crouch_ratio_threshold
            disp_thresh = gesture_engine.thresholds.crouch_displacement_threshold

            ratio_color = (0, 215, 255) if ratio <= ratio_thresh else (255, 255, 255)
            disp_color = (0, 215, 255) if disp <= disp_thresh else (255, 255, 255)

            cv2.putText(frame, f"Height Ratio (R_t): {ratio:.3f} (Req: <= {ratio_thresh:.2f})", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ratio_color, 1)
            cv2.putText(frame, f"Downward Disp: {disp:+.3f} (Req: <= {disp_thresh:+.2f})", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.42, disp_color, 1)
            cv2.putText(frame, f"Latency: {gesture_engine.processing_latency_ms:.2f}ms", (20, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            # Crouch Visual Guide
            if baseline is not None:
                by = int(baseline.hip_center_y * actual_h)
                b_scale_px = int(baseline.body_scale * actual_h)
                crouch_hip_y_px = int(by - disp_thresh * b_scale_px) # Downward is positive in pixel y

                cv2.line(frame, (0, by), (actual_w, by), (0, 0, 255), 1)
                cv2.line(frame, (0, crouch_hip_y_px), (actual_w, crouch_hip_y_px), (0, 215, 255), 2)
                cv2.putText(frame, "CROUCH THRESHOLD", (actual_w - 200, crouch_hip_y_px + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1)

            cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 7: Crouch Detection", frame)

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
    run_crouch_test(idx)
