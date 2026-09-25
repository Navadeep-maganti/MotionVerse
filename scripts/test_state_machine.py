"""
Step 10 Gesture State Machine & Cooldown Test Script for MotionVerse.
Demonstrates discrete jump debouncing and continuous gesture holding states.
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
from app.core.gestures.common.gesture_state import GestureState


def run_state_machine_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 10: Gesture State Machine & Cooldown Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))
    gesture_engine = BodyGestureEngine(
        BodyGestureThresholds(
            move_left_trigger=0.20,
            move_left_release=0.12,
            move_right_trigger=0.20,
            move_right_release=0.12,
            crouch_trigger_ratio=0.76,
            crouch_release_ratio=0.84,
            jump_displacement_threshold=0.10,
            jump_velocity_threshold=0.60
        ),
        jump_cooldown_sec=0.45
    )

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

            features = extractor.extract(landmarks, baseline_hip=base_hip, baseline_scale=base_scale, timestamp=ts)
            status = calibrator.update(landmarks, features, current_time=ts)
            gesture_result = gesture_engine.process(features, baseline=baseline, timestamp=ts)

            if landmarks is not None:
                frame = detector.draw_skeleton(frame, landmarks)

            # HUD Background Panel
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (430, 270), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 10: State Machine | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            calib_status_str = "ACTIVE" if status == CalibrationStatus.SUCCESS else ("CALIBRATING..." if status == CalibrationStatus.COLLECTING else "NOT CALIBRATED [c]")
            cv2.putText(frame, f"Calibration: {calib_status_str}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if status == CalibrationStatus.SUCCESS else (0, 255, 255), 1)

            # State Machine & Action Display
            state_val = gesture_result.debug_info.get("state", "IDLE") if gesture_result.debug_info else "IDLE"
            action_val = gesture_result.action.value

            if state_val == "TRIGGERED":
                state_color = (0, 255, 0)     # Bright Green
            elif state_val == "HOLDING":
                state_color = (0, 215, 255)   # Gold
            elif state_val == "COOLDOWN":
                state_color = (0, 140, 255)   # Orange
            else:
                state_color = (180, 180, 180) # Gray

            # Big Action & State Box
            cv2.rectangle(frame, (20, 75), (410, 125), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 75), (410, 125), state_color, 2)
            cv2.putText(frame, f"DISPATCH: {action_val}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(frame, f"STATE: {state_val}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1)

            # Cooldown progress bar
            cooldown_rem = gesture_result.debug_info.get("cooldown_remaining", 0.0) if gesture_result.debug_info else 0.0
            if cooldown_rem > 0:
                prog = 1.0 - (cooldown_rem / 0.45)
                cv2.putText(frame, f"Jump Cooldown: {cooldown_rem:.2f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1)
                cv2.rectangle(frame, (20, 160), (320, 172), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 160), (20 + int(300 * prog), 172), (0, 140, 255), -1)
            else:
                cv2.putText(frame, "Jump Cooldown: READY", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

            # Telemetry
            cv2.putText(frame, f"Norm X: {features.normalized_x:+.3f} | Y Disp: {features.normalized_y_displacement:+.3f} | Ratio: {features.body_height_ratio:.2f}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(frame, f"Pipeline Latency: {gesture_engine.processing_latency_ms:.2f}ms", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 10: State Machine & Cooldown", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == 32:
                calibrator.start()
            elif key == ord('r'):
                calibrator.reset()
                extractor.reset()
                gesture_engine.reset()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_state_machine_test(idx)
