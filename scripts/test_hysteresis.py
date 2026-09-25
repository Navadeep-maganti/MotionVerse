"""
Step 9 Hysteresis Test Script for MotionVerse.
Demonstrates dual-threshold Schmitt-trigger bands for left/right and crouch gestures.
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


def run_hysteresis_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 9: Hysteresis (Dual Thresholds) Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)
    calibrator = CalibrationManager(CalibrationConfig(target_duration_sec=2.0, min_samples=25))
    gesture_engine = BodyGestureEngine(BodyGestureThresholds(
        move_left_trigger=0.20,
        move_left_release=0.12,
        move_right_trigger=0.20,
        move_right_release=0.12,
        crouch_trigger_ratio=0.76,
        crouch_release_ratio=0.84
    ))

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
            cv2.rectangle(hud, (10, 10), (420, 260), (25, 25, 25), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

            fps = camera.fps
            cv2.putText(frame, f"Step 9: Hysteresis | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            calib_status_str = "ACTIVE" if status == CalibrationStatus.SUCCESS else ("CALIBRATING..." if status == CalibrationStatus.COLLECTING else "NOT CALIBRATED [c]")
            cv2.putText(frame, f"Calibration: {calib_status_str}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if status == CalibrationStatus.SUCCESS else (0, 255, 255), 1)

            # Action Banner
            act = gesture_result.action
            if act == Action.MOVE_LEFT:
                act_color = (255, 100, 0)
                act_text = "◄◄ MOVE_LEFT (Active)"
            elif act == Action.MOVE_RIGHT:
                act_color = (0, 140, 255)
                act_text = "MOVE_RIGHT (Active) ►►"
            elif act == Action.CROUCH:
                act_color = (0, 215, 255)
                act_text = "▼▼ CROUCH (Active) ▼▼"
            elif act == Action.JUMP:
                act_color = (0, 255, 0)
                act_text = "▲▲ JUMP (Active) ▲▲"
            else:
                act_color = (180, 180, 180)
                act_text = "NONE (Neutral Deadzone)"

            cv2.rectangle(frame, (20, 75), (400, 120), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 75), (400, 120), act_color, 2)
            cv2.putText(frame, f"ACTION: {act_text}", (30, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.60, act_color, 2)

            # Hysteresis Telemetry
            cv2.putText(frame, f"Norm X: {features.normalized_x:+.3f} | Trigger: ±0.20 | Release: ±0.12", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(frame, f"Height Ratio: {features.body_height_ratio:.3f} | Trigger: 0.76 | Release: 0.84", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(frame, f"Latency: {gesture_engine.processing_latency_ms:.2f}ms (Filter: {gesture_engine.filtering_latency_ms:.2f}ms)", (20, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Visual Dual-Threshold Lines (Left/Right Trigger & Release)
            if baseline is not None:
                bx = int(baseline.hip_center_x * actual_w)
                b_scale_px = int(baseline.body_scale * actual_w)

                left_trig_px = int(bx - 0.20 * b_scale_px)
                left_rel_px = int(bx - 0.12 * b_scale_px)
                right_rel_px = int(bx + 0.12 * b_scale_px)
                right_trig_px = int(bx + 0.20 * b_scale_px)

                # Trigger lines (Solid)
                cv2.line(frame, (left_trig_px, 0), (left_trig_px, actual_h), (255, 80, 0), 2)
                cv2.line(frame, (right_trig_px, 0), (right_trig_px, actual_h), (0, 120, 255), 2)

                # Release lines (Dashed / Thinner)
                cv2.line(frame, (left_rel_px, 0), (left_rel_px, actual_h), (200, 150, 100), 1)
                cv2.line(frame, (right_rel_px, 0), (right_rel_px, actual_h), (100, 180, 200), 1)

                cv2.putText(frame, "L-Trig", (left_trig_px - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 0), 1)
                cv2.putText(frame, "L-Rel", (left_rel_px + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 100), 1)
                cv2.putText(frame, "R-Rel", (right_rel_px - 35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 200), 1)
                cv2.putText(frame, "R-Trig", (right_trig_px + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 255), 1)

            cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 9: Hysteresis", frame)

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
    run_hysteresis_test(idx)
