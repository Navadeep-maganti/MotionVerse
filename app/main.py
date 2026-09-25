"""
MotionVerse — Phase 1 Body Controller Test Application.
Runs the complete real-time pipeline:
Webcam -> Bounded Frame Buffer -> MediaPipe Pose -> Feature Extraction -> Calibration -> Temporal Filtering -> Hysteresis -> State Machine -> Generic Action.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.camera.camera_manager import CameraManager
from app.core.camera.camera_config import CameraConfig
from app.core.calibration.calibration_manager import CalibrationStatus, CalibrationConfig
from app.core.gestures.common.actions import Action
from app.core.gestures.common.gesture_state import GestureState
from app.modes.body_mode import BodyModeController


def draw_hud_panel(
    frame: np.ndarray,
    camera_manager: CameraManager,
    controller: BodyModeController,
    gesture_result,
    features,
    landmarks,
    debug_mode: bool,
    actual_w: int,
    actual_h: int
) -> np.ndarray:
    """Draw a clean, informative HUD dashboard over the camera frame."""
    hud = frame.copy()
    panel_w = 420 if not debug_mode else 480
    panel_h = 320 if not debug_mode else 440
    cv2.rectangle(hud, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.78, frame, 0.22, 0, frame)

    fps = camera_manager.fps
    is_body_detected = landmarks is not None and features.is_valid
    calib_status = controller.calibrator.status
    baseline = controller.baseline

    # 1. Header & System Status
    cv2.putText(frame, f"MotionVerse — Body Controller", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Buffer: LATEST", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # 2. Calibration Status
    if calib_status == CalibrationStatus.SUCCESS:
        calib_text = "Calibration: READY (ACTIVE)"
        calib_col = (0, 255, 0)
    elif calib_status == CalibrationStatus.COLLECTING:
        prog = controller.calibrator.progress
        calib_text = f"CALIBRATING... {int(prog * 100)}%"
        calib_col = (0, 165, 255)
        # Progress bar
        cv2.rectangle(frame, (20, 80), (20 + int(260 * prog), 90), (0, 255, 0), -1)
    elif calib_status == CalibrationStatus.FAILED:
        calib_text = f"Calibration: FAILED (Press [c])"
        calib_col = (0, 0, 255)
    else:
        calib_text = "Calibration: NOT CALIBRATED (Press [c])"
        calib_col = (0, 255, 255)

    cv2.putText(frame, calib_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, calib_col, 1)

    # 3. Action Dispatch Banner
    action_val = gesture_result.action.value
    state_val = gesture_result.debug_info.get("state", "IDLE") if gesture_result.debug_info else "IDLE"

    if action_val == "JUMP":
        act_color = (0, 255, 0)       # Green
        symbol = "▲▲ JUMP ▲▲"
    elif action_val == "CROUCH":
        act_color = (0, 215, 255)     # Gold
        symbol = "▼▼ CROUCH ▼▼"
    elif action_val == "MOVE_LEFT":
        act_color = (255, 120, 0)     # Cyan/Blue
        symbol = "◄◄ MOVE_LEFT"
    elif action_val == "MOVE_RIGHT":
        act_color = (0, 140, 255)     # Orange
        symbol = "MOVE_RIGHT ►►"
    else:
        act_color = (180, 180, 180)   # Gray
        symbol = "NONE (Neutral)"

    # Action Card Box
    box_y = 100
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 50), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 50), act_color, 2)
    cv2.putText(frame, f"ACTION: {symbol}", (30, box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, act_color, 2)
    cv2.putText(frame, f"State: {state_val} | Engine: ACTIVE", (30, box_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # 4. Body Features Telemetry
    y_start = box_y + 70
    body_str = "YES" if is_body_detected else "NO (Searching...)"
    body_col = (0, 255, 0) if is_body_detected else (0, 0, 255)

    cv2.putText(frame, f"Body Detected: {body_str}", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.42, body_col, 1)

    if features.is_valid and baseline is not None:
        lines = [
            f"X Displacement:   {features.normalized_x:+.3f} (±0.20 trig / ±0.12 rel)",
            f"Y Displacement:   {features.normalized_y_displacement:+.3f} (Jump req: >= +0.10)",
            f"Vert Velocity:    {features.vertical_velocity:+.2f}/s (Jump req: >= +0.60)",
            f"Horiz Velocity:   {features.horizontal_velocity:+.2f}/s",
            f"Body Ratio (R_t): {features.body_height_ratio:.3f} (Crouch req: <= 0.76)"
        ]
        for i, l in enumerate(lines):
            cv2.putText(frame, l, (20, y_start + 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Stand in view of camera & press [c] to calibrate", (20, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # 5. Debug Overlay (Toggled with 'd')
    if debug_mode:
        dbg_y = y_start + 120
        cv2.line(frame, (20, dbg_y), (panel_w, dbg_y), (80, 80, 80), 1)
        dbg_lines = [
            f"-- Latency Telemetry --",
            f"Camera Capture:  {camera_manager.capture_latency_ms:.2f}ms",
            f"Pose Inference:  {controller.detector.inference_latency_ms:.2f}ms",
            f"Filtering Time:  {controller.engine.filtering_latency_ms:.3f}ms",
            f"Gesture Engine:  {controller.engine.processing_latency_ms:.3f}ms",
            f"Total Est. Pipe: {camera_manager.capture_latency_ms + controller.detector.inference_latency_ms + controller.engine.processing_latency_ms:.2f}ms",
            f"Jump Cooldown:   {controller.engine.state_machine.cooldown_tracker.remaining_cooldown_sec(Action.JUMP):.2f}s"
        ]
        for i, l in enumerate(dbg_lines):
            cv2.putText(frame, l, (20, dbg_y + 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Footer Controls
    cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [d]: Toggle Debug | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    return frame


def run_body_controller_app(camera_index: int = 0):
    """Main execution loop for Phase 1 Body Controller."""
    print("=" * 65)
    print("   MotionVerse — Real-Time Body Controller Test Application")
    print("=" * 65)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    controller = BodyModeController()

    if not camera.start():
        print("[ERROR] Camera unavailable. Please ensure a webcam is connected and accessible.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[STATUS] Camera Connected ({actual_w}x{actual_h} @ target 30 FPS).")
    print("[STATUS] Latest-frame buffer active (zero stale frames).")
    print("[STATUS] Ready. Press 'c' to calibrate.\n")

    debug_mode = True

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process frame end-to-end
            gesture_result, landmarks, features = controller.process_frame(frame, timestamp=ts)

            # Draw skeleton
            if landmarks is not None:
                frame = controller.detector.draw_skeleton(frame, landmarks)

            # Draw HUD
            frame = draw_hud_panel(
                frame,
                camera,
                controller,
                gesture_result,
                features,
                landmarks,
                debug_mode,
                actual_w,
                actual_h
            )

            cv2.imshow("MotionVerse — Phase 1 Body Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] Exit requested by user.")
                break
            elif key == ord('c') or key == 32:
                print("\n[INFO] Initiating standing calibration (stand still)...")
                controller.start_calibration()
            elif key == ord('r'):
                print("\n[INFO] Pipeline reset.")
                controller.reset()
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug overlay: {'ON' if debug_mode else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera.stop()
        controller.detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Resources released cleanly. MotionVerse shutdown complete.")

    return True


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_body_controller_app(camera_index=idx)
