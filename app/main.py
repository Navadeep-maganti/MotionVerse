"""
MotionVerse — Phase 1 Body Controller Test Application.
Runs the refactored real-time discrete pipeline:
Webcam -> Bounded Frame Buffer -> MediaPipe Pose -> Feature Extraction -> Calibration -> Temporal Filtering -> Dynamic Reference Hysteresis -> Discrete Generic Actions.
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
from app.core.calibration.calibration_manager import CalibrationStatus
from app.core.gestures.common.actions import Action
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
    panel_w = 460
    panel_h = 350 if not debug_mode else 470
    cv2.rectangle(hud, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.80, frame, 0.20, 0, frame)

    fps = camera_manager.fps
    is_body_detected = landmarks is not None and features.is_valid
    calib_status = controller.calibrator.status
    baseline = controller.baseline
    engine = controller.engine

    # 1. Header & FPS
    cv2.putText(frame, f"MotionVerse — Discrete Body Controller", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Buffer: LATEST (No Stale Frames)", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # 2. Calibration Status
    if calib_status == CalibrationStatus.SUCCESS:
        calib_text = "Calibration: READY (ACTIVE)"
        calib_col = (0, 255, 0)
    elif calib_status == CalibrationStatus.COLLECTING:
        prog = controller.calibrator.progress
        calib_text = f"CALIBRATING... {int(prog * 100)}%"
        calib_col = (0, 165, 255)
        cv2.rectangle(frame, (20, 80), (20 + int(260 * prog), 90), (0, 255, 0), -1)
    elif calib_status == CalibrationStatus.FAILED:
        calib_text = "Calibration: FAILED (Press [c])"
        calib_col = (0, 0, 255)
    else:
        calib_text = "Calibration: NOT CALIBRATED (Press [c])"
        calib_col = (0, 255, 255)

    cv2.putText(frame, calib_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, calib_col, 1)

    # 3. Action Dispatch Banner
    action_val = gesture_result.action.value
    h_state = gesture_result.debug_info.get("horizontal_state", "READY") if gesture_result.debug_info else "READY"
    v_state = gesture_result.debug_info.get("vertical_state", "READY") if gesture_result.debug_info else "READY"

    if action_val == "JUMP":
        act_color = (0, 255, 0)       # Green
        symbol = "▲▲ JUMP (DISCRETE EVENT) ▲▲"
    elif action_val == "CROUCH":
        act_color = (0, 215, 255)     # Gold
        symbol = "▼▼ CROUCH (DISCRETE EVENT) ▼▼"
    elif action_val == "MOVE_LEFT":
        act_color = (255, 120, 0)     # Cyan/Blue
        symbol = "◄◄ MOVE_LEFT (1 EVENT)"
    elif action_val == "MOVE_RIGHT":
        act_color = (0, 140, 255)     # Orange
        symbol = "MOVE_RIGHT (1 EVENT) ►►"
    else:
        act_color = (180, 180, 180)   # Gray
        symbol = "NONE (Idle / Settled)"

    # Action Card Box
    box_y = 98
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 48), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 48), act_color, 2)
    cv2.putText(frame, f"DISPATCH: {symbol}", (30, box_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, act_color, 2)
    cv2.putText(frame, f"H-State: {h_state} | V-State: {v_state}", (30, box_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 4. Dynamic Horizontal Telemetry
    y_start = box_y + 68
    body_str = "YES" if is_body_detected else "NO (Searching...)"
    body_col = (0, 255, 0) if is_body_detected else (0, 0, 255)
    cv2.putText(frame, f"Body Detected: {body_str}", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.40, body_col, 1)

    if features.is_valid and baseline is not None and gesture_result.debug_info:
        dbg = gesture_result.debug_info
        curr_x = dbg.get("current_x", features.hip_center_x)
        ref_x = dbg.get("reference_x", curr_x)
        delta_x = dbg.get("delta_x", 0.0)
        h_vel = dbg.get("horizontal_velocity", 0.0)

        lines = [
            f"--- Dynamic Horizontal Reference ---",
            f"Current X:     {curr_x:.3f}",
            f"Reference X:   {ref_x:.3f} (Updates on settle)",
            f"Delta X (Dx):  {delta_x:+.3f} (Trigger: >= ±0.18)",
            f"Horiz Velocity: {h_vel:+.2f}/s (Settle req: <= 0.12)",
            f"--- Vertical Telemetry ---",
            f"Y Displacement: {features.normalized_y_displacement:+.3f} (Jump req: >= +0.10)",
            f"Body Ratio:     {features.body_height_ratio:.3f} (Crouch req: <= 0.76)"
        ]
        for i, l in enumerate(lines):
            col = (0, 255, 255) if "---" in l else (255, 255, 255)
            cv2.putText(frame, l, (20, y_start + 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    else:
        cv2.putText(frame, "Stand in view of camera & press [c] to calibrate", (20, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # 5. Debug Latency Overlay
    if debug_mode and gesture_result.debug_info:
        dbg_y = y_start + 155
        cv2.line(frame, (20, dbg_y), (panel_w, dbg_y), (80, 80, 80), 1)
        dbg_lines = [
            f"-- Latency Telemetry --",
            f"Camera Capture:  {camera_manager.capture_latency_ms:.2f}ms",
            f"Pose Inference:  {controller.detector.inference_latency_ms:.2f}ms",
            f"Filtering Time:  {controller.engine.filtering_latency_ms:.3f}ms",
            f"Gesture Engine:  {controller.engine.processing_latency_ms:.3f}ms",
            f"Total Pipe Est:  {camera_manager.capture_latency_ms + controller.detector.inference_latency_ms + controller.engine.processing_latency_ms:.2f}ms"
        ]
        for i, l in enumerate(dbg_lines):
            cv2.putText(frame, l, (20, dbg_y + 14 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)

    # Visual Dynamic Reference Crosshair
    if baseline is not None and engine.horizontal_reference_x is not None:
        rx_px = int(engine.horizontal_reference_x * actual_w)
        hy_px = int(features.hip_center_y * actual_h) if features.is_valid else int(baseline.hip_center_y * actual_h)
        b_scale_px = int(baseline.body_scale * actual_w)

        # Dynamic Reference Marker (Cyan Crosshair)
        cv2.drawMarker(frame, (rx_px, hy_px), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)

        # Trigger Boundaries relative to current dynamic reference
        left_trig_px = int(rx_px - 0.18 * b_scale_px)
        right_trig_px = int(rx_px + 0.18 * b_scale_px)
        cv2.line(frame, (left_trig_px, 0), (left_trig_px, actual_h), (255, 100, 0), 1)
        cv2.line(frame, (right_trig_px, 0), (right_trig_px, actual_h), (0, 140, 255), 1)

    # Footer Controls
    cv2.putText(frame, "[c]: Calibrate | [r]: Reset | [d]: Toggle Debug | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    return frame


def run_body_controller_app(camera_index: int = 0):
    """Main execution loop for Phase 1 Discrete Body Controller."""
    print("=" * 65)
    print("   MotionVerse — Discrete Real-Time Body Controller")
    print("=" * 65)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    controller = BodyModeController()

    if not camera.start():
        print("[ERROR] Camera unavailable. Please check webcam connection.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[STATUS] Camera Connected ({actual_w}x{actual_h} @ target 30 FPS).")
    print("[STATUS] Dynamic Reference & Discrete Event State Machine active.")
    print("[STATUS] Ready. Stand in view and press 'c' to calibrate.\n")

    debug_mode = True

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            gesture_result, landmarks, features = controller.process_frame(frame, timestamp=ts)

            if landmarks is not None:
                frame = controller.detector.draw_skeleton(frame, landmarks)

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
                print("\n[INFO] Exit requested.")
                break
            elif key == ord('c') or key == 32:
                print("\n[INFO] Starting standing calibration (stand still)...")
                controller.start_calibration()
            elif key == ord('r'):
                print("\n[INFO] Resetting pipeline and dynamic references.")
                controller.reset()
            elif key == ord('d'):
                debug_mode = not debug_mode

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        camera.stop()
        controller.detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_body_controller_app(camera_index=idx)
