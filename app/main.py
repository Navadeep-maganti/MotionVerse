"""
MotionVerse — Browser Game Controller Application.
Runs the complete end-to-end pipeline:
Webcam -> MediaPipe Pose -> Feature Extraction -> Calibration -> Gesture Engine -> Generic Actions -> GameMapper -> KeyboardController -> Browser Game.
"""

import sys
import os
import time
from typing import Optional, Dict, Any, Tuple, List
import cv2
import numpy as np

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.camera.camera_manager import CameraManager
from app.core.camera.camera_config import CameraConfig
from app.core.calibration.calibration_manager import CalibrationStatus
from app.core.gestures.common.actions import Action
from app.games.game_profile import GameProfile
from app.games.game_mapper import GameMapper
from app.input.keyboard_controller import KeyboardController
from app.modes.body_mode import BodyModeController, ControllerMode


def list_available_cameras(max_tested: int = 4) -> List[int]:
    """Scan and list indices of connected camera devices."""
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
        else:
            cap_fallback = cv2.VideoCapture(i)
            if cap_fallback.isOpened():
                available.append(i)
                cap_fallback.release()
    return available


def draw_hud_panel(
    frame: np.ndarray,
    camera_manager: CameraManager,
    controller: BodyModeController,
    gesture_result,
    features,
    landmarks,
    mapped_key: str,
    dispatch_status: str,
    debug_mode: bool,
    actual_w: int,
    actual_h: int
) -> np.ndarray:
    """Draw dashboard over camera feed with game integration & dynamic reference telemetry."""
    hud = frame.copy()
    panel_w = 480
    panel_h = 390 if not debug_mode else 520
    cv2.rectangle(hud, (10, 10), (10 + panel_w, 10 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(hud, 0.82, frame, 0.18, 0, frame)

    fps = camera_manager.fps
    is_body_detected = landmarks is not None and features.is_valid
    calib_status = controller.calibrator.status
    baseline = controller.baseline
    engine = controller.engine
    is_armed = controller.is_armed
    game_name = controller.mapper.current_game_name
    cam_id = camera_manager.config.camera_index

    # 1. Header & FPS
    cv2.putText(frame, f"MotionVerse — Browser Game Controller", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Cam: [{cam_id}] | Game: {game_name}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # 2. Safety Arming Banner
    if is_armed:
        arm_text = "● INPUT: ARMED (SENDING REAL KEYSTROKES)"
        arm_col = (0, 255, 0)   # Bright Green
    else:
        arm_text = "○ INPUT: DISARMED (Press [a] to ARM)"
        arm_col = (0, 180, 255) # Orange/Yellow

    cv2.putText(frame, arm_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, arm_col, 2)

    # 3. Calibration Status
    if calib_status == CalibrationStatus.SUCCESS:
        calib_text = "Calibration: READY (Active)"
        calib_col = (0, 255, 0)
    elif calib_status == CalibrationStatus.COLLECTING:
        prog = controller.calibrator.progress
        calib_text = f"CALIBRATING... {int(prog * 100)}%"
        calib_col = (0, 165, 255)
        cv2.rectangle(frame, (20, 102), (20 + int(260 * prog), 110), (0, 255, 0), -1)
    elif calib_status == CalibrationStatus.FAILED:
        calib_text = "Calibration: FAILED (Press [c])"
        calib_col = (0, 0, 255)
    else:
        calib_text = "Calibration: NOT CALIBRATED (Press [c])"
        calib_col = (0, 255, 255)

    cv2.putText(frame, calib_text, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, calib_col, 1)

    # 4. Action & Game Mapping Card Box
    action_val = gesture_result.action.value
    h_state = gesture_result.debug_info.get("horizontal_state", "READY") if gesture_result.debug_info else "READY"
    v_state = gesture_result.debug_info.get("vertical_state", "READY") if gesture_result.debug_info else "READY"

    if action_val == "JUMP":
        act_color = (0, 255, 0)
        symbol = "▲▲ JUMP"
    elif action_val == "CROUCH":
        act_color = (0, 215, 255)
        symbol = "▼▼ CROUCH"
    elif action_val == "MOVE_LEFT":
        act_color = (255, 120, 0)
        symbol = "◄◄ MOVE_LEFT"
    elif action_val == "MOVE_RIGHT":
        act_color = (0, 140, 255)
        symbol = "MOVE_RIGHT ►►"
    else:
        act_color = (160, 160, 160)
        symbol = "NONE (Idle)"

    box_y = 118
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 54), (35, 35, 35), -1)
    cv2.rectangle(frame, (20, box_y), (panel_w, box_y + 54), act_color, 2)
    
    key_str = f"KEY: {mapped_key.upper()}" if mapped_key else "KEY: -"
    cv2.putText(frame, f"ACTION: {symbol}  -->  {key_str}", (30, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, act_color, 2)
    cv2.putText(frame, f"DISPATCH: {dispatch_status} | H-State: {h_state} | V-State: {v_state}", (30, box_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1)

    # 5. Dynamic Horizontal Reference Telemetry
    y_start = box_y + 72
    body_str = "YES" if is_body_detected else "NO (Searching...)"
    body_col = (0, 255, 0) if is_body_detected else (0, 0, 255)
    cv2.putText(frame, f"Body Detected: {body_str}", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.38, body_col, 1)

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
            f"Horiz Velocity: {h_vel:+.2f}/s",
            f"--- Vertical Telemetry ---",
            f"Y Displacement: {features.normalized_y_displacement:+.3f} (Jump req: >= +0.10)",
            f"Body Ratio:     {features.body_height_ratio:.3f} (Crouch req: <= 0.76)"
        ]
        for i, l in enumerate(lines):
            col = (0, 255, 255) if "---" in l else (255, 255, 255)
            cv2.putText(frame, l, (20, y_start + 16 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)
    else:
        cv2.putText(frame, "Stand in view of camera & press [c] to calibrate", (20, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # 6. Latency & Performance Breakdown
    if debug_mode and gesture_result.debug_info:
        dbg_y = y_start + 145
        cv2.line(frame, (20, dbg_y), (panel_w, dbg_y), (80, 80, 80), 1)
        tot_ms = (camera_manager.capture_latency_ms + 
                  controller.detector.inference_latency_ms + 
                  controller.engine.processing_latency_ms +
                  controller.mapper.mapping_latency_ms +
                  controller.keyboard.dispatch_latency_ms)
        dbg_lines = [
            f"-- Latency Breakdown --",
            f"Camera Capture:     {camera_manager.capture_latency_ms:.2f}ms",
            f"Pose Inference:     {controller.detector.inference_latency_ms:.2f}ms",
            f"Gesture Engine:     {controller.engine.processing_latency_ms:.3f}ms",
            f"Game Mapping:       {controller.mapper.mapping_latency_ms:.3f}ms",
            f"Keyboard Dispatch:  {controller.keyboard.dispatch_latency_ms:.3f}ms",
            f"Total Pipe Latency: {tot_ms:.2f}ms"
        ]
        for i, l in enumerate(dbg_lines):
            cv2.putText(frame, l, (20, dbg_y + 13 + i * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 255, 255), 1)

    # Dynamic Reference Visual Markers
    if baseline is not None and engine.horizontal_reference_x is not None:
        rx_px = int(engine.horizontal_reference_x * actual_w)
        hy_px = int(features.hip_center_y * actual_h) if features.is_valid else int(baseline.hip_center_y * actual_h)
        b_scale_px = int(baseline.body_scale * actual_w)

        # Dynamic Reference Marker (Cyan Crosshair)
        cv2.drawMarker(frame, (rx_px, hy_px), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)

        # Trigger Boundaries
        left_trig_px = int(rx_px - 0.18 * b_scale_px)
        right_trig_px = int(rx_px + 0.18 * b_scale_px)
        cv2.line(frame, (left_trig_px, 0), (left_trig_px, actual_h), (255, 100, 0), 1)
        cv2.line(frame, (right_trig_px, 0), (right_trig_px, actual_h), (0, 140, 255), 1)

    # Footer Controls
    cv2.putText(frame, "[a]: ARM/DISARM | [c]: Calib | [r]: Reset | [d]: Debug | [ESC/q]: Emergency Stop", 
                (15, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    return frame


def run_motionverse_game_controller(
    camera_index: int = 0,
    profile_path: Optional[str] = None,
    mock_input: bool = False
):
    """Main execution loop for MotionVerse Game Controller."""
    print("=" * 68)
    print("   MotionVerse — Browser Game Controller (End-to-End Prototype)")
    print("=" * 68)

    # 1. Load Game Profile
    default_profile_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "game_profiles", "subway_surfers.json"
    )
    active_profile_path = profile_path or default_profile_path

    if os.path.exists(active_profile_path):
        profile = GameProfile.load_from_file(active_profile_path)
        print(f"[PROFILE] Loaded Game Profile: '{profile.game_name}' ({profile.game_type})")
        print(f"[PROFILE] Action Bindings: {profile.bindings}")
    else:
        print(f"[WARN] Profile not found at {active_profile_path}. Using fallback default bindings.")
        profile = GameProfile(
            game_name="Generic Browser Runner",
            bindings={"JUMP": "up", "CROUCH": "down", "MOVE_LEFT": "left", "MOVE_RIGHT": "right"}
        )

    # 2. Initialize Camera & Controller
    print(f"[CAMERA] Initializing camera device index [{camera_index}]...")
    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)

    mapper = GameMapper(profile)
    keyboard = KeyboardController(key_tap_duration_ms=profile.key_tap_duration_ms, mock_mode=mock_input)
    controller = BodyModeController(game_mapper=mapper, keyboard_controller=keyboard)

    if not camera.start():
        print(f"[ERROR] Camera device index [{camera_index}] unavailable.")
        print("[INFO] Detecting connected camera devices...")
        cams = list_available_cameras()
        if cams:
            print(f"[INFO] Available camera indices on your system: {cams}")
            print(f"[TIP] Try launching with: python app/main.py {cams[0]}")
        else:
            print("[ERROR] No working cameras detected. Check USB webcam connection.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[STATUS] Camera Connected (Device [{camera_index}], {actual_w}x{actual_h} @ target 30 FPS).")
    print(f"[STATUS] Mock Input Mode: {mock_input}")
    print("[STATUS] Input is initially DISARMED for safety.")
    print("--------------------------------------------------------------------")
    print("INSTRUCTIONS:")
    print("  1. Open your browser game (e.g. Subway Surfers: https://poki.com/en/g/subway-surfers)")
    print("  2. Click inside the browser game window so it receives keyboard focus.")
    print("  3. Stand in view of webcam and press 'c' in MotionVerse window to calibrate.")
    print("  4. Press 'a' to ARM the controller and start playing!")
    print("  5. Press 'ESC' or 'q' at any time for Emergency Stop.")
    print("--------------------------------------------------------------------\n")

    debug_mode = True

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            gesture_result, landmarks, features, mapped_key, dispatch_status = controller.process_frame(frame, timestamp=ts)

            # Terminal log only when an action is dispatched
            if gesture_result.action != Action.NONE and controller.is_armed and mapped_key:
                print(f"[{time.strftime('%H:%M:%S')}] ACTION: {gesture_result.action.value:<10} | KEY: {mapped_key.upper():<6} | Latency: {keyboard.dispatch_latency_ms:.1f}ms")

            if landmarks is not None:
                frame = controller.detector.draw_skeleton(frame, landmarks)

            frame = draw_hud_panel(
                frame,
                camera,
                controller,
                gesture_result,
                features,
                landmarks,
                mapped_key,
                dispatch_status,
                debug_mode,
                actual_w,
                actual_h
            )

            cv2.imshow("MotionVerse — Browser Game Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC or q -> Emergency Stop
                print("\n[INFO] Emergency Stop requested.")
                controller.emergency_stop()
                break
            elif key == ord('a'):  # Toggle Arming
                is_armed = controller.toggle_arm()
                state_str = "ARMED (LIVE)" if is_armed else "DISARMED (SAFE)"
                print(f"\n[SAFETY] Keyboard Input is now: {state_str}")
            elif key == ord('c') or key == 32:  # Calibrate
                print("\n[INFO] Starting standing calibration (stand still)...")
                controller.start_calibration()
            elif key == ord('r'):  # Reset
                print("\n[INFO] Resetting pipeline and dynamic references.")
                controller.reset()
            elif key == ord('d'):  # Toggle Debug
                debug_mode = not debug_mode

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        controller.emergency_stop()
        camera.stop()
        controller.detector.close()
        cv2.destroyAllWindows()
        print("[INFO] MotionVerse Game Controller safely shut down.")


if __name__ == "__main__":
    cam_idx = 0
    mock = False
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--camera", "-c") and i + 1 < len(args):
            cam_idx = int(args[i + 1])
            i += 2
            continue
        elif arg.isdigit():
            cam_idx = int(arg)
        elif arg == "--mock":
            mock = True
        elif arg == "--list-cameras":
            cams = list_available_cameras()
            print(f"[INFO] Available camera indices: {cams}")
            sys.exit(0)
        i += 1

    run_motionverse_game_controller(camera_index=cam_idx, mock_input=mock)
