"""
Step 1 Camera Test Script for MotionVerse.
Opens the webcam feed, calculates and displays FPS and capture latency on the frame,
and gracefully exits when 'q' or ESC is pressed.
"""

import sys
import os
import cv2
import time

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.camera.camera_manager import CameraManager
from app.core.camera.camera_config import CameraConfig


def run_camera_test(camera_index: int = 0, width: int = 640, height: int = 480):
    print("=" * 60)
    print("MotionVerse — Step 1: Camera System Test")
    print("=" * 60)
    print(f"Attempting to open camera index {camera_index} at {width}x{height}...")

    config = CameraConfig(
        camera_index=camera_index,
        width=width,
        height=height,
        target_fps=30,
        flip_horizontal=True,
        use_threading=True
    )

    manager = CameraManager(config)
    success = manager.start()

    if not success:
        print("[ERROR] Failed to open camera. Please check your webcam connection or camera permissions.")
        return False

    actual_w, actual_h = manager.get_actual_resolution()
    print(f"[SUCCESS] Camera opened successfully! Resolution: {actual_w}x{actual_h}")
    print("Press 'q' or 'ESC' on the video window to stop the test.\n")

    try:
        while manager.is_opened:
            ret, frame, ts = manager.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            fps = manager.fps
            latency = manager.capture_latency_ms
            dropped = manager.buffer.dropped_frames

            # Overlay telemetry
            cv2.putText(
                frame,
                f"MotionVerse Camera Test | FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Res: {actual_w}x{actual_h} | Latency: {latency:.1f}ms | Dropped: {dropped}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' or 'ESC' to exit",
                (20, actual_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("MotionVerse — Camera Test (Step 1)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nExit key pressed.")
                break
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        manager.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera released cleanly. Test completed.")

    return True


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_camera_test(camera_index=camera_idx)
