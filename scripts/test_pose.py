"""
Step 2 Pose Detection Test Script for MotionVerse.
Connects CameraManager with PoseDetector, rendering the skeleton and telemetry in real time.
"""

import sys
import os
import cv2
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.camera.camera_manager import CameraManager
from app.core.camera.camera_config import CameraConfig
from app.core.vision.pose.pose_detector import PoseDetector, PoseDetectorConfig


def run_pose_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 2: MediaPipe Pose Detection Test")
    print("=" * 60)

    cam_config = CameraConfig(
        camera_index=camera_index,
        width=640,
        height=480,
        target_fps=30,
        flip_horizontal=True
    )
    detector_config = PoseDetectorConfig(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    camera = CameraManager(cam_config)
    detector = PoseDetector(detector_config)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}). MediaPipe Pose initialized.")
    print("Press 'q' or 'ESC' on the video window to stop the test.\n")

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detect pose
            landmarks = detector.detect(frame, timestamp=ts)

            # Draw skeleton
            if landmarks is not None:
                frame = detector.draw_skeleton(frame, landmarks)
                status_text = "Person: DETECTED"
                status_color = (0, 255, 0)
            else:
                status_text = "Person: NOT DETECTED"
                status_color = (0, 0, 255)

            # Telemetry display
            fps = camera.fps
            infer_ms = detector.inference_latency_ms
            total_lat = camera.capture_latency_ms + infer_ms

            cv2.putText(
                frame,
                f"MotionVerse Step 2 | FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{status_text} | Total Lat: {total_lat:.1f}ms",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' or 'ESC' to exit",
                (20, actual_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("MotionVerse — Step 2: Pose Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nExit requested by user.")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")

    return True


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_pose_test(idx)
