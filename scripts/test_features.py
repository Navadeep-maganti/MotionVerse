"""
Step 3 Body Features Test Script for MotionVerse.
Extracts and displays all normalized coordinates, scale, displacements, and velocities in real time.
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


def run_features_test(camera_index: int = 0):
    print("=" * 60)
    print("MotionVerse — Step 3: Body Feature Extraction Test")
    print("=" * 60)

    cam_config = CameraConfig(camera_index=camera_index, width=640, height=480, target_fps=30)
    camera = CameraManager(cam_config)
    detector = PoseDetector(PoseDetectorConfig(model_complexity=1))
    extractor = BodyFeatureExtractor(min_visibility=0.5)

    if not camera.start():
        print("[ERROR] Failed to start camera.")
        return False

    actual_w, actual_h = camera.get_actual_resolution()
    print(f"[SUCCESS] Camera started ({actual_w}x{actual_h}). Running feature extractor.")
    print("Tip: Press 'b' to set the current frame as baseline, 'r' to reset baseline.")
    print("Press 'q' or 'ESC' to exit.\n")

    baseline_hip = None
    baseline_scale = None

    try:
        while camera.is_opened:
            ret, frame, ts = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detect pose
            landmarks = detector.detect(frame, timestamp=ts)

            # Extract features
            features = extractor.extract(
                landmarks,
                baseline_hip=baseline_hip,
                baseline_scale=baseline_scale,
                timestamp=ts
            )

            # Draw skeleton and centers
            if landmarks is not None and features.is_valid:
                frame = detector.draw_skeleton(frame, landmarks)

                # Draw Hip and Shoulder Centers
                hx_px, hy_px = int(features.hip_center_x * actual_w), int(features.hip_center_y * actual_h)
                sx_px, sy_px = int(features.shoulder_center_x * actual_w), int(features.shoulder_center_y * actual_h)

                cv2.circle(frame, (hx_px, hy_px), 8, (0, 255, 0), -1)  # Green for Hip
                cv2.circle(frame, (sx_px, sy_px), 8, (255, 0, 0), -1)  # Blue for Shoulder
                cv2.line(frame, (sx_px, sy_px), (hx_px, hy_px), (255, 255, 0), 2)

                # If baseline is set, draw baseline reference marker
                if baseline_hip is not None:
                    bhx_px = int(baseline_hip[0] * actual_w)
                    bhy_px = int(baseline_hip[1] * actual_w)
                    cv2.drawMarker(frame, (bhx_px, bhy_px), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # Telemetry & Feature HUD Panel
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (10, 10), (320, 240), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

            fps = camera.fps
            infer_ms = detector.inference_latency_ms

            cv2.putText(frame, f"Step 3: Features | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if features.is_valid:
                lines = [
                    f"Hip: ({features.hip_center_x:.2f}, {features.hip_center_y:.2f})",
                    f"Scale (B): {features.body_scale:.3f}",
                    f"Norm X: {features.normalized_x:+.3f}",
                    f"Norm Y Disp: {features.normalized_y_displacement:+.3f}",
                    f"V_x (Horiz): {features.horizontal_velocity:+.2f}/s",
                    f"V_y (Vert):  {features.vertical_velocity:+.2f}/s",
                    f"Height Ratio: {features.body_height_ratio:.2f}",
                    f"Baseline: {'SET' if baseline_hip else 'AUTO'}"
                ]
            else:
                lines = [
                    "Body: NOT DETECTED",
                    "Please stand in full view of camera",
                    f"Baseline: {'SET' if baseline_hip else 'NONE'}"
                ]

            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, 55 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(frame, "[b]: Set Baseline | [r]: Reset | [q]: Quit", (20, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("MotionVerse — Step 3: Feature Extraction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b') and features.is_valid:
                baseline_hip = (features.hip_center_x, features.hip_center_y)
                baseline_scale = features.body_scale
                print(f"[BASELINE SET] Hip: {baseline_hip}, Scale: {baseline_scale:.3f}")
            elif key == ord('r'):
                baseline_hip = None
                baseline_scale = None
                extractor.reset()
                print("[BASELINE RESET]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_features_test(idx)
