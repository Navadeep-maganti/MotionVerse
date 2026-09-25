"""
MediaPipe Pose Detector module for MotionVerse.
Performs real-time pose inference, landmark extraction, and skeleton visualization.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

from app.core.vision.pose.pose_landmarks import Point3D, PoseLandmarks


@dataclass
class PoseDetectorConfig:
    """Configuration options for MediaPipe Pose."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0: Lite, 1: Full, 2: Heavy (1 is best balance)
    smooth_landmarks: bool = True
    enable_segmentation: bool = False


# Connections for skeleton rendering
POSE_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # Head connection
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder")
]


class PoseDetector:
    """
    Wraps MediaPipe Pose landmarker, extracts clean PoseLandmarks,
    tracks inference performance, and renders visual skeleton overlays.
    """

    def __init__(self, config: Optional[PoseDetectorConfig] = None):
        self.config = config or PoseDetectorConfig()
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            enable_segmentation=self.config.enable_segmentation,
            static_image_mode=False
        )

        self._inference_latency_ms: float = 0.0
        self._is_person_detected: bool = False

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Optional[PoseLandmarks]:
        """
        Process a BGR frame and return detected PoseLandmarks.
        
        Args:
            frame: BGR numpy image from camera.
            timestamp: Frame capture timestamp.
            
        Returns:
            PoseLandmarks object if a person is detected with valid landmarks, else None.
        """
        if frame is None or frame.size == 0:
            self._is_person_detected = False
            return None

        ts = timestamp if timestamp is not None else time.perf_counter()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        t_start = time.perf_counter()
        results = self._pose.process(rgb_frame)
        t_end = time.perf_counter()
        self._inference_latency_ms = (t_end - t_start) * 1000.0

        if not results.pose_landmarks:
            self._is_person_detected = False
            return None

        self._is_person_detected = True
        return self._convert_landmarks(results.pose_landmarks.landmark, ts)

    def _convert_landmarks(self, raw_landmarks, timestamp: float) -> PoseLandmarks:
        """Convert raw MediaPipe landmarks list into structured PoseLandmarks object."""
        mp_lm = self._mp_pose.PoseLandmark

        def extract_point(idx: int) -> Optional[Point3D]:
            if idx < len(raw_landmarks):
                lm = raw_landmarks[idx]
                return Point3D(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility)
                )
            return None

        return PoseLandmarks(
            nose=extract_point(mp_lm.NOSE.value),
            left_shoulder=extract_point(mp_lm.LEFT_SHOULDER.value),
            right_shoulder=extract_point(mp_lm.RIGHT_SHOULDER.value),
            left_elbow=extract_point(mp_lm.LEFT_ELBOW.value),
            right_elbow=extract_point(mp_lm.RIGHT_ELBOW.value),
            left_wrist=extract_point(mp_lm.LEFT_WRIST.value),
            right_wrist=extract_point(mp_lm.RIGHT_WRIST.value),
            left_hip=extract_point(mp_lm.LEFT_HIP.value),
            right_hip=extract_point(mp_lm.RIGHT_HIP.value),
            left_knee=extract_point(mp_lm.LEFT_KNEE.value),
            right_knee=extract_point(mp_lm.RIGHT_KNEE.value),
            left_ankle=extract_point(mp_lm.LEFT_ANKLE.value),
            right_ankle=extract_point(mp_lm.RIGHT_ANKLE.value),
            timestamp=timestamp
        )

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: PoseLandmarks,
        color_connections: Tuple[int, int, int] = (0, 255, 255),
        color_joints: Tuple[int, int, int] = (0, 128, 255),
        joint_radius: int = 5,
        line_thickness: int = 2,
        min_visibility: float = 0.5
    ) -> np.ndarray:
        """
        Draw the pose skeleton and landmark joints on the given image frame.
        """
        if landmarks is None or frame is None:
            return frame

        h, w = frame.shape[:2]
        lm_dict = landmarks.to_dict()

        # 1. Draw connections
        for name_a, name_b in POSE_CONNECTIONS:
            pt_a = lm_dict.get(name_a)
            pt_b = lm_dict.get(name_b)

            if pt_a and pt_b and pt_a.is_visible(min_visibility) and pt_b.is_visible(min_visibility):
                p1 = pt_a.to_pixel(w, h)
                p2 = pt_b.to_pixel(w, h)
                cv2.line(frame, p1, p2, color_connections, line_thickness, cv2.LINE_AA)

        # 2. Draw joints
        for pt in lm_dict.values():
            if pt and pt.is_visible(min_visibility):
                px, py = pt.to_pixel(w, h)
                cv2.circle(frame, (px, py), joint_radius, color_joints, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), joint_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    @property
    def inference_latency_ms(self) -> float:
        """Inference time of the last processed frame in milliseconds."""
        return self._inference_latency_ms

    @property
    def is_person_detected(self) -> bool:
        """Whether a person was detected in the last processed frame."""
        return self._is_person_detected

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
