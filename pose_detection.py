
import io
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker.task"


def detect_pose(image_bytes, visibility_threshold=0.5):
    """Detect pose landmarks using MediaPipe Tasks."""

    if not os.path.exists(MODEL_PATH):
        return None, (
            "Pose model not found. Download pose_landmarker.task "
            "and ensure it is in the project folder."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_image = np.array(image)
    except Exception:
        return None, "Unable to read the uploaded image."

    height, width = rgb_image.shape[:2]

    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(mp_image)

    except Exception as exc:
        return None, f"Pose detection failed: {exc}"

    if not results.pose_landmarks:
        return None, (
            "No body landmarks detected. Try a clear photograph "
            "with the person fully visible."
        )

    points = results.pose_landmarks[0]

    # MediaPipe's 33 pose landmark indices
    landmark_indices = {
        "nose": 0,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    landmarks = {}

    for name, index in landmark_indices.items():
        point = points[index]
        visibility = float(
            getattr(point, "visibility", 1.0)
        )

        landmarks[name] = {
            "x": float(point.x),
            "y": float(point.y),
            "z": float(point.z),
            "visibility": visibility,
            "visible": visibility >= visibility_threshold,
        }

    # Draw landmarks and connections
    annotated = rgb_image.copy()

    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    for start, end in connections:
        p1 = points[start]
        p2 = points[end]

        x1, y1 = int(p1.x * width), int(p1.y * height)
        x2, y2 = int(p2.x * width), int(p2.y * height)

        cv2.line(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    for point in points:
        x, y = int(point.x * width), int(point.y * height)
        cv2.circle(annotated, (x, y), 4, (255, 0, 0), -1)

    return {
        "original_image": rgb_image,
        "annotated_image": annotated,
        "height": height,
        "width": width,
        "landmarks": landmarks,
    }, None
