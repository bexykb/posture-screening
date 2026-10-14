import io
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker.task"
)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17),
    (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18),
    (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29),
    (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30),
    (30, 32), (28, 32)
]


def detect_pose(image_bytes, visibility_threshold=0.5):
    """Detect pose landmarks using MediaPipe Tasks."""

    if not os.path.isfile(MODEL_PATH):
        return None, (
            "Pose model not found. Ensure pose_landmarker.task "
            "is in the project folder."
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        rgb_image = np.array(image, dtype=np.uint8)

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
            min_pose_presence_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:
            results = landmarker.detect(mp_image)

    except Exception as exc:
        return None, f"Pose detection failed: {exc}"

    if not results.pose_landmarks:
        return None, (
            "No body landmarks detected. Try a clear image "
            "with the person visible and good lighting."
        )

    points = results.pose_landmarks[0]

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
        "right_foot_index": 32
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
            "visible": visibility >= visibility_threshold
        }

    annotated = rgb_image.copy()

    for start, end in POSE_CONNECTIONS:
        p1 = points[start]
        p2 = points[end]

        x1, y1 = int(p1.x * width), int(p1.y * height)
        x2, y2 = int(p2.x * width), int(p2.y * height)

        if (
            0 <= x1 < width and 0 <= y1 < height
            and 0 <= x2 < width and 0 <= y2 < height
        ):
            cv2.line(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    for point in points:
        x, y = int(point.x * width), int(point.y * height)
