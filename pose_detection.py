import streamlit as st
import io
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image as PILImage

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import (
    Image as MPImage,
    ImageFormat
)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker.task"
)


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29),
    (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30),
    (30, 32), (28, 32)
]


LANDMARK_INDICES = {
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


def detect_pose(image_bytes, visibility_threshold=0.5):
    """Detect body landmarks from uploaded image bytes."""

    # Check model file
    if not os.path.isfile(MODEL_PATH):
        return None, (
            "Pose model not found. Add pose_landmarker.task "
            "to the same project folder as pose_detection.py."
        )

    # Load image
    try:
        if hasattr(image_bytes, "read"):
            image_bytes = image_bytes.read()

        image = PILImage.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        rgb_image = np.ascontiguousarray(
            np.asarray(image, dtype=np.uint8)
        )

    except Exception as exc:
        return None, f"Unable to read uploaded image: {exc}"

    height, width = rgb_image.shape[:2]

    if height == 0 or width == 0:
        return None, "Uploaded image is empty."

    # Create MediaPipe image and run detection
    try:
      try:
    rgb_image = np.ascontiguousarray(
        rgb_image, dtype=np.uint8
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    st.write("MediaPipe version:", mp.__version__)
    st.write("Image shape:", rgb_image.shape)
    st.write("Image dtype:", rgb_image.dtype)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:
        results = landmarker.detect(mp_image)

except Exception as exc:
    import traceback
    return None, traceback.format_exc()
