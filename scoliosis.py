
import streamlit as st
import numpy as np

from pose_detection import detect_pose


def calculate_tilt_angle(point_left, point_right):
    """
    Calculate the angle of a line joining two landmarks
    relative to the horizontal image axis.
    """

    dx = point_right["x"] - point_left["x"]
    dy = point_right["y"] - point_left["y"]

    # Convert normalized coordinates to image-space coordinates
    # using a common scale-independent angle calculation.
    angle = np.degrees(np.arctan2(dy, dx))

    # Keep the absolute tilt relative to horizontal
    tilt = abs(angle)

    # Express tilt as the smaller angle from horizontal
    if tilt > 90:
        tilt = 180 - tilt

    return round(float(tilt), 2)


def analyze_scoliosis_image(image_bytes):
    """
    Analyze visible shoulder and hip alignment.
    This does not calculate a Cobb angle.
    """

    result, error = detect_pose(image_bytes)

    if error:
        return None, error

    landmarks = result["landmarks"]

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    for name in required:
        if not landmarks[name]["visible"]:
            return None, (
                f"Could not clearly detect the {name.replace('_', ' ')}. "
                "Try a clear, straight back-view photograph."
            )

    shoulder_angle = calculate_tilt_angle(
        landmarks["left_shoulder"],
        landmarks["right_shoulder"],
    )

    hip_angle = calculate_tilt_angle(
        landmarks["left_hip"],
        landmarks["right_hip"],
    )

    return {
        "original_image": result["original_image"],
        "annotated_image": result["annotated_image"],
        "shoulder_tilt_angle": shoulder_angle,
        "hip_tilt_angle": hip_angle,
    }, None


def render_scoliosis_page():
    st.header("Scoliosis — Visual Screening")

    st.write(
        "Upload a straight back-view photograph with both shoulders "
        "and both sides of the pelvis visible."
    )

    st.warning(
        "This feature measures visible body alignment only. It does not "
        "reconstruct the spine or calculate a clinical Cobb angle."
    )

    uploaded_file = st.file_uploader(
        "Upload a back-view image",
        type=["jpg", "jpeg", "png"],
        key="scoliosis_upload",
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Uploaded photograph",
        use_container_width=True,
    )

    if st.button("Analyze scoliosis indicators"):
        with st.spinner("Detecting body landmarks..."):
            analysis, error = analyze_scoliosis_image(image_bytes)

        if error:
            st.error(error)
            return

        st.image(
            analysis["annotated_image"],
            caption="Detected body landmarks",
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Shoulder tilt",
            f'{analysis["shoulder_tilt_angle"]}°',
        )

        col2.metric(
            "Hip tilt",
            f'{analysis["hip_tilt_angle"]}°',
        )

        st.info(
            "Larger visible tilts may indicate body asymmetry, but these "
            "measurements alone cannot determine whether scoliosis is present."
        )

        st.caption(
            "Reference: Wang et al., A novel portable and radiation-free "
            "method for assessing scoliosis (2025). The reference's Cobb-angle "
            "categories apply to spinal measurements, not these tilt angles."
        )
