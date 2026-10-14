
import streamlit as st

from pose_detection import detect_pose


def analyze_lordosis_image(image_bytes):
    """Detect landmarks for preliminary side-view posture screening."""

    result, error = detect_pose(image_bytes)

    if error:
        return None, error

    landmarks = result["landmarks"]

    required = [
        "left_shoulder",
        "left_hip",
        "left_knee",
    ]

    for name in required:
        if not landmarks[name]["visible"]:
            return None, (
                f"Could not clearly detect {name.replace('_', ' ')}. "
                "Upload a clearer side-view photograph."
            )

    return {
        "annotated_image": result["annotated_image"],
        "landmarks": landmarks,
    }, None


def render_lordosis_page():
    st.header("Lordosis — Preliminary Posture Screening")

    st.write(
        "Upload a side-view photograph showing the trunk, hips, "
        "and knees as clearly as possible."
    )

    st.warning(
        "This prototype displays body landmarks. It does not calculate "
        "a clinical lumbar lordosis angle or diagnose a spinal condition."
    )

    uploaded_file = st.file_uploader(
        "Upload a side-view photograph",
        type=["jpg", "jpeg", "png"],
        key="lordosis_upload",
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Uploaded photograph",
        use_container_width=True,
    )

    if st.button("Analyze posture", key="analyze_lordosis"):
        with st.spinner("Detecting landmarks..."):
            analysis, error = analyze_lordosis_image(image_bytes)

        if error:
            st.error(error)
            return

        st.image(
            analysis["annotated_image"],
            caption="Detected MediaPipe landmarks",
            use_container_width=True,
        )

        st.success("Required landmarks detected.")

        st.info(
            "Review the landmark overlay as a preliminary visual aid. "
            "Lumbar curvature requires an appropriate validated measurement "
            "method and, where clinically indicated, professional assessment."
        )
