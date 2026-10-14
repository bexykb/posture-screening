
import streamlit as st

from pose_detection import detect_pose


REFERENCE = (
    "Greendale GA, Nili NS, Huang MH. The reliability and validity "
    "of three non-radiological measures of thoracic kyphosis and "
    "their relations to the standing radiological Cobb angle. "
    "Osteoporosis International, 2011. "
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3092935/"
)


def analyze_kyphosis_image(image_bytes):
    """Detect landmarks for preliminary side-view posture screening."""

    result, error = detect_pose(image_bytes)

    if error:
        return None, error

    landmarks = result["landmarks"]

    required = [
        "left_ear",
        "left_shoulder",
        "left_hip",
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


def render_kyphosis_page():
    st.header("Kyphosis — Preliminary Posture Screening")

    st.write(
        "Upload a side-view photograph with the head, shoulders, "
        "and hips visible."
    )

    st.warning(
        "MediaPipe landmarks are visual posture indicators. They do not "
        "measure the clinical thoracic kyphosis angle."
    )

    uploaded_file = st.file_uploader(
        "Upload a side-view photograph",
        type=["jpg", "jpeg", "png"],
        key="kyphosis_upload",
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Uploaded photograph",
        use_container_width=True,
    )

    if st.button("Analyze posture", key="analyze_kyphosis"):
        with st.spinner("Detecting landmarks..."):
            analysis, error = analyze_kyphosis_image(image_bytes)

        if error:
            st.error(error)
            return

        st.image(
            analysis["annotated_image"],
            caption="Detected MediaPipe landmarks",
            use_container_width=True,
        )

        st.success("Required landmarks detected.")

        st.subheader("Research measurement methods")

        st.write(
            "The reference paper evaluates the Debrunner kyphometer "
            "angle, Flexicurve kyphosis index, and Flexicurve kyphosis "
            "angle. These require their own measurement procedures."
        )

        st.info(
            "No Cobb-angle estimate is generated from the MediaPipe "
            "landmarks. A validated measurement procedure is needed "
            "before applying the paper's conversion equations."
        )

        st.caption(f"Reference: {REFERENCE}")
