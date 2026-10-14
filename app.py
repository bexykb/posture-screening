
import streamlit as st

from scoliosis import render_scoliosis_page
from kyphosis import render_kyphosis_page
from lordosis import render_lordosis_page


# Page configuration
st.set_page_config(
    page_title="Posture Screening System",
    page_icon="🧍",
    layout="wide"
)


# Application heading
st.title("Posture Screening System")

st.write(
    """
    A computer-vision prototype for preliminary visual screening
    of posture-related concerns using uploaded photographs.
    """
)


# Medical disclaimer
st.warning(
    "This application is an educational screening prototype, "
    "not a medical diagnostic tool. Its results cannot confirm "
    "or rule out a spinal condition. Consult a qualified healthcare "
    "professional for clinical assessment."
)


# Condition selection
st.sidebar.header("Screening Options")

condition = st.sidebar.selectbox(
    "Select a posture condition",
    [
        "Choose a condition",
        "Scoliosis",
        "Kyphosis",
        "Lordosis"
    ]
)


# Display the selected page
if condition == "Scoliosis":
    render_scoliosis_page()

elif condition == "Kyphosis":
    render_kyphosis_page()

elif condition == "Lordosis":
    render_lordosis_page()

else:
    st.info(
        "Choose Scoliosis, Kyphosis, or Lordosis from the sidebar "
        "to begin screening."
    )
