
# Posture Screening System

## Overview

The Posture Screening System is a computer-vision prototype that uses
MediaPipe Pose to detect visible body landmarks from uploaded photographs.

The application provides preliminary visual screening features for:

- Scoliosis-related visible shoulder and hip asymmetry
- Kyphosis-related posture review
- Lordosis-related posture review

The application is built using Python and Streamlit.

## Features

- Upload photographs in JPG, JPEG, or PNG format.
- Detect body landmarks using MediaPipe Pose.
- Display images annotated with detected landmarks.
- Calculate visible shoulder and hip tilt angles for the scoliosis page.
- Select the posture-screening module from the sidebar.

## Technology Stack

- Python
- Streamlit
- MediaPipe
- OpenCV
- NumPy
- Pillow

## Project Structure

```text
posture-screening/
├── app.py
├── pose_detection.py
├── scoliosis.py
├── kyphosis.py
├── lordosis.py
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the application with:

```bash
streamlit run app.py
```

## Image Guidelines

For scoliosis-related visual screening, upload a clear back-view image
showing both shoulders and the pelvis.

For kyphosis and lordosis posture review, upload a clear side-view image
showing the relevant body landmarks.

Good lighting and an unobstructed view can improve landmark detection.

## Limitations and Safety

This project is an educational prototype, not a medical diagnostic tool.

MediaPipe landmarks do not directly measure spinal curvature. Shoulder
and hip tilt measurements must not be interpreted as Cobb angles.

The current application does not independently diagnose scoliosis,
kyphosis, or lordosis. Clinical assessment and validated measurement
methods are required for medical conclusions.

## Research References

1. Scoliosis research:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11863395/

2. Thoracic kyphosis measurement research:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3092935/

## License

For educational and research use. Add an appropriate license before
redistributing the project.
