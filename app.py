import streamlit as st

st.set_page_config(
    page_title="Road Defect and Hit & Run Detector",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Road Defect and Hit & Run Detector")

st.write(
    "Upload a video to detect potholes, waterlogging, and hit-and-run incidents."
)

video = st.file_uploader(
    "🎥 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if video is not None:
    st.subheader("🎬 Uploaded Video")
    st.video(video)
