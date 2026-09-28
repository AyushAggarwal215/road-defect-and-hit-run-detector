import streamlit as st
import cv2
import tempfile
import os
import subprocess
import base64
import numpy as np
import requests
import imageio_ffmpeg
from pathlib import Path

from ultralytics import YOLO
import gdown


st.set_page_config(
    page_title="Road Defect and Hit & Run Detector",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Road Defect and Hit & Run Detector")
st.caption("AI-powered road analysis")


# ============================================================
# POTHOLE MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path("/tmp/pothole_best.pt")

DRIVE_FILE_ID = "1B0XF-Mnn62Wrv3G9IPeMaavkpO6ReAyl"


# ============================================================
# WATERLOGGING MODEL CONFIGURATION
# ============================================================

WATERLOGGING_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "theftddetection-lwj20/"
    "waterlogging-video-bounding-boxes-1789221984773"
)


# ============================================================
# LOAD POTHOLE MODEL
# ============================================================

@st.cache_resource
def load_pothole_model():

    if not MODEL_PATH.exists():

        with st.spinner("Downloading pothole model..."):

            downloaded = gdown.download(
                id=DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )

        if downloaded is None or not MODEL_PATH.exists():

            raise RuntimeError(
                "Pothole model download failed. "
                "Make sure the Google Drive file is shared "
                "as Anyone with the link → Viewer."
            )

    return YOLO(str(MODEL_PATH))


try:

    pothole_model = load_pothole_model()

    st.success("✅ Pothole model loaded successfully.")

except Exception as e:

    st.error(f"❌ Could not load pothole model: {e}")
    st.stop()


# ============================================================
# WATERLOGGING FUNCTIONS
# ============================================================

def extract_first_output(response_json):

    if not isinstance(response_json, dict):
        return None

    outputs = response_json.get("outputs")

    if isinstance(outputs, list) and outputs:
        return outputs[0] if isinstance(outputs[0], dict) else None

    if isinstance(outputs, dict):
        return outputs

    return response_json


def decode_workflow_image(value):

    if value is None:
        return None

    if isinstance(value, dict):

        for key in ("value", "image", "data", "base64"):

            if key in value:
                return decode_workflow_image(value[key])

        return None

    if not isinstance(value, str):
        return None

    try:

        encoded = (
            value.split(",", 1)[1]
            if value.startswith("data:") and "," in value
            else value
        )

        image_bytes = base64.b64decode(encoded)

        image = cv2.imdecode(
            np.frombuffer(
                image_bytes,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


def prediction_count(value):

    if value is None:
        return 0

    if isinstance(value, list):
        return len(value)

    if isinstance(value, dict):

        for key in ("predictions", "detections"):

            if isinstance(value.get(key), list):
                return len(value[key])

        for key in ("output", "result", "data"):

            if key in value:

                count = prediction_count(
                    value[key]
                )

                if count:
                    return count

    return 0


def run_waterlogging_on_frame(
    frame,
    api_key
):

    ok, encoded = cv2.imencode(
        ".jpg",
        frame
    )

    if not ok:

        raise RuntimeError(
            "Could not encode video frame."
        )

    image_b64 = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    payload = {
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64
            }
        }
    }

    response = requests.post(
        WATERLOGGING_WORKFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    output = (
        extract_first_output(
            response.json()
        ) or {}
    )

    output_image = None

    for key in (
        "output_image",
        "annotated_image",
        "visualization",
        "image"
    ):

        if key in output:

            output_image = decode_workflow_image(
                output[key]
            )

            if output_image is not None:
                break

    predictions = (
        output.get("predictions")
        or output.get("detections")
        or []
    )

    return output_image, predictions



# ============================================================
# SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Minimum confidence",
        0.10,
        0.90,
        0.35,
        0.05
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640],
        value=640
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload a road / accident video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ============================================================
# Pothole VIDEO PROCESSING
# ============================================================

def process_combined_video(
    input_path,
    output_path,
    confidence,
    image_size,
    api_key
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open the uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    avi_path = output_path.replace(
        ".mp4",
        ".avi"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    out = cv2.VideoWriter(
        avi_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create the temporary output video."
        )

    progress = st.progress(0)
    status = st.empty()

    frame_number = 0
    pothole_count = 0
    waterlogging_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # =================================================
            # POTHOLE DETECTION
            # =================================================

            pothole_results = pothole_model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                verbose=False
            )

            pothole_result = pothole_results[0]

            if pothole_result.boxes is not None:

                pothole_count += len(
                    pothole_result.boxes
                )

            annotated_frame = (
                pothole_result.plot()
            )


            # =================================================
            # WATERLOGGING DETECTION
            # =================================================

            water_image, water_predictions = (
                run_waterlogging_on_frame(
                    annotated_frame,
                    api_key
                )
            )

            water_count = prediction_count(
                water_predictions
            )

            waterlogging_count += water_count


            # =================================================
            # USE WATERLOGGING ANNOTATED IMAGE
            # =================================================

            if water_image is not None:

                annotated_frame = water_image


            # =================================================
            # WRITE FRAME
            # =================================================

            out.write(
                annotated_frame
            )

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )

                status.text(
                    f"Processing frame "
                    f"{frame_number} / "
                    f"{total_frames} "
                    f"• Potholes: "
                    f"{pothole_count} "
                    f"• Waterlogging: "
                    f"{waterlogging_count}"
                )

            else:

                status.text(
                    f"Processing frame "
                    f"{frame_number} "
                    f"• Potholes: "
                    f"{pothole_count} "
                    f"• Waterlogging: "
                    f"{waterlogging_count}"
                )

    finally:

        cap.release()
        out.release()

    progress.empty()
    status.empty()

    # =========================================================
    # CONVERT AVI → MP4
    # =========================================================

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        avi_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if os.path.exists(avi_path):

        os.remove(avi_path)

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg could not convert "
            "the output video."
        )

    return (
        frame_number,
        pothole_count,
        waterlogging_count
    )

# ============================================================
# MAIN APP
# ============================================================

if uploaded_video is not None:

    st.subheader("🎬 Original Video")

    st.video(uploaded_video)

    if st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        api_key = st.secrets.get("ROBOFLOW_API_KEY")

        if not api_key:
            st.error(
                "❌ ROBOFLOW_API_KEY is missing "
                "from Streamlit Secrets."
            )
            st.stop()

        input_path = None
        output_path = None

        try:

            # ------------------------------------------------
            # SAVE INPUT VIDEO
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as input_file:

                input_file.write(
                    uploaded_video.getbuffer()
                )

                input_path = input_file.name


            # ------------------------------------------------
            # CREATE OUTPUT VIDEO
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as output_file:

                output_path = output_file.name


            # ------------------------------------------------
            # PROCESS VIDEO
            # ------------------------------------------------

            with st.spinner(
                "🔍 Detecting potholes..."
            ):

                frames_processed, pothole_detections, waterlogging_detections = (
                    process_combined_video(
                        input_path,
                        output_path,
                        confidence,
                        image_size,
                        api_key
                    )
                )


            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.success(
                f"✅ Analysis complete — "
                f"{frames_processed:,} frames processed."
            )

            st.subheader("📊 Detection Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
            
                st.metric(
                    "Frames Processed",
                    f"{frames_processed:,}"
                )
            
            with col2:
            
                st.metric(
                    "Pothole Detections",
                    f"{pothole_detections:,}"
                )
            
            with col3:
            
                st.metric(
                    "Waterlogging Detections",
                    f"{waterlogging_detections:,}"
                )

            # ------------------------------------------------
            # OUTPUT VIDEO
            # ------------------------------------------------

            st.subheader(
                "🎥 Pothole Detection Output"
            )

            st.video(output_path)


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as f:

                video_bytes = f.read()


            st.download_button(
                "⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="pothole_detected.mp4",
                mime="video/mp4",
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"❌ Error while processing video: {e}"
            )


        finally:

            if (
                input_path
                and os.path.exists(input_path)
            ):

                os.remove(input_path)
