import streamlit as st
import cv2
import tempfile
import os
import subprocess
import base64
import numpy as np
import requests
import imageio_ffmpeg
import re
import easyocr
from pathlib import Path

from datetime import datetime, timezone

from ultralytics import YOLO
import gdown


st.set_page_config(
    page_title="Road Defect and Hit & Run Detector",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Road Defect and Hit & Run Detector")
st.caption("AI-powered road analysis")

BACKEND_URL = "https://urban-net-sih26124.onrender.com/api/edge/events"

BUS_ID = "BUS_17"
CAMERA_ID = "CAM_FRONT"


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
# HIT & RUN MODEL CONFIGURATION
# ============================================================

HIT_RUN_PLATE_MODEL_PATH = Path("best.pt")

HIT_RUN_BUS_ID = "BUS_17"
HIT_RUN_CAMERA_ID = "CAM_FRONT"



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
# LOAD HIT & RUN MODELS
# ============================================================

@st.cache_resource
def load_hit_run_models():

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = YOLO(
        str(HIT_RUN_PLATE_MODEL_PATH)
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return vehicle_model, plate_model, reader


try:

    hit_run_vehicle_model, hit_run_plate_model, hit_run_reader = (
        load_hit_run_models()
    )

    st.success(
        "✅ Hit & Run models loaded successfully."
    )

except Exception as e:

    st.error(
        f"❌ Could not load Hit & Run models: {e}"
    )

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
# VIDEO PROCESSING
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

    hit_run_tracks = {}
    collision_events = []
    active_collision_pairs = set()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break


            # =================================================
            # HIT & RUN VEHICLE TRACKING
            # =================================================

            hit_run_results = hit_run_vehicle_model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.4,
                verbose=False
            )

            boxes = hit_run_results[0].boxes

            current_objects = []

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                for obj_id, cls, box in zip(
                    ids,
                    classes,
                    xyxy
                ):

                    obj_id = int(obj_id)
                    cls = int(cls)

                    x1, y1, x2, y2 = box

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    previous = hit_run_tracks.get(obj_id)

                    speed = 0.0

                    if previous is not None:

                        px, py, _ = previous

                        speed = (
                            (cx - px) ** 2 +
                            (cy - py) ** 2
                        ) ** 0.5

                    hit_run_tracks[obj_id] = (
                        cx,
                        cy,
                        frame_number
                    )

                    # COCO classes:
                    # 0 = person
                    # 2 = car
                    # 3 = motorcycle
                    # 5 = bus
                    # 7 = truck

                    if cls in [0, 2, 3, 5, 7]:

                        current_objects.append({
                            "id": obj_id,
                            "class": cls,
                            "cx": cx,
                            "cy": cy,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "speed": speed
                        })


            # =================================================
            # HIT & RUN COLLISION DETECTION
            # =================================================

            current_collision_pairs = set()

            for i in range(len(current_objects)):

                for j in range(i + 1, len(current_objects)):

                    a = current_objects[i]
                    b = current_objects[j]

                    distance = (
                        (a["cx"] - b["cx"]) ** 2 +
                        (a["cy"] - b["cy"]) ** 2
                    ) ** 0.5

                    if distance < 60:

                        pair = tuple(
                            sorted([
                                a["id"],
                                b["id"]
                            ])
                        )

                        current_collision_pairs.add(pair)

                        # Only record when the pair
                        # becomes close for the first time
                        if pair not in active_collision_pairs:

                            collision_events.append({

                                "frame": frame_number,

                                "a": a,

                                "b": b,

                                "distance": distance

                            })

            active_collision_pairs = current_collision_pairs

            
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

    hit_run_count = len(collision_events)
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


    # =========================================================
    # HIT & RUN NUMBER PLATE DETECTION
    # =========================================================

    plate_number = None
    plate_confidence = 0.0
    offender_id = None
    vehicle_type = None
    video_timestamp = None

    if len(collision_events) > 0:

        collision = collision_events[0]

        collision_frame = collision["frame"]

        obj_a = collision["a"]
        obj_b = collision["b"]

        vehicle_classes = [2, 3, 5, 7]

        if obj_a["class"] in vehicle_classes:

            offender_id = obj_a["id"]
            offender_box = obj_a

        elif obj_b["class"] in vehicle_classes:

            offender_id = obj_b["id"]
            offender_box = obj_b

        else:

            offender_id = obj_a["id"]
            offender_box = obj_a

        # -----------------------------------------------------
        # VIDEO TIMESTAMP
        # -----------------------------------------------------

        timestamp_seconds = collision_frame / fps

        minutes = int(timestamp_seconds // 60)

        seconds = int(timestamp_seconds % 60)

        video_timestamp = (
            f"{minutes:02d}:{seconds:02d}"
        )

        # -----------------------------------------------------
        # VEHICLE TYPE
        # -----------------------------------------------------

        vehicle_class = offender_box["class"]

        vehicle_type_map = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        vehicle_type = vehicle_type_map.get(
            vehicle_class,
            "vehicle"
        )

        # -----------------------------------------------------
        # OPEN VIDEO AGAIN
        # -----------------------------------------------------

        plate_cap = cv2.VideoCapture(input_path)

        start_frame = max(
            0,
            collision_frame - 30
        )

        end_frame = (
            collision_frame + 60
        )

        plate_cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        current_frame = start_frame

        plate_results = []

        while current_frame <= end_frame:

            ret, plate_frame = plate_cap.read()

            if not ret:
                break

            # -------------------------------------------------
            # NUMBER PLATE DETECTION
            # -------------------------------------------------

            results = hit_run_plate_model.predict(
                plate_frame,
                imgsz=640,
                conf=0.15,
                verbose=False
            )

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    plate_detection_conf = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].cpu().numpy()
                    )

                    crop = plate_frame[
                        max(0, y1):max(y1 + 1, y2),
                        max(0, x1):max(x1 + 1, x2)
                    ]

                    if crop.size == 0:
                        continue

                    # -------------------------------------------------
                    # UPSCALE
                    # -------------------------------------------------

                    crop = cv2.resize(
                        crop,
                        None,
                        fx=3,
                        fy=3,
                        interpolation=cv2.INTER_CUBIC
                    )

                    # -------------------------------------------------
                    # OCR
                    # -------------------------------------------------

                    ocr_results = hit_run_reader.readtext(
                        crop,
                        detail=1,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )

                    for _, text, ocr_conf in ocr_results:

                        cleaned = re.sub(
                            r"[^A-Z0-9]",
                            "",
                            text.upper()
                        )

                        if len(cleaned) >= 5:

                            combined_confidence = (
                                float(ocr_conf)
                                * plate_detection_conf
                            )

                            plate_results.append({
                                "plate": cleaned,
                                "confidence": combined_confidence,
                                "frame": current_frame
                            })

            current_frame += 1

        plate_cap.release()

        # -----------------------------------------------------
        # SELECT BEST PLATE
        # -----------------------------------------------------

        if len(plate_results) > 0:

            plate_results.sort(
                key=lambda x: x["confidence"],
                reverse=True
            )

            best_plate = plate_results[0]

            plate_number = best_plate["plate"]

            plate_confidence = min(
                best_plate["confidence"],
                1.0
            )
    
    return (
        frame_number,
        pothole_count,
        waterlogging_count,
        hit_run_count,
        offender_id,
        vehicle_type,
        video_timestamp,
        plate_number,
        plate_confidence
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

                frames_processed, pothole_detections, waterlogging_detections, hit_run_detections, offender_id, vehicle_type, video_timestamp, plate_number, plate_confidence = (
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

            col1, col2, col3, col4 = st.columns(4)

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

            with col4:
            
                st.metric(
                    "Hit & Run Candidates",
                    f"{hit_run_detections:,}"
                )


            # ------------------------------------------------
            # HIT & RUN RESULT
            # ------------------------------------------------

            if hit_run_detections > 0:

                st.subheader("🚨 Hit & Run Detection")

                st.error(
                    "🚨 Possible HIT-AND-RUN detected"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Offender Track ID",
                        str(offender_id)
                    )

                with col2:

                    st.metric(
                        "Vehicle Type",
                        vehicle_type
                    )

                with col3:

                    st.metric(
                        "Video Timestamp",
                        video_timestamp
                    )

                if plate_number is not None:

                    st.success(
                        f"🚘 Registration Number: "
                        f"{plate_number}"
                    )

                    st.write(
                        f"**OCR Confidence:** "
                        f"{plate_confidence * 100:.1f}%"
                    )

                else:

                    st.warning(
                        "Number plate could not be read clearly."
                    )

                # ------------------------------------------------
                # HIT & RUN BACKEND EVENT
                # ------------------------------------------------

                hit_run_payload = {
                    "eventType": "HIT_AND_RUN",

                    "busId": BUS_ID,

                    "cameraId": CAMERA_ID,

                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "location": {
                        "latitude": 28.6139,
                        "longitude": 77.2090,
                        "address": "New Delhi"
                    },

                    "detection": {
                        "confidence": float(
                            plate_confidence
                        ),
                        "severity": "CRITICAL"
                    },

                    "model": {
                        "name": "hit-and-run-yolo",
                        "version": "1.0"
                    },

                    "evidence": {
                        "imageUrl": None
                    },

                    "metadata": {
                        "offendingVehicleReg": plate_number,
                        "offendingVehicleDetails": vehicle_type,
                        "offenderTrackId": offender_id,
                        "videoTimestamp": video_timestamp
                    }
                }

            else:

                st.info(
                    "ℹ️ No possible hit-and-run collision "
                    "was detected."
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
