
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ==================================================
# 1. Configuration
# ==================================================

st.set_page_config(
    page_title="VeyloraX - Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

MODEL_PATH = "src/models/vehicle_detection.pt"


# ==================================================
# 2. Style
# ==================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }

    .title {
        text-align: center;
        color: #10b981;
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. Load YOLO model
# ==================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model introuvable : {MODEL_PATH}. "
        "Placez vehicle_detection.pt dans le même dossier que app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()


# ==================================================
# 4. Header and navigation
# ==================================================

st.markdown(
    '<div class="title">VeyloraX</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Vehicle Detection'
    '</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Upload Image", "Upload Video"]
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)


# ==================================================
# 5. Image detection function
# ==================================================

def detect_image(image):
    image_array = np.array(image.convert("RGB"))
    image_array = np.ascontiguousarray(image_array)

    results = model.predict(
        source=image_array,
        conf=confidence,
        verbose=False
    )

    # Ultralytics returns an annotated BGR image
    annotated_bgr = results[0].plot()

    # Convert BGR to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(annotated_rgb), results[0]


# ==================================================
# 6. Page 1: Upload Image
# ==================================================

if page == "Upload Image":

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Detection Result")

            if st.button("Detect Vehicles", key="detect_image"):

                with st.spinner("Detecting vehicles..."):
                    annotated_image, result = detect_image(image)

                st.image(
                    annotated_image,
                    use_container_width=True
                )

                st.success(
                    f"{len(result.boxes)} detection(s) found."
                )

                if len(result.boxes) > 0:
                    st.markdown("### Detection Details")

                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]
                        score = float(box.conf[0].item())

                        st.write(
                            f"**{class_name}** — "
                            f"Confidence: {score:.2%}"
                        )

                # Download annotated image
                from io import BytesIO

                buffer = BytesIO()
                annotated_image.save(buffer, format="PNG")

                st.download_button(
                    "Download Result Image",
                    data=buffer.getvalue(),
                    file_name="vehicle_detection_result.png",
                    mime="image/png"
                )


# ==================================================
# 7. Page 2: Upload Video
# ==================================================

elif page == "Upload Video":

    st.subheader("Upload a Video")

    st.write(
        "Upload a video to detect vehicles in its frames "
        "and draw bounding boxes."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:

        st.markdown("### Original Video")
        st.video(uploaded_video)

        if st.button("Detect Vehicles in Video", key="detect_video"):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                # Save uploaded video temporarily
                extension = os.path.splitext(
                    uploaded_video.name
                )[1].lower()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as input_file:
                    input_file.write(uploaded_video.getvalue())
                    input_path = input_file.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    raise ValueError(
                        "Impossible d'ouvrir la vidéo."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError(
                        "Les informations de la vidéo sont invalides."
                    )

                # Create output video
                output_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                output_path = output_file.name
                output_file.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise ValueError(
                        "Impossible de créer la vidéo résultat."
                    )

                progress = st.progress(0)
                status = st.empty()

                frame_count = 0
                total_detections = 0

                # Process every video frame
                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    # OpenCV BGR -> RGB
                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    results = model.predict(
                        source=frame_rgb,
                        conf=confidence,
                        verbose=False
                    )

                    result = results[0]
                    total_detections += len(result.boxes)

                    # Draw boxes and labels
                    annotated_rgb = result.plot()

                    # RGB -> BGR for OpenCV video writer
                    annotated_bgr = cv2.cvtColor(
                        annotated_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                    writer.write(annotated_bgr)

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.text(
                        f"Processing frame {frame_count}"
                        f" / {total_frames}"
                    )

                cap.release()
                cap = None

                writer.release()
                writer = None

                if frame_count == 0:
                    raise ValueError(
                        "Aucune frame n'a pu être traitée."
                    )

                progress.progress(1.0)
                status.success("Video processing completed!")

                st.markdown("### Detection Result")

                st.video(output_path)

                st.success(
                    f"Frames processed: {frame_count} | "
                    f"Total detections across frames: {total_detections}"
                )

                # Download processed video
                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "Download Processed Video",
                        data=video_file.read(),
                        file_name="vehicle_detection_result.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Erreur pendant le traitement : {e}")

            finally:
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

                # Keep output_path available for Streamlit
                # during this run so the video can be displayed.
