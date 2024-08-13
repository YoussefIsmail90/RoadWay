import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Streamlit app title
st.title("Road Damage Detection with YOLOv8")

st.sidebar.header("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Convert image to format suitable for YOLOv8
        image_np = np.array(image)

        # Display a spinner while the model is processing
        with st.spinner('Processing image...'):
            results = model.predict(source=image_np, conf=confidence_threshold)

        # Display results
        st.subheader("Detection Results")
        for result in results:
            img_with_boxes = result.plot()
            st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

    elif file_type == 'video':
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Process video and display
        st.subheader("Video Detection Results")
        with st.spinner('Processing video...'):
            vid = cv2.VideoCapture(video_path)
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vid.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(tempfile.gettempdir(), "detected_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                # Convert frame to YOLOv8 compatible format
                results = model.predict(source=frame, conf=confidence_threshold)
                for result in results:
                    frame_with_boxes = result.plot()

                out.write(frame_with_boxes)

            vid.release()
            out.release()

        st.video(output_path)

        # Option to download the processed video
        with open(output_path, "rb") as f:
            st.download_button(label="Download Processed Video", data=f, file_name="detected_road_damage.mp4")

# Real-time Webcam Detection
st.sidebar.subheader("Real-Time Detection")
webcam_index = st.sidebar.number_input("Webcam Index", value=0, min_value=0, step=1)

if st.sidebar.button("Start Webcam"):
    st.subheader("Webcam Live Feed")

    # Access the webcam
    webcam = cv2.VideoCapture(webcam_index)

    if not webcam.isOpened():
        st.error("Could not open webcam. Please check the webcam index and permissions.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = webcam.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Convert frame to YOLOv8 compatible format
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

    webcam.release()

