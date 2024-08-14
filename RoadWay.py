import streamlit as st
import requests
from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO
from io import BytesIO
from threading import Thread
import time

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Function to process video
def process_video(video_path, frame_interval):
    video_cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
            results = model.predict(source=frame)
            frame_with_boxes = results[0].plot()
            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
    
    video_cap.release()

# Function to handle URL video
def download_video(video_url):
    try:
        response = requests.get(video_url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading video: {e}")
        return None

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with open(tfile.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return tfile.name
    except Exception as e:
        st.error(f"Error saving video: {e}")
        return None

def handle_url_video(video_url, frame_interval):
    video_path = download_video(video_url)
    if video_path:
        process_video(video_path, frame_interval)

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video or Provide a URL")

option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video"))

confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.01)

if option == "URL Video":
    video_url = st.sidebar.text_input("Enter Video URL")
    if video_url:
        frame_interval = st.sidebar.slider("Process Every nth Frame", 1, 30, 5)
        st.subheader("Processing Video from URL...")
        
        # Use threading to handle URL video processing in the background
        Thread(target=handle_url_video, args=(video_url, frame_interval)).start()
