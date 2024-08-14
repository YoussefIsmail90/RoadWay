import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
import os
from io import BytesIO
import streamlit.components.v1 as components
import threading

# Load environment variables from .env file
load_dotenv()

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Function to display the map with Folium
def display_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup="Reported Damage Location",
        icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

# Function to process image
def process_image(image_np):
    results = model.predict(source=image_np, conf=confidence_threshold)
    img_with_boxes = results[0].plot()
    return img_with_boxes

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
            results = model.predict(source=frame, conf=confidence_threshold)
            frame_with_boxes = results[0].plot()
            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
    
    video_cap.release()

# Function to handle URL video
def handle_url_video(video_url, frame_interval):
    video_cap = cv2.VideoCapture(video_url)
    stframe = st.empty()
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
            results = model.predict(source=frame, conf=confidence_threshold)
            frame_with_boxes = results[0].plot()
            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

    video_cap.release()

# Function to get location from JavaScript
def get_location_js():
    location_js = """
    <script>
    function sendLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const queryParams = new URLSearchParams({ lat, lon }).toString();
                window.parent.postMessage(queryParams, "*");
            });
        } else {
            alert("Geolocation is not supported by this browser.");
        }
    }
    sendLocation();
    </script>
    """
    components.html(location_js, height=0, width=0)

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video or Provide a URL")

# Options for the user to select
option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video"))

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.5  # Default value
)

# Initialize default location
latitude, longitude = 30.0444, 31.2357  # Default to Cairo, Egypt

# Display JavaScript for geolocation
get_location_js()

# Handle incoming messages from JavaScript
def handle_message():
    global latitude, longitude
    message = st.experimental_get_query_params().get('data', [None])[0]
    if message:
        params_dict = dict(param.split('=') for param in message.split('&'))
        latitude = float(params_dict.get('lat', latitude))
        longitude = float(params_dict.get('lon', longitude))

handle_message()

# Main application logic
if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            img_with_boxes = process_image(image_np)
        
        st.subheader("Detection Results")
        st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.subheader("Processing Video...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name, frame_interval=5)  # Adjust frame interval as needed
        
        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "URL Image":
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL.', use_column_width=True)
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            img_with_boxes = process_image(image_np)
        
        st.subheader("Detection Results")
        st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "URL Video":
    video_url = st.sidebar.text_input("Enter Video URL")
    if video_url:
        frame_interval = st.sidebar.slider("Process Every nth Frame", 1, 30, 5)
        st.subheader("Processing Video from URL...")
        threading.Thread(target=handle_url_video, args=(video_url, frame_interval)).start()

        # Display the map with the detected location
        display_map(latitude, longitude)
