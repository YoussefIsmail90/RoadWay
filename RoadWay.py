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

# Load environment variables from .env file
load_dotenv()

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Function to get location based on IP using ipinfo.io
def get_location_from_address(address):
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("API key for Google Maps is missing.")
        return 30.0444, 31.2357  # Default to Cairo, Egypt

    try:
        response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}')
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            lat = location['lat']
            lon = location['lng']
            return lat, lon
        else:
            print("Failed to get location from address.")
            return 30.0444, 31.2357  # Default to Cairo, Egypt
    except requests.RequestException as e:
        print(f"Failed to get location: {e}")
        return 30.0444, 31.2357  # Default to Cairo, Egypt

latitude, longitude = get_ip_location()

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
def process_video(video_path):
    video_cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
        results = model.predict(source=frame, conf=confidence_threshold)
        frame_with_boxes = results[0].plot()

        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
    
    video_cap.release()

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
        process_video(tfile.name)
        
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

        # Display the map with the detected location
        display_map(latitude, longitude)
