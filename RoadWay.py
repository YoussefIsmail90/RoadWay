import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video or Provide a URL")

# Options for the user to select
option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video", "Real-Time"))

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.01  # Default value
)

# Sidebar for Latitude and Longitude input
st.sidebar.header("Set Map Location")
latitude = st.sidebar.number_input("Latitude", value=30.0444, format="%.6f")  # Default is Cairo, Egypt
longitude = st.sidebar.number_input("Longitude", value=31.2357, format="%.6f")  # Default is Cairo, Egypt

# Function to display map with Folium
def display_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon], 
        popup="Reported Damage Location", 
        icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

def real_time_video_processing():
    st.subheader("Real-Time Video Processing")
    
    camera_index = 0  # Try different indices if necessary
    video_cap = cv2.VideoCapture(camera_index)
    
    # Check if the webcam was successfully opened
    if not video_cap.isOpened():
        st.error(f"Unable to access the webcam with index {camera_index}. Please check your camera settings.")
        return
    
    stframe = st.empty()
    
    while True:
        ret, frame = video_cap.read()
        
        if not ret:
            st.error("Failed to grab frame from webcam. Please check your camera.")
            video_cap.release()
            return
        
        frame = cv2.resize(frame, (640, 360))
        results = model.predict(source=frame, conf=confidence_threshold)
        
        for result in results:
            frame_with_boxes = result.plot()
        
        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        if st.button("Stop Real-Time Processing"):
            video_cap.release()
            st.success("Real-time processing stopped.")
            break

# Handle different options
if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            results = model.predict(source=image_np, conf=confidence_threshold)
        
        st.subheader("Detection Results")
        for result in results:
            img_with_boxes = result.plot()
            st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

        display_map(latitude, longitude)

elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.subheader("Processing Video...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        display_map(latitude, longitude)

elif option == "URL Image":
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL.', use_column_width=True)
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            results = model.predict(source=image_np, conf=confidence_threshold)
        
        st.subheader("Detection Results")
        for result in results:
            img_with_boxes = result.plot()
            st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

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
                frame = cv2.resize(frame, (640, 360))
                results = model.predict(source=frame, conf=confidence_threshold)
                for result in results:
                    frame_with_boxes = result.plot()

                stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        display_map(latitude, longitude)

elif option == "Real-Time":
    real_time_video_processing()

else:
    st.write("Select an option from the sidebar.")
