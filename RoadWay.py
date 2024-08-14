import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
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
option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video"))

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

# Function to get device location using Nominatim with retry mechanism
def get_device_location(address):
    geolocator = Nominatim(user_agent="geoapiExercises", timeout=10)
    retries = 5  # Number of retries
    delay = 1  # Initial delay in seconds
    
    while retries > 0:
        try:
            location = geolocator.geocode(address)
            return location
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            retries -= 1
            if retries == 0:
                st.error("Unable to retrieve location data after multiple attempts.")
                return None
            else:
                st.warning(f"Error obtaining device location: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff: double the delay each time

# Function to display map with Folium
def display_map(lat, lon):
    # Create a folium map centered on the provided latitude and longitude
    m = folium.Map(location=[lat, lon], zoom_start=6)
    
    # Add a marker to the map
    folium.Marker(
        [lat, lon], 
        popup="Reported Damage Location", 
        icon=folium.Icon(color="red")
    ).add_to(m)
    
    # Display the map in the Streamlit app
    st_folium(m, width=700, height=500)

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

        # Display the map with the selected coordinates
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

            frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for faster processing
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Display the map with the selected coordinates
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

        # Display the map with the selected coordinates
        display_map(latitude, longitude)

elif option == "URL Video":
    video_url = st.sidebar.text_input("Enter Video URL")
    if video_url:
        frame_interval = st.sidebar.slider("Process Every nth Frame", 1, 30, 5)  # Show frame interval only for URL Video
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
                frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for faster processing
                results = model.predict(source=frame, conf=confidence_threshold)
                for result in results:
                    frame_with_boxes = result.plot()

                stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Display the map with the selected coordinates
        display_map(latitude, longitude)

else:
    st.error("Invalid option selected.")

