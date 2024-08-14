import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import geocoder
import folium
from streamlit_folium import st_folium

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Streamlit app title
st.title("Real-Time Roadway Infrastructure Monitoring System")

st.sidebar.header("Settings")

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.5  # Default value
)

# Function to display map with Folium
def display_map(lat, lon):
    # Create a folium map centered on the provided latitude and longitude
    m = folium.Map(location=[lat, lon], zoom_start=6)
    
    # Add a marker to the map
    folium.Marker(
        [lat, lon], 
        popup="Real-Time Location", 
        icon=folium.Icon(color="red")
    ).add_to(m)
    
    # Display the map in the Streamlit app
    st_folium(m, width=700, height=500)

# Function to get the current location
def get_current_location():
    # Use geocoder to get the current location based on the IP address
    g = geocoder.ip('me')
    latlng = g.latlng
    return latlng[0], latlng[1] if latlng else (None, None)

# Real-time video processing using webcam
if st.button("Start Real-Time Detection"):
    # Get the current location
    latitude, longitude = get_current_location()
    if latitude is None or longitude is None:
        st.error("Unable to get location. Please ensure your device's location services are enabled.")
    else:
        st.success(f"Current location: Latitude {latitude}, Longitude {longitude}")
        
        # Open webcam
        video_cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                st.warning("Failed to capture video frame.")
                break

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 360))
            results = model.predict(source=frame, conf=confidence_threshold)
            
            # Display the detection results
            for result in results:
                frame_with_boxes = result.plot()
            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

        video_cap.release()

        # Display the map with the real-time location
        display_map(latitude, longitude)

