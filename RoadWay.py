import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from geopy.geocoders import Nominatim
import geopy

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Streamlit app title
st.title("Real-Time Roadway Infrastructure Monitoring System")

# Sidebar for Latitude and Longitude input
st.sidebar.header("Set Map Location")

# Function to get the device's location
def get_device_location():
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode("Cairo, Egypt")  # Default fallback location (Cairo, Egypt)
        return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Error obtaining device location: {e}")
        return None, None

# Function to display map with Folium
def display_map(lat, lon):
    if lat is not None and lon is not None:
        m = folium.Map(location=[lat, lon], zoom_start=12)
        folium.Marker(
            [lat, lon], 
            popup="Reported Damage Location", 
            icon=folium.Icon(color="red")
        ).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("Unable to retrieve location data.")

# Custom video transformer using streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.confidence_threshold = 0.5  # Default confidence threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Resize the frame for faster processing
        resized_frame = cv2.resize(img, (640, 360))

        # Perform object detection
        results = model.predict(source=resized_frame, conf=self.confidence_threshold)

        # Draw bounding boxes on the frame
        for result in results:
            img_with_boxes = result.plot()

        return img_with_boxes

# Initialize real-time camera feed
webrtc_ctx = webrtc_streamer(
    key="example", 
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# Obtain the device's current location
latitude, longitude = get_device_location()

# Display the map with the current location
display_map(latitude, longitude)

# Adjust the confidence threshold dynamically
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        0.0,  # Minimum value
        1.0,  # Maximum value
        0.5  # Default value
    )
