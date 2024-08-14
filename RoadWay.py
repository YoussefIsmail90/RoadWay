import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from PIL import Image
import numpy as np
import tempfile
import cv2
import requests
from io import BytesIO
from ultralytics import YOLO

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Function to display the JavaScript widget for getting location
def display_location_widget():
    # JavaScript code to get location and send it to Streamlit
    js_code = """
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    // Pass the latitude and longitude to Streamlit
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    window.parent.postMessage({lat: lat, lon: lon}, "*");
                },
                function(error) {
                    console.error("Error getting location: " + error.message);
                }
            );
        } else {
            console.error("Geolocation is not supported by this browser.");
        }
    }
    // Automatically get location on page load
    window.onload = getLocation;
    </script>
    """
    components.html(js_code, height=0)

# Function to display map with Folium
def display_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup="Reported Damage Location",
        icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video or Provide a URL")

# Display the location widget
st.subheader("Getting Your Location...")
display_location_widget()

# JavaScript to Python communication
def get_location_from_js():
    location = st.experimental_get_query_params()
    if 'lat' in location and 'lon' in location:
        return float(location['lat'][0]), float(location['lon'][0])
    return None, None

latitude, longitude = get_location_from_js()

if latitude and longitude:
    st.write(f"Detected Location: Latitude {latitude}, Longitude {longitude}")

# Options for the user to select
option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video"))

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.5  # Default value
)

def process_image(image_np):
    # Perform object detection
    results = model.predict(source=image_np, conf=confidence_threshold)
    img_with_boxes = results[0].plot()
    return img_with_boxes

def process_video(video_path):
    # Process video frame by frame
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
        if latitude and longitude:
            display_map(latitude, longitude)

elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.subheader("Processing Video...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)
        
        # Display the map with the detected location
        if latitude and longitude:
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
        if latitude and longitude:
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
        if latitude and longitude:
            display_map(latitude, longitude)
