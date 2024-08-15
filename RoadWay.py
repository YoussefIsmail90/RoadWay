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
from threading import Thread

# Load environment variables from .env file
load_dotenv()

# Load YOLO models
try:
    model_existing = YOLO('yolov8n_custom_classes.pt')
    # st.success("YOLOv8n model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load YOLOv8n model: {e}")
    st.stop()

try:
    model_new = YOLO('yolov8_road_damage.pt')
    # st.success("YOLOv8 Road Damage model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load YOLOv8 Road Damage model: {e}")
    st.stop()

# Desired class indices for yolov8n_custom_classes.pt
desired_classes = {
    'Bus': 5,
    'Train': 6,
    'Truck': 7,
    'Traffic light': 9,
    'Fire hydrant': 10,
    'Stop sign': 11,
    'Bicycle': 1,
    'Car': 2,
    'Motorbike': 3
}
desired_class_indices = list(desired_classes.values())

# Function to display the map with Folium
def display_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup="Reported Damage Location",
        icon=folium.Icon(color="red")
    ).add_to(m)
    st_folium(m, width=700, height=500)

# Function to overlay detections from two models
def overlay_detections(image_np, results_existing, results_new):
    img_with_boxes_existing = results_existing[0].plot()
    img_with_boxes_new = results_new[0].plot()
    
    # Convert to PIL Image for overlay
    img_with_boxes_existing_pil = Image.fromarray(img_with_boxes_existing)
    img_with_boxes_new_pil = Image.fromarray(img_with_boxes_new)
    
    # Convert to RGBA for better blending
    img_with_boxes_existing_pil = img_with_boxes_existing_pil.convert("RGBA")
    img_with_boxes_new_pil = img_with_boxes_new_pil.convert("RGBA")
    
    # Combine the images
    combined_img_pil = Image.blend(img_with_boxes_existing_pil, img_with_boxes_new_pil, alpha=0.5)
    combined_img = np.array(combined_img_pil)
    
    return combined_img

# Function to process image with both models
def process_image(image_np):
    # Convert grayscale to RGB if necessary
    if len(image_np.shape) == 2:  # Grayscale image
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
        image_rgb = image_np
    else:
        st.error("Unexpected image format. Ensure the image has three channels.")
        return image_np  # Return the original image or handle the error as needed

    # Process the image with the YOLOv8n model and filter by desired classes
    results_existing = model_existing.predict(source=image_rgb, conf=confidence_threshold, classes=desired_class_indices)
    
    # Process the image with the YOLOv8 road damage model (no filtering)
    results_new = model_new.predict(source=image_rgb, conf=confidence_threshold)
    
    # Combine detection results
    combined_img = overlay_detections(image_rgb, results_existing, results_new)
    
    return combined_img

# Function to process video with both models
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
            
            # Convert BGR to RGB if necessary
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Apply class filter to the YOLOv8n model's predictions
            results_existing = model_existing.predict(source=frame_rgb, conf=confidence_threshold, classes=desired_class_indices)
            
            # Process the video frame with the YOLOv8 road damage model (no filtering)
            results_new = model_new.predict(source=frame_rgb, conf=confidence_threshold)
            
            combined_frame = overlay_detections(frame_rgb, results_existing, results_new)
            
            stframe.image(combined_frame, caption="Combined Detection Results", channels="BGR", use_column_width=True)
    
    video_cap.release()

# Function to handle URL video
def download_video(video_url):
    response = requests.get(video_url, stream=True)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(tfile.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tfile.name

def handle_url_video(video_url, frame_interval):
    video_path = download_video(video_url)
    process_video(video_path, frame_interval)

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
            combined_img = process_image(image_np)
        
        st.subheader("Detection Results")
        st.image(combined_img, caption="Combined Detection Results", use_column_width=True)

        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.subheader("Processing Video...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name, frame_interval=5)
        display_map(latitude, longitude)

elif option == "URL Image":
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL.', use_column_width=True)
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            combined_img = process_image(image_np)
        
        st.subheader("Detection Results")
        st.image(combined_img, caption="Combined Detection Results", use_column_width=True)

        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "URL Video":
    video_url = st.sidebar.text_input("Enter Video URL")
    if video_url:
        st.subheader("Processing Video from URL...")
        handle_url_video(video_url, frame_interval=5)
        display_map(latitude, longitude)
