import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO
from geopy.geocoders import Nominatim
from gmplot import gmplot
import os

# Load YOLO models
try:
    model_existing = YOLO('yolov8n_custom_classes.pt')
except Exception as e:
    st.error(f"Failed to load YOLOv8n model: {e}")
    st.stop()

try:
    model_new = YOLO('yolov8_road_damage.pt')
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

# Function to overlay detections from two models
def overlay_detections(image_np, results_existing, results_new):
    img_with_boxes_existing = results_existing[0].plot()
    img_with_boxes_new = results_new[0].plot()
    
    img_with_boxes_existing_pil = Image.fromarray(img_with_boxes_existing)
    img_with_boxes_new_pil = Image.fromarray(img_with_boxes_new)
    
    img_with_boxes_existing_pil = img_with_boxes_existing_pil.convert("RGBA")
    img_with_boxes_new_pil = img_with_boxes_new_pil.convert("RGBA")
    
    combined_img_pil = Image.blend(img_with_boxes_existing_pil, img_with_boxes_new_pil, alpha=0.5)
    combined_img = np.array(combined_img_pil)
    
    return combined_img

# Function to process image with both models
def process_image(image_np):
    if len(image_np.shape) == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = image_np
    else:
        st.error("Unexpected image format. Ensure the image has three channels.")
        return image_np

    results_existing = model_existing.predict(source=image_rgb, conf=confidence_threshold, classes=desired_class_indices)
    results_new = model_new.predict(source=image_rgb, conf=confidence_threshold)
    
    combined_img = overlay_detections(image_rgb, results_existing, results_new)
    
    return combined_img

# Function to extract GPS coordinates from image metadata
def get_image_gps(image):
    try:
        exif_data = image._getexif()
        if exif_data is None:
            st.warning("No EXIF data found.")
            return None, None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_info = value
                break

        if not gps_info:
            st.warning("No GPS information found in EXIF data.")
            return None, None

        def _convert_to_degrees(value):
            d = float(value[0])
            m = float(value[1]) / 60.0
            s = float(value[2]) / 3600.0
            return d + m + s

        lat = _convert_to_degrees(gps_info[2])
        lon = _convert_to_degrees(gps_info[4])

        if gps_info[1] == 'S':
            lat = -lat
        if gps_info[3] == 'W':
            lon = -lon

        return lat, lon
    except Exception as e:
        st.warning(f"Error extracting GPS data: {e}")
        return None, None

# Function to display the map using gmplot
def display_map(lat, lon):
    gmap = gmplot.GoogleMapPlotter(lat, lon, 13)
    gmap.marker(lat, lon, color='red')
    map_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    gmap.draw(map_file.name)
    st.components.v1.html(open(map_file.name, 'r').read(), height=500)

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video")

option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video"))

confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  
    1.0,  
    0.5  
)

if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        lat, lon = get_image_gps(image)
        if lat and lon:
            display_map(lat, lon)
        else:
            st.warning("Location could not be determined from the image metadata.")
        
        image_np = np.array(image)
        with st.spinner('Processing image...'):
            combined_img = process_image(image_np)
        
        st.subheader("Detection Results")
        st.image(combined_img, caption="Combined Detection Results", use_column_width=True)

elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.subheader("Processing Video...")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name, frame_interval=5)

# Function to process video with both models
def process_video(video_path, frame_interval):
    video_cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_count = 0
    
    while video_cap.isOpened():
        ret, frame = cv2.VideoCapture(video_path).read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (640, 360)) 
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            results_existing = model_existing.predict(source=frame_rgb, conf=confidence_threshold, classes=desired_class_indices)
            results_new = model_new.predict(source=frame_rgb, conf=confidence_threshold)
            
            combined_frame = overlay_detections(frame_rgb, results_existing, results_new)
            
            stframe.image(combined_frame, caption="Combined Detection Results", channels="BGR", use_column_width=True)
    
    video_cap.release()
