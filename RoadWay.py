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

# Function to display the JavaScript widget
def display_location_widget():
    with open("location_widget.html", "r") as file:
        html_code = file.read()
    components.html(html_code, height=100)

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
st.subheader("Get Your Location")
display_location_widget()

# JavaScript to Python communication
message = st.experimental_get_query_params()
latitude, longitude = None, None
if 'latitude' in message and 'longitude' in message:
    latitude = float(message['latitude'][0])
    longitude = float(message['longitude'][0])
    st.write(f"Detected Location: Latitude {latitude}, Longitude {longitude}")

# Options for the user to select
option = st.sidebar.selectbox("Choose Input Type", ("Upload Image", "Upload Video", "URL Image", "URL Video"))

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.01  # Default value
)

if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Simulate image processing here
        st.subheader("Image Processing Results")
        st.write("Detected image with simulated results.")
        
        # Display the map with the selected coordinates
        if latitude and longitude:
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

            frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
            # Simulate video processing here
            stframe.image(frame, channels="BGR", use_column_width=True)
        
        video_cap.release()
        
        # Display the map with the selected coordinates
        if latitude and longitude:
            display_map(latitude, longitude)

elif option == "URL Image":
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL.', use_column_width=True)
        
        # Simulate image processing here
        st.subheader("Image Processing Results")
        st.write("Detected image with simulated results.")
        
        # Display the map with the selected coordinates
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
                # Simulate video processing here
                stframe.image(frame, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Display the map with the selected coordinates
        if latitude and longitude:
            display_map(latitude, longitude)
