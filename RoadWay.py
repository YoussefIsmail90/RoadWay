import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
import geocoder  # For getting GPS location

# Load the YOLOv8 model
try:
    model = YOLO('yolov8_road_damage.pt')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the YOLOv8 model: {e}")
    st.stop()

# Streamlit app title
st.title("Real-Time Roadway Infrastructure Monitoring System")

st.sidebar.header("Select Input Mode")

# Options for the user to select
option = st.sidebar.selectbox("Choose Input Type", ("Real-Time Camera", "Upload Image", "Upload Video", "URL Image", "URL Video"))

# Set the default confidence threshold value
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.01  # Default value
)

# Function to display map with Folium
def display_map(lat, lon):
    # Create a folium map centered on the provided latitude and longitude
    m = folium.Map(location=[lat, lon], zoom_start=12)
    
    # Add a marker to the map
    folium.Marker(
        [lat, lon], 
        popup="Reported Damage Location", 
        icon=folium.Icon(color="red")
    ).add_to(m)
    
    # Display the map in the Streamlit app
    st_folium(m, width=700, height=500)

if option == "Real-Time Camera":
    st.subheader("Real-Time Camera Feed")

    # Get the device's current location using geocoder (IP-based location)
    g = geocoder.ip('me')
    latitude = g.latlng[0]
    longitude = g.latlng[1]
    
    if st.button("Start Camera"):
        video_cap = cv2.VideoCapture(0)  # Open the device camera

        stframe = st.empty()
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Display the map with the detected location
        display_map(latitude, longitude)

elif option == "Upload Image":
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

        # Get GPS location and display the map
        g = geocoder.ip('me')
        latitude = g.latlng[0]
        longitude = g.latlng[1]
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
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Get GPS location and display the map
        g = geocoder.ip('me')
        latitude = g.latlng[0]
        longitude = g.latlng[1]
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

        # Get GPS location and display the map
        g = geocoder.ip('me')
        latitude = g.latlng[0]
        longitude = g.latlng[1]
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
                frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
                results = model.predict(source=frame, conf=confidence_threshold)
                for result in results:
                    frame_with_boxes = result.plot()

                stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

        # Get GPS location and display the map
        g = geocoder.ip('me')
        latitude = g.latlng[0]
        longitude = g.latlng[1]
        display_map(latitude, longitude)

