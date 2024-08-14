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
import sqlite3
import os

# Initialize the database
db_path = "roadway_monitoring.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create table to store detected damages if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS damages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        damage_type TEXT,
        confidence REAL,
        latitude REAL,
        longitude REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

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

# Set the default confidence threshold value to 0.5
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    0.0,  # Minimum value
    1.0,  # Maximum value
    0.5  # Default value
)

# Add GPS coordinates input
latitude = st.sidebar.text_input("Enter Latitude", "40.7128")
longitude = st.sidebar.text_input("Enter Longitude", "-74.0060")

def save_detection_data(damage_type, confidence, latitude, longitude, image_path):
    c.execute("INSERT INTO damages (damage_type, confidence, latitude, longitude, image_path) VALUES (?, ?, ?, ?, ?)",
              (damage_type, confidence, latitude, longitude, image_path))
    conn.commit()

def display_map():
    # Retrieve all damages from the database
    c.execute("SELECT damage_type, confidence, latitude, longitude FROM damages")
    damages = c.fetchall()

    # Create a folium map centered at the average location of all damages
    if damages:
        avg_lat = sum([damage[2] for damage in damages]) / len(damages)
        avg_lon = sum([damage[3] for damage in damages]) / len(damages)
    else:
        avg_lat, avg_lon = 40.7128, -74.0060  # Default to NYC coordinates

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

    # Add markers for each detected damage
    for damage in damages:
        folium.Marker(
            location=[damage[2], damage[3]],
            popup=f"Damage: {damage[0]}, Confidence: {damage[1]:.2f}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

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

            # Save detection data
            for box in result.boxes:
                damage_type = box.label
                confidence = box.confidence
                save_detection_data(damage_type, confidence, latitude, longitude, uploaded_file.name)

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

            # Downscale frame for faster processing
            frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for faster processing
            results = model.predict(source=frame, conf=confidence_threshold)
            for result in results:
                frame_with_boxes = result.plot()

                # Save detection data
                for box in result.boxes:
                    damage_type = box.label
                    confidence = box.confidence
                    save_detection_data(damage_type, confidence, latitude, longitude, uploaded_file.name)

            stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

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

            # Save detection data
            for box in result.boxes:
                damage_type = box.label
                confidence = box.confidence
                save_detection_data(damage_type, confidence, latitude, longitude, image_url)

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
            if frame_count % frame_interval == 0:  # Process every nth frame
                # Downscale frame for faster processing
                frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for faster processing
                results = model.predict(source=frame, conf=confidence_threshold)
                for result in results:
                    frame_with_boxes = result.plot()

                    # Save detection data
                    for box in result.boxes:
                        damage_type = box.label
                        confidence = box.confidence
                        save_detection_data(damage_type, confidence, latitude, longitude, video_url)

                stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        video_cap.release()

# Display the map with detected damages
st.subheader("Damage Location Map")
display_map()

# Option to view detection data
if st.sidebar.button("View Detection Data"):
    st.subheader("Stored Detection Data")
    c.execute("SELECT * FROM damages")
    rows = c.fetchall()
    if rows:
        for row in rows:
            st.write(f"Damage Type: {row[2]}, Confidence: {row[3]:.2f}, Location: ({row[4]}, {row[5]}), Timestamp: {row[6]}")
    else:
        st.write("No detection data found.")

# Close the database connection
conn.close()

