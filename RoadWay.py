import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the YOLOv8 model
model = YOLO('yolov8_road_damage.pt')

# Streamlit app
st.title("Road Damage Detection with YOLOv8")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert image to format suitable for YOLOv8
    image_np = np.array(image)
    results = model.predict(source=image_np)

    # Display results
    st.subheader("Detection Results")
    for result in results:
        img_with_boxes = result.plot()
        st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

