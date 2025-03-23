import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/mask_detector_v2.h5")

# Preprocess the uploaded image
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img = cv2.resize(img, (150, 150))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

# Streamlit UI
st.title("😷 Face Mask Detector")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)[0][0]

    result = "Masked 😷" if prediction > 0.5 else "No Mask ❌"
    confidence = round(prediction, 2)

    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {confidence}")
