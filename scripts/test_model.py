import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("mask_detector_v2.h5")

# Load a test image
img_path = r"C:\Users\pankp\OneDrive\Desktop\pyprojects\dataset\Face Mask Dataset\Train\WithMask\23.png"
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Make prediction
prediction = model.predict(img_array)
label = "Mask Detected" if prediction[0][0] < 0.5 else "No Mask Detected"

print(f"Prediction: {label}")
