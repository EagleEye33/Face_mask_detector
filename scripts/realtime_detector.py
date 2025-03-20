import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model("mask_detector_v2.h5")

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use webcam (0 for default camera)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face, (150, 150))  # Resize to match model input size
        face_array = img_to_array(face_resized) / 255.0  # Normalize
        face_array = np.expand_dims(face_array, axis=0)  # Expand dimensions for model input

        # Predict mask or no mask
        prediction = model.predict(face_array)
        label = "Mask" if prediction[0][0] < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green for Mask, Red for No Mask

        # Draw rectangle and label on face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the output
    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
