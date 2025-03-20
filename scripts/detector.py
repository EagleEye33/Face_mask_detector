import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing with Augmentation
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_data = data_gen.flow_from_directory(r"C:\Users\pankp\OneDrive\Desktop\pyprojects\dataset\Face Mask Dataset\Train", target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')
val_data = data_gen.flow_from_directory(r"C:\Users\pankp\OneDrive\Desktop\pyprojects\dataset\Face Mask Dataset\Train", target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')

# Load Pretrained Model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# Freeze the base model layers so they don’t get trained
base_model.trainable = False

# Add Custom Layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the Model
history = model.fit(train_data, validation_data=val_data, epochs=20)

# Save the Model
model.save("mask_detector_v2.h5")
print("Model training complete and saved as 'mask_detector_v2.h5'")

