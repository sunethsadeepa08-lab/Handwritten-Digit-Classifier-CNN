import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. Loading dataset...")

# Search for the dataset file in the project folder
file_path = None
for f in os.listdir('.'):
    if f.startswith('mnist_test'):
        file_path = f
        break

if not file_path:
    raise FileNotFoundError("Could not find mnist_test dataset file!")

print(f"Loading file: {file_path}")

try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
except Exception as e:
    df = pd.read_csv(file_path)

# Separate features (pixels) and labels
X = df.drop('label', axis=1).values
y = df['label'].values

print("2. Preprocessing data...")
# Normalize pixel values from [0, 255] to [0.0, 1.0]
X = X / 255.0

# Reshape flattened pixel arrays into 28x28 3D image tensors
X_images = X.reshape(-1, 28, 28, 1)

print("3. Building CNN Architecture...")
# Define Convolutional Neural Network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("4. Training the Model...")
# Train the CNN model
model.fit(X_images, y, epochs=5, batch_size=64)

print("5. Saving the Model...")
# Save trained weights to disk
model.save('mnist_cnn_model.h5')
print("\nModel trained successfully and saved as 'mnist_cnn_model.h5'!")