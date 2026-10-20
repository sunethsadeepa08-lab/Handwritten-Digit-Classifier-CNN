import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import center_of_mass
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier AI", page_icon="🔢"
)

st.title("🔢 AI Handwritten Digit Classifier")
st.write(
    "Upload an image of a handwritten digit (0–9) to test the trained Deep Learning Model."
)


# Load Trained Model
@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()


# Advanced Shadow-Removal & Adaptive Preprocessing
def preprocess_image(image):
    gray = np.array(image.convert("L"))

    # Remove background shadows using Adaptive Gaussian Thresholding
    # Works for both white paper phone photos and digital paint dark/white canvas
    if np.mean(gray) > 127:
        # For paper photos with light background
        inv_gray = cv2.bitwise_not(gray)
        thresh = cv2.adaptiveThreshold(
            inv_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )
    else:
        # For dark canvas drawings
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Thicken thin ballpoint pen lines to match MNIST brush size
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours and filter out small noise dots
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the contour that resembles the actual digit (ignoring tiny dots)
        contours = [c for c in contours if cv2.contourArea(c) > 20]

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            digit = thresh[y : y + h, x : x + w]

            # Aspect ratio resize into 20x20 box
            aspect_ratio = w / float(h)
            if aspect_ratio > 1:
                new_w = 20
                new_h = max(1, int(20 / aspect_ratio))
            else:
                new_h = 20
                new_w = max(1, int(20 * aspect_ratio))

            digit_resized = cv2.resize(
                digit, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            # Pad into 28x28 centered frame
            padded = np.zeros((28, 28), dtype=np.uint8)
            pad_x = (28 - new_w) // 2
            pad_y = (28 - new_h) // 2
            padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = digit_resized

            # Center using Center of Mass
            cy, cx = center_of_mass(padded)
            if not np.isnan(cy) and not np.isnan(cx):
                shift_x = int(round(14 - cx))
                shift_y = int(round(14 - cy))
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                padded = cv2.warpAffine(padded, M, (28, 28))

            normalized = padded / 255.0
            return normalized.reshape(1, 28, 28, 1), padded

    resized = cv2.resize(gray, (28, 28)) / 255.0
    return resized.reshape(1, 28, 28, 1), cv2.resize(gray, (28, 28))


# File Uploader
uploaded_file = st.file_uploader(
    "Upload an Image (PNG/JPG)...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    processed_img, display_img = preprocess_image(img)

    st.image(
        display_img, caption="Preprocessed Image (Adaptive MNIST)", width=150
    )

    # Prediction
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    st.success(f"🎯 **Predicted Digit: {predicted_digit}**")
    st.info(f"**Confidence Level: {confidence:.2f}%**")