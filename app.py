import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import center_of_mass
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier AI",
    page_icon="🔢",
    layout="centered",
)

st.title("🔢 AI Handwritten Digit Classifier")
st.write(
    "Upload an image of a handwritten digit (paper photo or digital file) to classify using Deep Learning!"
)


# Load Trained CNN Model
@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()


# Robust Preprocessing Pipeline
def preprocess_image(image):
    img_gray = np.array(image.convert("L").resize((300, 300)))

    # Paper Photo Processing Pipeline
    if np.mean(img_gray) > 127:
        # Contrast Enhancement & Thresholding for Paper Photos
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        inv = cv2.bitwise_not(enhanced)
        thresh = cv2.adaptiveThreshold(
            inv,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
    else:
        # Dark Background Pipeline
        _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Contour Extraction and Centering
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit = thresh[y : y + h, x : x + w]

            # Aspect Ratio Resizing
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

            padded = np.zeros((28, 28), dtype=np.uint8)
            pad_x = (28 - new_w) // 2
            pad_y = (28 - new_h) // 2
            padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = digit_resized

            # Center of Mass Alignment (MNIST Standard)
            cy, cx = center_of_mass(padded)
            if not np.isnan(cy) and not np.isnan(cx):
                shift_x = int(round(14 - cx))
                shift_y = int(round(14 - cy))
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                padded = cv2.warpAffine(padded, M, (28, 28))

            normalized = padded / 255.0
            return normalized.reshape(1, 28, 28, 1), padded

    resized = cv2.resize(img_gray, (28, 28)) / 255.0
    return resized.reshape(1, 28, 28, 1), cv2.resize(img_gray, (28, 28))


# Image Uploader Section
uploaded_file = st.file_uploader(
    "Upload an Image of Digit (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img_input = Image.open(uploaded_file)

    # Prediction Execution
    processed_img, display_img = preprocess_image(img_input)

    col1, col2 = st.columns(2)

    with col1:
        st.write("📷 **Uploaded Original Image:**")
        st.image(img_input, use_container_width=True)

    with col2:
        st.write("🔬 **AI Preprocessed Image (28x28):**")
        st.image(display_img, width=160)

    # Model Inference
    prediction = model.predict(processed_img)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.subheader("🔍 Prediction Results:")
    st.success(f"🎯 **Predicted Digit: {predicted_digit}**")
    st.info(f"📊 **Confidence Level: {confidence:.2f}%**")