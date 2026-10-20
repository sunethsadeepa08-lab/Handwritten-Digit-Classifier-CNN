import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import center_of_mass
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier AI", page_icon="🔢"
)

st.title("🔢 AI Handwritten Digit Classifier")
st.write(
    "Draw a digit on the canvas or upload an image (paper photo) to classify using Deep Learning!"
)


# Load Trained Model
@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()


# Ultimate Preprocessing Pipeline (Handles Ballpoint Pen on Paper & Digital Canvas)
def preprocess_image(image):
    # 1. Resize image to standard processing resolution (300x300)
    img_gray = np.array(image.convert("L").resize((300, 300)))

    # 2. Adaptive Pipeline for Paper Photos vs Canvas Drawings
    if np.mean(img_gray) > 127:
        # Paper Photo Pipeline: CLAHE contrast enhancement for ballpoint pen lines
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        inv = cv2.bitwise_not(enhanced)

        # Adaptive Gaussian Thresholding to isolate pen strokes from paper shadows
        thresh = cv2.adaptiveThreshold(
            inv,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )

        # Dilation: Thickens thin 0.5mm pen strokes to match MNIST marker brush size
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
    else:
        # Canvas Pipeline: Thresholding for white strokes on dark background
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 3. Find digit contours and ignore tiny paper noise
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            digit = thresh[y : y + h, x : x + w]

            # Fit into 20x20 box preserving aspect ratio
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

            # Center using Center of Mass (Standard MNIST Pipeline)
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


# Option Selection: Draw on Canvas or Upload Image
option = st.radio("Choose Input Method:", ("Draw on Canvas", "Upload Image"))

img_to_process = None

if option == "Draw on Canvas":
    st.write("🖌️ **Draw any digit (0–9) using your mouse/touch pad:**")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if (
        canvas_result.image_data is not None
        and np.max(canvas_result.image_data) > 0
    ):
        img_to_process = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

else:
    uploaded_file = st.file_uploader(
        "Upload an Image (PNG/JPG)...", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)

# Processing & Prediction Execution
if img_to_process is not None:
    processed_img, display_img = preprocess_image(img_to_process)

    st.image(
        display_img,
        caption="Preprocessed Image (MNIST Centered & Noise Filtered)",
        width=150,
    )

    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    st.success(f"🎯 **Predicted Digit: {predicted_digit}**")
    st.info(f"**Confidence Level: {confidence:.2f}%**")