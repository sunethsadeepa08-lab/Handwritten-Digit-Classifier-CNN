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
    "Upload a photo of a handwritten digit (paper) or draw on the canvas to test the CNN model."
)


# Load Trained Model
@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()


# Advanced Preprocessing Pipeline
def preprocess_image(image):
    img_gray = np.array(image.convert("L").resize((300, 300)))

    # Paper Photo vs Canvas Detection
    if np.mean(img_gray) > 127:
        # Paper Photo Pipeline: CLAHE contrast enhancement & Adaptive Thresholding
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
        # Canvas Pipeline
        _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit = thresh[y : y + h, x : x + w]

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

            # Center of Mass Alignment
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


# Option Selection
option = st.radio("Choose Input Method:", ("Upload Image", "Draw on Canvas"))

img_to_process = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an Image of Digit (PNG/JPG/JPEG)...", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)

else:
    st.write("🖌️ **Draw a digit (0–9) on the canvas below:**")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    if st.button("🎯 Classify Canvas Digit"):
        if canvas_result is not None:
            raw_data = None
            try:
                raw_data = canvas_result.image_data
            except Exception:
                raw_data = None

            if raw_data is not None:
                arr = raw_data.astype("uint8")
                if np.any(arr[:, :, :3] > 0):
                    img_to_process = Image.fromarray(arr).convert("RGB")
                else:
                    st.warning("⚠️ Please draw a digit on the canvas first!")
            else:
                st.warning("⚠️ Please draw a digit on the canvas first!")

# Prediction Execution
if img_to_process is not None:
    processed_img, display_img = preprocess_image(img_to_process)

    st.subheader("🔍 Prediction Results:")
    st.image(
        display_img,
        caption="Preprocessed MNIST Matrix (Centered & Filtered)",
        width=150,
    )

    prediction = model.predict(processed_img)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.success(f"🎯 **Predicted Digit: {predicted_digit}**")
    st.info(f"**Confidence Level: {confidence:.2f}%**")