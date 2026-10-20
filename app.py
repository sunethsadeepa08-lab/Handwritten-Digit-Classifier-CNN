import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import center_of_mass
from tensorflow.keras.models import load_model

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Handwritten Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional CSS Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #313244;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🔢 World-Class AI Handwritten Digit Classifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Powered by Deep Convolutional Neural Networks (CNN) & Multi-Stage Computer Vision Pipeline</div>',
    unsafe_allow_html=True,
)


# Load Trained CNN Model with Caching
@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()


# Ultimate Multi-Stage Image Processing Pipeline
def preprocess_image(image):
    # 1. Convert Image to Grayscale Numpy Array
    img_gray = np.array(image.convert("L"))

    # High Quality Resizing to Processing Canvas (300x300)
    img_gray = cv2.resize(img_gray, (300, 300), interpolation=cv2.INTER_CUBIC)

    # 2. Bilateral Noise Reduction (Preserves Edges while Denoising)
    denoised = cv2.bilateralFilter(
        img_gray, d=9, sigmaColor=75, sigmaSpace=75
    )

    # 3. Automatic Foreground/Background Inversion Detection
    if np.mean(denoised) > 127:
        denoised = cv2.bitwise_not(denoised)

    # 4. CLAHE Contrast Limited Adaptive Histogram Equalization (Shadow Removal)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Hybrid Thresholding (Otsu's Binarization + Adaptive Thresholding)
    _, thresh_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh_adapt = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)

    # 6. Morphological Dilation (Stroke Normalization for Thin Pen Strokes)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 7. Contour Detection and Primary Digit Extraction
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            digit = thresh[y : y + h, x : x + w]

            # 8. Aspect-Ratio Preserving Resizing to 20x20
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

            # 9. Center Padding into 28x28 Matrix
            padded = np.zeros((28, 28), dtype=np.uint8)
            pad_x = (28 - new_w) // 2
            pad_y = (28 - new_h) // 2
            padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = digit_resized

            # 10. SciPy Center-of-Mass Alignment (MNIST Mathematical Standard)
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


# Sidebar Configuration & Project Information
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg",
        width=120,
    )
    st.title("📌 System Overview")
    st.info("""
    **Model:** Deep Convolutional Neural Network (CNN)
    **Dataset:** MNIST (60,000 Training Samples)
    **Features:**
    - Bilateral Denoising
    - Adaptive Contrast Enhancement
    - Automatic Mass Centering
    - Real-Time Probability Distribution
    """)
    st.write("---")
    st.write(" Developed for Academic & Professional AI Deployment")

# File Uploader Container
st.write("### 📂 Upload Digit Image")
uploaded_file = st.file_uploader(
    "Choose a PNG, JPG, or JPEG file...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img_input = Image.open(uploaded_file)

    # Perform Image Preprocessing
    processed_img, display_img = preprocess_image(img_input)

    # Model Inference
    prediction = model.predict(processed_img)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.write("---")

    # Display Visual Outputs
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.markdown("#### 📷 Original Image")
        st.image(img_input, use_container_width=True)

    with col2:
        st.markdown("#### 🔬 Processed Matrix (28x28)")
        st.image(display_img, width=200)

    with col3:
        st.markdown("#### 🎯 AI Prediction Result")
        st.success(f"### Predicted Digit: **{predicted_digit}**")
        st.info(f"### Confidence: **{confidence:.2f}%**")

    st.write("---")

    # Probability Bar Chart Visualization across all digits (0-9)
    st.markdown("### 📊 Class Probability Distribution (0–9)")
    prob_df = pd.DataFrame(
        {"Digit": [str(i) for i in range(10)], "Probability (%)": prediction[0] * 100}
    )

    st.bar_chart(prob_df.set_index("Digit"))
else:
    st.info(
        "💡 **Tip:** Upload any handwritten digit image (e.g., photo taken from your mobile phone or a drawing) to see real-time preprocessing and AI predictions."
    )