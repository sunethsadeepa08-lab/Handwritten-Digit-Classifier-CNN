import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Configure page layout
st.set_page_config(page_title="Handwritten Digit Classifier AI", page_icon="🔢")

st.title("🔢 AI Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9) to test the trained Deep Learning Model.")

# Load saved CNN model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

model = load_my_model()

# File uploader widget
uploaded_file = st.file_uploader("Upload an Image (PNG/JPG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert('L')
    
    # Check if background is light/white, then invert colors (Black background, White digit)
    img_array_temp = np.array(image)
    if np.mean(img_array_temp) > 127:
        image = ImageOps.invert(image)
        
    st.image(image, caption='Preprocessed Image (MNIST Style)', width=150)
    
    # Preprocess image to match training input (28x28 normalized tensor)
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Perform prediction
    with st.spinner('AI model is predicting...'):
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

    # Display results
    st.success(f"🎯 Predicted Digit: **{predicted_digit}**")
    st.info(f"Confidence Level: {confidence:.2f}%")
    
    st.write("Prediction Probabilities across classes (0-9):")
    st.bar_chart(prediction[0])