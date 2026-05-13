import streamlit as st
from PIL import Image
import numpy as np  # Ensure numpy is imported for np.max and np.argmax

# Importing from your project structure
from model.model import leak_model
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# UI SETUP
st.set_page_config(page_title="Equipment Fault Detection system", page_icon="⚙️", layout="centered")

st.title("⚙️ Equipment Fault Detection system")
st.write("Upload a sensor data image to classify it into 6 categories.")

# LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# CLASS LABELS (Ensure these match your training)
class_names = ["scratches", "crazing", "patches", "inclusion", "pitted_surface", "rolled-in_scale"]

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        # 2. THE FIX: Preprocess and then Predict to create the 'prediction' variable
        # We use the 'preprocess' and 'predict' functions from your folders
        processed_image = preprocess(image)
        prediction = predict(model, processed_image)

        # 3. Calculate confidence and label NOW that prediction exists
        confidence = np.max(prediction) * 100
        label = class_names[np.argmax(prediction)]

        # 4. Display Results
        if confidence < 50:
            st.warning(f"**Analysis Inconclusive**")
            st.write(f"The model is only {confidence:.2f}% sure. Please upload a clearer photo.")
        else:
            st.success(f"**Prediction: {label}**")
            st.write(f"Confidence: {confidence:.2f}%")