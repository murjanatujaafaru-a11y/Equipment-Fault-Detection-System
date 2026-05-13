import streamlit as st
from PIL import Image
import numpy as np 

# Importing from your project structure
from model.model import leak_model
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# 1. UI SETUP
st.set_page_config(page_title="Equipment Fault Detection system", page_icon="⚙️", layout="centered")
st.title("⚙️ Equipment Fault Detection system")
st.write("Upload a sensor data image to classify it into 6 categories.")

# 2. LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# 3. CLASS LABELS
class_names = ["scratches", "crazing", "patches", "inclusion", "pitted_surface", "rolled-in_scale"]

# 4. FILE UPLOAD & PROCESSING
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        # --- THE FIX STARTS HERE ---
        # First: Preprocess the image
        processed_image = preprocess(image)
        
        # Second: Generate the 'prediction' variable
        prediction = predict(model, processed_image)

        # Third: Now that 'prediction' is defined, calculate confidence and label
        confidence = np.max(prediction) * 100
        label = class_names[np.argmax(prediction)]
        # --- THE FIX ENDS HERE ---

        # 5. DISPLAY RESULTS
        if confidence < 50:
            st.warning(f"**Analysis Inconclusive**")
            st.write(f"The model is only {confidence:.2f}% sure. Please upload a clearer photo.")
        else:
            st.success(f"**Prediction: {label}**")
            st.write(f"Confidence: {confidence:.2f}%")