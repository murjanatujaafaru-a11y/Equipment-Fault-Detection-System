import streamlit as st
from PIL import Image
import numpy as np

# These imports pull the logic from your folders
from model.model import leak_model
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# 1. UI SETUP
st.set_page_config(page_title="Equipment Fault Detection", page_icon="⚙️")
st.title("⚙️ Equipment Fault Detection System")

# 2. LOAD MODEL (Cached for speed)
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()
class_names = ["scratches", "crazing", "patches", "inclusion", "pitted_surface", "rolled-in_scale"]

# 3. FILE UPLOAD
uploaded_file = st.file_uploader("Upload sensor data image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        # --- THE LOGIC FLOW ---
        # A. Preprocess the image first
        processed_img = preprocess(image)
        
        # B. Generate the prediction variable HERE
        prediction = predict(model, processed_img)
        
        # C. Now that 'prediction' is defined, calculate confidence
        confidence = np.max(prediction) * 100
        label = class_names[np.argmax(prediction)]

        # 4. SHOW RESULTS
        # ... (After your prediction and confidence calculations)

        # 4. SHOW RESULTS (To match WhatsApp Image 2026-05-10 at 21.59.25.jpeg)
        
        # Display the green box for the prediction
        st.success(f"Prediction: **{label}**")
        
        # Display the confidence as standard white/gray text below the box
        st.write(f"**Confidence: {confidence:.2f}%**")