import streamlit as st
from PIL import Image
import os

import sys
import pathlib

# This gets the directory where app.py is located
current_dir = pathlib.Path(__file__).parent.resolve()

# Add the directory to sys.path if it's not already there
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    
# Now try the imports
from services.inference import predict
from utils.preprocessing import preprocess

# Adds the current directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# UI SETUP
st.set_page_config(page_title="Pipeline Leak", layout="centered")

st.title("⛑️ Pipeline Leak")
st.write("Upload a pipe leak image to classify it into 6 categories.")

@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")

model = get_model()

# CLASS LABELS(As per the model's training)
# idx_to_class = {
#     0: "crazing",
#     1: "inclusion",
#     2: "patches",
#     3: "pitted_surface",
#     4: "rolled-in_scale",
#     5: "scratches"
# }

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning("Model not loaded.")
    else:
        tensor = preprocess(image)
        result, confidence = predict(model, tensor)
        # result, confidence = predict(model, tensor, idx_to_class)

        st.success(f"Prediction: **{result}**")
        st.write(f"Confidence: {confidence:.2%}")