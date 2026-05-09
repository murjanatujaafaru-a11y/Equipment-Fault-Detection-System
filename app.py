import streamlit as st
from PIL import Image
import os

import sys
import pathlib


# 1. Get the absolute path of the directory containing app.py
root_path = os.path.dirname(os.path.abspath(__file__))

# 2. Add that path to the front of sys.path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 3. Debug print (this will show up in your Streamlit logs)
print(f"DEBUG: Root path added to sys.path: {root_path}")
print(f"DEBUG: Files found in root: {os.listdir(root_path)}")

# 4. NOW do the imports
from utils.preprocessing import preprocess
from model import leak_model.py


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