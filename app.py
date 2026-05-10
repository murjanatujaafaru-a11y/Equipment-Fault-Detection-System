import streamlit as st
from PIL import Image

# This tells Python: "From the file model.py, import the function/class leak_model"
from model.model import leak_model
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# UI SETUP
st.set_page_config(page_title="Equipment_Fault_Detection", layout="centered")

st.title("⛑️ Equipment Fault Detection")
st.write("Upload a fault detection image to classify it into 6 categories.")

# LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# CLASS LABELS(As per the model's training)
# idx_to_class = {
#     0: "scratches",
#     1: "crazing",
#     2: "patches",
#     3: "inclusion",
#     4: "pitted_surface",
#     5: "rolled-in_scale",
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