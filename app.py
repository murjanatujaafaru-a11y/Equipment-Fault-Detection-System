import streamlit as st
from PIL import Image

# This tells Python: "From the file model.py, import the function/class leak_model"
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

    
        # Assuming 'prediction' is the output of your model
confidence = np.max(prediction) * 100
label = class_names[np.argmax(prediction)]

if confidence < 50:
    result_text = "Analysis Inconclusive"
    sub_text = f"The model is only {confidence:.2f}% sure. Please upload a clearer photo."
    color = "orange"
else:
    result_text = f"Prediction: {label}"
    sub_text = f"Confidence: {confidence:.2f}%"
    color = "green"

# Then pass these variables to your UI (e.g., st.success or st.warning)