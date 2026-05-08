import torch
import torch.nn.functional as F
import sys
import os

# Ensure the root directory is in the path for this sub-module too
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SurfaceDefectCNN
import gdown
import json
import os
# from huggingface_hub import hf_hub_download

#To Load our model from `Hugging Face`, uncomment the below code and comment out the `Google Drive download` section
# def load_model():
#     try:
#         # Downloads and caches the model automatically
#         model_path = hf_hub_download(
#             repo_id="your-username/wildscan-model",
#             filename="leak_model.pth",
#             cache_dir="model"
            # token=st.secrets["HF_TOKEN"]  # only needed for private repos
#         )

#         model = build_model()
#         model.load_state_dict(torch.load(model_path, map_location="cpu"))
#         model.eval()
#         return model

#     except Exception as e:
#         raise RuntimeError(f"Model loading failed: {e}")

# Google Drive download
def download_model():
    if not os.path.exists("model/leak_model"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1s4Ts_VwyCq7i0dg-RERZIrRCNkf6BJO7",  # Google Drive file ID
            output="model/leak_model",
            quiet=False
        )
        print("Model downloaded successfully.")

#Load class mapping
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    download_model()  # downloads only if file doesn't exist
    model = build_model()
    model.load_state_dict(
        torch.load("model/leak_model", map_location="cpu")
    )
    model.eval()
    return model

# Predict
def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
    return label, confidence.item()