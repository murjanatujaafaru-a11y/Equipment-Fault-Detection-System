import torch
import torch.nn.functional as F
# Assuming build_model is defined in model.py or elsewhere
from model import leak_model as build_model 
import gdown
import json
import os

# Google Drive download
def download_model():
    # Everything inside this function is indented 4 spaces
    if not os.path.exists("model/leak_model.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1FB3IirDTbhp4eR_MQSZYlO9CZHAANEhc",  
            output="model/leak_model.pth",
            quiet=False
        )
        print("Model downloaded successfully.")

# Load class mapping
# These should NOT be inside a function if you want them available globally
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(BASE_DIR, "model", "class_to_idx.json")

# Load class mapping
with open(json_path, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    download_model()  # downloads only if file doesn't exist
    model = build_model()
    model.load_state_dict(
        torch.load("model/leak_model.pth", map_location="cpu")
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