import torch
import torch.nn.functional as F
import gdown
import json
import os
import numpy as np
from model.model import leak_model

# 1. Google Drive download logic
def download_model():
    if not os.path.exists("model/leak_model.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1FB3IirDTbhp4eR_MQSZYlO9CZHAANEhc",  
            output="model/leak_model.pth",
            quiet=False
        )
        print("Model downloaded successfully.")

# 2. Path logic for class mapping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(BASE_DIR, "model", "class_to_idx.json")

# Load class mapping globally
with open(json_path, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# 3. Model Loading
def load_model():
    download_model()
    num_classes = 6
    model = leak_model(num_classes=num_classes)
    
    # Load weights with strict=False to handle naming mismatches
    state_dict = torch.load("model/leak_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

# 4. THE BIG FIX: Correcting the function signature and logic
def predict(model, processed_img):
    """
    Takes the model and the preprocessed image tensor, 
    returns the raw probability array.
    """
    model.eval()
    with torch.no_grad():
        # Run the model directly on the tensor
        output = model(processed_img)
        
        # Convert logits to probabilities (0.0 to 1.0)
        probabilities = F.softmax(output, dim=1)
        
        # Return as a flat numpy array for the app.py math
        return probabilities.cpu().numpy().flatten()