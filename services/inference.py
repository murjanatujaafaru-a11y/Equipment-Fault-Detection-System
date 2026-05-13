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
# We use absolute paths to ensure Streamlit Cloud finds the files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(BASE_DIR, "model", "class_to_idx.json")

# Load class mapping globally
with open(json_path, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# 3. Model Loading with STRICT CHECK
def load_model():
    # Ensure the model exists before trying to load it
    download_model() 

    # 1. Create the architecture
    model = leak_model(num_classes=6)
    
    # 2. Get the correct path to your weights
    base_path = os.path.dirname(__file__)
    weights_path = os.path.join(base_path, '../model/leak_model.pth')
    
    # 3. Load the weights
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        # Load to CPU
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # THE STRICT FIX: strict=True ensures your .pth file matches your model architecture perfectly
        model.load_state_dict(state_dict, strict=True) 
        
        model.eval() # CRITICAL: Sets the model to evaluation mode
        print("SUCCESS: Trained weights loaded successfully.")
        return model
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load trained weights: {e}")
        # We return None instead of a random model so the app knows it's broken
        return None 


# 4. Prediction Logic
def predict(model, processed_img):
    """
    Takes the model and the preprocessed image tensor, 
    returns the raw probability array.
    """
    if model is None:
        return np.zeros(6) # Return zeros if model isn't loaded

    model.eval()
    with torch.no_grad():
        # Run the model
        output = model(processed_img)
        
        # Convert logits to probabilities (0.0 to 1.0)
        probabilities = F.softmax(output, dim=1)
        
        # Return as a flat numpy array
        return probabilities.cpu().numpy().flatten()