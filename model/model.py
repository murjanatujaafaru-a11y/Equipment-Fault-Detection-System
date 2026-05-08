def predict(data):
    # Initialize the class
    model = SurfaceDefectCNN() 
    
    # Load your saved weights
    model.load_state_dict(torch.load('model/leak_model.pth', map_location='cpu'))
    model.eval()
    # ... rest of your code