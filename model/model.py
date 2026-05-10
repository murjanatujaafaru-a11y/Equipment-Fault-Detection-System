import torch.nn as nn
from torchvision import models

class leak_model(nn.Module):
    def __init__(self, num_classes=6):
        super(leak_model, self).__init__()
        # Load ResNet18 - this defaults to 3-channel (RGB) input
        self.model = models.resnet18(weights=None)
        
        # Adjust the final layer for your 6 surface defect classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)