import torch.nn as nn
from torchvision import models

# Level 0: No spaces
class leak_model(nn.Module):
    # Level 1: Exactly 4 spaces
    def __init__(self, num_classes=6):
        # Level 2: Exactly 8 spaces
        super(leak_model, self).__init__()
        # Load ResNet18
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    # Level 1: Exactly 4 spaces (aligned with def __init__)
    def forward(self, x):
        # Level 2: Exactly 8 spaces
        return self.model(x)