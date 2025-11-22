import torch
import torch.nn as nn
from torchvision import models

# Path to your checkpoint
ckpt_path = r"C:\ml df project\DeepfakeBench\logs\training\sbi_2025-09-04-23-11-01\test\avg\ckpt_best.pth"

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint keys:", ckpt.keys())

# Define a dummy model using EfficientNet as backbone
# Adjust final layer to match your checkpoint's output
# Here, assuming the last layer is backbone.last_layer
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        # Replace classifier with the shape of your checkpoint's last layer
        self.backbone.classifier = nn.Linear(1280, ckpt['backbone.last_layer.weight'].shape[0])

    def forward(self, x):
        return self.backbone(x)

# Initialize model
model = CustomModel()

# Load state dict (some keys may need strict=False)
state_dict = {k.replace("backbone.", ""): v for k, v in ckpt.items() if "backbone." in k}
model.backbone.load_state_dict(state_dict, strict=False)

# Test dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size if different
output = model(dummy_input)
print("Output shape:", output.shape)
