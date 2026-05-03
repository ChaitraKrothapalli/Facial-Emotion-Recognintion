import sys
import os
sys.path.append(os.path.abspath("."))

import torch

from src.data.load_data import get_dataloaders
from src.models.transfer_models import get_resnet50


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

_, _, test_loader = get_dataloaders(batch_size=32)

model = get_resnet50(num_classes=7)
model.load_state_dict(torch.load("models/best_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total

print(f"Test Accuracy: {test_acc:.2f}%")