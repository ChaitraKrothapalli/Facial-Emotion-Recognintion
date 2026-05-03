import sys
import os
sys.path.append(os.path.abspath("."))

import torch
from sklearn.metrics import classification_report

from src.data.load_data import get_dataloaders
from src.models.transfer_models import get_resnet50


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_, _, test_loader = get_dataloaders(batch_size=32)

model = get_resnet50(num_classes=7)
model.load_state_dict(torch.load("models/best_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

print(classification_report(all_labels, all_preds, target_names=class_names))