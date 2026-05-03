import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import matplotlib.pyplot as plt

from src.data.load_data import get_dataloaders
from src.models.transfer_models import get_resnet50


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_, _, test_loader = get_dataloaders(batch_size=32)

model = get_resnet50(num_classes=7)
model.load_state_dict(torch.load("models/best_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

images = images.cpu()

plt.figure(figsize=(12, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Pred: {class_names[preds[i]]}")
    plt.axis("off")

plt.savefig("outputs/sample_predictions.png")
plt.show()