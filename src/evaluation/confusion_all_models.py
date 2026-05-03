import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.load_data import get_dataloaders
from src.models.transfer_models import (
    get_resnet18,
    get_resnet50,
    get_mobilenet_v2,
    get_efficientnet_b0
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_, _, test_loader = get_dataloaders(batch_size=32)

class_names = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]


def evaluate_model(model, model_path, model_name):
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

    plt.title(f"Confusion Matrix - {model_name}")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/confusion_{model_name}.png")
    plt.show()


# ----------- Run for each model -----------

# ResNet18
model = get_resnet18()
evaluate_model(model, "models/best_resnet18.pth", "resnet18")

# ResNet50
model = get_resnet50()
evaluate_model(model, "models/best_resnet50.pth", "resnet50")

# MobileNetV2
model = get_mobilenet_v2()
evaluate_model(model, "models/best_mobilenet_v2.pth", "mobilenet_v2")

# EfficientNet-B0
model = get_efficientnet_b0()
evaluate_model(model, "models/best_efficientnet_b0.pth", "efficientnet_b0")