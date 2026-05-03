import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.load_data import get_dataloaders
from src.models.transfer_models import get_efficientnet_b0


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

model = get_efficientnet_b0(num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

epochs = 20
best_val_acc = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_acc)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_efficientnet_b0.pth")
        print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

print("EfficientNet-B0 training completed")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("Best model saved at models/best_efficientnet_b0.pth")