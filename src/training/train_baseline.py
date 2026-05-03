import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.load_data import get_dataloaders
from src.models.baseline_cnn import BaselineCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

model = BaselineCNN(num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0
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

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline_cnn.pth")

print("Baseline CNN training completed ")
print("Model saved at models/baseline_cnn.pth")