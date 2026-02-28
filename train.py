import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import config

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Using device:", config.DEVICE)

# Image transforms
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Load pretrained model
model = models.resnet18(weights="DEFAULT")

# Modify output layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

best_accuracy = 0

# Training loop
for epoch in range(config.EPOCHS):

    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

    # TRAIN
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(model.state_dict(), config.MODEL_PATH)

        print("Model saved (best accuracy)")

print("\nTraining completed.")
print("Best Validation Accuracy:", best_accuracy)