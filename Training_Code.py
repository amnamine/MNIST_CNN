# -*- coding: utf-8 -*-
"""MNIIST using CNNs

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/mniist-using-cnns-c030d4e5-e370-411d-b881-09ecf07b4f6c.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20241219/auto/storage/goog4_request%26X-Goog-Date%3D20241219T223840Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6c3e801079e55f08380c5300250daf661313604805b79e71111ae2ca330740e97def210105aed6ce956ea44144afc0bf9acbb42e49d5e6155d52f2e3385d67ad24ba31b83dfe963f3495c5256918019fa92a36a847567d59872ef077e714d929b7cb12b18b57b8074e2e717af11eff87ff477cbb7dd5724ab1e49f7dec086ec46e522b35f79707c9ed5f829b938889959c3fd63bf9d33c73ceaa71ae5e199a5d6f8ccdb55e6de4534577144b30daf2bcaaba76abec19eb01dc5b8c2b3c993632b622f524e99bad8799db55a97076a6b095f0716effc39e998515faa7dad2a413e5d8ed186710b889480ef950689e8e7d0c9aff7607429cb3280fbc67bd830756
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Conv layer 2
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply Conv2 + ReLU + Pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output
        x = torch.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model on test data
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No gradient calculations during evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")