# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 20:07:06 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# -------------------------------------------------------------------
# Baseline: Deep CNN without residuals or batchnorm
# -------------------------------------------------------------------
class DeepCNN(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, 32, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = 32
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32*14*14, 10)  # after one pooling

    def forward(self, x):
        x = self.conv(x)         # repeated convs
        x = self.pool(x)         # downsample once
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------------------------------------------
# Residual Block with BatchNorm
# -------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# -------------------------------------------------------------------
# Residual CNN with BatchNorm
# -------------------------------------------------------------------
class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = ResidualBlock(32)
        self.fc = nn.Linear(32*14*14, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------------------------------------------
# Training and Evaluation
# -------------------------------------------------------------------
def train_model(model, train_loader, test_loader, num_epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_acc = 100 * correct / total

        # test accuracy
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        test_acc = 100 * correct_test / total_test
        print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%")

    return test_acc

# -------------------------------------------------------------------
# Run comparison
# -------------------------------------------------------------------
print("\nTraining Deep CNN (no residuals/batchnorm)...")
deep_model = DeepCNN(depth=6)
deep_params = sum(p.numel() for p in deep_model.parameters())
deep_acc = train_model(deep_model, train_loader, test_loader)

print("\nTraining Residual CNN (with batchnorm)...")
res_model = ResidualCNN()
res_params = sum(p.numel() for p in res_model.parameters())
res_acc = train_model(res_model, train_loader, test_loader)

# -------------------------------------------------------------------
# Final Comparison
# -------------------------------------------------------------------
print("\nFinal Results:")
print(f"Deep CNN: {deep_acc:.1f}% accuracy, {deep_params:,} parameters")
print(f"Residual CNN: {res_acc:.1f}% accuracy, {res_params:,} parameters")
print(f"Improvement: {res_acc - deep_acc:.1f} percentage points")
