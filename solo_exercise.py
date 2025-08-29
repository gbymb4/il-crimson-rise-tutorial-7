# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 20:07:22 2025

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
# Baseline CNN from Session 6 (already implemented for comparison)
# -------------------------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------------------------------------------
# YOUR TASK: Residual CNN with BatchNorm
# -------------------------------------------------------------------
"""
TODO 1: Implement a ResidualBlock class
    - Two convolution layers with the same number of channels
    - BatchNorm after each convolution
    - Add skip connection (identity mapping)
    - Apply ReLU after addition
"""

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # TODO: define conv1, bn1, conv2, bn2
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement forward pass with skip connection
        raise NotImplementedError

"""
TODO 2: Implement ResidualCNN
    - Start with a conv + batchnorm + relu
    - Add at least one residual block
    - Add pooling to reduce dimensions
    - Add another residual block
    - Flatten and output with a linear layer
"""
class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(self, x):
        # TODO: forward pass
        raise NotImplementedError

# -------------------------------------------------------------------
# Training and Evaluation Loop
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

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_acc = 100 * correct / total

        # Test accuracy
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
# Baseline Run
# -------------------------------------------------------------------
print("\nTraining baseline CNN...")
baseline_model = BaselineCNN().to(device)
baseline_acc = train_model(baseline_model, train_loader, test_loader)
baseline_params = sum(p.numel() for p in baseline_model.parameters())
print(f"Baseline CNN: {baseline_acc:.1f}% accuracy, {baseline_params:,} parameters")

# -------------------------------------------------------------------
# TODO 3: Train your ResidualCNN and compare
# -------------------------------------------------------------------
"""
Steps:
1. Uncomment and complete ResidualBlock + ResidualCNN
2. Create an instance of ResidualCNN
3. Train it with train_model()
4. Compare accuracy and parameter count with baseline
"""

# Example usage after implementation:
# residual_model = ResidualCNN().to(device)
# residual_acc = train_model(residual_model, train_loader, test_loader)
# residual_params = sum(p.numel() for p in residual_model.parameters())
# print(f"Residual CNN: {residual_acc:.1f}% accuracy, {residual_params:,} parameters")

# -------------------------------------------------------------------
# Reflection Questions (answer after implementation)
# -------------------------------------------------------------------
"""
1. Did your Residual CNN outperform the baseline CNN? By how much?
2. How did BatchNorm affect training stability and accuracy?
3. Why do skip connections help deeper networks train better?
4. Compare parameter counts: which model is more efficient?
"""
