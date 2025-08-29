Here’s a structured follow-up session plan that naturally builds on your CNN session, focusing on the **vanishing gradient problem**, **residual blocks**, and **batch normalization**. I’ve included both (1) an instructor-led demo/example script and (2) a student solo exercise.

---

# Machine Learning Session 7: Deeper CNNs, Residual Connections & Batch Normalization

## Tackling Vanishing Gradients in Deep Networks

### Session Overview

**Duration**: 1 hour
**Prerequisites**: Completed Session 6 (CNNs)
**Goal**: Understand vanishing gradients, explore batch normalization and residual connections, and see how they enable deep networks
**Focus**: Vanishing gradient problem, ResNet blocks, batch norm in PyTorch

---

### Session Timeline

| Time        | Activity                                            |
| ----------- | --------------------------------------------------- |
| 0:00 - 0:05 | 1. Touching Base & Session Overview                 |
| 0:05 - 0:25 | 2. Demo: Vanishing Gradients, BatchNorm & ResBlocks |
| 0:25 - 0:50 | 3. Solo Exercise: Implementing Residual CNN         |
| 0:50 - 1:00 | 4. Wrap-up & Key Insights                           |

---

## 1. Touching Base & Session Overview (5 minutes)

* Review Session 6: CNNs, parameter efficiency, translation invariance
* Ask: *“What happens if we stack more layers? Why don’t we just make CNNs 100 layers deep?”*
* Introduce **vanishing gradients**:

  * In deep nets, early layers get very small gradients
  * Network can’t learn useful features
  * Worse with sigmoids/tanh, still happens with ReLU for very deep nets
* Preview solutions:

  * **Batch Normalization** (stabilizes activations, smoother training)
  * **Residual Connections** (skip-connections let gradients flow more easily)

---

## 2. Demo: Vanishing Gradients, BatchNorm & Residual Blocks (20 minutes)

### Live Demo Script

```python
"""
Session 7 Example: Residual CNN vs Deep CNN on MNIST

This demo shows:
1. Vanishing gradient issues in deeper plain CNNs
2. How BatchNorm + Residual connections stabilize training
3. Performance comparison between a plain CNN and Residual CNN
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
```

---

## 3. Solo Exercise: Implementing a Residual CNN (25 minutes)

### Task:

Modify the MNIST CNN from Session 6 to include **batch normalization** and a **residual block**. Compare performance with the baseline CNN.

```python
"""
Session 7 Solo Exercise: Residual CNN with Batch Normalization

Goal: Implement a CNN for MNIST that includes residual connections and batch normalization.
You will compare its performance to a baseline CNN from Session 6.
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

```

### Reflection Questions

1. Did your Residual CNN train faster/more stably than the plain CNN?
2. How does batch norm change the training dynamics (learning rate stability, convergence)?
3. Why do skip connections help very deep networks?
4. What is the relationship between residuals and identity mapping?

---

## 4. Wrap-up & Key Insights (5 minutes)

* **Vanishing Gradients**: Deep nets fail to learn if gradients disappear
* **Batch Normalization**: Normalizes activations, stabilizes training
* **Residual Connections**: Shortcut paths let gradients flow easily
* **Impact**: ResNets made 100+ layer networks trainable and practical

**Looking Ahead**:

* Next session: Explore **transfer learning with pretrained CNNs** (e.g., ResNet on ImageNet)

---

👉 This way, Session 7 is a natural progression: from *basic CNNs* (Session 6) → *deeper networks, why they fail, and the architectural tricks that fixed them*.

Would you like me to also **integrate a visualization** of vanishing gradients (like plotting gradient norms layer by layer) into the demo so learners can *see* the problem directly?
