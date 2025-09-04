"""
Session 7 Homework: Deep Networks & Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# TODO: Implement VeryDeepCNN with 8+ conv layers, no batchnorm, single pooling
class VeryDeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

# TODO: Same architecture but add BatchNorm2d after each conv
class BatchNormCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

# TODO: ResidualBlock with two convs, batchnorm, skip connection
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        pass
    
    def forward(self, x):
        # TODO: return f(x) + x pattern
        pass

# TODO: CNN using 2-3 ResidualBlocks
class ResidualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        pass
    
    def forward(self, x):
        pass

# TODO: Training loop that tracks gradient norms of first layer
def train_with_gradients(model, epochs=3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    grad_norms = []
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # TODO: forward, loss, backward
            # TODO: compute and store gradient norm
            # TODO: optimizer step
            
            if batch_idx % 300 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}')
    
    return grad_norms

# TODO: Evaluate accuracy on test set
def test_accuracy(model):
    pass

# TODO: Compare all three models - train and measure accuracy + gradient norms
def compare_models():
    models = {
        'VeryDeep': VeryDeepCNN(),
        'BatchNorm': BatchNormCNN(), 
        'Residual': ResidualCNN()
    }
    
    # TODO: train each model and collect results
    pass

if __name__ == "__main__":
    # TODO: Run comparison and analyze results
    pass

# Reflection: Which worked best and why? How did gradients differ?