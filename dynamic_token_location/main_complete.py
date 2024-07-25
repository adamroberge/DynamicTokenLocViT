# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import random
from dynamic_vit_viz import vit_register_dynamic_viz
from dynamic_vit import vit_register_dynamic
from train_complete import train_model
from test_complete import test_model

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


# Define data transforms with augmentation for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define data transforms for testing
test_transform = transforms.Compose([
    transforms.Resize(32),  # Ensure images are resized to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load CIFAR-10 datasets
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

# Initialize the model
model = vit_register_dynamic(img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                 drop_path_rate=0., init_scale=1e-4,
                                 mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=None)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Train the model
train_model(model, train_loader, loss_fn, optimizer, num_epochs=200, device=device)

# Test the model
test_model(model, test_loader, device)