import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress tracking
from vit_dynamic import vit_register_dynamic
from dynamic_losses import ClassificationLoss


# Define data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=2)

# Initialize the model
model = vit_register_dynamic(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=10,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    num_register_tokens=4,
    cls_pos=5,
    reg_pos=6
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
loss_fn = ClassificationLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Wrap the train_loader with tqdm for progress tracking
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

print("Training complete")


# Evaluate the model
model.eval()
correct = 0
total = 0

# Wrap the test_loader with tqdm for progress tracking
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Accuracy on the test set: {100 * correct / total:.2f}%")


# Define data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=2)

# Initialize the model
model = vit_register_dynamic(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=10,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    num_register_tokens=4,
    cls_pos=5,
    reg_pos=6
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
loss_fn = ClassificationLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Wrap the train_loader with tqdm for progress tracking
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Training complete")

# Evaluate the model
model.eval()
correct = 0
total = 0
# Wrap the test_loader with tqdm for progress tracking
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Accuracy on the test set: {100 * correct / total:.2f}%")
