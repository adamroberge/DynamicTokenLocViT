# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from dynamic_vit_viz import vit_register_dynamic_viz
from custom_summary import custom_summary

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Define data transforms with augmentation
transform = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
# train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

def train_model(model, train_loader, loss_fn, optimizer, num_epochs=200, device=device, best_model_path='best_model.pth'):
    model.to(device)  # Ensure the model is on the right device
    best_accuracy = 0.0
    training_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create a progress bar for the current epoch
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, targets) in pbar:
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

            # Update the progress bar
            pbar.set_postfix({'loss': running_loss / (i + 1), 'accuracy': 100 * correct / total})

        accuracy = 100 * correct / total
        training_accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

    print("Training complete")

    # Plot training accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), training_accuracies, marker='o')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.savefig('training_accuracy_over_epochs_224.pdf', format='pdf')
    plt.show()

# if __name__ == '__main__':
#     # Initialize the model
#     model = vit_register_dynamic_viz(img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=384, depth=12,
#                                      num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
#                                      drop_path_rate=0., init_scale=1e-4,
#                                      mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=None)

#     # Move the model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Print model summary 
#     custom_summary(model, (3, 32, 32))

#     # Define the loss function and optimizer
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=5e-4)

#     # Train the model
#     train_model(model, train_loader, loss_fn, optimizer, num_epochs=200, device=device)
