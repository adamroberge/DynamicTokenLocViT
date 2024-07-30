# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from trainable_cls_reg import TrainableVitRegisterDynamicViz  # Import the correct class name
from custom_summary import custom_summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainable_train_model(model, train_loader, loss_fn, optimizer, num_epochs, device):
    best_accuracy = 0.0
    best_model_path = 'best_trainable_model.pth'
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Create a progress bar for the current epoch
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

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

            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            total += targets.size(0)  # Accumulate total number of targets

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

            # Update the progress bar
            pbar.set_postfix({'loss': running_loss / total, 'accuracy': 100 * correct / total})

        accuracy = 100 * correct / total
        training_accuracies.append(accuracy)
        epoch_loss = running_loss / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # # Validation phase
        # model.eval()
        # val_correct = 0
        # val_total = 0
        # with torch.no_grad():
        #     for val_inputs, val_targets in val_loader:
        #         val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
        #         val_outputs = model(val_inputs)
        #         _, val_predicted = torch.max(val_outputs.data, 1)
        #         val_total += val_targets.size(0)
        #         val_correct += (val_predicted == val_targets).sum().item()

        # val_accuracy = 100 * val_correct / val_total
        # validation_accuracies.append(val_accuracy)
        # print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

    print("Training complete")

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), training_accuracies, marker='o', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), validation_accuracies, marker='x', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig('trainable_accuracy_over_epochs_224.pdf', format='pdf')
    plt.show()
