import torch
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from timm.utils import NativeScaler

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

def train_model(model, train_loader, loss_fn, optimizer, lr_scheduler, num_epochs, device):
    best_accuracy = 0.0
    best_model_path = 'best_model.pth'
    training_accuracies = []
    scaler = torch.cuda.amp.GradScaler()

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
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            total += targets.size(0)  # Accumulate total number of targets

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

            # Update the progress bar
            pbar.set_postfix({'loss': running_loss / total, 'accuracy': 100 * correct / total})

        # Update the learning rate
        lr_scheduler.step(epoch + 1)  # Update learning rate with the current epoch

        accuracy = 100 * correct / total
        training_accuracies.append(accuracy)
        epoch_loss = running_loss / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

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
    plt.savefig('output_dir/training_accuracy_over_epochs_224.pdf', format='pdf')
    plt.show()
