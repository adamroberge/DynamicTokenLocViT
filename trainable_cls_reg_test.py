# test.py
import torch
from tqdm import tqdm
import numpy as np
import random
from trainable_cls_reg import TrainableVitRegisterDynamicViz  # Import the correct class name
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

def trainable_test_model(model, test_loader, device):
    model.to(device)  # Ensure the model is on the right device
    # Load the best model for evaluation
    best_model_path = 'best_trainable_model.pth'
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Evaluate the model
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

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy