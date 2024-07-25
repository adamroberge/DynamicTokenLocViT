# test.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from dynamic_vit_viz import vit_register_dynamic_viz

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(32),  # Ensure images are resized to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load CIFAR-10 dataset
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

def test_model(model, test_loader, device):
    model.to(device)  # Ensure the model is on the right device
    # Load the best model for evaluation
    best_model_path = 'best_model.pth'
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Move the model to GPU if available
    model.to(device)

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

if __name__ == '__main__':
    # Initialize the model
    model = vit_register_dynamic_viz(img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                     num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                     drop_path_rate=0., init_scale=1e-4,
                                     mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=None)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Test the model
    test_model(model, test_loader, device)
