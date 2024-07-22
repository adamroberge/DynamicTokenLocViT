import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# Ensure vit_dynamic contains your deit_small_patch16_LS definition
from original_vit_deit import deit_small_patch16_LS, deit_small_patch16
from dynamic_vit import vit_models, vit_register_dynamic
from custom_summary import custom_summary

# Define data transforms without augmentation
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
# model = deit_small_patch16_LS(img_size=224, num_classes=10)
# model = deit_small_patch16(img_size=224, num_classes=10)
# model = vit_models(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
#                  num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., init_scale=1e-4,
#                  mlp_ratio_clstk=4.0)
model = vit_register_dynamic(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                             num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                             drop_path_rate=0., init_scale=1e-4,
                             mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=0)


custom_summary(model, (3, 224, 224))


# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 100
best_accuracy = 0.0
best_model_path = 'best_model.pth'
training_accuracies = []

# Get a single batch of training data for overfitting test
one_batch = next(iter(train_loader))
inputs, targets = one_batch
inputs, targets = inputs.to(device), targets.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    training_accuracies.append(accuracy)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

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
plt.show()
plt.savefig('training_accuracy_over_epochs.pdf', format='pdf')

# Load the best model for evaluation
model.load_state_dict(torch.load(best_model_path))
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