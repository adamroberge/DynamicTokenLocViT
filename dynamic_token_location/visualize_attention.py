import os
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as pth_transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from dynamic_vit_viz import vit_register_dynamic_viz
from train_model import train_model


# Set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Argument parser for command-line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument("--output_dir", default='.', help='Path where to save visualizations.')
    parser.add_argument('--layer_num', default=-1, type=int, help='Layer number to visualize attention from.')

    args = parser.parse_args()

    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Define data transforms
    transform = pth_transforms.Compose([
        pth_transforms.Resize(224),  # Resize images to 224x224
        pth_transforms.ToTensor(),   # Convert images to tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize with mean and std
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

    # Get one image from the test dataset
    for images, _ in test_loader:
        img = images[0].unsqueeze(0)  # Extract the first image and add batch dimension
        break

    # Build the model
    model = vit_register_dynamic_viz(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                             num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                             drop_path_rate=0., init_scale=1e-4,
                             mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=None)   
    
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    # model = train_model(model, train_loader, loss_fn, optimizer, num_epochs=150, device=device)
 
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device (CPU or GPU)

    # Preprocess the image
    img = img.to(device)

    # Compute feature map sizes
    w_featmap = img.shape[-2] // 16  # Width of the feature map
    h_featmap = img.shape[-1] // 16  # Height of the feature map

    # Get self-attention from the specified layer
    if args.layer_num >= 0:
        attentions = model.get_selfattention(img, args.layer_num)
    else:
        attentions = model.get_selfattention(img, len(model.blocks) - 1)

    nh = attentions.shape[1]  # Number of heads

    # Print the shape before and after each operation for debugging
    print(f"Original attentions shape: {attentions.shape}")

    # Keep only the output patch attention and reshape
    # We are slicing attentions to keep only patch tokens (not the class token)
    attentions = attentions[0, :, 0, 1:]
    print(f"Shape after slicing: {attentions.shape}")

    # Ensure the total number of elements matches the expected size
    expected_size = nh * w_featmap * h_featmap
    actual_size = attentions.numel()
    print(f"Expected size: {expected_size}, Actual size: {actual_size}")

    if actual_size == expected_size:
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
    else:
        raise ValueError(f"The number of elements in the attention maps ({actual_size}) does not match the expected size ({expected_size}).")

    # Print shape after reshaping
    print(f"Shape after reshaping: {attentions.shape}")

    # Upsample the attention maps to the input image size
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
    print(f"Shape after upsampling: {attentions.shape}")

    # Save attention heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(images, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
