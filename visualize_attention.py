import os
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as pth_transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dynamic_vit_viz import vit_register_dynamic_viz

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Global variables 
num_img = 60
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Argument parser for command-line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention Maps')
    parser.add_argument("--output_dir", default='.', help='Path where to save visualizations.')
    parser.add_argument('--layer_num', default=7, type=int, help='Layer number to visualize attention from.')
    parser.add_argument('--model_path', default='best_model.pth', type=str, help='Path to the trained model.')
    parser.add_argument('--cls_pos', default='0', type=int, help='Layer number where cls token is added.')
    parser.add_argument('--reg_pos', default='0', type=int, help='Layer number where reg tokens are added.')
    
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
    train_dataset = CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

    # Get one image from the test dataset
    for images, labels in test_loader:
        img = images[num_img].unsqueeze(0)  # Extract an image and add batch dimension
        label = labels[num_img].item()  # Extract the label of the image
        label_name = cifar10_classes[label]  # Get the class name
        break

    # Build the model
    model = vit_register_dynamic_viz(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                     num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                     drop_path_rate=0., init_scale=1e-4,
                                     mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=args.cls_pos, reg_pos=args.reg_pos)   
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device (CPU or GPU)

    # Preprocess the image
    img = img.to(device)

    # Compute feature map sizes
    w_featmap = img.shape[-2] // model.patch_size  # Width of the feature map
    h_featmap = img.shape[-1] // model.patch_size  # Height of the feature map

    # Get self-attention from the specified layer
    if args.layer_num < 0:
        raise ValueError(f"The layer you are trying to print the attention map from ({args.layer_num}) should be a positive number smaller than {model.depth}")
    elif args.layer_num >= model.reg_pos and args.layer_num < model.depth:
        cls_attentions, reg_attentions_list = model.get_attention_map(img, args.layer_num)
    elif args.layer_num < model.reg_pos:
        cls_attentions, _ = model.get_attention_map(img, args.layer_num)
        reg_attentions_list = []
    else:
        raise ValueError(f"The layer you are trying to print the attention map from ({args.layer_num}) is bigger than the model's depth.")

    nh = cls_attentions.shape[1]  # Number of heads

    # Print the shape before and after each operation for debugging
    print(f"Original class attentions shape: {cls_attentions.shape}")
    print(f"Original register attentions shape: {len(reg_attentions_list)}")

    # Reshape class attentions
    cls_attentions = cls_attentions[0].reshape(nh, w_featmap, h_featmap)
    print(f"Class attentions shape after reshaping: {cls_attentions.shape}")

    # Upsample the class attention maps to the input image size
    cls_attentions = torch.nn.functional.interpolate(cls_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
    print(f"Class attentions shape after upsampling: {cls_attentions.shape}")

    # Process register attentions
    for i in range(len(reg_attentions_list)):
        reg_attentions_list[i] = reg_attentions_list[i][0].reshape(nh, w_featmap, h_featmap)
        reg_attentions_list[i] = torch.nn.functional.interpolate(reg_attentions_list[i].unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
        print(f"Register attentions shape after upsampling for token {i+1}: {reg_attentions_list[i].shape}")

    # Save attention heatmaps in a single PDF
    with PdfPages(os.path.join(args.output_dir, "attention_maps.pdf")) as pdf:
        # Save class attentions
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))

        # Plot the original image
        ax = axes[0, 0]
        ax.imshow(np.transpose(images[num_img].cpu().numpy(), (1, 2, 0)))
        ax.axis('off')
        ax.set_title(f'Original Image: {label_name} at layer {args.layer_num}')

        for ax in axes[0, 1:]:
            ax.axis('off')

        # Plot the class attention maps
        for j in range(nh):
            row = (j + 3) // 3  # Calculate the row index (start from row 1)
            col = (j + 3) % 3   # Calculate the column index
            ax = axes[row, col]
            ax.imshow(cls_attentions[j], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Class Token Attention Head {j+1}')

        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Save register attentions
        for i, reg_attentions in enumerate(reg_attentions_list):
            fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))

            # Plot the original image
            ax = axes[0, 0]
            ax.imshow(np.transpose(images[num_img].cpu().numpy(), (1, 2, 0)))
            ax.axis('off')
            ax.set_title(f'Original Image: {label_name} at layer {args.layer_num} \n Register tokens added at layer {model.reg_pos}')

            for ax in axes[0, 1:]:
                ax.axis('off')

            # Plot the register attention maps
            for j in range(nh):
                row = (j + 3) // 3  # Calculate the row index (start from row 1)
                col = (j + 3) % 3   # Calculate the column index
                ax = axes[row, col]
                ax.imshow(reg_attentions[j], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Register Token {i+1} Attention Head {j+1}')

            # Save the figure to the PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"All attention heads saved in {os.path.join(args.output_dir, 'attention_maps.pdf')}.")
