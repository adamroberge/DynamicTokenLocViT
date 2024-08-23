import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms as pth_transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# Fixing the issue by removing 'module.' prefix from state_dict keys
def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

# Argument parser for command-line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention Maps')
    parser.add_argument("--output_dir", default='/home/adam/dynamic_vit/DynamicTokenLocViT/result', help='Path where to save visualizations.')
    parser.add_argument('--model_path', default='/home/adam/dynamic_vit/DynamicTokenLocViT/result/best_checkpoint.pth', type=str, help='Path to the trained model.')
    parser.add_argument('--layer_num', default=5, type=int, help='Layer number to visualize attention from.')
    parser.add_argument('--num_reg', default=4, type=int, help='Number of register tokens')    
    parser.add_argument('--cls_pos', default=0, type=int, help='Position of cls token')    
    parser.add_argument('--reg_pos', default=0, type=int, help='Position of register tokens')    
    parser.add_argument('--img_num', default=0, type=int, help="Number of image inside the batch")
    parser.add_argument('--image_path', default='', type=str, help='Path to the image for visualization.')
    
    args = parser.parse_args()

    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define data transforms for ImageNet1k
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256),  # Resize the shorter side to 256 pixels
        pth_transforms.CenterCrop(224),  # Crop the central 224x224 patch
        pth_transforms.ToTensor(),  # Convert image to tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize with ImageNet mean and std
    ])

    # Load an image for visualization
    if args.image_path:
        from PIL import Image
        img = Image.open(args.image_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # Transform and add batch dimension
    else:
        # Load a sample from the ImageNet validation set
        dataset_val = datasets.ImageNet(root='/home/adam/data/in1k/', split='val', transform=transform)
        data_loader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=2)
        for images, labels in data_loader_val:
            img = images[args.img_num].unsqueeze(0)  # Take an image and add batch dimension
            label = labels[args.img_num].item()  # Take the according label of the image
            break

    # Build the model with ImageNet1k parameters
    model = vit_register_dynamic_viz(
        img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=192, depth=12,
        num_heads=3, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        init_scale=1e-4, mlp_ratio_clstk=4.0, num_register_tokens=args.num_reg,
        cls_pos=args.cls_pos, reg_pos=args.reg_pos
    )
    
    # Load the model's state_dict from the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_state_dict = checkpoint['model']  # Extract model state dict
    model_state_dict = remove_module_prefix(model_state_dict)  # Remove 'module.' prefix
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    custom_summary(model, (3, 224, 224))

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

    # Reshape class attentions
    cls_attentions = cls_attentions[0].reshape(nh, w_featmap, h_featmap)
    cls_attentions = torch.nn.functional.interpolate(cls_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()

    # Process register attentions
    for i in range(len(reg_attentions_list)):
        reg_attentions_list[i] = reg_attentions_list[i][0].reshape(nh, w_featmap, h_featmap)
        reg_attentions_list[i] = torch.nn.functional.interpolate(reg_attentions_list[i].unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()

    # Save attention heatmaps in a single PDF
    with PdfPages(Path(args.output_dir) / f"attention_maps_layer_{args.layer_num}_of_image_{args.img_num}_cls_{args.cls_pos}_reg_{args.reg_pos}.pdf") as pdf:
        # Save class attentions
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))

        # Plot the original image
        ax = axes[0, 0]
        ax.imshow(img[0].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Original Image at layer {args.layer_num} \n Class token added at layer {model.cls_pos}')

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
            ax.imshow(img[0].permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
            ax.set_title(f'Original Image at layer {args.layer_num} \n Register tokens added at layer {model.reg_pos}')

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

    print(f"All attention heads saved in {os.path.join(args.output_dir, 'attention_maps_layer_{args.layer_num}_of_image_{args.img_num}_cls_{args.cls_pos}_reg_{args.reg_pos}.pdf')}.")
