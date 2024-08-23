import os
import torch
import argparse
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dynamic_vit_viz import vit_register_dynamic_viz


def get_args_parser():
    parser = argparse.ArgumentParser('Visualization Script', add_help=False)
    parser.add_argument('--model_path', default='best_model.pth', type=str, help='Path to the trained model.')
    parser.add_argument('--layer_num', default=6, type=int, help='Layer number to visualize attention from.')
    parser.add_argument('--cls_pos', default=3, type=int, help='Position of cls token.')
    parser.add_argument('--reg_pos', default=0, type=int, help='Position of register tokens.')
    parser.add_argument('--data_path', default='/home/adam/data/in1k', type=str, help='Path to the ImageNet1k dataset.')
    parser.add_argument('--output_dir', default='output_dir', type=str, help='Directory to save the attention maps.')
    parser.add_argument('--input-size', default=224, type=int, help='Image size for model input.')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--img_num', default=0, type=int, help="Number of image inside the batch")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for computation')
    
    return parser


def visualize_attention(model, img, label, class_names, layer_num, output_dir):
    model.eval()
    
    # Move the image to device
    device = next(model.parameters()).device
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    # Compute feature map sizes
    w_featmap = img.shape[-2] // model.patch_size  # Width of the feature map
    h_featmap = img.shape[-1] // model.patch_size  # Height of the feature map

    # Get attention maps from the specified layer
    cls_attentions, reg_attentions_list = model.get_attention_map(img, layer_num)

    # Number of attention heads
    nh = cls_attentions.shape[1]

    # Process class attentions
    cls_attentions = cls_attentions[0].reshape(nh, w_featmap, h_featmap)
    cls_attentions = torch.nn.functional.interpolate(cls_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()

    # Process register attentions
    reg_attentions_processed = []
    for reg_attentions in reg_attentions_list:
        reg_attentions = reg_attentions[0].reshape(nh, w_featmap, h_featmap)
        reg_attentions = torch.nn.functional.interpolate(reg_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
        reg_attentions_processed.append(reg_attentions)

    # Function to calculate the grid size based on the number of heads
    def calc_grid_size(n):
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        return rows, cols

    # Save attention heatmaps to a single PDF
    with PdfPages(Path(output_dir) / f"attention_maps_layer_{layer_num}_of_image_{args.img_num}_cls_{args.cls_pos}_reg_{args.reg_pos}.pdf") as pdf:
        rows, cols = calc_grid_size(nh + 1)  # Add 1 for the original image
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

        # Plot the original image
        axes = axes.flatten()  # Flatten the axes for easier indexing
        axes[0].imshow(np.transpose(img.cpu().squeeze().numpy(), (1, 2, 0)))
        axes[0].axis('off')
        axes[0].set_title(f'Original Image: {class_names[label]} at layer {layer_num}')

        # Plot class attentions
        for j in range(nh):
            axes[j + 1].imshow(cls_attentions[j], cmap='viridis')
            axes[j + 1].axis('off')
            axes[j + 1].set_title(f'Class Token Attention Head {j+1}')

        # Remove any unused subplots
        for ax in axes[nh+1:]:
            ax.axis('off')

        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Save register attentions
        for i, reg_attentions in enumerate(reg_attentions_processed):
            rows, cols = calc_grid_size(nh + 1)
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

            # Plot the original image
            axes = axes.flatten()
            axes[0].imshow(np.transpose(img.cpu().squeeze().numpy(), (1, 2, 0)))
            axes[0].axis('off')
            axes[0].set_title(f'Original Image: {class_names[label]} at layer {layer_num}')

            # Plot register attentions
            for j in range(nh):
                axes[j + 1].imshow(reg_attentions[j], cmap='viridis')
                axes[j + 1].axis('off')
                axes[j + 1].set_title(f'Register Token {i+1} Attention Head {j+1}')

            # Remove any unused subplots
            for ax in axes[nh+1:]:
                ax.axis('off')

            # Save the figure to the PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Attention maps saved in {output_dir}.")


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define data transforms for testing
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load ImageNet1K validation dataset
    test_dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize the model
    model = vit_register_dynamic_viz(img_size=args.input_size, patch_size=args.patch_size, in_chans=3, num_classes=args.nb_classes, embed_dim=384, depth=12,
                                     num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                     drop_path_rate=0.05, init_scale=1e-4,
                                     mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=args.cls_pos, reg_pos=args.reg_pos)

    # Load the model weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Move the model to the correct device
    model.to(args.device)

    # Get one image and label from the validation dataset
    for images, labels in test_loader:
        img = images[args.img_num]  # Take the first image from the batch
        label = labels[args.img_num].item()  # Take the label of the first image
        break

    # Visualize attention maps
    visualize_attention(model, img, label, test_dataset.classes, args.layer_num, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attention Map Visualization Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
