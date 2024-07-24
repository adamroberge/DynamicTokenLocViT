import os
import argparse
import torch
import torchvision
from torchvision import datasets, transforms as pth_transforms
import matplotlib.pyplot as plt
from dynamic_vit_viz import vit_register_dynamic_viz

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
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Get one image from the test dataset
    for images, _ in test_loader:
        img = images[0].unsqueeze(0)  # Extract the first image and add batch dimension
        break

    # Build the model
    model = vit_register_dynamic_viz(img_size=224, patch_size=16, num_classes=10, depth=12)
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

    # Keep only the output patch attention and reshape
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  
    # attentions[0, :, 0, 1:] results in shape (12, 196, 195)
    # reshape(12, 196 * 195) results in shape (12, 38160)

    # Reshape attention maps to match feature map dimensions
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # Assuming w_featmap = h_featmap = 14
    # reshape(12, 38160) to (12, 14, 14)

    # Upsample the attention maps to the input image size
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    # attentions.unsqueeze(0) results in shape (1, 12, 14, 14)
    # interpolate with scale_factor=16 results in shape (1, 12, 224, 224)
    # [0] removes batch dimension resulting in shape (12, 224, 224)
    # .cpu().numpy() converts to NumPy array

    # Save attention heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(images, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
