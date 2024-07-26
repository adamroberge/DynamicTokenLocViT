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
num_img = 16

# Argument parser for command-line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument("--output_dir", default='.', help='Path where to save visualizations.')
    parser.add_argument('--layer_num', default=-1, type=int, help='Layer number to visualize attention from.')
    parser.add_argument('--model_path', default='best_model.pth', type=str, help='Path to the trained model.')
    
    args = parser.parse_args()

    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # # Define data transforms
    transform = pth_transforms.Compose([
        pth_transforms.Resize(224),  # Resize images to 224x224
        pth_transforms.ToTensor(),   # Convert images to tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize with mean and std
    ])

    # Load CIFAR-10 dataset
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    # train_dataset = CIFAR10(root='./data/CIFAR100', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

    # Get one image from the test dataset
    for images, _ in test_loader:
        img = images[num_img].unsqueeze(0)  # Extract an image and add batch dimension
        break

    # Build the model
    model = vit_register_dynamic_viz(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                     num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                     drop_path_rate=0., init_scale=1e-4,
                                     mlp_ratio_clstk=4.0, num_register_tokens=0, cls_pos=0, reg_pos=0)   
    
    # model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
     
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
        reg_attentions = model.get_register_token_attention(img, args.layer_num)
    else:
        attentions = model.get_selfattention(img, len(model.blocks) - 1)
        reg_attentions = model.get_register_token_attention(img, len(model.blocks) - 1)

    nh = attentions.shape[1]  # Number of heads

    # Print the shape before and after each operation for debugging
    print(f"Original attentions shape: {attentions.shape}")
    # attentions.shape: [1, nh, N, N] = [1, 12, 197, 197]

    # Keep only the output patch attention and reshape
    # We are slicing attentions to keep only patch tokens (not the class token)
    attentions = attentions[0, :, 0, 1:]
    print(f"Shape after slicing: {attentions.shape}")
    # attentions.shape: [nh, N-1] = [12, 196]

    # Ensure the total number of elements matches the expected size
    expected_size = nh * w_featmap * h_featmap
    actual_size = attentions.numel()
    print(f"Expected size: {expected_size}, Actual size: {actual_size}")
    # expected_size: nh * w_featmap * h_featmap = 12 * 14 * 14 = 2352
    # actual_size: numel of attentions = nh * (N-1) = 12 * 196 = 2352

    if actual_size == expected_size:
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        # attentions.shape: [nh, w_featmap, h_featmap] = [12, 14, 14]
    else:
        raise ValueError(f"The number of elements in the attention maps ({actual_size}) does not match the expected size ({expected_size}).")

    # Print shape after reshaping
    print(f"Shape after reshaping: {attentions.shape}")
    # attentions.shape: [nh, w_featmap, h_featmap] = [12, 14, 14]

    # Upsample the attention maps to the input image size
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
    print(f"Shape after upsampling: {attentions.shape}")
    # attentions.shape: [nh, img_size, img_size] = [12, 224, 224]

    # Ensure that reg_attentions has correct dimensions
    num_register_tokens = 4
    reg_attentions = reg_attentions[0, :, -num_register_tokens:, 1:]  # Slice the register tokens from the end
    print(f"NUM REGISTER TOKENS: {num_register_tokens}")
    reg_attentions = reg_attentions.reshape(num_register_tokens, nh, w_featmap, h_featmap)
    reg_attentions = torch.nn.functional.interpolate(reg_attentions, scale_factor=16, mode="nearest").cpu().detach().numpy()

    # Save attention heatmaps in a single PDF
    with PdfPages(os.path.join(args.output_dir, "attention_maps.pdf")) as pdf:
        # Create a grid for the plots
        fig, axes = plt.subplots(nrows=5 + nh * (num_register_tokens + 1), ncols=3, figsize=(15, 5 * (5 + nh * (num_register_tokens + 1))))

        # Plot the original image
        ax = axes[0, 0]
        ax.imshow(np.transpose(images[num_img].cpu().numpy(), (1, 2, 0)))
        ax.axis('off')
        ax.set_title('Original Image')

        for ax in axes[0, 1:]:
            ax.axis('off')

        # Plot the class token attention maps
        for j in range(nh):
            row = (j + 3) // 3
            col = (j + 3) % 3
            ax = axes[row, col]
            ax.imshow(attentions[j], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Class Token Attention Head {j+1}')

        # Plot the register token attention maps
        for k in range(num_register_tokens):
            for j in range(nh):
                row = 5 + k * nh + j
                col = (5 + k * nh + j) % 3
                ax = axes[row, col]
                ax.imshow(reg_attentions[k, j], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Register Token {k+1} Attention Head {j+1}')

        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"All attention heads saved in {os.path.join(args.output_dir, 'attention_maps.pdf')}.")

# import os
# import argparse
# import torch
# import torch.nn as nn 
# import torch.optim as optim
# import torchvision
# from torchvision.datasets import CIFAR10, CIFAR100
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms as pth_transforms
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from dynamic_vit_viz import vit_register_dynamic_viz


# # Set random seed for reproducibility
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(seed)
# random.seed(seed)

# # Global variables 
# num_img = 16

# # Argument parser for command-line options
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Visualize Self-Attention maps')
#     parser.add_argument("--output_dir", default='.', help='Path where to save visualizations.')
#     parser.add_argument('--layer_num', default=-1, type=int, help='Layer number to visualize attention from.')
#     parser.add_argument('--model_path', default='best_model.pth', type=str, help='Path to the trained model.')
    
#     args = parser.parse_args()

#     # Set device to GPU if available, else CPU
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
#     # Define data transforms
#     transform = pth_transforms.Compose([
#         pth_transforms.Resize(224),  # Resize images to 224x224
#         pth_transforms.ToTensor(),   # Convert images to tensor
#         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize with mean and std
#     ])

#     # Load CIFAR-10 dataset
#     test_dataset = CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)

#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(seed))

#     # Get one image from the test dataset
#     for images, _ in test_loader:
#         img = images[num_img].unsqueeze(0)  # Extract an image and add batch dimension
#         break

#     # Build the model
#     model = vit_register_dynamic_viz(img_size=224,  patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
#                                      num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
#                                      drop_path_rate=0., init_scale=1e-4,
#                                      mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=0, reg_pos=0)   
    
#     model.load_state_dict(torch.load(args.model_path, map_location=device))

#     # Define the loss function and optimizer
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=5e-4)
     
#     model.eval()  # Set the model to evaluation mode
#     model.to(device)  # Move the model to the specified device (CPU or GPU)

#     # Preprocess the image
#     img = img.to(device)

#     # Compute feature map sizes
#     w_featmap = img.shape[-2] // 16  # Width of the feature map
#     h_featmap = img.shape[-1] // 16  # Height of the feature map

#     # Get self-attention from the specified layer
#     if args.layer_num >= 0:
#         cls_attentions, reg_attentions = model.get_attention_map(img, args.layer_num)
#     else:
#         cls_attentions, reg_attentions = model.get_attention_map(img, len(model.blocks) - 1)

#     nh = cls_attentions.shape[1]  # Number of heads

#     # Print the shape before and after each operation for debugging
#     print(f"Original class attentions shape: {cls_attentions.shape}")
#     print(f"Original register attentions shape: {reg_attentions.shape}")

#     # Keep only the output patch attention and reshape
#     cls_attentions = cls_attentions[0, :, 1:]  # Removed extra dimension
#     print(f"Shape after slicing class attentions: {cls_attentions.shape}")

#     # Ensure that cls_attentions has correct dimensions
#     cls_expected_size = nh * w_featmap * h_featmap
#     cls_actual_size = cls_attentions.numel()
#     print(f"Class Expected size: {cls_expected_size}, Class Actual size: {cls_actual_size}")
#     if cls_actual_size == cls_expected_size:
#         cls_attentions = cls_attentions.reshape(nh, w_featmap, h_featmap)
#     else:
#         raise ValueError(f"The number of elements in the class attention maps ({cls_actual_size}) does not match the expected size ({cls_expected_size}).")

#     # Ensure that reg_attentions has correct dimensions
#     reg_expected_size = model.num_register_tokens * nh * w_featmap * h_featmap
#     reg_actual_size = reg_attentions.numel()
#     print(f"Register Expected size: {reg_expected_size}, Register Actual size: {reg_actual_size}")
#     if reg_actual_size == reg_expected_size:
#         reg_attentions = reg_attentions.reshape(model.num_register_tokens, nh, w_featmap, h_featmap)
#     else:
#         raise ValueError(f"The number of elements in the register attention maps ({reg_actual_size}) does not match the expected size ({reg_expected_size}).")

#     print(f"Shape after reshaping class attentions: {cls_attentions.shape}")

#     cls_attentions = torch.nn.functional.interpolate(cls_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
#     print(f"Shape after upsampling class attentions: {cls_attentions.shape}")

#     # Ensure that reg_attentions has correct dimensions
#     num_register_tokens = model.num_register_tokens
#     reg_attentions = reg_attentions[0, :, :, 1:model.num_patches+1]  # Slice the register tokens from the end
#     print(f"NUM REGISTER TOKENS: {num_register_tokens}")
#     reg_attentions = reg_attentions.reshape(num_register_tokens * nh, w_featmap, h_featmap)
#     reg_attentions = torch.nn.functional.interpolate(reg_attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()

#     # Save attention heatmaps in a single PDF
#     with PdfPages(os.path.join(args.output_dir, "attention_maps.pdf")) as pdf:
#         # Create a grid for the plots
#         fig, axes = plt.subplots(nrows=5 + nh * (num_register_tokens + 1), ncols=3, figsize=(15, 5 * (5 + nh * (num_register_tokens + 1))))

#         # Plot the original image
#         ax = axes[0, 0]
#         ax.imshow(np.transpose(images[num_img].cpu().numpy(), (1, 2, 0)))
#         ax.axis('off')
#         ax.set_title('Original Image')

#         for ax in axes[0, 1:]:
#             ax.axis('off')

#         # Plot the class token attention maps
#         for j in range(nh):
#             row = (j + 3) // 3
#             col = (j + 3) % 3
#             ax = axes[row, col]
#             ax.imshow(cls_attentions[j], cmap='viridis')
#             ax.axis('off')
#             ax.set_title(f'Class Token Attention Head {j+1}')

#         # Plot the register token attention maps
#         for k in range(num_register_tokens):
#             for j in range(nh):
#                 row = 5 + k * nh + j
#                 col = (5 + k * nh + j) % 3
#                 ax = axes[row, col]
#                 ax.imshow(reg_attentions[k * nh + j], cmap='viridis')
#                 ax.axis('off')
#                 ax.set_title(f'Register Token {k+1} Attention Head {j+1}')

#         # Save the figure to the PDF
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close(fig)

#     print(f"All attention heads saved in {os.path.join(args.output_dir, 'attention_maps.pdf')}.")
