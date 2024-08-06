# in_main.py
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import random
from dynamic_vit_viz import vit_register_dynamic_viz
from in_train import train_model
from in_test import test_model
from custom_summary import custom_summary

from timm.data import create_transform
from timm.scheduler import create_scheduler


def get_args_parser():
    parser = argparse.ArgumentParser('Training and Evaluation Script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--eval-crop-ratio', default=1.0, type=float)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--nb-classes', type=int, default=1000)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--drop-path', type=float, default=0.05)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.3)
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--data-path', default='/home/adam/data/in1k', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='output_dir', type=str)

    return parser

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_transform = create_transform(
        input_size=args.input_size,  # Ensure images are resized to 224x224
        is_training=True,
        color_jitter=args.color_jitter,  # Adjusted for more variability
        auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment policy
        interpolation='bicubic',  # Interpolation method
        re_prob=args.reprob,  # Random Erase probability
        re_mode='pixel',  # Random Erase mode
        re_count=1,  # Random Erase count
    )

    # Define data transforms for testing
    test_transform = transforms.Compose([
        transforms.Resize(int(args.input_size / args.eval_crop_ratio)),  # Resize maintaining aspect ratio
        transforms.CenterCrop(args.input_size),  # Center crop
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load ImageNet1K datasets
    train_dataset = ImageFolder(root=os.path.join(args.data_path, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=lambda _: np.random.seed(args.seed))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=lambda _: np.random.seed(args.seed))

    # Initialize the model
    model = vit_register_dynamic_viz(img_size=args.input_size, patch_size=args.patch_size, in_chans=3, num_classes=args.nb_classes, embed_dim=384, depth=12,
                                     num_heads=6, mlp_ratio=4., drop_rate=args.drop, attn_drop_rate=0.,
                                     drop_path_rate=args.drop_path, init_scale=1e-4,
                                     mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=6, reg_pos=0)

    custom_summary(model, (3, args.input_size, args.input_size))

    # Move the model to GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create learning rate scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Train the model
    train_model(model, train_loader, loss_fn, optimizer, lr_scheduler, num_epochs=args.epochs, device=device)

    # Test the model
    test_model(model, test_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ImageNet1K Training and Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
