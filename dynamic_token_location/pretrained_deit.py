import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from timm.models import create_model
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor


def to_tensor(img):
    transform_fn = Compose([Resize(249, 3), CenterCrop(
        224), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_fn(img)


def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_img2_on_axes(ax, img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    ax.imshow(img1)
    ax.imshow(img2, alpha=alpha)
    ax.axis('off')


def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(
            B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward


img = Image.open('image.jpg')
x = to_tensor(img)

model = create_model('deit_small_distilled_patch16_224', pretrained=True)

# Apply the forward wrapper to all blocks
for blk in model.blocks:
    blk.attn.forward = my_forward_wrapper(blk.attn)

y = model(x.unsqueeze(0))

# Collect attention maps for each block
attn_maps = [blk.attn.attn_map.mean(dim=1).squeeze(
    0).detach() for blk in model.blocks]
cls_weights = [blk.attn.cls_attn_map.mean(
    dim=1).detach() for blk in model.blocks]

img_resized = x.permute(1, 2, 0) * 0.5 + 0.5

# Plot attention maps for patches
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
axs = axs.flatten()
for i, (attn_map) in enumerate(attn_maps):
    axs[i].imshow(attn_map.cpu().numpy(), cmap='viridis')
    axs[i].set_title(f'Layer {i + 1} Attention Map', fontsize=10)
    cbar = fig.colorbar(axs[i].images[0], ax=axs[i], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
plt.tight_layout()
plt.show()

# Plot class token attention maps
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
axs = axs.flatten()
for i, cls_weight in enumerate(cls_weights):
    cls_resized = F.interpolate(cls_weight.view(1, 1, int(cls_weight.size(-1) ** 0.5), int(
        cls_weight.size(-1) ** 0.5)), (224, 224), mode='bilinear').view(224, 224, 1)
    axs[i].imshow(img_resized)
    axs[i].imshow(cls_resized.squeeze(2).cpu().numpy(),
                  cmap='viridis', alpha=0.8)
    axs[i].set_title(f'Layer {i + 1} CLS Attention Map', fontsize=10)
plt.tight_layout()
plt.show()
