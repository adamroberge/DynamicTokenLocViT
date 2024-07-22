# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # Dimension after permute: (3, B, self.num_heads, N, C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        q = q * self.scale  # q / sqrt(d) where d is head_dim

        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N, N)
        # softmax along the last dimension, which is the sequence length dim (N) a.k.a number of tokens
        attn = attn.softmax(dim=-1)
        # dropout layer applied to the attention weights.
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # applies a linear transformation to each token, resulting in a tensor of the same shape (B, N, C).
        x = self.proj(x)
        # applies dropout to the output of the linear projection.
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    ''''
    Block

    Architecture:
    -------------
    1. LayerNorm
    2. Attention
    3. DropPath (optional)
    4. LayerNorm
    5. MLP (Multilayer Perceptron)
    6. DropPath (optional)

    Purpose:
    --------
    1. Normalize the input tensor.
    2. Apply the attention mechanism to the normalized tensor.
    3. Optionally apply stochastic depth (DropPath) for regularization.
    4. Normalize the tensor after the attention layer.
    5. Apply an MLP to the normalized tensor to introduce non-linearity.
    6. Optionally apply stochastic depth (DropPath) after the MLP for regularization.

    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    '''
    Layer_scale_init_Block

    Architecture:
    -------------
    1. LayerNorm
    2. Attention
    3. DropPath (optional)
    4. Scale (gamma_1)
    5. LayerNorm
    6. MLP (Multilayer Perceptron)
    7. DropPath (optional)
    8. Scale (gamma_2)

    Purpose:
    --------
    1. Normalize the input tensor.
    2. Apply the attention mechanism to the normalized tensor.
    3. Optionally apply stochastic depth (DropPath) for regularization.
    4. Apply learnable scaling factor (gamma_1) to the output of the attention mechanism.
    5. Normalize the tensor after the attention layer.
    6. Apply an MLP to the normalized tensor to introduce non-linearity.
    7. Optionally apply stochastic depth (DropPath) after the MLP for regularization.
    8. Apply learnable scaling factor (gamma_2) to the output of the MLP.

    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    '''
    Layer_scale_init_Block_paralx2

    Architecture:
    -------------
    1. LayerNorm (norm1)
    2. Attention (attn)
    3. DropPath (optional, applied to attn)
    4. LayerNorm (norm11)
    5. Attention (attn1)
    6. DropPath (optional, applied to attn1)
    7. Scale (gamma_1 and gamma_1_1)
    8. Residual Connection (combining scaled attention outputs)
    9. LayerNorm (norm2)
    10. MLP (mlp)
    11. DropPath (optional, applied to mlp)
    12. LayerNorm (norm21)
    13. MLP (mlp1)
    14. DropPath (optional, applied to mlp1)
    15. Scale (gamma_2 and gamma_2_1)
    16. Residual Connection (combining scaled MLP outputs)

    Purpose:
    --------
    1. Normalize the input tensor (norm1 and norm11).
    2. Apply the attention mechanism (attn and attn1) to the normalized tensors.
    3. Optionally apply stochastic depth (DropPath) for regularization after attention mechanisms.
    4. Apply learnable scaling factors (gamma_1 and gamma_1_1) to the output of the attention mechanisms.
    5. Combine the scaled attention outputs using residual connections.
    6. Normalize the tensor after the attention layer (norm2 and norm21).
    7. Apply an MLP to the normalized tensor to introduce non-linearity (mlp and mlp1).
    8. Optionally apply stochastic depth (DropPath) for regularization after MLPs.
    9. Apply learnable scaling factors (gamma_2 and gamma_2_1) to the output of the MLPs.
    10. Combine the scaled MLP outputs using residual connections.

    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + \
            self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + \
            self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    '''
    Block_paralx2

    Architecture:
    -------------
    1. LayerNorm (norm1)
    2. Attention (attn)
    3. DropPath (optional, applied to attn)
    4. LayerNorm (norm11)
    5. Attention (attn1)
    6. DropPath (optional, applied to attn1)
    7. Residual Connection (combining attention outputs)
    8. LayerNorm (norm2)
    9. MLP (mlp)
    10. DropPath (optional, applied to mlp)
    11. LayerNorm (norm21)
    12. MLP (mlp1)
    13. DropPath (optional, applied to mlp1)
    14. Residual Connection (combining MLP outputs)

    Purpose:
    --------
    1. Normalize the input tensor (norm1 and norm11).
    2. Apply the attention mechanism (attn and attn1) to the normalized tensors.
    3. Optionally apply stochastic depth (DropPath) for regularization after attention mechanisms.
    4. Combine the outputs of the attention mechanisms using residual connections.
    5. Normalize the tensor after the attention layers (norm2 and norm21).
    6. Apply an MLP to the normalized tensor to introduce non-linearity (mlp and mlp1).
    7. Optionally apply stochastic depth (DropPath) for regularization after MLPs.
    8. Combine the outputs of the MLPs using residual connections.

    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + \
            self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + \
            self.drop_path(self.mlp1(self.norm21(x)))
        return x


class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    '''
    Architecture:
    -------------
    1. Convert input image to patches
    2. Apply Conv2d (4x4 kernel, stride 4) to reduce spatial dimensions
    3. Apply normalization (BatchNorm) and GELU activation
    4. Apply Conv2d (2x2 kernel, stride 2) to further reduce dimensions
    5. Apply normalization (BatchNorm) and GELU activation
    6. Apply Conv2d (2x2 kernel, stride 2) to get desired embedding dimension
    7. Apply normalization (BatchNorm)

    Purpose:
    --------
    1. Transform input images into a sequence of patches
    2. Reduce spatial dimensions while increasing the channel dimensions
    3. Normalize and apply non-linearity to aid in training
    4. Flatten and transpose the tensor to prepare it for transformer layers

    '''

    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(
                                              embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(
                                              embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                          ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    '''
    Architecture:
    -------------
    1. Patch Embedding Layer
    2. Position Embedding
    3. Class Token Initialization
    4. Stochastic Depth DropPath Rates
    5. Transformer Encoder Blocks
    6. Layer Normalization
    7. Classification Head

    Purpose:
    --------
    1. Embed input image into patch tokens
    2. Add positional encoding to maintain spatial information
    3. Introduce a class token to aggregate information for classification
    4. Apply stochastic depth to each Transformer block for regularization
    5. Normalize and transform tokens through multiple Transformer blocks
    6. Normalize the final token representation
    7. Map the output to the desired number of classes

    '''

    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, **kwargs):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [
            dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]  # Get the batch size from the input tensor
        # Apply patch embedding to the input image, converting it to a sequence of patch tokens
        x = self.patch_embed(x)

        # Expand the class token to match the batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Add positional embeddings to the patch tokens to retain spatial information
        x = x + self.pos_embed

        # Concatenate the class token to the beginning of the token sequence
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass the token sequence through each transformer block
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        # Apply layer normalization to the output of the last transformer block
        x = self.norm(x)

        # Return only the class token's representation for classification
        return x[:, 0]

    def forward(self, x):
        # Compute the forward pass through the transformer to get the feature representation
        x = self.forward_features(x)

        # Apply dropout to the feature representation if a dropout rate is specified
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate),
                          training=self.training)

        # Pass the feature representation through the classification head to get the final output
        x = self.head(x)

        # Return the final class scores
        return x


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)


@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False,   **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_small_patch16(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, block_layers=Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_' + \
            str(img_size)+'_'
        if pretrained_21k:
            name += '21k_v1.pth'
        else:
            name += '1k_v1.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


# @register_model
# def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
#     model = vit_models(
#         img_size=img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paral_LS, **kwargs)

#     return model


# @register_model
# def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
#     model = vit_models(
#         img_size=img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paral_LS, **kwargs)
#     return model


@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model


@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    # model.default_cfg = _cfg()

    return model

# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)


@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k=False,  **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
