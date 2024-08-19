import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torchsummary import summary
import numpy as np
from collections import OrderedDict

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Attention(nn.Module):
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # (B, num_heads, N, N)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn # Return both x and attention


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention: # if return_attention is true, only return the attn
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
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
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))[0]) + \
            self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x))[0])
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + \
            self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x


class Block_paralx2(nn.Module):
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
        x = x + self.drop_path(self.attn(self.norm1(x))[0]) + \
            self.drop_path(self.attn1(self.norm11(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x))) + \
            self.drop_path(self.mlp1(self.norm21(x)))
        return x


class hMLP_stem(nn.Module):
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


class vit_register_dynamic_viz(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block, Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp, dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, num_register_tokens=4, reg_pos=None, cls_pos=None, **kwargs):
        super().__init__()

        self.reg_pos = reg_pos
        self.cls_pos = cls_pos
        
        self.patch_size = patch_size
        self.depth = depth
        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_register_tokens = num_register_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.num_register_tokens, embed_dim))

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
        trunc_normal_(self.register_tokens, std=.02)
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
        return {'pos_embed', 'cls_token', 'reg_tokens'}  # Add register tokens

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x):
        B = x.shape[0]  # Get the batch size from the input tensor
        x = self.patch_embed(x)  # Apply patch embedding to the input image

        # Initialize cls_tokens and register_tokens placeholders
        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(
            B, -1, -1) if self.register_tokens is not None else None

        # Add positional embeddings to the patch tokens
        # x = x + self.pos_embed[:, :self.num_patches, :]
        x = x + self.pos_embed
        return x, cls_tokens, register_tokens

    def forward_features(self, x, cls_pos=None, reg_pos=None):
        x, cls_tokens, register_tokens = self.prepare_tokens(x)
        # Pass the token sequence through each transformer block
        for i, blk in enumerate(self.blocks):
            if i == reg_pos and register_tokens is not None:
                x = torch.cat((x, register_tokens), dim=1)
            if i == cls_pos:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        # Apply layer normalization to the output of the last transformer block
        x = self.norm(x)

        # Extract the class token if it's been added in the transformer blocks
        # if cls_pos is not None and cls_pos < len(self.blocks):
        x_cls = x[:, 0]
        x_regs = x[:, -self.num_register_tokens:] if self.num_register_tokens > 0 else None

        return x_cls, x_regs
    
    def forward(self, x):
        # Compute the forward pass through the transformer
        x_cls, x_regs = self.forward_features(x, self.cls_pos, self.reg_pos)

        if self.dropout_rate:
            x_cls = F.dropout(x_cls, p=float(
                self.dropout_rate), training=self.training)

        # Pass the class token representation through the classification head
        x_cls = self.head(x_cls)
        
        return x_cls  # Return the final class scores
    
    def get_last_selfattention(self, x):
        x, cls_tokens, reg_tokens = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_selfattention(self, x, layer): # Only cls token
        cls_pos = self.cls_pos
        reg_pos = self.reg_pos
        num_reg = self.num_register_tokens

        x, cls_tokens, reg_tokens = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks):
            if i == reg_pos and reg_tokens is not None:
                x = torch.cat((x, reg_tokens), dim=1)
            if i == cls_pos:
                x = torch.cat((cls_tokens, x), dim=1)
            if i == layer:
                # Get the attention map from the specified layer
                attn = blk(x, return_attention=True) # (1, 12, 197, 197)
                break
            x = blk(x)
        return attn

    def get_register_token_attention(self, x, layer):
        cls_pos = self.cls_pos
        reg_pos = self.reg_pos
        num_reg = self.num_register_tokens

        x, cls_tokens, reg_tokens = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks):
            if i == reg_pos and reg_tokens is not None:
                x = torch.cat((x, reg_tokens), dim=1)
            if i == cls_pos:
                x = torch.cat((cls_tokens, x), dim=1)
            if i == layer:
                # Get the attention map from the specified layer
                attn = blk(x, return_attention=True)
                reg_attn = attn[:, :, -num_reg:, :-num_reg]  # Extract attention from register tokens to patch tokens
                break
            x = blk(x)
        return reg_attn
    
    def get_attention_map(self, x, layer): # Both cls and reg tokens
        cls_pos = self.cls_pos
        reg_pos = self.reg_pos
        num_reg = self.num_register_tokens

        if layer < reg_pos:
            raise ValueError(f"Cannot access register tokens at layer {layer} since they are added at layer {reg_pos}")

        x, cls_tokens, reg_tokens = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks):
            if i == cls_pos:
                x = torch.cat((cls_tokens, x), dim=1)
            if i == reg_pos and reg_tokens is not None:
                x = torch.cat((x, reg_tokens), dim=1)
            if i == layer:
                # Get the attention map from the specified layer
                attn = blk(x, return_attention=True) # [1, 12, 201, 201]
                # Attention from cls token to patch tokens [1, 12, 1, 196]
                cls_attn = attn[:, :, 0, 1:self.num_patches+1]  # self.num_patches+1 because it's exclusive of end index
                # Attention from register tokens to patch tokens 
                reg_attn_list = [attn[:, :, -num_reg + i, 1:self.num_patches+1] for i in range(num_reg)]  # List of attention maps for each register token
                break

        return cls_attn, reg_attn_list
