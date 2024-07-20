import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
import numpy as np
from collections import OrderedDict

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# from cait import Class_Attention


class Class_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # changed input to both x(patch+reg) and cls_token (instead of only x from the original cait model)
    def forward(self, x, cls_token):
        B, N, C = x.shape
        # Ensure cls_token is expanded to match batch size
        cls_token = cls_token.expand(B, -1, -1)
        # Concatenate cls_token with the input
        x = torch.cat((cls_token, x), dim=1)

        q = self.q(x[:, 0:1]).reshape(B, 1, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N + 1, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N + 1, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_token = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        cls_token = self.proj(cls_token)
        cls_token = self.proj_drop(cls_token)

        return cls_token


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


class vit_register_dynamic(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block, Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp, dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, num_register_tokens=0, reg_pos=None, cls_pos=None, **kwargs):
        super().__init__()

        self.reg_pos = reg_pos
        self.cls_pos = cls_pos

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        # self.pos_embed = nn.Parameter(torch.zeros(
        #     1, num_patches + num_register_tokens + 1, embed_dim))  # patches + registers + 1 for cls
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None

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

        self.project = nn.Linear(
            embed_dim * (1 + num_register_tokens), embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.register_tokens is not None:
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, cls_pos=None, reg_pos=None):
        B = x.shape[0]  # Get the batch size from the input tensor
        x = self.patch_embed(x)  # Apply patch embedding to the input image

        # Initialize cls_tokens and register_tokens placeholders
        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(
            B, -1, -1) if self.register_tokens is not None else None

        # Add positional embeddings to the patch tokens
        x = x + self.pos_embed[:, 1 + self.num_register_tokens:, :]

        # Pass the token sequence through each transformer block
        for i, blk in enumerate(self.blocks):
            if i == reg_pos and register_tokens is not None:
                x = torch.cat((register_tokens, x), dim=1)
            if i == cls_pos:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        # Apply layer normalization to the output of the last transformer block
        x = self.norm(x)

        # Extract the class token if it's been added in the transformer blocks
        if cls_pos is not None and cls_pos < len(self.blocks):
            x_cls = x[:, 0]
            x_regs = x[:, 1:1 + self.num_register_tokens]
        else:  # If the CLS token is added at the end (CaiT-like)
            x_cls = self.class_attention_block(x, cls_tokens).squeeze(1)
            x_regs = x[:, :self.num_register_tokens]

        return x_cls, x_regs

    def forward(self, x):
        # Compute the forward pass through the transformer
        x_cls, x_regs = self.forward_features(x, self.cls_pos, self.reg_pos)

        if self.dropout_rate:
            x_cls = F.dropout(x_cls, p=float(
                self.dropout_rate), training=self.training)

        if self.register_tokens is not None:
            # Concatenate class token with register tokens
            x_cls = torch.cat((x_regs, x_cls.unsqueeze(1)), dim=1)
            # Flatten the concatenated tokens
            x_cls = x_cls.view(x_cls.size(0), -1)
            # Project to the original embedding dimension
            x_cls = self.project(x_cls)

        # Pass the class token representation through the classification head
        x_cls = self.head(x_cls)

        return x_cls  # Return the final class scores


# Define model parameters
img_size = 224
patch_size = 16
in_chans = 3
num_classes = 10
embed_dim = 384  # Tiny: 192, Small: 384, Big: 768
depth = 12
num_heads = 6  # Tiny: 3, Small:6 , Big: 12
mlp_ratio = 4.0
qkv_bias = True
qk_scale = None
drop_rate = 0.1
attn_drop_rate = 0.1
drop_path_rate = 0.1
num_register_tokens = 4
cls_pos = 5
reg_pos = 6

# Create an instance of the model
model = vit_register_dynamic(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    qkv_bias=qkv_bias,
    qk_scale=qk_scale,
    drop_rate=drop_rate,
    attn_drop_rate=attn_drop_rate,
    drop_path_rate=drop_path_rate,
    num_register_tokens=num_register_tokens,
    cls_pos=cls_pos,
    reg_pos=reg_pos
)

# Create a random input tensor with the appropriate shape
batch_size = 2
x = torch.randn(batch_size, in_chans, img_size, img_size)

# Use the model with dynamic insertion points
output = model(x)

# # Check the final output dimensions
# assert output.shape == (
#     batch_size, num_classes), f"Final output shape mismatch: {output.shape}"

# print("All checks passed successfully!")

# summary(model, (3, 224, 224))


def custom_summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Batch size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(register_hook)

    # Make a forward pass
    # print(x.shape)
    model(*x)

    # Remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    block_counter = 0  # Initialize block counter

    for layer in summary:
        # Add separator for each Block
        if "Block" in layer:
            block_counter += 1  # Increment block counter
            print("----------------------------------------------------------------")
            print(f"Block-{block_counter}")
            print("----------------------------------------------------------------")

        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")


custom_summary(model, (3, 224, 224))
