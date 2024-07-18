import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from cait import __all__, Class_Attention, LayerScale_Block_CA, Attention_talking_head, LayerScale_Block, cait_models


@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size=224, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_XXS36_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size=224, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_XXS36(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_XS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_S24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size=224, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_S24(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_S36(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_M36(pretrained=False, **kwargs):
    model = cait_models(
        img_size=384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def cait_M48(pretrained=False, **kwargs):
    model = cait_models(
        img_size=448, patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]

        model.load_state_dict(checkpoint_no_module)

    return model
