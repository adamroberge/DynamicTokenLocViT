# Saved attention figures

[← Project overview](../../README.md) · [Figure provenance](../figures/README.md)

These are the original research PDFs, relocated from the repository root and `output_dir/`. Their contents are unchanged. The filenames retain the original block index, image index, and token-position settings where available.

| Figure | Saved configuration |
| --- | --- |
| [CIFAR-10 attention maps](cifar10_attention_maps.pdf) | CIFAR-10 example with class and register attention. |
| [ImageNet: block 0, image 5](attention_maps_layer_0_of_image_5_cls_0_reg_0.pdf) | `cls_pos=0`, `reg_pos=0` |
| [ImageNet: block 3, image 5](attention_maps_layer_3_of_image_5_cls_0_reg_0.pdf) | `cls_pos=0`, `reg_pos=0` |
| [ImageNet: block 11, image 5](attention_maps_layer_11_of_image_5_cls_0_reg_0.pdf) | `cls_pos=0`, `reg_pos=0` |
| [ImageNet: block 3, image 0](attention_maps_layer_3_of_image_0_cls_0_reg_0.pdf) | `cls_pos=0`, `reg_pos=0` |

The ImageNet PDFs contain six attention heads and four register tokens. The first page shows class-token attention; the next four pages show attention from register tokens 1 through 4. Input images were plotted after normalization in the original scripts.

New run outputs belong in ignored local directories such as `output_dir/` or `result/`. Keep curated documentation examples here.
