# README figure provenance

[← Project overview](../../README.md) · [Original PDFs](../examples/README.md)

| Asset | Source and interpretation |
| --- | --- |
| [`token-placement.svg`](token-placement.svg) | Editable schematic of `prepare_tokens`, `forward_features`, and `forward` in [`models/dynamic_vit_viz.py`](../../models/dynamic_vit_viz.py). Illustrates 224/16 image/patch sizes, 12 blocks, four registers, `reg_pos=3`, and `cls_pos=6`. This example configuration differs from the archived attention maps. |
| [`token-attention.png`](token-attention.png) | Panels from the [block 11, image 5 PDF](../examples/attention_maps_layer_11_of_image_5_cls_0_reg_0.pdf): plotted input and class-token head 1 on page 1, register-token-1 head 1 on page 2, and register-token-4 head 1 on page 5. |
| [`layer-attention.png`](layer-attention.png) | Plotted input plus class-token head 1 from page 1 of the image 5 PDFs at [block 0](../examples/attention_maps_layer_0_of_image_5_cls_0_reg_0.pdf), [block 3](../examples/attention_maps_layer_3_of_image_5_cls_0_reg_0.pdf), and [block 11](../examples/attention_maps_layer_11_of_image_5_cls_0_reg_0.pdf). |

The PNGs were made by rendering, cropping, and arranging panels from the existing PDFs. No model was run and no attention values were recomputed. Labels were added for readability; the underlying panels retain their original colors and orientation.

The saved ImageNet sample is labeled `n01440764` in the PDFs. Both token types enter at block 0 in these examples. Each heatmap uses its original plot's color scale, so these panels show qualitative patterns rather than a shared quantitative scale. The plotted input tensors were normalized, giving the displayed images exaggerated colors.
