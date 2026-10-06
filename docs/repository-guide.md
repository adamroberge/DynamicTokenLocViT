# Repository guide

[← Project overview](../README.md) · [Usage guide](usage.md) · [Saved figures](examples/README.md)

Python files are grouped by purpose into packages. Run entry points with `python -m` from the repository root; internal imports use the same package structure.

## Models

| File | Purpose |
| --- | --- |
| [`models/dynamic_vit_viz.py`](../models/dynamic_vit_viz.py) | Main `vit_register_dynamic_viz` implementation; configurable class/register insertion and per-layer attention extraction. |
| [`models/dynamic_vit.py`](../models/dynamic_vit.py) | `vit_models` baseline, `vit_register_dynamic`, and transformer building blocks. |
| [`models/trainable_cls_reg.py`](../models/trainable_cls_reg.py) | Alternative `TrainableVitRegisterDynamicViz` implementation and associated attention/block classes. Includes model-summary code at module scope. |
| [`models/baselines/deit.py`](../models/baselines/deit.py) | DeiT model definitions, including distilled variants. |
| [`models/baselines/original_vit_deit.py`](../models/baselines/original_vit_deit.py) | Reference ViT/DeiT definitions and registered model variants. |

## Training and evaluation

| Workflow | Entry point | Supporting files |
| --- | --- | --- |
| Main workflow | [`training/main.py`](../training/main.py) | [`training/engine.py`](../training/engine.py), [`data_utils/datasets.py`](../data_utils/datasets.py), [`common/utils.py`](../common/utils.py) |
| Optional teacher distillation | [`training/distillation.py`](../training/distillation.py) | Main workflow helpers plus [`training/losses.py`](../training/losses.py) |
| Simple CIFAR-10 workflow | [`training/cifar/main.py`](../training/cifar/main.py) | [`training/cifar/train.py`](../training/cifar/train.py), [`training/cifar/evaluate.py`](../training/cifar/evaluate.py) |
| Simple ImageNet workflow | [`training/imagenet/main.py`](../training/imagenet/main.py) | [`training/imagenet/train.py`](../training/imagenet/train.py), [`training/imagenet/evaluate.py`](../training/imagenet/evaluate.py) |
| Alternative token-model workflow | [`training/trainable_tokens/main.py`](../training/trainable_tokens/main.py) | [`training/trainable_tokens/train.py`](../training/trainable_tokens/train.py), [`training/trainable_tokens/evaluate.py`](../training/trainable_tokens/evaluate.py) |

Each dataset-specific training folder contains `main.py`, `train.py`, and `evaluate.py`. The former root files ending in `_test.py` are now the corresponding `evaluate.py` modules. Historical scratch scripts live in `research/`.

## Attention visualization

| File | Expected model/checkpoint |
| --- | --- |
| [`visualization/attention.py`](../visualization/attention.py) | Tiny model, 192 embedding dimensions and 3 heads; checkpoint dictionary with a `model` key. Supports an explicit image path. |
| [`visualization/imagenet.py`](../visualization/imagenet.py) | Small model, 384 embedding dimensions and 6 heads; bare state dictionary. Uses the ImageNet validation directory. |
| [`visualization/cifar.py`](../visualization/cifar.py) | CIFAR-10 model, 384 embedding dimensions and 12 heads; bare state dictionary. |

See [checkpoint compatibility](usage.md#checkpoint-compatibility) before pairing a training script with a visualizer.

## Shared helpers

| File | Purpose |
| --- | --- |
| [`data_utils/datasets.py`](../data_utils/datasets.py) | Dataset selection and image transforms for the main workflows. |
| [`data_utils/augment.py`](../data_utils/augment.py) | Additional image augmentation. |
| [`data_utils/samplers.py`](../data_utils/samplers.py) | Repeated-augmentation sampler. |
| [`training/engine.py`](../training/engine.py) | Training epoch and evaluation functions. |
| [`training/losses.py`](../training/losses.py) | Distillation loss. |
| [`common/utils.py`](../common/utils.py) | Logging, distributed helpers, checkpoint handling, and other utilities. |
| [`common/custom_summary.py`](../common/custom_summary.py) | Model summary using hooks and a forward pass. |

## Research scratchpad

These scripts preserve development notes and small research probes. Some contain hard-coded paths, optional imports, or references to earlier modules.

| File or directory | Purpose |
| --- | --- |
| [`research/`](../research/README.md) | Dataset inspection, tensor packing, seed/summary exploration, class attention, and an earlier training routine. |
| [`research/test_one_batch.py`](../research/test_one_batch.py) | One-batch model/training exploration. |
| [`research/verify_pth.py`](../research/verify_pth.py) | Checkpoint inspection using an original machine-specific path. |
| [`docs/archive/commands.sh`](archive/commands.sh) | Historical experiment settings with launch paths updated to the new modules. Includes incomplete commands and local paths. |

## Entry-point migration

| Former script | Run from the repository root |
| --- | --- |
| `main.py` | `python -m training.main` |
| `main_distillation.py` | `python -m training.distillation` |
| `cifar_main.py` | `python -m training.cifar.main` |
| `in_main.py` | `python -m training.imagenet.main` |
| `trainable_cls_reg_main.py` | `python -m training.trainable_tokens.main` |
| `viz_attn_main.py` | `python -m visualization.attention` |
| `viz_attn_in_main.py` | `python -m visualization.imagenet` |
| `cifar_visualize_attention.py` | `python -m visualization.cifar` |
| `test_one_batch.py` | `python -m research.test_one_batch` |
| `verify_pth.py` | `python -m research.verify_pth` |

Pass the same script arguments after the module name. For distributed training, use `torchrun --nproc_per_node=N --module training.main --distributed` with your experiment arguments. The former `test/` directory is now `research/`, and baseline model definitions are in `models/baselines/`.

## Documentation and local artifacts

| Location | Contents |
| --- | --- |
| [`docs/usage.md`](usage.md) | Setup, data layout, example commands, token positions, and checkpoint formats. |
| [`docs/figures/`](figures/README.md) | Editable model diagram and previews extracted from saved PDFs. |
| [`docs/examples/`](examples/README.md) | Original CIFAR-10 and ImageNet attention-map PDFs. |
| `data/`, `cifar_data/` | Ignored local dataset downloads. |
| `result/`, `output_dir/`, `outputs/`, `checkpoints/` | Ignored local logs, checkpoints, and generated figures. |

The PDFs previously stored at the repository root and in `output_dir/` now live in `docs/examples/`. The former root `commands.sh` now lives in `docs/archive/`. Downloaded CIFAR data and the generated `result/log.txt` are excluded from version control; local copies can remain in place.
