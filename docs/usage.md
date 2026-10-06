# Usage guide

[← Project overview](../README.md) · [File guide](repository-guide.md) · [Saved figures](examples/README.md)

Run commands from the repository root using `python -m package.module`. This makes the project's package imports available without an editable install. The examples below follow the source's argument names and checkpoint formats.

## Setup and data

Install the pinned dependencies in [`requirements.txt`](../requirements.txt) as shown in the [README](../README.md#setup). Model definitions import `torchsummary`, and the simple training/evaluation helpers import `tqdm`. These imports are additional to the original dependency pins:

```bash
python -m pip install tqdm torchsummary
```

The main ImageNet workflow uses an `ImageFolder` layout:

```text
data/imagenet/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

Supply your dataset root with `--data-path`. For CIFAR datasets, pass the parent directory containing the extracted dataset folder, such as `data/CIFAR10/`. The main CIFAR-10 loader requests an automatic download; the main CIFAR-100 loader expects the data to exist already. See [`data_utils/datasets.py`](../data_utils/datasets.py) for the selection logic.

Downloaded data, checkpoints, and generated outputs are ignored by Git. The saved figures used in the README are kept separately in [`docs/examples/`](examples/README.md).

## Main training workflow

[`training/main.py`](../training/main.py) selects the dynamic visualization model by default. Its active configuration uses 12 blocks, an embedding dimension of 192, and 3 attention heads.

Example with the class token inserted before block 6 and four register tokens inserted before block 3:

```bash
python -m training.main \
  --model vit_register_dynamic_viz \
  --data-set IMNET \
  --data-path ./data/imagenet \
  --input-size 224 --patch-size 16 --nb-classes 1000 \
  --num_reg 4 --cls_pos 6 --reg_pos 3 \
  --output_dir ./result/cls6_reg3
```

For a distributed run, use the same arguments with `torchrun --nproc_per_node=N --module training.main --distributed`, replacing `N` with the number of GPUs to use. Dataset, optimization, and batch-size settings should be chosen for your experiment.

`training/main.py` writes `checkpoint.pth`, `best_checkpoint_cls_6_reg_3.pth` for the configuration above, and `log.txt` inside `--output_dir`.

[`training/distillation.py`](../training/distillation.py) exposes the same token-placement flags and adds `--distillation-type`, `--teacher-model`, and `--teacher-path`. Its default `--distillation-type none` leaves teacher distillation disabled. It saves its best checkpoint as `best_checkpoint.pth`.

## Evaluation only

Use `--eval` and `--resume` with the same dataset and model configuration used to train the checkpoint:

```bash
python -m training.main \
  --eval --resume ./result/cls6_reg3/best_checkpoint_cls_6_reg_3.pth \
  --data-set IMNET --data-path ./data/imagenet \
  --input-size 224 --patch-size 16 --nb-classes 1000 \
  --num_reg 4 --cls_pos 6 --reg_pos 3 \
  --output_dir ./result/cls6_reg3
```

## Visualize a main-workflow checkpoint

[`visualization/attention.py`](../visualization/attention.py) expects a checkpoint dictionary with a `model` key. Its architecture matches the active tiny configuration in `training/main.py`: 224-pixel images, 16-pixel patches, 1,000 classes, embedding dimension 192, and 3 heads.

Use an explicit `--image_path` to select an image and bypass the script's original ImageNet dataset path. Create the output directory before running this visualizer:

```bash
mkdir -p output_dir
python -m visualization.attention \
  --model_path ./result/cls6_reg3/best_checkpoint_cls_6_reg_3.pth \
  --image_path ./path/to/image.jpg \
  --num_reg 4 --cls_pos 6 --reg_pos 3 \
  --layer_num 11 \
  --output_dir ./output_dir
```

The PDF contains one page for class-token attention and one page for each register token. Each page shows the input image and the attention heads.

### Token positions and layer selection

- Positions and `--layer_num` use zero-based block indices. A 12-block model uses indices `0` through `11`.
- Tokens enter **before** the block at their configured position.
- For both class and register attention, choose `layer_num >= max(cls_pos, reg_pos)` and `layer_num < depth`.
- Match the checkpoint's token count and insertion positions. Token positions are configuration values and are not inferred from the checkpoint weights.
- CLI spellings differ between scripts: the main workflows use `--data-path`, while `visualization/imagenet.py` uses `--data_path`.

## Checkpoint compatibility

| Workflow or visualizer | Model configuration | Checkpoint format |
| --- | --- | --- |
| `training/main.py`, `training/distillation.py` | 192 embedding dimensions, 3 heads; image/patch sizes and class count are configurable | Dictionary containing `model`, optimizer, and other training state |
| `visualization/attention.py` | 192 embedding dimensions, 3 heads; 224/16 image/patch sizes, 1,000 classes | Loads `checkpoint['model']`; strips a distributed `module.` prefix |
| `training/imagenet/main.py`, `visualization/imagenet.py` | 384 embedding dimensions, 6 heads; four register tokens | Bare model state dictionary |
| `training/cifar/main.py` | 384 embedding dimensions, 6 heads; 224/16 image/patch sizes, 10 classes, four register tokens | Bare model state dictionary |
| `visualization/cifar.py` | 384 embedding dimensions, 12 heads; 224/16 image/patch sizes, 10 classes, four register tokens | Bare model state dictionary |

The CIFAR training and visualization scripts currently configure different numbers of attention heads. A state dictionary loading successfully does not make those attention configurations equivalent; review the settings before interpreting their maps together. The six-head archived ImageNet PDFs also come from a different configuration than the active three-head main visualizer.

## Other research workflows

- [`training/cifar/main.py`](../training/cifar/main.py) is a fixed CIFAR-10 training/evaluation script. It downloads to `./data/CIFAR10`, uses `cls_pos=6`, `reg_pos=0`, and saves `best_model.pth` through `training/cifar/train.py`. It has no command-line parser; edit its configuration directly.
- [`training/imagenet/main.py`](../training/imagenet/main.py) is a simpler ImageNet training/evaluation workflow. It uses four register tokens and accepts `--data-path`, `--cls_pos`, and `--reg_pos`. Its training helper saves a bare state dictionary to `best_model.pth`; the helper's plot path uses `output_dir/`.
- [`visualization/imagenet.py`](../visualization/imagenet.py) loads a bare state dictionary and accepts `--data_path`, `--model_path`, `--cls_pos`, `--reg_pos`, `--layer_num`, and `--output_dir`.
- [`visualization/cifar.py`](../visualization/cifar.py) accepts `--model_path`, `--cls_pos`, `--reg_pos`, `--layer_num`, and `--output_dir`. Its model dimensions and head count are set in the source.
- [`training/trainable_tokens/main.py`](../training/trainable_tokens/main.py) preserves alternative token-model work. The training and evaluation calls are currently commented out; the active script prepares data and summarizes the model.

The [historical command notebook](archive/commands.sh) records earlier experiment settings and original machine paths. Use it as context alongside the current source and this guide.
