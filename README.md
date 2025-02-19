# Dynamic Token Location for ViT Visualization

This repository presents an implementation of an enhanced **Vision Transformer (ViT)** model with dynamic token allocation, incorporating both class tokens and register tokens. Built on the foundational code from [DeiT (Data-efficient Image Transformers) by Facebook](https://github.com/facebookresearch/deit), this project enhances the standard ViT architecture by allowing dynamic and customizable token management at various layers.

Inspired by research from the papers ["Vision Transformers Need Registers"](https://arxiv.org/abs/2309.16588) and ["Going Deeper with Image Transformers"](https://arxiv.org/abs/2103.17239), this implementation introduces **flexible class and register token placement**—a unique feature that enables users to choose where these tokens are injected within the model architecture. This dynamic approach improves token adaptability, making the transformer more responsive to different tasks and datasets.

Additionally, **attention map visualization** tools are provided, enabling users to better understand token interactions and their influence on the model's decision-making process. This contributes to **improved model interpretability** by allowing deeper insights into how the attention mechanism operates across layers and tokens.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/adanroberge/cls-register.git
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/adamroberge/cls-register.git
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**:
    Ensure that the CIFAR-10 or CIFAR-100 dataset is available in the `./data/CIFAR10` or `./data/CIFAR100` directory respectively. If you want to train on the ImageNet 1k dataset, download the data through [ImageNet](https://www.image-net.org/)

4. **Train the model**:
    Train the Vision Transformer model on the CIFAR-10, CIFAR-100, or ImageNet1k dataset. The model prepared to train on the ImageNet1k has dynamic token locations you can set and returns the evaluation as well after training.
    More details on the training function are in ```cifar_train.py```
    ```bash
    python cifar_main.py --data_path ./data/CIFAR10 --model_path ./models/best_model.pth
    ```
    Example (4 reg tokens with cls token added at the 6th block and register tokens added at the 3rd block):
    ```bash
    torchrun --nnodes=1 --nproc_per_node=4 main_distillation.py --distributed --num_reg 4 --cls_pos 6 --reg_pos 3
    ```
    For more detailed commands, check the [commands](commands.sh) file.

6. **Evaluate the model**:
    Evaluate the trained model on the CIFAR-10 or CIFAR-100 test set.
    More details on the training function are in ```cifar_test.py```
    ```bash
    python cifar_main.py --data_path ./data/CIFAR10 --model_path ./models/best_model.pth
    ```

8. **Visualize Attention Maps**:
    Visualize the self-attention maps of the model for a specific layer.

    **Options**:
    - `--model_path`: Path to the trained model file (default: `best_model.pth`).
    - `--layer_num`: The layer number to visualize attention maps from (default: 2).
    - `--output_dir`: Directory to save the attention maps (default: `.`).

    Example:
    ```bash
    python visualize_attention.py --model_path ./models/best_model.pth --layer_num 5 --output_dir ./attention_maps
    ```
    Check [CIFAR10 Attention Map](cifar10_attention_maps.pdf) for an example result.

## Arguments

- `--model_path`: Path to the trained model.
- `--layer_num`: Layer number to visualize attention from.
- `--output_dir`: Directory to save the visualizations.


### Model Architecture

The Vision Transformer (ViT) model used in this repository includes dynamic tokens, specifically class and register tokens. The register tokens are added at a specified layer and influence the subsequent layers.


## `vit_register_dynamic_viz` Class

The `vit_register_dynamic_viz` class extends the standard ViT model to include dynamic tokens.

### Key Parameters:
- `img_size`: Size of the input images.
- `patch_size`: Size of the patches.
- `num_classes`: Number of output classes.
- `embed_dim`: Embedding dimension.
- `depth`: Number of transformer layers.
- `num_heads`: Number of attention heads.
- `mlp_ratio`: MLP ratio.
- `num_register_tokens`: Number of register tokens.
- `cls_pos`: Layer to add the class token.
- `reg_pos`: Layer to add the register tokens.


## Visualization

The visualization script extracts self-attention maps from the specified layer and saves them as a PDF. The class token's attention map and each register token's attention map are saved on different pages.


## Example Output

The attention maps are saved in a PDF file in the specified output directory. Each page includes the original image and the attention maps for the class and register tokens.

※ Please note that the attention maps generated in this version are preliminary and may be of lower quality. We are actively working on improving the generation process to enhance the resolution and accuracy of these visualizations. Future updates will include higher-quality attention maps and additional features to better support your analysis needs.


## Important Notes

- Ensure that the layer number specified for visualization is valid and does not exceed the model's depth.
- The visualization script will raise an error if attempting to access register tokens from a layer before they are added.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

### Steps to Contribute
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

