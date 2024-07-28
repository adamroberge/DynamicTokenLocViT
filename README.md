# Dynamic ViT Visualization

This repository contains code for visualizing self-attention maps in Vision Transformers (ViT) with dynamic tokens, specifically class and register tokens. The code allows for visualizing attention maps from different layers of the ViT model on the CIFAR-10 dataset.

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
    Ensure that the CIFAR-10 or CIFAR-100 dataset is available in the `./data/CIFAR10` or `./data/CIFAR100` directory respectively. The dataset will be automatically downloaded if not already present.

4. **Train the model**:
    Train the Vision Transformer model on the CIFAR-10 or CIFAR-100 dataset.
    More details on the training function are in ```cifar_train.py```
    ```bash
    python cifar_main.py --data_path ./data/CIFAR10 --model_path ./models/best_model.pth
    ```

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

This project is licensed under the MIT License. See the `LICENSE` file for details.

