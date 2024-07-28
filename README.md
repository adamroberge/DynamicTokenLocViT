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
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**:
    Ensure that the CIFAR-10 dataset is available in the `./data/CIFAR10` directory. The dataset will be automatically downloaded if not already present.

4. **Train the model**:
    Train the Vision Transformer model on the CIFAR-10 dataset.
    ```bash
    python train.py --data_path ./data/CIFAR10 --model_path ./models/best_model.pth
    ```

5. **Evaluate the model**:
    Evaluate the trained model on the CIFAR-10 test set.
    ```bash
    python evaluate.py --data_path ./data/CIFAR10 --model_path ./models/best_model.pth
    ```

6. **Visualize Attention Maps**:
    Visualize the self-attention maps of the model for a specific layer.
    ```bash
    python visualize_attention.py --model_path ./models/best_model.pth --layer_num 2 --output_dir ./attention_maps
    ```

    **Options**:
    - `--model_path`: Path to the trained model file (default: `best_model.pth`).
    - `--layer_num`: The layer number to visualize attention maps from (default: 2).
    - `--output_dir`: Directory to save the attention maps (default: `.`).

    Example:
    ```bash
    python visualize_attention.py --model_path ./models/best_model.pth --layer_num 5 --output_dir ./attention_maps
    ```




