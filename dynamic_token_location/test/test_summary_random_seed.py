import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary 
import matplotlib.pyplot as plt
import numpy as np
import random

# # Ensure vit_dynamic contains your deit_small_patch16_LS definition
# from original_vit_deit import deit_small_patch16_LS, deit_small_patch16
# from dynamic_vit import vit_models, vit_register_dynamic
# from dynamic_vit_viz import vit_register_dynamic_viz
# from custom_summary import custom_summary

def print_model_parameters(model, num_params=30):
    """
    Print the names and values of the first num_params parameters of a model.
    
    Args:
        model (torch.nn.Module): The model whose parameters are to be printed.
        num_params (int): The number of parameters to print. Defaults to 30.
    """
    print(f"Printing the first {num_params} parameters of the model:\n")
    
    # Iterate through the model's parameters
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= num_params:
            break
        print(f"Parameter {i + 1}:")
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values: {param.detach().cpu().numpy().flatten()[:10]} ...")  # Print first 10 values for brevity
        print("-" * 50)


def compare_model_parameters(model, initial_state):
    for name, param in model.named_parameters():
        initial_param = initial_state[name]
        if not torch.equal(param.data, initial_param.data):
            print(f"Parameter {name} has changed.")
            return False
    print("All parameters are identical.")
    return True
