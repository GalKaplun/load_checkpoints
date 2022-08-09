"""
Utility functions for loading models and training.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np


def load_resnet50_from_checkpoint(checkpoint_path):
    """
    Load a ResNet50 model from a checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint.

    Returns:
        The loaded model.
    """
    model = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    fixed_state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    return model