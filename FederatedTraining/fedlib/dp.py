"""
Useful tools
"""
import numpy as np
import random
import torch

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)