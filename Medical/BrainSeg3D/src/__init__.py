"""
BrainSeg3D: 3D U-Net for Brain Tumor Segmentation

This package provides tools for segmenting brain tumors from multi-modal MRI data using
a 3D U-Net architecture.
"""

from .model import UNet3D, ConvBlock, Encoder, Decoder
from .data import BraTSDataset, DataLoader
from .train import train_model, validate
from .visualize import visualize_results, visualize_multiple_slices
from .augmentation import get_transforms

__version__ = '0.1.0'
