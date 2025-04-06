# BrainSeg3D

A 3D U-Net implementation for brain tumor segmentation using volumetric medical imaging data.

## Overview
This project implements a 3D U-Net architecture for volumetric brain segmentation tasks. The model is designed to work with multi-modal MRI scans and segment brain tumors into different regions.

## Features
- Complete 3D U-Net architecture implementation
- Volumetric data processing pipelines
- 3D visualization tools
- Data augmentation for medical volumes
- Evaluation metrics for segmentation tasks

## Dataset
This project uses a subset of the BraTS (Brain Tumor Segmentation Challenge) dataset accessed through MONAI's sample data. This dataset contains multi-modal MRI scans (T1, T1ce, T2, FLAIR) with segmentation masks for tumor regions.

## Google Colab Implementation
The complete implementation is available as a Google Colab notebook:

[BrainSeg3D Colab Notebook](https://colab.research.google.com/drive/13rQii6yytiY6dZ6VMWCnEtiso0hkTuXb?usp=sharing)

## Project Structure
```
BrainSeg3D/
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py              # 3D U-Net implementation
│   ├── data.py               # Data loading and preprocessing
│   ├── train.py              # Training and evaluation functions
│   └── visualize.py          # 3D visualization tools
├── notebooks/                # Jupyter notebooks
│   └── BrainSeg3D_Complete.ipynb  # Complete implementation notebook
├── assets/                   # Example outputs and visualizations
│   ├── segmentation_3d.png   # 3D segmentation visualization
│   └── model_architecture.png # U-Net architecture diagram
└── requirements.txt          # Required dependencies
```

## 3D U-Net Architecture
The implemented 3D U-Net architecture includes:
- 3D convolutions for volumetric processing
- Skip connections between encoder and decoder
- Multiple resolution scales to capture different feature sizes
- Instance normalization for stable training

## Results
The model achieves segmentation of brain tumors into:
- Background (label 0)
- Tumor core (label 1)
- Enhancing tumor (label 2)

Evaluation metrics include:
- Dice similarity coefficient
- Hausdorff distance
- Sensitivity, specificity, and precision

## Setup and Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook in Google Colab
# or
# Import the model from src/
from brainseg3d.model import UNet3D

# Initialize model
model = UNet3D(in_channels=4, out_channels=3)
```

## Applications to fMRI Analysis
The 3D spatial modeling techniques developed in this project provide a foundation for spatiotemporal analysis in fMRI:
- Structural segmentation as anatomical references for functional data
- Feature extraction from volumetric brain regions
- Transfer learning from structural to functional analysis