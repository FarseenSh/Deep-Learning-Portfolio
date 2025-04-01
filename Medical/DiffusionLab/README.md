# DiffusionLab

A PyTorch implementation of diffusion models for image generation with a focus on medical imaging applications.

## Overview
This project implements a denoising diffusion probabilistic model (DDPM) for image generation. While the implementation starts with standard image datasets (MNIST), the techniques developed here provide a foundation for application to medical imaging data.

## Features
- Complete implementation of DDPM in PyTorch
- Class-conditional image generation
- Visualization of the diffusion process
- Experimentation with different noise schedules
- Training and sampling pipelines

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
# Training a diffusion model
python src/train.py --dataset mnist --batch_size 64

# Generating images
python src/sample.py --model_path checkpoints/diffusion_model.pth --num_samples 10
```

## Project Structure
```
DiffusionLab/
├── src/                      # Source code
│   ├── __init__.py
│   ├── diffusion.py          # Core diffusion model implementation
│   ├── model.py              # U-Net architecture for noise prediction
│   ├── train.py              # Training script
│   └── sample.py             # Sampling and generation script
├── notebooks/                # Jupyter notebooks with examples
│   ├── 01_diffusion_basics.ipynb
│   ├── 02_training_visualization.ipynb
│   └── 03_conditional_generation.ipynb
└── assets/                   # Generated images and visualizations
    └── diffusion_steps.png   # Visualization of denoising steps
```

## Visualization
The diffusion process gradually transforms noise into structured images. This process can be visualized as a sequence of denoising steps:

[Diffusion steps visualization would be here]

## Applications to Medical Imaging
While this implementation uses standard image datasets for demonstration, the approach has significant potential for medical imaging:

- Generating synthetic medical images for training data augmentation
- Conditional generation based on clinical parameters
- Anomaly detection through reconstruction error

## Future Work
- Extension to 3D volumetric data for medical applications
- Integration with segmentation tasks
- Implementation of classifier-free guidance
- Application to specific medical imaging modalities
