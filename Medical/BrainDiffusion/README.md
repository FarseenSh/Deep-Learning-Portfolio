# BrainDiffusion

A state-of-the-art implementation of 3D diffusion models for brain MRI generation using PyTorch and TorchIO.

## Overview
BrainDiffusion implements cutting-edge diffusion probabilistic models specifically for volumetric brain imaging data. The project demonstrates how to apply denoising diffusion models to 3D MRI volumes, enabling the generation of high-quality synthetic brain scans with applications in medical imaging research, data augmentation, and brain decoding.

## Features
- Complete 3D diffusion model implementation for volumetric brain data
- Memory-efficient 3D U-Net architecture with gradient checkpointing
- Patch-based training for handling large 3D volumes
- Optimized 3D attention mechanisms
- Interactive 3D visualization using Plotly
- Support for brain region conditioning
- DDIM sampling for faster generation
- Comprehensive evaluation metrics for 3D volumes

## Dataset
This project uses the IXI (Information eXtraction from Images) dataset, containing 582 T1-weighted brain MRI scans in NIfTI (.nii.gz) format. The dataset provides high-quality 3D brain volumes for training and evaluating diffusion models.

## Setup
```bash
# Install required dependencies
pip install -r requirements.txt

# For Google Colab, mount your drive with the IXI dataset
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
```python
# Train the diffusion model
python src/train.py --data_dir /path/to/IXI-T1 --batch_size 2 --epochs 100

# Generate brain volumes
python src/generate.py --model_path checkpoints/best_model.pt --num_samples 5

# Interactive visualization
python src/visualize.py --volume_path generated/sample_001.nii.gz
```

## Project Structure
```
BrainDiffusion/
├── src/                      # Source code
│   ├── __init__.py
│   ├── diffusion.py          # Core diffusion model implementation
│   ├── unet3d.py             # 3D U-Net architecture
│   ├── dataset.py            # TorchIO dataset and loaders
│   ├── train.py              # Training script
│   ├── generate.py           # Generation script
│   ├── visualize.py          # 3D visualization utilities
│   └── evaluate.py           # Evaluation metrics
├── notebooks/                # Jupyter notebooks with examples
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_diffusion_training.ipynb
│   ├── 03_3d_visualization.ipynb
│   └── 04_brain_generation.ipynb
└── examples/                 # Example outputs and visualizations
    ├── real_vs_generated.png
    ├── diffusion_process.png
    └── brain_regions.png
```

## 3D Visualization
BrainDiffusion includes interactive 3D visualization tools using Plotly that allow for:
- Volumetric rendering of brain MRI scans
- Interactive exploration of generated brain volumes
- Visualization of the diffusion process from noise to brain
- Comparative visualization of real vs. generated samples

## Applications to Neuroimaging
- Data augmentation for medical imaging models
- Anonymized synthetic dataset generation
- Brain decoding and cognitive state modeling
- Anomaly detection in neuroimaging
- Conditional generation of specific brain structures

## Colab Notebook
[Open in Google Colab](https://colab.research.google.com/drive/your-notebook-link-here)

## Memory Optimization Techniques
BrainDiffusion implements several memory optimization techniques essential for working with 3D volumes:
- Patch-based sampling with TorchIO
- Gradient checkpointing in ResNet blocks
- Mixed precision training
- Memory-efficient attention mechanisms
- DDIM sampling for faster inference

## Implementation Highlights
- Attention mechanisms adapted for 3D data
- FiLM conditioning for brain region control
- Custom 3D positional embeddings
- Classifier-free guidance for controlled generation
- Comprehensive brain-specific evaluation metrics

## Future Work
- Extension to 4D spatiotemporal fMRI data
- Integration of anatomical priors for improved generation
- Multi-modal brain imaging synthesis
- Fine-grained control of generated brain regions
- Application to pathological brain condition modeling
