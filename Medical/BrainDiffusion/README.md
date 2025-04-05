# BrainDiffusion

A state-of-the-art implementation of 3D diffusion models for brain MRI generation.

## Overview
This project implements cutting-edge diffusion probabilistic models specifically for 3D brain imaging data. It demonstrates how modern generative AI techniques can be applied to neuroimaging to create realistic brain MRI volumes and potentially be extended to functional MRI (fMRI) analysis.

## Features
- Full 3D diffusion model implementation for volumetric brain data
- Memory-efficient architecture with gradient checkpointing
- Patch-based training for handling large 3D volumes
- Brain region conditioning capabilities
- Interactive 3D visualization of brain volumes
- Efficient sampling using DDIM for faster inference
- Quantitative evaluation metrics for generated volumes

## Dataset
This project uses the IXI Dataset (Information eXtraction from Images), specifically the T1-weighted MRI scans of 582 subjects in .nii.gz format. The IXI dataset is a collection of nearly 600 MR images from normal, healthy subjects that provides a valuable resource for developing and testing medical imaging algorithms.

## Dependencies
- PyTorch - Deep learning framework
- TorchIO - Specialized library for medical imaging in PyTorch
- Plotly - For interactive 3D visualization
- nibabel - For reading neuroimaging file formats
- einops - For tensor manipulation

## Notebook Contents
The included Jupyter notebook demonstrates:

1. **Environment Setup**: Configuration and library installation
2. **Data Loading**: Importing and exploration of IXI brain MRI data with TorchIO
3. **Preprocessing Pipeline**: Normalization and preparation of 3D volumes
4. **3D Visualization**: Interactive exploration of brain volumes using Plotly
5. **Diffusion Model Implementation**: Complete implementation of 3D diffusion model components
6. **U-Net Architecture**: Memory-efficient 3D U-Net with attention mechanisms
7. **Training Process**: Patch-based training with optimization techniques
8. **Sampling and Generation**: DDIM sampling for efficient brain volume generation
9. **Visualization of Results**: Comparison between real and generated brain volumes
10. **Evaluation**: Quantitative assessment of generation quality

## How to Use
1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open the Jupyter notebook: `jupyter notebook BrainDiffusion.ipynb` or use the Colab link
4. Run the cells to see the complete implementation

## Colab Link
https://colab.research.google.com/drive/1IlneCfzyJeozj1bEplJT9AH1subhMAUp?usp=sharing

## Results
The project demonstrates the ability to generate realistic 3D brain volumes using diffusion models. The implementation includes memory optimization techniques critical for working with volumetric data, providing a foundation for applications in data augmentation, anomaly detection, and potentially extending to full 4D spatiotemporal fMRI data.
