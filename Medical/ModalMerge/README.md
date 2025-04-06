# ModalMerge: Cross-Modal Learning Framework

A PyTorch-based framework for multi-modal learning that enables conditioning one data modality on another through cross-attention mechanisms. This project focuses on medical imaging applications, particularly for fMRI data analysis.

## Project Overview

ModalMerge addresses the challenge of relating different data modalities in a principled way. By leveraging cross-attention mechanisms, the framework allows information from one modality (e.g., structural brain images) to guide the generation or analysis of another modality (e.g., functional brain activity).

This approach has significant applications in medical imaging, particularly neuroimaging, where multiple complementary modalities are often available (structural MRI, functional MRI, EEG, clinical data, etc.).

## Technical Approach

### Architecture

The framework employs a modular architecture consisting of:

1. **Encoders**: Modality-specific encoders that extract features from different data types:
   - Image encoders using CNN backbones (ResNet)
   - Time series encoders using LSTM or Transformer architectures
   - Text encoders with embedding layers and RNNs
   - Tabular data encoders with fully-connected networks

2. **Cross-Attention Module**: The core component that enables conditioning between modalities:
   - Multi-head attention mechanism
   - Ability to focus on relevant parts of the conditioning modality
   - Visualization of attention patterns

3. **Decoders**: Modality-specific decoders that generate data from attended embeddings:
   - Image decoders using transposed convolutions
   - Time series decoders using LSTM or Transformer architectures
   - Text decoders with language modeling capabilities
   - Tabular data decoders with fully-connected networks

### Key Features

- **Multi-Modal Support**: Works with images, time series, text, and tabular data
- **Flexible Conditioning**: Any supported modality can condition any other modality
- **Medical Imaging Focus**: Specific adaptations for medical imaging data
- **Attention Visualization**: Tools to understand cross-modal relationships
- **Modular Design**: Easily extendable to new modalities and architectures

## Applications

The framework demonstrates several applications:

### 1. Image to Time Series

Converting visual information to temporal patterns - useful for:
- Predicting temporal dynamics from static images
- Generating physiological signals from medical scans
- Forecasting time-evolving properties based on initial conditions

### 2. Time Series to Image

Generating visual representations from temporal data - useful for:
- Visualizing patterns in complex time series data
- Creating visual summaries of temporal information
- Mapping functional data to anatomical structures

### 3. Brain Structure to Function Mapping

A specialized medical application demonstrating:
- Mapping structural brain imaging to functional activity
- Predicting fMRI signals from anatomical MRI
- Region-of-interest (ROI) based functional analysis

## Datasets

The framework can work with various datasets, including:

### Medical Imaging Datasets

- **Brain Imaging Datasets**:
  - Human Connectome Project (HCP)
  - OpenNeuro datasets
  - Brain Tumor Segmentation (BraTS) dataset
  - Alzheimer's Disease Neuroimaging Initiative (ADNI)

- **fMRI-Specific Datasets**:
  - Task-based fMRI datasets
  - Resting-state fMRI collections
  - Multi-site datasets for robustness testing

### Synthetic Data Generation

For development and testing, the framework includes utilities to generate:
- Synthetic brain-like images with controllable features
- Simulated fMRI time series with realistic properties
- Paired data with known relationships between modalities

## Implementation Details

- **Framework**: PyTorch
- **Core Models**: ResNet, LSTM, Transformer
- **Attention Mechanism**: Multi-head attention with customized cross-modal operations
- **Visualization**: Matplotlib-based tools for attention and generation visualization
- **Training**: Modality-specific loss functions with early stopping

## Future Work

### Planned Extensions

1. **Advanced Conditioning Methods**:
   - FiLM (Feature-wise Linear Modulation)
   - AdaIN (Adaptive Instance Normalization)
   - Diffusion model conditioning

2. **Additional Modalities**:
   - 3D volumetric data processing
   - Point cloud representations
   - Graph-structured data

### fMRI-Specific Improvements

1. Integration with standard neuroimaging preprocessing pipelines
2. Support for 4D data (3D volumes over time)
3. Brain parcellation techniques for ROI-based analysis
4. Incorporation of functional connectivity metrics
5. Spatiotemporal attention mechanisms specific to brain data

## Usage Examples

The detailed implementation is provided in the Jupyter notebook. To use the framework:

```python
# Create a model for mapping brain structure to function
model = MultiModalModel(
    condition_type='image',  # Structural MRI
    target_type='time_series',  # fMRI time series
    condition_encoder='resnet50',
    target_decoder='transformer',
    embedding_dim=512,
    num_attention_heads=16
)

# Train the model
trainer = Trainer(model)
trainer.train(train_loader, val_loader, epochs=30)

# Generate functional data from structural image
fmri_prediction = model(structural_mri)

# Visualize attention patterns
Visualizer.plot_attention(model, structural_mri)
```

---

This project demonstrates advanced techniques for multi-modal learning with specific applications to medical imaging and fMRI analysis, relevant to the Emory-BMI-GSoC project "Advancing Brain Decoding and Cognitive Analysis: Leveraging Diffusion Models for Spatiotemporal Pattern Recognition in fMRI Data."
