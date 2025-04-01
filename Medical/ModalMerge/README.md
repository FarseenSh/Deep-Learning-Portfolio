# ModalMerge

A framework for multi-modal learning and conditioning techniques with applications in medical imaging.

## Overview
This project explores methods for conditioning one data modality on another through cross-attention mechanisms. While demonstrated with various data types, the techniques are particularly relevant for integrating different forms of medical imaging and clinical data.

## Features
- Cross-attention mechanisms for multi-modal learning
- Conditioning frameworks for various data types
- Integration of image, time-series, and structured data
- Visualization of cross-modal relationships
- Applications in medical data integration

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
from modalmerge import MultiModalModel, DataLoader

# Load sample multi-modal data
image_data, time_series_data, labels = DataLoader.load_sample_data()

# Create a model that conditions time-series generation on images
model = MultiModalModel(
    condition_type='image',
    target_type='time_series',
    condition_encoder='resnet18',
    target_decoder='transformer'
)

# Train the model
model.train(
    condition_data=image_data,
    target_data=time_series_data,
    epochs=50
)

# Generate time-series from a new image
new_time_series = model.generate(new_image)

# Visualize the attention between modalities
model.visualize_cross_attention(new_image, new_time_series)
```

## Project Structure
```
ModalMerge/
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py              # Multi-modal model implementation
│   ├── attention.py          # Cross-attention mechanisms
│   ├── encoders.py           # Various encoders for different modalities
│   ├── decoders.py           # Various decoders for different modalities
│   └── visualize.py          # Visualization utilities
└── notebooks/                # Jupyter notebooks with examples
    ├── 01_image_to_text.ipynb
    ├── 02_text_to_image.ipynb
    ├── 03_image_to_timeseries.ipynb
    └── 04_multimodal_medical.ipynb
```

## Cross-Modal Attention
The cross-attention mechanism allows the model to focus on relevant parts of the conditioning modality:

[Cross-attention visualization would be here]

## Supported Modality Pairs
The project demonstrates conditioning between various modality pairs:

| Condition Modality | Target Modality   | Use Case Example               |
|--------------------|-------------------|--------------------------------|
| Image              | Text              | Medical image captioning       |
| Text               | Image             | Text-guided image generation   |
| Image              | Time-series       | Predicting signals from scans  |
| Time-series        | Image             | Visualizing signal patterns    |
| Structured data    | Image             | Clinical data to imaging       |

## Applications to Medical Imaging
This project has direct applications to multi-modal medical data analysis:

- Conditioning image generation on clinical parameters
- Integrating structural and functional imaging data
- Combining imaging with genetic or demographic information
- Cross-modal retrieval in medical databases

## Connection to fMRI Analysis
The multi-modal techniques explored here are particularly relevant for fMRI research:

- Integrating anatomical MRI with functional data
- Conditioning brain activity patterns on experimental stimuli
- Combining fMRI with other clinical measurements
- Generating synthetic fMRI data based on structural imaging
