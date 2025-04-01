# fMRI Explorer

A Python toolkit for loading, preprocessing, and visualizing fMRI data.

## Overview
This project demonstrates fundamental neuroimaging techniques using Nilearn and other Python libraries. It includes preprocessing pipelines, visualization techniques, and functional connectivity analysis.

## Features
- Motion correction and spatial normalization
- Brain activation visualization
- Region-of-interest analysis
- Functional connectivity matrices
- Basic statistical analysis

## Dataset
This project uses data from the OpenNeuro dataset DS002422, including sample preprocessed fMRI scans.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
from fmri_explorer import load_data, preprocess, visualize

# Load sample data
data = load_data('sample.nii.gz')

# Preprocess the data
preprocessed = preprocess(data, motion_correction=True)

# Create visualization
visualize.plot_activation_map(preprocessed)
```

## Project Structure
```
fMRI-Explorer/
├── src/                  # Source code
│   ├── __init__.py
│   ├── load_data.py      # Data loading utilities
│   ├── preprocess.py     # Preprocessing pipeline
│   └── visualize.py      # Visualization functions
├── notebooks/            # Jupyter notebooks with examples
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_visualization.ipynb
└── data/                 # Sample data and data loaders
    └── README.md         # Information about data sources
```

## Results
The preprocessing pipeline improves data quality by reducing motion artifacts and normalizing to standard space, enabling more accurate analysis of brain activation patterns.
