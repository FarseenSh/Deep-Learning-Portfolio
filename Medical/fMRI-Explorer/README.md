# fMRI Explorer

A comprehensive Python toolkit for analyzing functional Magnetic Resonance Imaging (fMRI) data.

## Overview
This project provides a complete pipeline for neuroimaging analysis using industry-standard tools. It implements a systematic workflow from raw data processing to advanced statistical analysis.

## Features
- Data loading and exploration of neuroimaging datasets
- Comprehensive preprocessing pipeline with Nilearn
- Multiple visualization techniques for brain activity
- Region-of-interest (ROI) analysis with anatomical atlases
- Functional connectivity analysis with correlation matrices
- Statistical modeling using General Linear Model (GLM)
- Advanced surface-based visualization

## Dataset
This project uses the ADHD-200 dataset, a publicly available collection of resting-state fMRI data from the 1000 Functional Connectomes Project. It also utilizes the standard MNI152 template for anatomical reference.

## Dependencies
- nibabel - For reading neuroimaging file formats
- nilearn - Specialized library for neuroimaging analysis
- matplotlib - For visualization
- pandas - For data manipulation
- scikit-learn - For statistical tools and models

## Notebook Contents
The included Jupyter notebook demonstrates:

1. **Data Loading**: Import and initial exploration of fMRI data
2. **Raw Data Visualization**: Basic orthogonal views of the brain
3. **Preprocessing**: Signal cleaning, normalization, and filtering
4. **Brain Visualization**: Multiple methods to visualize brain structure and function
5. **ROI Analysis**: Extraction of time series from anatomical regions
6. **Functional Connectivity**: Analysis of brain network organization
7. **Statistical Analysis**: Application of the General Linear Model
8. **Surface Mapping**: Advanced visualization on cortical surfaces

## How to Use
1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open the Jupyter notebook: `jupyter notebook fMRI_Explorer.ipynb`
4. Run the cells to see the complete analysis pipeline

## Colab Link:
https://colab.research.google.com/drive/1PiWaW7KFMUK_50U6DYb8qjMio6qfk32S?usp=sharing

## Results
The project demonstrates how preprocessing improves data quality by reducing noise and artifacts, allowing for meaningful analysis of brain activation patterns and functional connectivity networks.
