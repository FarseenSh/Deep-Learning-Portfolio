# ModalMerge Examples and Results

This directory contains example outputs and visualizations from the ModalMerge framework, demonstrating its capabilities in multi-modal learning and conditioning.

## Example Visualizations

### Cross-Attention Visualizations

These visualizations show how the model attends to different parts of the conditioning modality when generating the target modality:

- Attention maps between image regions and time series points
- Attention entropy analysis showing focus vs. dispersion
- Multi-head attention visualization

### Image to Time Series Generation

Examples of generating time series data from images:

- Synthetic image to time series pairs
- Attention-guided generation process
- Comparison of generated vs. ground truth time series

### Brain Structure to Function Mapping

Medical imaging specific examples:

- Structural MRI to functional time series mapping
- Region-of-interest (ROI) based analysis
- Attention patterns on brain regions

## Evaluation Results

This section includes quantitative evaluation results:

- MSE, MAE for time series reconstruction
- PSNR, SSIM for image reconstruction
- Dynamic Time Warping (DTW) for time series similarity
- Model training curves

## Usage Notes

The example outputs provided here are generated from the Jupyter notebooks in the `notebooks/` directory. To reproduce these results:

1. Run the corresponding notebook
2. Follow the visualization sections
3. Adjust parameters as needed for your specific use case

## Citation

If you use these examples in your work, please cite:

```
@misc{modalmerge2025,
  author = {Farseen Shahanas},
  title = {ModalMerge: A Framework for Multi-Modal Learning and Conditioning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/FarseenSh/Deep-Learning-Portfolio}}
}
```
