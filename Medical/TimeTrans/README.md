# TimeTrans: Transformer Architecture for fMRI Temporal Pattern Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Nilearn](https://img.shields.io/badge/Nilearn-0.9.0%2B-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-yellow)

## Overview

TimeTrans is a specialized transformer architecture designed for analyzing temporal patterns in functional Magnetic Resonance Imaging (fMRI) data. By leveraging the power of self-attention mechanisms, this project enables:

- **Temporal pattern recognition** in brain activity sequences
- **Cognitive state decoding** from fMRI time series
- **Interpretable analysis** through attention visualization
- **Identification of key timepoints** in neural dynamics

Traditional approaches to fMRI analysis often treat each timepoint independently or use recurrent models like LSTMs that can struggle with long-range dependencies. TimeTrans addresses these limitations by using transformer-based attention mechanisms that can directly model relationships between any timepoints in an fMRI sequence, regardless of their temporal distance.

## Dataset

This project uses the Localizer fMRI dataset, available through the Nilearn library. The dataset features:

- Task-based fMRI data with multiple cognitive conditions
- TR (repetition time) of 2.4 seconds
- Various task conditions including calculation, sentences, button presses, and more
- Whole-brain functional images with 3D volumes over time

### Preprocessing Pipeline

The raw fMRI data undergoes several preprocessing steps:
1. **Brain masking** using a focused mask based on signal intensity
2. **Standardization** (Z-scoring) to normalize signal intensity across voxels
3. **Detrending** to remove linear signal drifts
4. **Frequency filtering** with high-pass (0.01 Hz) and low-pass (0.1 Hz) to focus on task-relevant frequencies
5. **Dimensionality reduction** via PCA to 50 components, preserving most of the variance while making training computationally feasible

Each timepoint (TR) is labeled according to the cognitive task being performed at that time. To create meaningful sequences for temporal analysis, a sliding window approach with a window size of 10-15 TRs is used, allowing the model to learn patterns across time.

## Technical Approach

### Architecture

TimeTrans implements a specialized transformer architecture for fMRI time series:

<details>
<summary><b>Architecture Details (Click to expand)</b></summary>

#### Core Components

1. **Input Projection**
   - Linear projection from PCA components (50) to model dimension (64)
   - Adapts the reduced fMRI features to the transformer's working dimension

2. **Positional Encoding**
   - Both fixed sinusoidal and learnable implementations
   - Fixed encoding uses sine/cosine functions at different frequencies
   - Learnable encoding uses parameters initialized with Kaiming normal distribution
   - Encodes the temporal position of each TR in the sequence
   - Critical for the model to understand temporal ordering

3. **Transformer Encoder Layers**
   - Multi-head self-attention (4 heads) to capture relationships between timepoints
   - Layer normalization applied before attention for training stability
   - Feedforward networks with dimension 128 and ReLU activation
   - Dropout of 0.2 for regularization
   - Residual connections to prevent gradient degradation

4. **Classification Head**
   - Global average pooling across time dimension
   - Two-layer MLP (64→32→num_classes) for final classification of cognitive states
   - Dropout of 0.2 between layers
</details>

### Self-Attention for Temporal Dynamics

The key innovation of TimeTrans is its use of self-attention to model temporal relationships in fMRI data:

- Each timepoint can attend to any other timepoint in the sequence
- Attention weights reveal which temporal relationships are most informative
- Multiple attention heads capture different types of temporal patterns
- Complex temporal dynamics can be modeled without assumptions about their structure

### Comparison with LSTM Baseline

The project includes an LSTM baseline for comparison, which represents the traditional approach to sequence modeling. The transformer architecture offers several advantages:

- Better handling of long-range dependencies in temporal sequences
- Parallel computation for faster training
- More interpretable attention patterns
- No recurrent state bottleneck

### Training Process

- Batch size of 32 (or smaller based on dataset size)
- Adam optimizer with learning rate 0.001
- Cross-entropy loss for multi-class classification
- 20 training epochs
- Train/validation/test split with stratified sampling to maintain class distribution

## Results and Visualizations

The model successfully decodes cognitive states from fMRI temporal patterns:

- Classification accuracy of approximately 70-80% (significantly above chance level)
- Attention patterns reveal which timepoints are most informative for different cognitive tasks
- Visualization tools provide insight into how the model makes decisions

### Attention Visualization

TimeTrans includes comprehensive tools for visualizing attention patterns:

- **Attention Heatmaps**: Displaying the full attention matrix between all timepoints
- **Temporal Profiles**: Average attention weights across time to identify key moments
- **Comparative Analysis**: Side-by-side visualization of input signals and attention patterns
- **Peak Detection**: Automatic identification of high-attention regions, including statistical thresholding (mean + std)
- **Classification Insights**: Comparison between correctly and incorrectly classified samples to understand model behavior

These visualizations not only help understand model behavior but also provide neuroscientific insights about which moments in brain activity are most distinctive for different cognitive processes.

## Implementation Details

### Libraries and Frameworks

- **PyTorch**: Deep learning framework for model implementation
- **Nilearn**: Neuroimaging library for fMRI data handling
- **NumPy/Pandas**: Data processing and manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Data preprocessing and evaluation metrics

## Usage Examples

### Training the Model

```python
# Initialize model
transformer = TimeSeriesTransformer(
    input_dim=input_dim,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.2,
    num_classes=num_classes
)

# Train
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(transformer, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(transformer, val_loader, criterion)
```

### Visualizing Attention

```python
# Get attention weights for a sample
sample = X_test_seq[sample_idx].unsqueeze(0)
outputs, attn_weights = model(sample)

# Visualize attention
visualize_attention(transformer, X_test_seq, y_test_seq, sample_idx, class_name)
```

## Code Organization

The project is organized into several logical blocks:

1. **Setup and Imports**: Installation of dependencies and basic configuration
2. **Data Loading**: Downloading and loading the Localizer dataset
3. **Data Exploration**: Visualization and analysis of fMRI data and task events
4. **Preprocessing**: Brain masking, standardization, PCA, and sequence creation
5. **Positional Encoding**: Implementation of fixed and learnable position embeddings
6. **Transformer Implementation**: Core self-attention and encoder layer modules
7. **Full Model**: Complete TimeTrans architecture with classification head
8. **LSTM Baseline**: Alternative LSTM implementation for comparison
9. **Training**: Data preparation, training loop, and evaluation
10. **Visualization**: Attention visualization and result interpretation

## Future Work

Potential extensions to this project include:

- Application to resting-state fMRI for functional connectivity analysis
- Cross-subject transfer learning for generalized cognitive state decoding
- Integration with other neuroimaging modalities (EEG, MEG)
- Extended transformer architectures like performers or linformers for longer sequences
- Incorporation of anatomical priors through graph-based transformers

## Acknowledgments

This project builds upon several open-source libraries and resources:

- [Nilearn](https://nilearn.github.io/) for neuroimaging data handling
- [PyTorch](https://pytorch.org/) for deep learning implementation
- [Localizer dataset](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_localizer_first_level.html) for fMRI data
- Research on [transformers for time series](https://arxiv.org/abs/2001.08317)
- Studies on [neural decoding from fMRI](https://www.sciencedirect.com/science/article/pii/S1053811920305103)