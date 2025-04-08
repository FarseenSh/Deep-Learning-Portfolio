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
- Various task types including calculation, sentence reading, and button presses
- Multiple subjects with whole-brain functional images

### Preprocessing Pipeline

The raw fMRI data undergoes several preprocessing steps:
1. **Brain masking** to focus analysis on brain voxels only
2. **Standardization** to normalize signal intensity
3. **Detrending** to remove linear signal drifts
4. **Frequency filtering** (0.01-0.1 Hz) to focus on task-relevant frequencies
5. **Dimensionality reduction** via PCA to make training computationally feasible

Each timepoint (TR) is labeled according to the cognitive task being performed, creating a sequence labeling problem where the model must learn to recognize temporal patterns associated with different cognitive states.

## Technical Approach

### Architecture

TimeTrans implements a specialized transformer architecture for fMRI time series:

<details>
<summary><b>Architecture Details (Click to expand)</b></summary>

#### Core Components

1. **Input Projection**
   - Linear projection from PCA components to model dimension
   - Adapts the reduced fMRI features to the transformer's working dimension

2. **Positional Encoding**
   - Both fixed sinusoidal and learnable implementations
   - Encodes the temporal position of each TR in the sequence
   - Critical for the model to understand temporal ordering

3. **Transformer Encoder Layers**
   - Multi-head self-attention to capture relationships between timepoints
   - Layer normalization for training stability
   - Feedforward networks with ReLU activation
   - Residual connections to prevent gradient degradation

4. **Classification Head**
   - Global average pooling across time dimension
   - MLP for final classification of cognitive states
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

## Results and Visualizations

The model successfully decodes cognitive states from fMRI temporal patterns:

- Classification accuracy exceeds traditional approaches
- Attention patterns reveal which timepoints are most informative for different cognitive tasks
- Visualization tools provide insight into how the model makes decisions

### Attention Visualization

TimeTrans includes comprehensive tools for visualizing attention patterns:

- Heatmaps showing relationships between timepoints
- Temporal attention profiles highlighting key moments in the fMRI sequence
- Comparison between correctly and incorrectly classified samples
- Identification of high-attention regions that drive classification decisions

These visualizations not only help understand model behavior but also provide neuroscientific insights about which moments in brain activity are most distinctive for different cognitive processes.

## Implementation Details

### Libraries and Frameworks

- **PyTorch**: Deep learning framework for model implementation
- **Nilearn**: Neuroimaging library for fMRI data handling
- **NumPy/Pandas**: Data processing and manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Data preprocessing and evaluation metrics

### Training Process

- Dimensionality reduction via PCA
- Sequence creation with sliding windows
- Stratified train/validation/test splitting
- Cross-entropy loss optimization
- Adam optimizer with learning rate 0.001
- 20 training epochs

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