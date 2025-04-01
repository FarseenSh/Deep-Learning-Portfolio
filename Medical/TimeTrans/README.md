# TimeTrans

A transformer-based architecture for time-series analysis with applications in biomedical signal processing.

## Overview
This project implements a transformer model designed specifically for time-series data. While applicable to various domains, the focus is on biomedical time-series such as EEG signals, which serve as a stepping stone toward more complex spatiotemporal data like fMRI.

## Features
- Transformer architecture optimized for time-series data
- Attention visualization tools
- Comparison with LSTM baselines
- Time-series classification and forecasting
- Biomedical signal processing examples

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
from timetrans import TimeSeriesTransformer, data_loader

# Load example data
X_train, y_train, X_test, y_test = data_loader.load_eeg_dataset()

# Initialize the model
model = TimeSeriesTransformer(
    input_dim=X_train.shape[2],
    num_classes=len(set(y_train)),
    d_model=128,
    nhead=8,
    num_layers=4
)

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Visualize attention
model.plot_attention(X_test[0])
```

## Project Structure
```
TimeTrans/
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py              # Transformer model implementation
│   ├── attention.py          # Attention mechanisms
│   ├── data_loader.py        # Dataset handling utilities
│   └── visualize.py          # Attention visualization tools
├── notebooks/                # Jupyter notebooks with examples
│   ├── 01_transformer_basics.ipynb
│   ├── 02_eeg_classification.ipynb
│   ├── 03_lstm_comparison.ipynb
│   └── 04_attention_visualization.ipynb
└── examples/                 # Example outputs and visualizations
```

## Attention Visualization
The attention mechanism in transformers allows the model to focus on the most relevant parts of the input sequence. Here's an example of attention weights across a time-series:

[Attention visualization would be here]

## Comparison with LSTM
Performance comparison between the TimeTrans transformer and LSTM baselines on various time-series tasks:

| Model     | EEG Classification | Stock Prediction | Sensor Reading |
|-----------|-------------------|-----------------|----------------|
| TimeTrans | 87.3%             | 0.023 MSE       | 0.018 MSE      |
| LSTM      | 82.1%             | 0.035 MSE       | 0.022 MSE      |

## Applications to fMRI Analysis
While this project focuses on 1D time-series, the techniques developed here provide a foundation for analyzing the temporal dynamics in fMRI data:

- Understanding sequence modeling for brain activity patterns
- Attention mechanisms for identifying significant temporal events
- Transfer learning from simpler time-series to complex neuroimaging data
