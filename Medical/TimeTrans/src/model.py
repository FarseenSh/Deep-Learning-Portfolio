"""
TimeTrans Model Implementation

This module implements a transformer architecture optimized for time-series data,
with specific modifications for handling temporal dependencies in biomedical signals.
Also includes a comparable LSTM baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Custom positional encoding for time-series data.
    
    This implementation includes options for both standard sinusoidal encoding
    and learnable positional embeddings, with special consideration for
    temporal data characteristics.
    """
    def __init__(self, d_model, max_seq_length=1000, dropout=0.1, learnable=False):
        """
        Initialize the positional encoding layer.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length to pre-compute encodings for
            dropout: Dropout rate
            learnable: Whether to use learnable positional embeddings
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        if learnable:
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
            nn.init.kaiming_normal_(self.pe)
        else:
            # Fixed sinusoidal embeddings
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            # Apply sine to even indices
            pe[:, 0::2] = torch.sin(position * div_term)
            
            # Apply cosine to odd indices
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Add batch dimension and register as buffer (not a parameter)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        self.learnable = learnable
        
    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        # Select the relevant part of positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformerEncoder(nn.Module):
    """
    Transformer encoder block modified for time-series data.
    
    Implements multi-head self-attention with specific adaptations
    for handling temporal dependencies in biomedical time-series.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        Initialize the encoder block.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super(TimeSeriesTransformerEncoder, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward neural network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def forward(self, src, src_mask=None):
        """
        Forward pass through the encoder block.
        
        Args:
            src: Input tensor of shape [batch_size, seq_length, d_model]
            src_mask: Optional mask for source sequence
            
        Returns:
            Output tensor after self-attention and feedforward layers
        """
        # Multi-head self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attn(
            query=src, 
            key=src, 
            value=src, 
            attn_mask=src_mask,
            need_weights=True
        )
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights
        
        # Residual connection and layer normalization
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Feedforward network with residual connection and layer norm
        ff_output = self.feedforward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src


class TimeSeriesTransformer(nn.Module):
    """
    Complete Transformer architecture for time-series analysis.
    
    This model is specifically designed for biomedical time-series data,
    with adaptations for temporal dynamics and specialized positional encoding.
    """
    def __init__(self, 
                 input_dim, 
                 d_model=128, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=512, 
                 dropout=0.1, 
                 num_classes=None,
                 max_seq_length=1000,
                 learnable_pos=False,
                 task_type='classification'):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            num_classes: Number of output classes (for classification)
            max_seq_length: Maximum sequence length
            learnable_pos: Whether to use learnable positional embeddings
            task_type: 'classification' or 'forecasting'
        """
        super(TimeSeriesTransformer, self).__init__()
        
        self.task_type = task_type
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout,
            learnable=learnable_pos
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TimeSeriesTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output head
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
        elif task_type == 'forecasting':
            self.output_head = nn.Linear(d_model, input_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask for padding
            
        Returns:
            Output predictions
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.layers:
            x = layer(x, mask)
            attention_weights.append(layer.attention_weights)
            
        # For classification tasks, use the encoding of the last token
        if self.task_type == 'classification':
            # Global average pooling over sequence length
            x = torch.mean(x, dim=1)
            
        # Apply output head
        output = self.output_head(x)
        
        return output, attention_weights
        
    def get_attention_weights(self, x, mask=None):
        """
        Retrieve attention weights for visualization.
        
        Args:
            x: Input tensor
            mask: Optional mask
            
        Returns:
            List of attention weight matrices
        """
        # Forward pass
        _, attention_weights = self.forward(x, mask)
        return attention_weights


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for comparison with the transformer architecture.
    
    Implements both unidirectional and bidirectional LSTM with optional
    attention mechanism for sequence classification or forecasting.
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim=128, 
                 num_layers=2, 
                 dropout=0.1, 
                 bidirectional=True, 
                 num_classes=None,
                 use_attention=False,
                 task_type='classification'):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of output classes (for classification)
            use_attention: Whether to use attention mechanism
            task_type: 'classification' or 'forecasting'
        """
        super(LSTMBaseline, self).__init__()
        
        self.task_type = task_type
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # Output head
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task_type == 'forecasting':
            self.output_head = nn.Linear(hidden_dim * self.num_directions, input_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
    def attention_net(self, lstm_output):
        """
        Apply attention mechanism to LSTM outputs.
        
        Args:
            lstm_output: Output tensor from LSTM of shape [batch_size, seq_length, hidden_dim*num_directions]
            
        Returns:
            Context vector and attention weights
        """
        # Calculate attention scores
        attn_weights = self.attention(lstm_output).squeeze(-1)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to LSTM outputs
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, soft_attn_weights
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Output predictions and attention weights (if applicable)
        """
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            if self.task_type == 'classification':
                context, attention_weights = self.attention_net(output)
                output = self.output_head(context)
            else:  # forecasting
                # Apply attention for each time step
                seq_len = output.size(1)
                forecasts = []
                all_attention_weights = []
                
                for i in range(seq_len):
                    context, attn_weights = self.attention_net(output[:, :i+1, :])
                    forecast = self.output_head(context)
                    forecasts.append(forecast)
                    all_attention_weights.append(attn_weights)
                    
                output = torch.stack(forecasts, dim=1)
                attention_weights = all_attention_weights
        else:
            if self.task_type == 'classification':
                # Use the last hidden state
                if self.bidirectional:
                    # Concatenate forward and backward final hidden states
                    hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden_final = hidden[-1, :, :]
                    
                output = self.output_head(hidden_final)
            else:  # forecasting
                # Apply output head to each time step
                batch_size, seq_len, _ = output.shape
                output = self.output_head(output.reshape(-1, self.hidden_dim * self.num_directions))
                output = output.reshape(batch_size, seq_len, -1)
        
        return output, attention_weights
        
    def get_attention_weights(self, x):
        """
        Retrieve attention weights for visualization (if attention is enabled).
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights or None
        """
        if not self.use_attention:
            return None
            
        # Forward pass
        _, attention_weights = self.forward(x)
        return attention_weights


def create_model(model_type, config):
    """
    Factory function to create a model based on configuration.
    
    Args:
        model_type: 'transformer' or 'lstm'
        config: Dictionary with model configuration
        
    Returns:
        Instantiated model
    """
    if model_type == 'transformer':
        return TimeSeriesTransformer(**config)
    elif model_type == 'lstm':
        return LSTMBaseline(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
