"""
Neural network architectures for trading intelligence
Production-ready PyTorch models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TradingLSTM(nn.Module):
    """
    LSTM network for time series prediction in trading
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(TradingLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.linear = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        output = self.dropout(last_output)

        # Linear transformation
        output = self.linear(output)

        return output

class TradingTransformer(nn.Module):
    """
    Transformer network for trading prediction
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1, max_seq_length: int = 512):
        super(TradingTransformer, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        self.input_projection.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        self.output_projection.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Apply dropout
        x = self.dropout(x)

        # Output projection
        output = self.output_projection(x)

        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model: int, max_length: int = 512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class TradingCNN(nn.Module):
    """
    Convolutional neural network for trading patterns
    """

    def __init__(self, input_size: int, num_filters: int = 64, kernel_sizes: Tuple[int, ...] = (3, 5, 7)):
        super(TradingCNN, self).__init__()

        self.input_size = input_size
        self.num_filters = num_filters

        # Convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Output layers
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 64)
        self.output = nn.Linear(64, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            conv.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Transpose for conv1d: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)

        # Apply convolutions
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            pooled = self.global_max_pool(conv_out).squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)

        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        output = self.output(x)

        return output