"""
Transformer Sequence Embedder for Trader-AI

Provides sequence embedding for consensus checking and uncertainty estimation.
NOT for direct price prediction - used as a disagreement signal with TimesFM.

Based on: https://github.com/lj-valencia/TrasnformerTimeSeries
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback embedder")


@dataclass
class EmbeddingResult:
    """Result of sequence embedding"""
    embedding: np.ndarray
    uncertainty_proxy: float
    confidence: float
    metadata: Dict


class PositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Add positional encoding to input"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Simple transformer encoder for sequence embedding"""

    def __init__(self, input_dim: int = 5, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model)
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        return x


class TransformerEmbedder:
    """
    Transformer for sequence embedding (NOT direct price prediction)

    Usage:
    1. Embed OHLCV sequences to fixed-size vectors
    2. Compare embeddings across time for regime detection
    3. Estimate uncertainty via embedding variance
    4. Consensus check: compare with TimesFM predictions
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 input_dim: int = 5, use_fallback: bool = True):
        """
        Initialize transformer embedder

        Args:
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            input_dim: Input dimension (OHLCV = 5)
            use_fallback: Use fallback if PyTorch unavailable
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.use_fallback = use_fallback

        self.model = None
        self.is_initialized = False

        if TORCH_AVAILABLE:
            self._initialize_model()
        elif use_fallback:
            logger.info("Using fallback PCA-based embedder")
            self.is_initialized = True
        else:
            raise RuntimeError("PyTorch not available and fallback disabled")

    def _initialize_model(self):
        """Initialize the transformer model"""
        try:
            self.model = TransformerEncoder(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers
            )
            self.model.eval()
            self.is_initialized = True
            logger.info(f"Transformer embedder initialized: d_model={self.d_model}, layers={self.num_layers}")
        except Exception as e:
            logger.error(f"Failed to initialize transformer: {e}")
            if self.use_fallback:
                self.is_initialized = True

    def embed_sequence(self, ohlcv_window: np.ndarray) -> np.ndarray:
        """
        Convert price sequence to embedding vector

        Args:
            ohlcv_window: OHLCV data of shape (seq_len, 5) or (seq_len,) for close only

        Returns:
            Embedding vector of shape (d_model,)
        """
        if not self.is_initialized:
            raise RuntimeError("Embedder not initialized")

        # Handle different input shapes
        if ohlcv_window.ndim == 1:
            # Close prices only - expand to pseudo-OHLCV
            ohlcv_window = np.column_stack([
                ohlcv_window,  # Open
                ohlcv_window * 1.001,  # High
                ohlcv_window * 0.999,  # Low
                ohlcv_window,  # Close
                np.ones_like(ohlcv_window)  # Volume (normalized)
            ])

        if TORCH_AVAILABLE and self.model is not None:
            return self._embed_with_transformer(ohlcv_window)
        else:
            return self._embed_fallback(ohlcv_window)

    def _embed_with_transformer(self, ohlcv: np.ndarray) -> np.ndarray:
        """Embed using transformer model"""
        # Normalize input
        normalized = self._normalize_ohlcv(ohlcv)

        # Convert to tensor
        x = torch.FloatTensor(normalized).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            encoded = self.model(x)
            # Use mean pooling over sequence
            embedding = encoded.mean(dim=1).squeeze().numpy()

        return embedding

    def _embed_fallback(self, ohlcv: np.ndarray) -> np.ndarray:
        """Fallback embedding using statistical features"""
        # Normalize
        normalized = self._normalize_ohlcv(ohlcv)

        # Extract statistical features
        features = []

        for col in range(normalized.shape[1]):
            series = normalized[:, col]
            features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                series[-1] - series[0],  # Trend
                np.mean(np.abs(np.diff(series))),  # Volatility
            ])

        # Pad/truncate to d_model size
        features = np.array(features)
        if len(features) < self.d_model:
            features = np.pad(features, (0, self.d_model - len(features)))
        else:
            features = features[:self.d_model]

        return features

    def _normalize_ohlcv(self, ohlcv: np.ndarray) -> np.ndarray:
        """Normalize OHLCV data"""
        # Use first close as reference
        ref_price = ohlcv[0, 3] if ohlcv[0, 3] != 0 else 1.0

        normalized = ohlcv.copy().astype(np.float32)
        normalized[:, :4] = normalized[:, :4] / ref_price - 1.0  # Normalize prices

        # Normalize volume (if present)
        if normalized.shape[1] > 4:
            vol_mean = np.mean(normalized[:, 4])
            if vol_mean > 0:
                normalized[:, 4] = normalized[:, 4] / vol_mean - 1.0

        return normalized

    def get_uncertainty_proxy(self, embedding: np.ndarray) -> float:
        """
        Estimate forecast uncertainty from embedding variance

        Higher variance in embedding = more uncertain/volatile market

        Args:
            embedding: Embedding vector

        Returns:
            Uncertainty proxy (0-1 scale)
        """
        # Variance-based uncertainty
        variance = np.var(embedding)

        # Normalize to 0-1 (empirically calibrated)
        uncertainty = np.tanh(variance * 10)

        return float(uncertainty)

    def get_embedding_result(self, ohlcv_window: np.ndarray) -> EmbeddingResult:
        """
        Get complete embedding result with metadata

        Args:
            ohlcv_window: OHLCV data

        Returns:
            EmbeddingResult with embedding, uncertainty, and confidence
        """
        embedding = self.embed_sequence(ohlcv_window)
        uncertainty = self.get_uncertainty_proxy(embedding)

        # Confidence inversely related to uncertainty
        confidence = 1.0 - uncertainty

        return EmbeddingResult(
            embedding=embedding,
            uncertainty_proxy=uncertainty,
            confidence=confidence,
            metadata={
                'd_model': self.d_model,
                'seq_len': len(ohlcv_window),
                'method': 'transformer' if (TORCH_AVAILABLE and self.model) else 'fallback'
            }
        )

    def calculate_disagreement(self,
                              transformer_pred: float,
                              timesfm_pred: float) -> float:
        """
        Calculate disagreement between Transformer and TimesFM predictions

        High disagreement = reduce position size

        Args:
            transformer_pred: Transformer-based prediction
            timesfm_pred: TimesFM prediction

        Returns:
            Disagreement score (0-1)
        """
        # Absolute difference normalized
        diff = abs(transformer_pred - timesfm_pred)

        # Normalize (assuming predictions are % changes in reasonable range)
        disagreement = np.tanh(diff * 10)

        return float(disagreement)

    def embed_for_comparison(self, ohlcv_window: np.ndarray) -> Dict:
        """
        Embed sequence and prepare for comparison with TimesFM

        Returns dict with embedding and simple forecast
        """
        result = self.get_embedding_result(ohlcv_window)

        # Simple forecast: trend from embedding
        # Use first and last few elements of embedding as trend proxy
        embedding = result.embedding
        trend = np.mean(embedding[-10:]) - np.mean(embedding[:10])

        # Normalize to expected return scale
        forecast = np.tanh(trend) * 0.05  # Max +/- 5% forecast

        return {
            'embedding': embedding,
            'forecast': forecast,
            'uncertainty': result.uncertainty_proxy,
            'confidence': result.confidence
        }


def generate_transformer_features(ohlcv_df, timesfm_forecast: float = 0.0) -> Dict[str, float]:
    """
    Generate transformer-based features for the feature vector

    Args:
        ohlcv_df: OHLCV DataFrame
        timesfm_forecast: TimesFM prediction for disagreement calculation

    Returns:
        Dict with transformer features
    """
    try:
        embedder = TransformerEmbedder(use_fallback=True)

        # Extract OHLCV array
        ohlcv = ohlcv_df[['open', 'high', 'low', 'close', 'volume']].values

        # Get embedding and forecast
        result = embedder.embed_for_comparison(ohlcv)

        # Calculate disagreement with TimesFM
        disagreement = embedder.calculate_disagreement(result['forecast'], timesfm_forecast)

        return {
            'transformer_uncertainty': result['uncertainty'],
            'transformer_disagreement': disagreement
        }

    except Exception as e:
        logger.warning(f"Transformer feature generation failed: {e}")
        return {
            'transformer_uncertainty': 0.5,
            'transformer_disagreement': 0.0
        }


if __name__ == "__main__":
    # Test transformer embedder
    print("=== Testing Transformer Embedder ===")

    # Create synthetic OHLCV data
    np.random.seed(42)
    n_bars = 100

    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars)) * 0.5
    low = close - np.abs(np.random.randn(n_bars)) * 0.5
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.random.uniform(1000000, 5000000, n_bars)

    ohlcv = np.column_stack([open_price, high, low, close, volume])

    # Initialize embedder
    embedder = TransformerEmbedder(d_model=64, use_fallback=True)

    # Test embedding
    print("\n1. Testing sequence embedding...")
    result = embedder.get_embedding_result(ohlcv)
    print(f"   Embedding shape: {result.embedding.shape}")
    print(f"   Uncertainty: {result.uncertainty_proxy:.4f}")
    print(f"   Confidence: {result.confidence:.4f}")
    print(f"   Method: {result.metadata['method']}")

    # Test comparison
    print("\n2. Testing forecast comparison...")
    comparison = embedder.embed_for_comparison(ohlcv)
    print(f"   Transformer forecast: {comparison['forecast']:+.4f}")

    # Simulate TimesFM prediction
    timesfm_pred = 0.02  # +2% prediction
    disagreement = embedder.calculate_disagreement(comparison['forecast'], timesfm_pred)
    print(f"   TimesFM forecast: {timesfm_pred:+.4f}")
    print(f"   Disagreement: {disagreement:.4f}")

    # Test feature generation
    print("\n3. Testing feature generation...")
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'])
    features = generate_transformer_features(df, timesfm_forecast=0.01)
    print(f"   Features: {features}")

    print("\n=== Transformer Embedder Test Complete ===")
