"""
Enhanced HRM 32D Training - FAST RESUME VERSION
Continues from checkpoint or starts fresh with optimized settings
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.intelligence.enhanced_hrm_features import EnhancedHRMFeatureEngine
from src.strategies.black_swan_strategies import BlackSwanStrategyToolbox
from src.strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
from src.strategies.enhanced_market_state import create_enhanced_market_state
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FastHRMConfig32D:
    """Fast training config for 32D HRM"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    n_strategies: int = 8
    use_layernorm: bool = True
    use_residual: bool = True
    hidden_dim: int = 512
    max_iterations: int = 20000  # Much faster
    batch_size: int = 32  # Larger batches
    learning_rate: float = 0.0001  # Higher LR
    validation_every: int = 100  # More frequent validation

class EnhancedHRM32D(nn.Module):
    """Enhanced HRM with 32-dimensional input"""

    def __init__(self, config: FastHRMConfig32D):
        super().__init__()
        self.config = config

        # Input projection: 32 features -> d_model
        self.input_projection = nn.Linear(32, config.d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Strategy selection head
        self.strategy_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_strategies)
        )

        # Layer norm if enabled
        if config.use_layernorm:
            self.layer_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass"""
        # Input projection
        x = self.input_projection(x)

        # Add batch dim if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply layer norm
        if self.config.use_layernorm:
            x = self.layer_norm(x)

        # Transformer encoding
        encoded = self.encoder(x)

        # Pool over sequence dim
        pooled = encoded.mean(dim=1)

        # Strategy selection
        logits = self.strategy_head(pooled)

        return logits

class FastDataGenerator:
    """Optimized data generator for fast training"""

    def __init__(self, feature_engine: EnhancedHRMFeatureEngine):
        self.feature_engine = feature_engine

        # Pre-generate pools of data for speed
        self.vix_pool = [self._generate_vix_history() for _ in range(10)]
        self.price_pool = [self._generate_price_history() for _ in range(10)]
        self.news_pool = self._generate_news_pool()

        # Pre-define scenarios for consistency
        self.scenarios = [
            {'vix_level': 45, 'spy_returns_5d': -0.15, 'spy_returns_20d': -0.08,
             'put_call_ratio': 2.5, 'market_breadth': 0.15, 'volume_ratio': 3.0,
             'correlation': 0.95, 'gini_coefficient': 0.65, 'sector_dispersion': 0.025},

            {'vix_level': 28, 'spy_returns_5d': -0.05, 'spy_returns_20d': -0.02,
             'put_call_ratio': 1.6, 'market_breadth': 0.35, 'volume_ratio': 2.0,
             'correlation': 0.75, 'gini_coefficient': 0.52, 'sector_dispersion': 0.015},

            {'vix_level': 18, 'spy_returns_5d': 0.01, 'spy_returns_20d': 0.02,
             'put_call_ratio': 1.0, 'market_breadth': 0.6, 'volume_ratio': 1.2,
             'correlation': 0.5, 'gini_coefficient': 0.42, 'sector_dispersion': 0.008},

            {'vix_level': 14, 'spy_returns_5d': 0.06, 'spy_returns_20d': 0.08,
             'put_call_ratio': 0.7, 'market_breadth': 0.8, 'volume_ratio': 1.6,
             'correlation': 0.6, 'gini_coefficient': 0.35, 'sector_dispersion': 0.012}
        ]

        self.toolbox = BlackSwanStrategyToolbox()
        self.strategy_names = list(self.toolbox.strategies.keys())

    def _generate_vix_history(self):
        """Generate VIX history"""
        vix = np.random.normal(22, 8, 500)
        return np.clip(vix, 10, 70)

    def _generate_price_history(self):
        """Generate price history"""
        returns = np.random.normal(0.0002, 0.015, 500)
        return 400 * np.exp(np.cumsum(returns))

    def _generate_news_pool(self):
        """Generate news pool"""
        return [
            "Market rallies on strong earnings",
            "Fed signals policy shift",
            "Volatility spikes on geopolitical concerns",
            "Tech sector leads gains",
            "Economic data beats expectations",
            "Market breadth improves",
            "Risk-off sentiment dominates",
            "Bullish momentum continues"
        ]

    def generate_batch(self, batch_size: int):
        """Generate training batch quickly"""
        features = []
        labels = []

        for _ in range(batch_size):
            # Random scenario with noise
            scenario = copy.deepcopy(np.random.choice(self.scenarios))
            for key in scenario:
                scenario[key] *= np.random.uniform(0.8, 1.2)

            # Random data from pools
            vix = self.vix_pool[np.random.randint(len(self.vix_pool))]
            prices = self.price_pool[np.random.randint(len(self.price_pool))]
            news = np.random.choice(self.news_pool, size=3).tolist()

            try:
                # Generate features
                enhanced = self.feature_engine.create_enhanced_features(
                    base_market_features=scenario,
                    vix_history=vix,
                    price_history=prices,
                    news_articles=news,
                    symbol='SPY'
                )

                # Simple label generation (best strategy based on scenario)
                if scenario['vix_level'] > 35:
                    label = self.strategy_names.index('crisis_alpha')
                elif scenario['vix_level'] > 25:
                    label = self.strategy_names.index('tail_hedge')
                elif scenario['spy_returns_5d'] > 0.05:
                    label = self.strategy_names.index('gamma_squeeze')
                else:
                    label = self.strategy_names.index('volatility_harvest')

                features.append(enhanced.combined_features)
                labels.append(label)

            except Exception as e:
                logger.debug(f"Feature generation failed: {e}")
                continue

        if len(features) == 0:
            return None, None

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def train_fast():
    """Fast training with resume capability"""

    print("="*80)
    print("FAST ENHANCED HRM TRAINING - 32D FEATURES")
    print("="*80)
    print()

    # Check for existing checkpoint
    checkpoint_path = project_root / 'models' / 'enhanced_hrm_32d_grokfast.pth'
    resume_iteration = 0

    # Initialize components
    print("Initializing components...")
    feature_engine = EnhancedHRMFeatureEngine()
    data_generator = FastDataGenerator(feature_engine)

    # Create model
    config = FastHRMConfig32D()
    model = EnhancedHRM32D(config)

    # Try to load checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'iteration' in checkpoint:
                    resume_iteration = checkpoint.get('iteration', 0)
                    print(f"Resuming from iteration {resume_iteration}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}, starting fresh")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Training history
    history = {
        'iterations': [],
        'losses': [],
        'accuracies': []
    }

    # Training loop
    print("Starting training...")
    print()

    best_accuracy = 0.0
    model.train()

    for iteration in range(resume_iteration, config.max_iterations):
        # Generate batch
        features, labels = data_generator.generate_batch(config.batch_size)
        if features is None:
            continue

        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        if iteration % config.validation_every == 0:
            with torch.no_grad():
                # Quick validation
                val_features, val_labels = data_generator.generate_batch(64)
                if val_features is not None:
                    val_features = val_features.to(device)
                    val_labels = val_labels.to(device)
                    val_logits = model(val_features)
                    val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()

                    print(f"Iter {iteration:5d}: Loss={loss.item():.4f}, Acc={val_acc:.3f}")

                    history['iterations'].append(iteration)
                    history['losses'].append(loss.item())
                    history['accuracies'].append(val_acc)

                    # Save best model
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        print(f"  [NEW BEST: {best_accuracy:.3f}]")

                        # Save checkpoint
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iteration': iteration,
                            'best_accuracy': best_accuracy,
                            'config': config,
                            'history': history
                        }, checkpoint_path)

                    # Early stopping if good enough
                    if val_acc > 0.85:
                        print(f"\nTarget accuracy reached! Best: {best_accuracy:.3f}")
                        break

    print()
    print("="*80)
    print(f"Training complete! Best accuracy: {best_accuracy:.3f}")
    print(f"Model saved to: {checkpoint_path}")
    print("="*80)

    # Save final history
    history_path = project_root / 'models' / 'enhanced_hrm_32d_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train_fast()