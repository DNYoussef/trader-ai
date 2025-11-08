"""
TRM Configuration Module

Centralized configuration for Tiny Recursive Model training and inference.
Based on TRM paper hyperparameters with adaptations for financial markets.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TRMModelConfig:
    """TRM model architecture configuration"""

    # Input/Output dimensions
    input_dim: int = 10  # Market features
    hidden_dim: int = 512  # TRM standard
    output_dim: int = 8  # 8 trading strategies

    # Recursion parameters
    num_latent_steps: int = 6  # n: latent reasoning steps per cycle
    num_recursion_cycles: int = 3  # T: deep recursion cycles

    # Regularization
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Model features
    use_residual: bool = True
    use_halt_signal: bool = True

    def __post_init__(self):
        """Validate configuration"""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.num_latent_steps > 0, "num_latent_steps must be positive"
        assert self.num_recursion_cycles > 0, "num_recursion_cycles must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    @property
    def effective_depth(self) -> int:
        """Calculate effective network depth: T(n+1)×n_layers"""
        n_layers = 2  # TRM uses 2-layer network
        return self.num_recursion_cycles * (self.num_latent_steps + 1) * n_layers

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TRMModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class TRMTrainingConfig:
    """TRM training configuration"""

    # Optimizer settings (AdamW per TRM paper)
    optimizer: str = 'adamw'
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01

    # Learning rate schedule
    warmup_steps: int = 2000  # TRM paper: 2,000 iterations
    max_steps: int = 100000
    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'constant'

    # Batch configuration
    batch_size: int = 768  # TRM paper standard
    gradient_accumulation_steps: int = 1

    # Training dynamics
    max_epochs: int = 10
    validation_frequency: int = 1000  # steps
    checkpoint_frequency: int = 5000  # steps

    # Loss weights
    strategy_loss_weight: float = 1.0  # Cross-entropy for classification
    halt_loss_weight: float = 0.5  # BCE for halting signal
    profit_loss_weight: float = 0.3  # RL reward component

    # GrokFast optimizer integration
    use_grokfast: bool = True
    grokfast_filter_type: str = 'ema'  # 'ema' or 'ma'
    grokfast_alpha: float = 0.98  # EMA coefficient
    grokfast_lamb: float = 2.0  # Gradient amplification

    # EMA model (exponential moving average)
    use_ema: bool = True
    ema_decay: float = 0.999  # TRM paper standard

    # Anti-memorization validation
    noise_injection_rate: float = 0.1  # 10% feature noise
    consistency_threshold: float = 0.7  # 70% agreement required

    # Early stopping
    patience: int = 10  # epochs without improvement
    min_delta: float = 1e-4  # minimum improvement

    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration"""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert 0 < self.ema_decay < 1, "ema_decay must be in (0, 1)"
        assert self.optimizer in ['adamw', 'adam', 'sgd'], f"Unknown optimizer: {self.optimizer}"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TRMTrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class MarketFeatureConfig:
    """Configuration for market feature extraction"""

    # Feature names (10 features matching market_data.py)
    feature_names: List[str] = field(default_factory=lambda: [
        'vix_level',           # Volatility Index
        'spy_returns_5d',      # 5-day momentum
        'spy_returns_20d',     # 20-day momentum
        'volume_ratio',        # Liquidity indicator
        'market_breadth',      # Breadth indicator
        'correlation',         # Asset correlation
        'put_call_ratio',      # Sentiment indicator
        'gini_coefficient',    # Inequality metric (Gary's framework)
        'sector_dispersion',   # Sector health
        'signal_quality_score' # Overall confidence
    ])

    # Normalization
    normalize_features: bool = True
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'

    # Feature engineering
    use_rolling_windows: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [5, 20, 60])

    # Missing data handling
    fill_method: str = 'forward_fill'  # 'forward_fill', 'mean', 'zero'
    max_missing_ratio: float = 0.1  # Max 10% missing data allowed

    def __post_init__(self):
        """Validate configuration"""
        assert len(self.feature_names) > 0, "Must have at least one feature"
        assert self.normalization_method in ['zscore', 'minmax', 'robust'], \
            f"Unknown normalization: {self.normalization_method}"
        assert 0 <= self.max_missing_ratio <= 1, "max_missing_ratio must be in [0, 1]"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MarketFeatureConfig':
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class StrategyConfig:
    """Configuration for 8 trading strategies"""

    # Strategy names
    strategy_names: List[str] = field(default_factory=lambda: [
        'ultra_defensive',      # 0: VIX>50, max drawdown protection
        'defensive',            # 1: Elevated risk, conservative positioning
        'balanced_safe',        # 2: Normal but cautious
        'balanced_growth',      # 3: Normal conditions, balanced allocation
        'growth',               # 4: Risk-on, higher equity exposure
        'aggressive_growth',    # 5: High momentum, aggressive positioning
        'contrarian_long',      # 6: Gary's inequality opportunity
        'tactical_opportunity'  # 7: Short-term edge, high conviction
    ])

    # Strategy selection thresholds
    confidence_threshold: float = 0.7  # Min confidence to trade
    halt_threshold: float = 0.5  # Min halting signal to proceed

    # Strategy-specific parameters
    strategy_params: Dict[str, Dict] = field(default_factory=lambda: {
        'ultra_defensive': {'max_equity': 0.2, 'cash_reserve': 0.5},
        'defensive': {'max_equity': 0.4, 'cash_reserve': 0.3},
        'balanced_safe': {'max_equity': 0.6, 'cash_reserve': 0.2},
        'balanced_growth': {'max_equity': 0.7, 'cash_reserve': 0.1},
        'growth': {'max_equity': 0.8, 'cash_reserve': 0.05},
        'aggressive_growth': {'max_equity': 0.9, 'cash_reserve': 0.0},
        'contrarian_long': {'max_equity': 0.85, 'cash_reserve': 0.0},
        'tactical_opportunity': {'max_equity': 0.75, 'cash_reserve': 0.1}
    })

    def __post_init__(self):
        """Validate configuration"""
        assert len(self.strategy_names) == 8, "Must have exactly 8 strategies"
        assert 0 <= self.confidence_threshold <= 1, "confidence_threshold must be in [0, 1]"

    def get_strategy_idx(self, strategy_name: str) -> int:
        """Get strategy index from name"""
        return self.strategy_names.index(strategy_name)

    def get_strategy_name(self, strategy_idx: int) -> str:
        """Get strategy name from index"""
        return self.strategy_names[strategy_idx]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'StrategyConfig':
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class TRMConfig:
    """Complete TRM configuration"""

    model: TRMModelConfig = field(default_factory=TRMModelConfig)
    training: TRMTrainingConfig = field(default_factory=TRMTrainingConfig)
    features: MarketFeatureConfig = field(default_factory=MarketFeatureConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)

    # Paths
    model_save_dir: str = "models/trm"
    checkpoint_dir: str = "checkpoints/trm"
    log_dir: str = "logs/trm"
    data_dir: str = "data/historical"

    # Experiment tracking
    experiment_name: str = "trm_trader_ai"
    wandb_project: Optional[str] = None  # Set to enable W&B logging

    # Random seed
    seed: int = 42

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'features': self.features.to_dict(),
            'strategies': self.strategies.to_dict(),
            'model_save_dir': self.model_save_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'data_dir': self.data_dir,
            'experiment_name': self.experiment_name,
            'wandb_project': self.wandb_project,
            'seed': self.seed
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TRMConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(
            model=TRMModelConfig.from_dict(config_dict['model']),
            training=TRMTrainingConfig.from_dict(config_dict['training']),
            features=MarketFeatureConfig.from_dict(config_dict['features']),
            strategies=StrategyConfig.from_dict(config_dict['strategies']),
            model_save_dir=config_dict['model_save_dir'],
            checkpoint_dir=config_dict['checkpoint_dir'],
            log_dir=config_dict['log_dir'],
            data_dir=config_dict['data_dir'],
            experiment_name=config_dict['experiment_name'],
            wandb_project=config_dict.get('wandb_project'),
            seed=config_dict['seed']
        )

    def print_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("TRM Configuration Summary")
        print("=" * 80)

        print("\n[Model Architecture]")
        print(f"  Input dimension: {self.model.input_dim}")
        print(f"  Hidden dimension: {self.model.hidden_dim}")
        print(f"  Output dimension: {self.model.output_dim}")
        print(f"  Latent steps (n): {self.model.num_latent_steps}")
        print(f"  Recursion cycles (T): {self.model.num_recursion_cycles}")
        print(f"  Effective depth: {self.model.effective_depth} layers")

        print("\n[Training]")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Max epochs: {self.training.max_epochs}")
        print(f"  GrokFast enabled: {self.training.use_grokfast}")
        print(f"  EMA enabled: {self.training.use_ema}")

        print("\n[Features]")
        print(f"  Number of features: {len(self.features.feature_names)}")
        print(f"  Feature names: {', '.join(self.features.feature_names[:3])}...")
        print(f"  Normalization: {self.features.normalization_method}")

        print("\n[Strategies]")
        print(f"  Number of strategies: {len(self.strategies.strategy_names)}")
        print(f"  Confidence threshold: {self.strategies.confidence_threshold}")
        print(f"  Strategies: {', '.join(self.strategies.strategy_names[:3])}...")

        print("\n[Paths]")
        print(f"  Model save dir: {self.model_save_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Experiment: {self.experiment_name}")

        print("=" * 80)


def get_default_config() -> TRMConfig:
    """Get default TRM configuration"""
    return TRMConfig()


def create_config_from_dict(config_dict: Dict) -> TRMConfig:
    """Create TRM configuration from dictionary"""
    return TRMConfig(
        model=TRMModelConfig.from_dict(config_dict.get('model', {})),
        training=TRMTrainingConfig.from_dict(config_dict.get('training', {})),
        features=MarketFeatureConfig.from_dict(config_dict.get('features', {})),
        strategies=StrategyConfig.from_dict(config_dict.get('strategies', {})),
        **{k: v for k, v in config_dict.items()
           if k not in ['model', 'training', 'features', 'strategies']}
    )


if __name__ == "__main__":
    # Test configuration creation and saving
    print("Creating default TRM configuration...")
    config = get_default_config()

    config.print_summary()

    # Test save/load
    test_path = "config/trm_config_test.json"
    print(f"\nSaving to {test_path}...")
    config.save(test_path)

    print(f"Loading from {test_path}...")
    loaded_config = TRMConfig.load(test_path)

    print("\n✅ Configuration test passed!")
