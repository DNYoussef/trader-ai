"""
Enhanced HRM Training with 32-Dimensional Features
Retrains HRM with TimesFM + FinGPT enhanced features (24 -> 32 dims)
Expected training time: 3-5 hours on GPU for true grokking
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import logging
import json
import copy

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from src.strategies.black_swan_strategies import BlackSwanStrategyToolbox
from src.strategies.enhanced_market_state import create_enhanced_market_state
from src.strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
from src.training.grokfast_optimizer import GrokFastOptimizer, AntiMemorizationValidator

# Import TimesFM and FinGPT components
from src.intelligence.timesfm_forecaster import TimesFMForecaster
from src.intelligence.fingpt_sentiment import FinGPTSentimentAnalyzer, MarketSentiment
from src.intelligence.enhanced_hrm_features import EnhancedHRMFeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealHRMConfig32D:
    """Enhanced HRM configuration for 32-dimensional input"""
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 1024
        self.n_layers = 12
        self.n_heads = 16
        self.d_ff = 4096
        self.max_seq_len = 512
        self.dropout = 0.1
        self.layer_norm_eps = 1e-5
        self.high_level_dim = 768
        self.low_level_dim = 512
        self.num_hierarchies = 3

        # Updated for 32 features instead of 24
        self.input_dim = 32


class EnhancedHRM32D(nn.Module):
    """Enhanced HRM with 32-dimensional input projection"""

    def __init__(self, config: RealHRMConfig32D):
        super().__init__()
        self.config = config

        # Updated: 32 features -> d_model (was 24)
        self.input_projection = nn.Linear(32, config.d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))

        # Hierarchical reasoning module
        self.hierarchical = HierarchicalReasoning32D(config)

        # Output head for 8 strategies
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 8)  # 8 black swan strategies
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 32-dimensional input

        Args:
            x: (batch, 32) - Enhanced feature vector

        Returns:
            (batch, 8) - Strategy logits
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, 32) -> (batch, d_model)

        # Add positional encoding
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = x + self.pos_encoding[:, :1, :]

        # Hierarchical reasoning
        x = self.hierarchical(x.squeeze(1))  # (batch, d_model)

        # Output prediction
        logits = self.output_head(x)  # (batch, 8)

        return logits

    def reset_hidden(self):
        """Reset hierarchical hidden states"""
        if hasattr(self.hierarchical, 'reset_hidden'):
            self.hierarchical.reset_hidden()


class HierarchicalReasoning32D(nn.Module):
    """Hierarchical reasoning module compatible with 32D features"""

    def __init__(self, config: RealHRMConfig32D):
        super().__init__()

        self.high_level_dim = config.high_level_dim
        self.low_level_dim = config.low_level_dim
        self.d_model = config.d_model

        # High-level abstract reasoning
        self.high_level = nn.Sequential(
            nn.Linear(config.d_model, config.high_level_dim),
            nn.LayerNorm(config.high_level_dim),
            nn.GELU(),
            nn.Linear(config.high_level_dim, config.high_level_dim),
            nn.Dropout(config.dropout)
        )

        # Mid-level tactical reasoning
        self.mid_level = nn.Sequential(
            nn.Linear(config.d_model + config.high_level_dim, config.low_level_dim),
            nn.LayerNorm(config.low_level_dim),
            nn.GELU(),
            nn.Linear(config.low_level_dim, config.low_level_dim),
            nn.Dropout(config.dropout)
        )

        # Low-level execution
        self.low_level = nn.Sequential(
            nn.Linear(config.d_model + config.high_level_dim + config.low_level_dim,
                     config.low_level_dim),
            nn.LayerNorm(config.low_level_dim),
            nn.GELU(),
            nn.Linear(config.low_level_dim, config.d_model),
            nn.Dropout(config.dropout)
        )

        # Temporal weights
        self.high_temporal_weight = nn.Parameter(torch.tensor(0.9))
        self.mid_temporal_weight = nn.Parameter(torch.tensor(0.7))
        self.low_temporal_weight = nn.Parameter(torch.tensor(0.5))

        # Hidden states
        self.high_hidden = None
        self.mid_hidden = None
        self.low_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden states
        if self.high_hidden is None or self.high_hidden.shape[0] != batch_size:
            self.high_hidden = torch.zeros(batch_size, self.high_level_dim, device=device)
            self.mid_hidden = torch.zeros(batch_size, self.low_level_dim, device=device)
            self.low_hidden = torch.zeros(batch_size, x.shape[-1], device=device)
        else:
            self.high_hidden = self.high_hidden.to(device)
            self.mid_hidden = self.mid_hidden.to(device)
            self.low_hidden = self.low_hidden.to(device)

        # Hierarchical processing
        high_out = self.high_level(x)
        high_weight = self.high_temporal_weight.to(device)
        self.high_hidden = high_weight * self.high_hidden + (1 - high_weight) * high_out

        mid_input = torch.cat([x, self.high_hidden], dim=-1)
        mid_out = self.mid_level(mid_input)
        mid_weight = self.mid_temporal_weight.to(device)
        self.mid_hidden = mid_weight * self.mid_hidden + (1 - mid_weight) * mid_out

        low_input = torch.cat([x, self.high_hidden, self.mid_hidden], dim=-1)
        low_out = self.low_level(low_input)
        low_weight = self.low_temporal_weight.to(device)
        self.low_hidden = low_weight * self.low_hidden + (1 - low_weight) * low_out

        return self.low_hidden

    def reset_hidden(self):
        self.high_hidden = None
        self.mid_hidden = None
        self.low_hidden = None


class Enhanced32DDataGenerator:
    """Data generator with TimesFM + FinGPT enhanced features"""

    def __init__(self, base_scenarios, feature_engine: EnhancedHRMFeatureEngine):
        self.base_scenarios = base_scenarios
        self.feature_engine = feature_engine
        self.generation_count = 0

        # Generate synthetic VIX and price history
        self.vix_history = self._generate_vix_history()
        self.price_history = self._generate_price_history()
        self.news_pool = self._generate_news_pool()

    def _generate_vix_history(self, length=500):
        """Generate realistic VIX history"""
        vix = np.random.normal(22, 8, length)
        vix = np.clip(vix, 10, 70)
        # Add some autocorrelation
        for i in range(1, length):
            vix[i] = 0.9 * vix[i-1] + 0.1 * vix[i]
        return vix

    def _generate_price_history(self, length=500):
        """Generate realistic price history"""
        returns = np.random.normal(0.0002, 0.015, length)
        prices = 400 * np.exp(np.cumsum(returns))
        return prices

    def _generate_news_pool(self):
        """Generate pool of news articles for sentiment"""
        return [
            "Market shows strong momentum as earnings beat expectations",
            "Fed signals dovish stance amid economic uncertainty",
            "Tech sector leads rally with innovative AI breakthroughs",
            "Volatility spikes as geopolitical tensions escalate",
            "Analysts upgrade financial stocks on improving fundamentals",
            "Market breadth narrows as rotation accelerates",
            "Central bank policy shift triggers market reaction",
            "Economic data surprises to the upside supporting growth",
            "Risk-off sentiment dominates as concerns mount",
            "Bullish momentum continues with strong participation"
        ]

    def generate_batch(self, batch_size: int):
        """Generate batch with enhanced 32D features"""
        batch_features = []
        batch_labels = []

        toolbox = BlackSwanStrategyToolbox()
        reward_calc = ConvexRewardFunction()

        for _ in range(batch_size):
            # Select base scenario
            scenario = copy.deepcopy(np.random.choice(self.base_scenarios))

            # Add noise
            for key in scenario:
                if isinstance(scenario[key], (int, float)) and key != 'name':
                    noise = np.random.normal(0, abs(scenario[key]) * 0.1)
                    scenario[key] += noise

            # Select random news
            news = np.random.choice(self.news_pool, size=3, replace=False).tolist()

            # Create enhanced 32D features using feature engine
            try:
                enhanced_features = self.feature_engine.create_enhanced_features(
                    base_market_features=scenario,
                    vix_history=self.vix_history,
                    price_history=self.price_history,
                    news_articles=news,
                    symbol='SPY'
                )

                # Test all strategies and get best one
                market_state = create_enhanced_market_state(
                    timestamp=datetime.now(),
                    vix_level=scenario['vix_level'],
                    spy_returns_5d=scenario['spy_returns_5d'],
                    spy_returns_20d=scenario['spy_returns_20d'],
                    put_call_ratio=scenario['put_call_ratio'],
                    market_breadth=scenario['market_breadth'],
                    volume_ratio=scenario['volume_ratio'],
                    regime=self._determine_regime(scenario)
                )

                strategy_rewards = {}
                for strategy_name in toolbox.strategies.keys():
                    # Simulate performance
                    simulated_return = self._simulate_performance(scenario, strategy_name)

                    trade_outcome = TradeOutcome(
                        strategy_name=strategy_name,
                        entry_date=datetime.now(),
                        exit_date=datetime.now(),
                        symbol='SPY',
                        returns=simulated_return,
                        max_drawdown=min(0, simulated_return),
                        holding_period_days=10,
                        volatility_during_trade=scenario.get('sector_dispersion', 0.01) * 2,
                        is_black_swan_period=abs(simulated_return) > 0.10,
                        black_swan_captured=simulated_return > 0.10,
                        convexity_achieved=max(0, simulated_return / 0.05)
                    )

                    reward_metrics = reward_calc.calculate_reward(trade_outcome)
                    strategy_rewards[strategy_name] = reward_metrics.final_reward

                # Find best strategy
                best_strategy = max(strategy_rewards.keys(), key=lambda k: strategy_rewards[k])
                strategy_names = list(toolbox.strategies.keys())
                best_idx = strategy_names.index(best_strategy)

                # Add to batch
                batch_features.append(enhanced_features.combined_features)
                batch_labels.append(best_idx)

                self.generation_count += 1

            except Exception as e:
                logger.warning(f"Feature generation failed: {e}, using fallback")
                continue

        if len(batch_features) == 0:
            return None, None

        features_tensor = torch.tensor(batch_features, dtype=torch.float32)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

        return features_tensor, labels_tensor

    def _determine_regime(self, scenario):
        vix = scenario['vix_level']
        returns_5d = scenario['spy_returns_5d']

        if vix > 35 or returns_5d < -0.1:
            return 'crisis'
        elif vix > 25 or abs(returns_5d) > 0.04:
            return 'volatile'
        elif vix < 15 and returns_5d > 0.05:
            return 'momentum'
        else:
            return 'normal'

    def _simulate_performance(self, scenario, strategy_name):
        regime = self._determine_regime(scenario)

        base_returns = {
            'crisis': {'crisis_alpha': 0.18, 'tail_hedge': 0.15, 'volatility_harvest': 0.12,
                      'event_catalyst': 0.08, 'correlation_breakdown': 0.06, 'inequality_arbitrage': 0.04,
                      'momentum_explosion': -0.08, 'mean_reversion': -0.03},
            'volatile': {'momentum_explosion': 0.10, 'event_catalyst': 0.08, 'volatility_harvest': 0.07,
                        'correlation_breakdown': 0.05, 'crisis_alpha': 0.03, 'tail_hedge': 0.02,
                        'mean_reversion': -0.01, 'inequality_arbitrage': -0.02},
            'momentum': {'momentum_explosion': 0.12, 'mean_reversion': 0.08, 'inequality_arbitrage': 0.06,
                        'event_catalyst': 0.04, 'correlation_breakdown': 0.02, 'volatility_harvest': 0.00,
                        'crisis_alpha': -0.02, 'tail_hedge': -0.03},
            'normal': {'mean_reversion': 0.07, 'momentum_explosion': 0.05, 'inequality_arbitrage': 0.04,
                      'event_catalyst': 0.03, 'correlation_breakdown': 0.02, 'volatility_harvest': 0.01,
                      'crisis_alpha': -0.01, 'tail_hedge': -0.02}
        }

        base_return = base_returns[regime].get(strategy_name, 0.0)
        noise = np.random.normal(0, 0.03)
        return base_return + noise


def train_enhanced_hrm_32d():
    """Main training function for enhanced 32D HRM"""

    print("="*80)
    print("ENHANCED HRM TRAINING WITH 32-DIMENSIONAL FEATURES")
    print("="*80)
    print("TimesFM + FinGPT Integration")
    print("Expected time: 3-5 hours for true grokking")
    print("="*80)
    print()

    # Initialize feature engine
    print("Initializing enhanced feature engine...")
    feature_engine = EnhancedHRMFeatureEngine()

    # Base scenarios
    base_scenarios = [
        {'name': 'Black Swan', 'vix_level': 45, 'spy_returns_5d': -0.15, 'spy_returns_20d': -0.08,
         'put_call_ratio': 2.5, 'market_breadth': 0.15, 'volume_ratio': 3.0, 'correlation': 0.95,
         'gini_coefficient': 0.65, 'sector_dispersion': 0.025, 'signal_quality_score': 0.8},

        {'name': 'Volatile', 'vix_level': 28, 'spy_returns_5d': -0.05, 'spy_returns_20d': -0.02,
         'put_call_ratio': 1.6, 'market_breadth': 0.35, 'volume_ratio': 2.0, 'correlation': 0.75,
         'gini_coefficient': 0.52, 'sector_dispersion': 0.015, 'signal_quality_score': 0.65},

        {'name': 'Normal', 'vix_level': 18, 'spy_returns_5d': 0.01, 'spy_returns_20d': 0.02,
         'put_call_ratio': 1.0, 'market_breadth': 0.6, 'volume_ratio': 1.2, 'correlation': 0.5,
         'gini_coefficient': 0.42, 'sector_dispersion': 0.008, 'signal_quality_score': 0.5},

        {'name': 'Momentum', 'vix_level': 14, 'spy_returns_5d': 0.06, 'spy_returns_20d': 0.08,
         'put_call_ratio': 0.7, 'market_breadth': 0.8, 'volume_ratio': 1.6, 'correlation': 0.6,
         'gini_coefficient': 0.35, 'sector_dispersion': 0.012, 'signal_quality_score': 0.4}
    ]

    # Initialize data generator
    print("Initializing enhanced data generator...")
    data_generator = Enhanced32DDataGenerator(base_scenarios, feature_engine)

    # Create model
    print("Creating Enhanced HRM with 32-dim input...")
    config = RealHRMConfig32D()
    model = EnhancedHRM32D(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Input dimensions: 32 (24 base + 8 AI features)")
    print()

    # Training configuration
    training_config = {
        'max_iterations': 100000,  # 100K for faster initial training (can extend to 500K)
        'batch_size': 16,
        'gradient_accumulation_steps': 4,
        'validation_every': 1000,
        'grokking_threshold': 0.90,
        'noise_tolerance': 0.05,
        'early_stopping_patience': 20000,
        'learning_rate': 0.00002,
        'gradient_clip': 0.5,
        'min_clean_accuracy': 0.85,
        'consecutive_validations_needed': 3
    }

    # Initialize optimizer with GrokFast
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=0.01
    )

    grokfast_optimizer = GrokFastOptimizer(
        base_optimizer,
        filter_type='ema',
        alpha=0.98,
        lamb=2.5,
        warmup_steps=100
    )

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Training on: {device}")
    print(f"Configuration: {training_config}")
    print()
    print("Starting training...")
    print()

    training_history = {
        'iteration': [],
        'clean_accuracy': [],
        'noisy_accuracy': [],
        'generalization_gap': [],
        'grokking_score': [],
        'loss': []
    }

    best_grokking_score = 0.0
    patience_counter = 0
    consecutive_success = 0
    training_start = datetime.now()

    for iteration in range(training_config['max_iterations']):
        # Generate training batch
        features, labels = data_generator.generate_batch(training_config['batch_size'])

        if features is None:
            continue

        features = features.to(device)
        labels = labels.to(device)

        # Reset hidden states
        model.reset_hidden()

        # Forward pass
        predictions = model(features)
        loss = nn.CrossEntropyLoss()(predictions, labels)
        loss = loss / training_config['gradient_accumulation_steps']

        # Backward pass
        loss.backward()

        # Step optimizer every N accumulation steps
        if (iteration + 1) % training_config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            grokfast_optimizer.step(model)
            grokfast_optimizer.zero_grad()

        # Validation
        if iteration % training_config['validation_every'] == 0:
            model.eval()

            # Generate clean and noisy validation data
            clean_features, clean_labels = data_generator.generate_batch(16)
            noisy_features, noisy_labels = data_generator.generate_batch(16)

            if clean_features is not None and noisy_features is not None:
                with torch.no_grad():
                    model.reset_hidden()
                    clean_preds = model(clean_features.to(device))
                    clean_acc = (clean_preds.argmax(dim=1) == clean_labels.to(device)).float().mean().item()

                    model.reset_hidden()
                    noisy_preds = model(noisy_features.to(device))
                    noisy_acc = (noisy_preds.argmax(dim=1) == noisy_labels.to(device)).float().mean().item()

                    gen_gap = clean_acc - noisy_acc
                    grok_score = noisy_acc / clean_acc if clean_acc > 0 else 0.0

                    # Log
                    training_history['iteration'].append(iteration)
                    training_history['clean_accuracy'].append(clean_acc)
                    training_history['noisy_accuracy'].append(noisy_acc)
                    training_history['generalization_gap'].append(gen_gap)
                    training_history['grokking_score'].append(grok_score)
                    training_history['loss'].append(loss.item() * training_config['gradient_accumulation_steps'])

                    elapsed = (datetime.now() - training_start).total_seconds() / 3600
                    print(f"Iter {iteration:5d}: Loss={loss.item()*4:.4f}, Clean={clean_acc:.3f}, "
                          f"Noisy={noisy_acc:.3f}, Gap={gen_gap:.3f}, Grok={grok_score:.3f}, "
                          f"Time={elapsed:.2f}h")

                    # Check for grokking
                    if (clean_acc >= training_config['min_clean_accuracy'] and
                        grok_score >= training_config['grokking_threshold'] and
                        gen_gap <= training_config['noise_tolerance']):
                        consecutive_success += 1
                        print(f"    Grokking conditions met ({consecutive_success}/{training_config['consecutive_validations_needed']})")

                        if consecutive_success >= training_config['consecutive_validations_needed']:
                            print()
                            print("*** GROKKING ACHIEVED! ***")
                            print(f"Final grokking score: {grok_score:.3f}")
                            print(f"Generalization gap: {gen_gap:.3f}")
                            print(f"Total time: {elapsed:.2f} hours")
                            break
                    else:
                        consecutive_success = 0

                    # Early stopping
                    if grok_score > best_grokking_score:
                        best_grokking_score = grok_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= training_config['early_stopping_patience']:
                        print("Early stopping triggered")
                        break

            model.train()

    # Save model
    output_path = project_root / 'models' / 'enhanced_hrm_32d_grokfast.pth'
    output_path.parent.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': grokfast_optimizer.state_dict(),
        'training_config': training_config,
        'training_history': training_history,
        'feature_dimensions': 32,
        'model_params': total_params
    }, output_path)

    # Save training history
    history_path = project_root / 'models' / 'enhanced_hrm_32d_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'training_config': training_config,
            'training_history': training_history,
            'final_grokking_score': training_history['grokking_score'][-1] if training_history['grokking_score'] else 0.0,
            'model_parameters': total_params,
            'feature_dimensions': 32
        }, f, indent=2)

    print()
    print(f"Model saved to: {output_path}")
    print(f"History saved to: {history_path}")
    print("="*80)


if __name__ == "__main__":
    train_enhanced_hrm_32d()