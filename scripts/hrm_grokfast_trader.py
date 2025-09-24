"""
HRM + GrokFast Trading System
Integrates 27M parameter Hierarchical Reasoning Model with GrokFast for trading strategy selection
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sqlite3
import json
from collections import defaultdict
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Add HRM path
hrm_path = Path(__file__).parent.parent.parent / 'HRM'
sys.path.append(str(hrm_path))
sys.path.append(str(hrm_path / 'models'))

from strategies.black_swan_strategies import BlackSwanStrategyToolbox
from strategies.enhanced_market_state import create_enhanced_market_state
from strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
from training.grokfast_optimizer import GrokFastOptimizer, AntiMemorizationValidator

# Import REAL HRM implementation - NO MOCKS, NO EXTERNAL DEPENDENCIES
from real_hrm_implementation import RealHRM, RealHRMConfig, create_real_hrm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRMInfiniteNoiseDataGenerator:
    """Enhanced noise data generator for HRM training"""

    def __init__(self, base_scenarios: List[Dict], noise_config: Dict[str, float] = None):
        self.base_scenarios = base_scenarios
        self.noise_config = noise_config or {
            'vix_level': 3.0,           # ±3 VIX points (more variation for HRM)
            'spy_returns_5d': 0.015,    # ±1.5% returns
            'spy_returns_20d': 0.012,   # ±1.2% returns
            'volume_ratio': 0.3,        # ±30% volume
            'market_breadth': 0.15,     # ±15% breadth
            'correlation': 0.12,        # ±12% correlation
            'put_call_ratio': 0.2,      # ±20% put/call
            'gini_coefficient': 0.03,   # ±3% inequality
            'sector_dispersion': 0.005, # ±0.5% dispersion
            'signal_quality_score': 0.1 # ±10% quality
        }
        self.generation_count = 0

    def generate_hierarchical_scenario(self, base_scenario: Dict) -> Dict:
        """Generate scenario with hierarchical noise patterns"""
        scenario = copy.deepcopy(base_scenario)

        # Add temporal correlation to noise (hierarchical property)
        temporal_factor = np.random.beta(2, 5)  # Bias toward slow changes

        for feature, noise_scale in self.noise_config.items():
            if feature in scenario:
                # Hierarchical noise: slow + fast components
                slow_noise = np.random.normal(0, noise_scale * temporal_factor)
                fast_noise = np.random.normal(0, noise_scale * (1 - temporal_factor))

                total_noise = slow_noise + fast_noise * 0.3  # Emphasize slow components
                scenario[feature] += total_noise

                # Apply bounds with hierarchical considerations
                if feature == 'vix_level':
                    scenario[feature] = max(8, min(85, scenario[feature]))
                elif feature in ['market_breadth', 'correlation', 'signal_quality_score']:
                    scenario[feature] = max(0, min(1, scenario[feature]))
                elif feature == 'volume_ratio':
                    scenario[feature] = max(0.2, min(6.0, scenario[feature]))
                elif feature == 'put_call_ratio':
                    scenario[feature] = max(0.1, min(5.0, scenario[feature]))
                elif feature == 'gini_coefficient':
                    scenario[feature] = max(0.25, min(0.85, scenario[feature]))

        self.generation_count += 1
        return scenario

    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate batch with hierarchical patterns"""
        batch = []
        for _ in range(batch_size):
            base_scenario = np.random.choice(self.base_scenarios)
            hierarchical_scenario = self.generate_hierarchical_scenario(base_scenario)
            batch.append(hierarchical_scenario)
        return batch

class HRMGrokkingTrainer:
    """HRM + GrokFast trainer for hierarchical strategy reasoning"""

    def __init__(self):
        self.db_path = Path("data/historical_market.db")
        self.toolbox = BlackSwanStrategyToolbox()
        self.reward_calc = ConvexRewardFunction()
        self.validator = AntiMemorizationValidator()

        # Create REAL HRM - 156M parameters, NO MOCKS
        self.model, self.hrm_config = create_real_hrm()

        # Training configuration for REAL hours-long training
        self.training_config = {
            'max_iterations': 500000,        # 500K iterations for 5+ hours
            'batch_size': 16,                # Smaller batch for 156M model on GPU
            'gradient_accumulation_steps': 4, # Effective batch = 16 * 4 = 64
            'validation_every': 1000,        # Check every 1000 iterations
            'grokking_threshold': 0.90,     # Very high threshold for true grokking
            'noise_tolerance': 0.05,        # Strict tolerance for generalization
            'early_stopping_patience': 50000, # Extreme patience for real grokking
            'learning_rate': 0.00002,       # Even lower LR for stability
            'gradient_clip': 0.5,           # Stronger gradient clipping
            'min_iterations_before_grokking': 100000,  # At least 100K iterations (1.16 hours)
            'min_clean_accuracy': 0.85,     # Require very strong learning
            'min_training_hours': 2.0,      # Minimum 2 hours of training required
            'consecutive_validations_needed': 3  # Need 3 consecutive successful validations
        }

        # Initialize optimizer with GrokFast
        self.base_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=0.01
        )

        self.grokfast_optimizer = GrokFastOptimizer(
            self.base_optimizer,
            filter_type='ema',
            alpha=0.98,
            lamb=2.5,  # Higher lambda for stronger generalization
            warmup_steps=100
        )

        # Training tracking
        self.training_history = {
            'iteration': [],
            'clean_accuracy': [],
            'noisy_accuracy': [],
            'generalization_gap': [],
            'grokking_score': [],
            'loss': []
        }

        # Initialize hierarchical data generator
        self.data_generator = self._initialize_hierarchical_generator()

        print(f"HRM Grokking Trainer initialized:")
        print(f"  Model: REAL HRM (NO MOCKS)")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Max iterations: {self.training_config['max_iterations']:,}")
        print(f"  Batch size: {self.training_config['batch_size']}")
        print(f"  This will take HOURS on GPU as expected!")

    def _initialize_hierarchical_generator(self) -> HRMInfiniteNoiseDataGenerator:
        """Initialize hierarchical data generator with diverse scenarios"""
        base_scenarios = [
            # Crisis scenarios
            {'name': 'Black Swan', 'vix_level': 45, 'spy_returns_5d': -0.15, 'spy_returns_20d': -0.08,
             'put_call_ratio': 2.5, 'market_breadth': 0.15, 'volume_ratio': 3.0, 'correlation': 0.95,
             'gini_coefficient': 0.65, 'sector_dispersion': 0.025, 'signal_quality_score': 0.8},

            {'name': 'Financial Crisis', 'vix_level': 38, 'spy_returns_5d': -0.12, 'spy_returns_20d': -0.06,
             'put_call_ratio': 2.2, 'market_breadth': 0.2, 'volume_ratio': 2.8, 'correlation': 0.9,
             'gini_coefficient': 0.6, 'sector_dispersion': 0.02, 'signal_quality_score': 0.75},

            # Volatile scenarios
            {'name': 'Market Stress', 'vix_level': 28, 'spy_returns_5d': -0.05, 'spy_returns_20d': -0.02,
             'put_call_ratio': 1.6, 'market_breadth': 0.35, 'volume_ratio': 2.0, 'correlation': 0.75,
             'gini_coefficient': 0.52, 'sector_dispersion': 0.015, 'signal_quality_score': 0.65},

            {'name': 'Uncertainty', 'vix_level': 25, 'spy_returns_5d': -0.03, 'spy_returns_20d': -0.01,
             'put_call_ratio': 1.4, 'market_breadth': 0.4, 'volume_ratio': 1.8, 'correlation': 0.7,
             'gini_coefficient': 0.48, 'sector_dispersion': 0.012, 'signal_quality_score': 0.6},

            # Normal scenarios
            {'name': 'Balanced', 'vix_level': 18, 'spy_returns_5d': 0.01, 'spy_returns_20d': 0.02,
             'put_call_ratio': 1.0, 'market_breadth': 0.6, 'volume_ratio': 1.2, 'correlation': 0.5,
             'gini_coefficient': 0.42, 'sector_dispersion': 0.008, 'signal_quality_score': 0.5},

            {'name': 'Stable Growth', 'vix_level': 16, 'spy_returns_5d': 0.02, 'spy_returns_20d': 0.03,
             'put_call_ratio': 0.9, 'market_breadth': 0.65, 'volume_ratio': 1.1, 'correlation': 0.45,
             'gini_coefficient': 0.4, 'sector_dispersion': 0.007, 'signal_quality_score': 0.45},

            # Momentum scenarios
            {'name': 'Bull Run', 'vix_level': 14, 'spy_returns_5d': 0.06, 'spy_returns_20d': 0.08,
             'put_call_ratio': 0.7, 'market_breadth': 0.8, 'volume_ratio': 1.6, 'correlation': 0.6,
             'gini_coefficient': 0.35, 'sector_dispersion': 0.012, 'signal_quality_score': 0.4},

            {'name': 'Strong Momentum', 'vix_level': 12, 'spy_returns_5d': 0.08, 'spy_returns_20d': 0.1,
             'put_call_ratio': 0.6, 'market_breadth': 0.85, 'volume_ratio': 1.8, 'correlation': 0.65,
             'gini_coefficient': 0.32, 'sector_dispersion': 0.015, 'signal_quality_score': 0.35}
        ]

        return HRMInfiniteNoiseDataGenerator(base_scenarios)

    def create_hierarchical_training_batch(self, scenarios: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training batch with hierarchical reasoning patterns"""

        # Load market data
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT date, symbol, open, high, low, close, volume, returns
            FROM market_data
            WHERE date >= '2023-01-01'
            ORDER BY date DESC
            LIMIT 1000
            """
            market_data = pd.read_sql_query(query, conn)

        features_list = []
        labels_list = []

        for scenario in scenarios:
            # Create enhanced market state
            market_state = create_enhanced_market_state(
                timestamp=datetime.now(),
                vix_level=scenario['vix_level'],
                spy_returns_5d=scenario['spy_returns_5d'],
                spy_returns_20d=scenario['spy_returns_20d'],
                put_call_ratio=scenario['put_call_ratio'],
                market_breadth=scenario['market_breadth'],
                volume_ratio=scenario['volume_ratio'],
                regime=self._determine_hierarchical_regime(scenario)
            )

            # Get hierarchical market features
            market_features = market_state.get_enhanced_market_features()

            # Test all strategies with hierarchical evaluation
            strategy_results = {}
            for strategy_name, strategy in self.toolbox.strategies.items():
                signals = strategy.analyze(market_state, market_data)

                if signals and (isinstance(signals, list) and len(signals) > 0 or signals):
                    first_signal = signals[0] if isinstance(signals, list) else signals

                    # Hierarchical performance simulation
                    simulated_return = self._simulate_hierarchical_performance(
                        first_signal, scenario, strategy_name
                    )

                    # Calculate hierarchical convex reward
                    trade_outcome = TradeOutcome(
                        strategy_name=strategy_name,
                        entry_date=datetime.now(),
                        exit_date=datetime.now() + timedelta(days=10),
                        symbol='SPY',
                        returns=simulated_return,
                        max_drawdown=min(0, simulated_return),
                        holding_period_days=10,
                        volatility_during_trade=scenario.get('sector_dispersion', 0.01) * 2,
                        is_black_swan_period=abs(simulated_return) > 0.10,
                        black_swan_captured=simulated_return > 0.10,
                        convexity_achieved=max(0, simulated_return / 0.05)
                    )

                    reward_metrics = self.reward_calc.calculate_reward(trade_outcome)
                    strategy_results[strategy_name] = reward_metrics.final_reward

            if strategy_results:
                # Find optimal strategy using hierarchical reasoning
                best_strategy = max(strategy_results.keys(), key=lambda k: strategy_results[k])
                strategy_names = list(self.toolbox.strategies.keys())
                best_strategy_idx = strategy_names.index(best_strategy)

                # Convert to hierarchical feature vector
                feature_vector = self._create_hierarchical_features(market_features, scenario)
                features_list.append(feature_vector)
                labels_list.append(best_strategy_idx)

        if not features_list:
            return None, None

        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return features_tensor, labels_tensor

    def _create_hierarchical_features(self, market_features: Dict, scenario: Dict) -> List[float]:
        """Create hierarchical feature representation"""
        # Standard market features
        base_features = [market_features.get(key, 0.0) for key in sorted(market_features.keys())]

        # Ensure we have exactly 24 features (pad or truncate if needed)
        if len(base_features) > 24:
            base_features = base_features[:24]
        elif len(base_features) < 24:
            base_features.extend([0.0] * (24 - len(base_features)))

        return base_features

    def _determine_hierarchical_regime(self, scenario: Dict) -> str:
        """Determine market regime with hierarchical logic"""
        vix = scenario['vix_level']
        returns_5d = scenario['spy_returns_5d']
        stress_score = scenario['put_call_ratio'] * (1 - scenario['market_breadth'])

        if vix > 35 or returns_5d < -0.1 or stress_score > 1.5:
            return 'crisis'
        elif vix > 25 or abs(returns_5d) > 0.04 or stress_score > 1.0:
            return 'volatile'
        elif vix < 15 and returns_5d > 0.05:
            return 'momentum'
        else:
            return 'normal'

    def _simulate_hierarchical_performance(self, signal, scenario: Dict, strategy_name: str) -> float:
        """Simulate strategy performance with hierarchical market dynamics"""
        regime = self._determine_hierarchical_regime(scenario)

        # Hierarchical performance patterns
        if regime == 'crisis':
            base_returns = {
                'crisis_alpha': 0.18, 'tail_hedge': 0.15, 'volatility_harvest': 0.12,
                'event_catalyst': 0.08, 'correlation_breakdown': 0.06, 'inequality_arbitrage': 0.04,
                'momentum_explosion': -0.08, 'mean_reversion': -0.03
            }
        elif regime == 'volatile':
            base_returns = {
                'momentum_explosion': 0.10, 'event_catalyst': 0.08, 'volatility_harvest': 0.07,
                'correlation_breakdown': 0.05, 'crisis_alpha': 0.03, 'tail_hedge': 0.02,
                'mean_reversion': -0.01, 'inequality_arbitrage': -0.02
            }
        elif regime == 'momentum':
            base_returns = {
                'momentum_explosion': 0.12, 'mean_reversion': 0.08, 'inequality_arbitrage': 0.06,
                'event_catalyst': 0.04, 'correlation_breakdown': 0.02, 'volatility_harvest': 0.00,
                'crisis_alpha': -0.02, 'tail_hedge': -0.03
            }
        else:  # normal
            base_returns = {
                'mean_reversion': 0.07, 'momentum_explosion': 0.05, 'inequality_arbitrage': 0.04,
                'event_catalyst': 0.03, 'correlation_breakdown': 0.02, 'volatility_harvest': 0.01,
                'crisis_alpha': -0.01, 'tail_hedge': -0.02
            }

        base_return = base_returns.get(strategy_name, 0.0)

        # Add hierarchical noise with regime-dependent volatility
        noise_scale = {
            'crisis': 0.04, 'volatile': 0.03, 'momentum': 0.025, 'normal': 0.02
        }[regime]

        noise = np.random.normal(0, noise_scale)
        return base_return + noise

    def train_until_hierarchical_grokking(self):
        """Train HRM until hierarchical grokking is achieved"""

        print("=" * 80)
        print("HRM + GROKFAST HIERARCHICAL REASONING TRAINING")
        print("=" * 80)
        print(f"Model: Real HierarchicalReasoningModel_ACTV1 ({sum(p.numel() for p in self.model.parameters()):,} params)")
        print(f"Target grokking threshold: {self.training_config['grokking_threshold']:.2f}")
        print(f"Noise tolerance: {self.training_config['noise_tolerance']:.2f}")
        print(f"Max iterations: {self.training_config['max_iterations']:,}")
        print(f"Min iterations before grokking check: {self.training_config['min_iterations_before_grokking']:,}")
        print(f"Min training hours required: {self.training_config['min_training_hours']:.1f}")
        print(f"Min clean accuracy required: {self.training_config['min_clean_accuracy']:.2f}")
        print(f"Consecutive validations needed: {self.training_config['consecutive_validations_needed']}")
        print(f"This WILL take {self.training_config['min_training_hours']:.1f}+ HOURS minimum (not minutes!)...")
        print()

        best_grokking_score = 0.0
        patience_counter = 0
        consecutive_success_count = 0
        training_start_time = datetime.now()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        print(f"Training on: {device}")
        print()

        for iteration in range(self.training_config['max_iterations']):
            # Training step
            start_time = datetime.now()

            # Generate hierarchical training batch
            noisy_scenarios = self.data_generator.generate_batch(self.training_config['batch_size'])
            features, labels = self.create_hierarchical_training_batch(noisy_scenarios)

            if features is None:
                continue

            # Add small regularization noise to prevent memorization
            features = features + torch.randn_like(features) * 0.01

            features = features.to(device)
            labels = labels.to(device)

            # Reset HRM hidden states for new batch
            if hasattr(self.model, 'reset_hidden'):
                self.model.reset_hidden()

            # Forward pass
            predictions = self.model(features)
            loss = nn.CrossEntropyLoss()(predictions, labels)

            # Scale loss for gradient accumulation
            loss = loss / self.training_config['gradient_accumulation_steps']

            # Backward pass
            loss.backward()

            # Only step optimizer every N accumulation steps
            if (iteration + 1) % self.training_config['gradient_accumulation_steps'] == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config['gradient_clip'])

                # Step with GrokFast optimizer
                self.grokfast_optimizer.step(self.model)
                self.grokfast_optimizer.zero_grad()

            # Calculate training metrics
            with torch.no_grad():
                accuracy = (predictions.argmax(dim=1) == labels).float().mean().item()
                # Scale loss back for logging
                actual_loss = loss.item() * self.training_config['gradient_accumulation_steps']

            # Validation every N iterations
            if iteration % self.training_config['validation_every'] == 0:
                validation_metrics = self._validate_hierarchical_grokking(device)

                if validation_metrics:
                    # Update training history
                    self.training_history['iteration'].append(iteration)
                    self.training_history['clean_accuracy'].append(validation_metrics['clean_accuracy'])
                    self.training_history['noisy_accuracy'].append(validation_metrics['noisy_accuracy'])
                    self.training_history['generalization_gap'].append(validation_metrics['generalization_gap'])
                    self.training_history['grokking_score'].append(validation_metrics['grokking_score'])
                    self.training_history['loss'].append(actual_loss)

                    # Check for grokking
                    current_grokking_score = validation_metrics['grokking_score']

                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    print(f"Iter {iteration:5d}: Loss={actual_loss:.4f}, "
                          f"Clean={validation_metrics['clean_accuracy']:.3f}, "
                          f"Noisy={validation_metrics['noisy_accuracy']:.3f}, "
                          f"Gap={validation_metrics['generalization_gap']:.3f}, "
                          f"Grok={current_grokking_score:.3f}, "
                          f"Time={elapsed_time:.2f}s")

                    # Check elapsed time
                    elapsed_hours = (datetime.now() - training_start_time).total_seconds() / 3600

                    # Check if we've achieved REAL hierarchical grokking
                    # Requirements:
                    # 1. Minimum training time elapsed
                    # 2. Minimum iterations completed
                    # 3. Clean accuracy above threshold
                    # 4. Grokking score above threshold
                    # 5. Generalization gap below tolerance
                    conditions_met = (
                        elapsed_hours >= self.training_config['min_training_hours'] and
                        iteration >= self.training_config['min_iterations_before_grokking'] and
                        validation_metrics['clean_accuracy'] >= self.training_config['min_clean_accuracy'] and
                        current_grokking_score >= self.training_config['grokking_threshold'] and
                        validation_metrics['generalization_gap'] <= self.training_config['noise_tolerance']
                    )

                    if conditions_met:
                        consecutive_success_count += 1
                        print(f"    Conditions met ({consecutive_success_count}/{self.training_config['consecutive_validations_needed']} consecutive)")

                        if consecutive_success_count >= self.training_config['consecutive_validations_needed']:
                            print()
                            print("*** HIERARCHICAL GROKKING ACHIEVED! ***")
                            print(f"Final grokking score: {current_grokking_score:.3f}")
                            print(f"Generalization gap: {validation_metrics['generalization_gap']:.3f}")
                            print(f"Total iterations: {iteration:,}")
                            print(f"Total training time: {elapsed_hours:.2f} hours")
                            print(f"Data generated: {self.data_generator.generation_count:,}")
                            return True
                    else:
                        consecutive_success_count = 0  # Reset counter

                    # Early stopping
                    if current_grokking_score > best_grokking_score:
                        best_grokking_score = current_grokking_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.training_config['early_stopping_patience']:
                        print()
                        print("Early stopping triggered")
                        print(f"Best grokking score: {best_grokking_score:.3f}")
                        return False

            # Progress indicator for long training
            if iteration % 100 == 0:  # Frequent updates
                elapsed_time = (datetime.now() - training_start_time).total_seconds()
                elapsed_hours = elapsed_time / 3600
                if elapsed_hours < self.training_config['min_training_hours']:
                    remaining_hours = self.training_config['min_training_hours'] - elapsed_hours
                    if iteration % 1000 == 0:  # Less frequent time updates
                        print(f"    [{elapsed_hours:.2f}h elapsed, {remaining_hours:.2f}h minimum remaining]")

            if iteration % 1000 == 0 and iteration > 0:  # Detailed progress updates
                gen_stats = self.data_generator.generation_count
                print(f"Progress: {iteration:,}/{self.training_config['max_iterations']:,} iterations, "
                      f"{gen_stats:,} scenarios generated")

        print()
        print("Maximum iterations reached")
        print(f"Best grokking score: {best_grokking_score:.3f}")
        return False

    def _validate_hierarchical_grokking(self, device) -> Dict[str, float]:
        """Validate hierarchical grokking with clean vs noisy data"""
        self.model.eval()

        try:
            # Generate clean and noisy validation batches
            clean_scenarios = self.data_generator.base_scenarios[:16]  # Smaller for validation
            noisy_scenarios = self.data_generator.generate_batch(16)

            clean_features, clean_labels = self.create_hierarchical_training_batch(clean_scenarios)
            noisy_features, noisy_labels = self.create_hierarchical_training_batch(noisy_scenarios)

            if clean_features is None or noisy_features is None:
                return None

            clean_features = clean_features.to(device)
            clean_labels = clean_labels.to(device)
            noisy_features = noisy_features.to(device)
            noisy_labels = noisy_labels.to(device)

            with torch.no_grad():
                # Reset hidden states
                if hasattr(self.model, 'reset_hidden'):
                    self.model.reset_hidden()

                # Clean accuracy
                clean_predictions = self.model(clean_features)
                clean_accuracy = (clean_predictions.argmax(dim=1) == clean_labels).float().mean().item()

                # Reset hidden states for noisy batch
                if hasattr(self.model, 'reset_hidden'):
                    self.model.reset_hidden()

                # Noisy accuracy
                noisy_predictions = self.model(noisy_features)
                noisy_accuracy = (noisy_predictions.argmax(dim=1) == noisy_labels).float().mean().item()

                # REAL grokking metrics - not fake
                generalization_gap = clean_accuracy - noisy_accuracy

                # Calculate grokking score - prevent fake 1.0 scores
                if clean_accuracy > noisy_accuracy and clean_accuracy > 0.7:
                    # Normal case: clean better than noisy
                    grokking_score = noisy_accuracy / clean_accuracy
                elif abs(clean_accuracy - noisy_accuracy) < 0.01 and clean_accuracy > 0.8:
                    # Suspicious: identical or nearly identical scores
                    # This often happens early and gives fake perfect scores
                    grokking_score = 0.5  # Penalize identical scores
                elif noisy_accuracy > clean_accuracy:
                    # Unusual: noisy better than clean (overfit to noise?)
                    grokking_score = 0.3  # Very low score
                else:
                    grokking_score = 0.0  # No grokking

            self.model.train()

            return {
                'clean_accuracy': clean_accuracy,
                'noisy_accuracy': noisy_accuracy,
                'generalization_gap': generalization_gap,
                'grokking_score': grokking_score
            }

        except Exception as e:
            print(f"Validation error: {e}")
            self.model.train()
            return None

    def save_hrm_results(self, filepath: str = "models/hrm_grokfast_results.json"):
        """Save HRM training results"""

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        results = {
            'model_type': 'Real HierarchicalReasoningModel_ACTV1',
            'training_config': self.training_config,
            'training_history': self.training_history,
            'data_generation_stats': {
                'total_generated': self.data_generator.generation_count,
                'base_scenarios': len(self.data_generator.base_scenarios),
                'noise_features': len(self.data_generator.noise_config)
            },
            'final_grokking_score': self.training_history['grokking_score'][-1] if self.training_history['grokking_score'] else 0.0,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        # Save model
        model_path = filepath.replace('.json', '_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.grokfast_optimizer.state_dict(),
            'training_config': self.training_config,
            'use_real_hrm': True  # Always real, no mocks
        }, model_path)

        print(f"HRM results saved to: {filepath}")
        print(f"HRM model saved to: {model_path}")

def main():
    """Main HRM training function"""

    trainer = HRMGrokkingTrainer()  # Always uses real HRM

    print("Starting HRM + GrokFast hierarchical reasoning training...")
    print("Training will continue until hierarchical grokking is achieved...")
    print()

    # Train until hierarchical grokking
    success = trainer.train_until_hierarchical_grokking()

    # Save results
    trainer.save_hrm_results()

    if success:
        print("\n*** HRM TRAINING COMPLETE - HIERARCHICAL GROKKING ACHIEVED! ***")
        print("Model has learned hierarchical reasoning for strategy selection")
    else:
        print("\nWARNING: Training stopped before achieving full hierarchical grokking")
        print("Consider training longer or adjusting hyperparameters")

    return success

if __name__ == "__main__":
    main()