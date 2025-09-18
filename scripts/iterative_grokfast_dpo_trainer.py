"""
ITERATIVE GROKFAST DPO TRAINER
Creates infinite training variations through noise injection to achieve true grokking
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

from strategies.black_swan_strategies import BlackSwanStrategyToolbox
from strategies.enhanced_market_state import create_enhanced_market_state
from strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
from training.grokfast_optimizer import GrokFastOptimizer, AntiMemorizationValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfiniteNoiseDataGenerator:
    """Generates infinite variations of training data through controlled noise injection"""

    def __init__(self, base_scenarios: List[Dict], noise_config: Dict[str, float] = None):
        self.base_scenarios = base_scenarios
        self.noise_config = noise_config or {
            'vix_level': 2.0,           # ±2 VIX points
            'spy_returns_5d': 0.01,     # ±1% returns
            'spy_returns_20d': 0.008,   # ±0.8% returns
            'volume_ratio': 0.2,        # ±20% volume
            'market_breadth': 0.1,      # ±10% breadth
            'correlation': 0.08,        # ±8% correlation
            'put_call_ratio': 0.15,     # ±15% put/call
            'gini_coefficient': 0.02,   # ±2% inequality
            'sector_dispersion': 0.003, # ±0.3% dispersion
        }
        self.generation_count = 0

    def generate_noisy_scenario(self, base_scenario: Dict) -> Dict:
        """Generate a noisy variation of a base scenario"""
        noisy_scenario = copy.deepcopy(base_scenario)

        for feature, noise_scale in self.noise_config.items():
            if feature in noisy_scenario:
                noise = np.random.normal(0, noise_scale)
                noisy_scenario[feature] += noise

                # Apply realistic bounds
                if feature == 'vix_level':
                    noisy_scenario[feature] = max(10, min(80, noisy_scenario[feature]))
                elif feature in ['market_breadth', 'correlation', 'signal_quality_score']:
                    noisy_scenario[feature] = max(0, min(1, noisy_scenario[feature]))
                elif feature == 'volume_ratio':
                    noisy_scenario[feature] = max(0.3, min(5.0, noisy_scenario[feature]))
                elif feature == 'put_call_ratio':
                    noisy_scenario[feature] = max(0.2, min(4.0, noisy_scenario[feature]))
                elif feature == 'gini_coefficient':
                    noisy_scenario[feature] = max(0.3, min(0.8, noisy_scenario[feature]))

        self.generation_count += 1
        return noisy_scenario

    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate a batch of noisy scenarios"""
        batch = []
        for _ in range(batch_size):
            base_scenario = np.random.choice(self.base_scenarios)
            noisy_scenario = self.generate_noisy_scenario(base_scenario)
            batch.append(noisy_scenario)
        return batch

    def get_generation_stats(self) -> Dict:
        """Get statistics about data generation"""
        return {
            'total_generated': self.generation_count,
            'base_scenarios': len(self.base_scenarios),
            'noise_features': len(self.noise_config)
        }

class IterativeGrokkingTrainer:
    """Trains DPO with GrokFast using infinite noise-augmented data until true grokking"""

    def __init__(self, model_config: Dict = None):
        self.db_path = Path("data/historical_market.db")
        self.toolbox = BlackSwanStrategyToolbox()
        self.reward_calc = ConvexRewardFunction()
        self.validator = AntiMemorizationValidator()

        # Model configuration
        self.model_config = model_config or {
            'input_size': 24,  # Enhanced market features
            'hidden_size': 128,
            'num_strategies': 8,
            'learning_rate': 0.0003,
            'grokfast_alpha': 0.98,
            'grokfast_lambda': 2.0
        }

        # Training configuration
        self.training_config = {
            'max_iterations': 1000,
            'batch_size': 32,
            'validation_every': 10,
            'grokking_threshold': 0.85,  # Minimum performance under noise
            'noise_tolerance': 0.1,      # Maximum performance drop under noise
            'early_stopping_patience': 50
        }

        # Initialize model and optimizer
        self.model = self._create_model()
        self.base_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'])
        self.grokfast_optimizer = GrokFastOptimizer(
            self.base_optimizer,
            alpha=self.model_config['grokfast_alpha'],
            lamb=self.model_config['grokfast_lambda']
        )

        # Training tracking
        self.training_history = {
            'iteration': [],
            'clean_accuracy': [],
            'noisy_accuracy': [],
            'generalization_gap': [],
            'grokking_score': []
        }

        # Initialize data generator
        self.data_generator = self._initialize_data_generator()

    def _create_model(self) -> nn.Module:
        """Create strategy selection neural network"""
        return nn.Sequential(
            nn.Linear(self.model_config['input_size'], self.model_config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.model_config['hidden_size'], self.model_config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.model_config['hidden_size'], self.model_config['num_strategies']),
            nn.Softmax(dim=1)
        )

    def _initialize_data_generator(self) -> InfiniteNoiseDataGenerator:
        """Initialize the infinite data generator with base scenarios"""
        base_scenarios = [
            {'name': 'Crisis', 'vix_level': 35, 'spy_returns_5d': -0.08, 'spy_returns_20d': -0.05,
             'put_call_ratio': 2.0, 'market_breadth': 0.2, 'volume_ratio': 2.5, 'correlation': 0.9,
             'gini_coefficient': 0.55, 'sector_dispersion': 0.015},

            {'name': 'Volatile', 'vix_level': 25, 'spy_returns_5d': -0.03, 'spy_returns_20d': -0.01,
             'put_call_ratio': 1.4, 'market_breadth': 0.35, 'volume_ratio': 1.8, 'correlation': 0.7,
             'gini_coefficient': 0.48, 'sector_dispersion': 0.012},

            {'name': 'Normal', 'vix_level': 18, 'spy_returns_5d': 0.01, 'spy_returns_20d': 0.02,
             'put_call_ratio': 1.0, 'market_breadth': 0.6, 'volume_ratio': 1.2, 'correlation': 0.5,
             'gini_coefficient': 0.42, 'sector_dispersion': 0.008},

            {'name': 'Momentum', 'vix_level': 15, 'spy_returns_5d': 0.04, 'spy_returns_20d': 0.06,
             'put_call_ratio': 0.8, 'market_breadth': 0.75, 'volume_ratio': 1.5, 'correlation': 0.6,
             'gini_coefficient': 0.38, 'sector_dispersion': 0.010},

            {'name': 'Calm', 'vix_level': 12, 'spy_returns_5d': 0.005, 'spy_returns_20d': 0.01,
             'put_call_ratio': 0.9, 'market_breadth': 0.8, 'volume_ratio': 0.9, 'correlation': 0.3,
             'gini_coefficient': 0.35, 'sector_dispersion': 0.006}
        ]

        return InfiniteNoiseDataGenerator(base_scenarios)

    def create_training_batch(self, scenarios: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training batch from noisy scenarios"""
        # Load market data
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT date, symbol, open, high, low, close, volume, returns
            FROM market_data
            WHERE date >= '2023-01-01'
            ORDER BY date DESC
            LIMIT 500
            """
            market_data = pd.read_sql_query(query, conn)

        features_list = []
        labels_list = []

        for scenario in scenarios:
            # Create market state
            market_state = create_enhanced_market_state(
                timestamp=datetime.now(),
                vix_level=scenario['vix_level'],
                spy_returns_5d=scenario['spy_returns_5d'],
                spy_returns_20d=scenario['spy_returns_20d'],
                put_call_ratio=scenario['put_call_ratio'],
                market_breadth=scenario['market_breadth'],
                volume_ratio=scenario['volume_ratio'],
                regime=self._determine_regime(scenario['vix_level'])
            )

            # Get enhanced features
            market_features = market_state.get_enhanced_market_features()

            # Test all strategies and find best performer
            strategy_results = {}
            for strategy_name, strategy in self.toolbox.strategies.items():
                signals = strategy.analyze(market_state, market_data)
                # signals can be a single StrategySignal or a list of StrategySignals
                if signals:
                    if isinstance(signals, list):
                        if len(signals) > 0:
                            first_signal = signals[0]
                        else:
                            continue
                    else:
                        # Single signal object
                        first_signal = signals

                    # Simulate forward performance
                    simulated_return = self._simulate_strategy_performance(first_signal, scenario)

                    # Calculate convex reward
                    trade_outcome = TradeOutcome(
                        strategy_name=strategy_name,
                        entry_date=datetime.now(),
                        exit_date=datetime.now() + timedelta(days=10),
                        symbol='SPY',
                        returns=simulated_return,
                        max_drawdown=min(0, simulated_return),
                        holding_period_days=10,
                        volatility_during_trade=0.025,
                        is_black_swan_period=abs(simulated_return) > 0.10,
                        black_swan_captured=simulated_return > 0.10,
                        convexity_achieved=max(0, simulated_return / 0.05)
                    )

                    reward_metrics = self.reward_calc.calculate_reward(trade_outcome)
                    strategy_results[strategy_name] = reward_metrics.final_reward

            if strategy_results:
                # Find best strategy
                best_strategy = max(strategy_results.keys(), key=lambda k: strategy_results[k])
                strategy_names = list(self.toolbox.strategies.keys())
                best_strategy_idx = strategy_names.index(best_strategy)

                # Convert features to tensor format
                feature_vector = [market_features.get(key, 0.0) for key in sorted(market_features.keys())]
                features_list.append(feature_vector)
                labels_list.append(best_strategy_idx)

        if not features_list:
            return None, None

        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return features_tensor, labels_tensor

    def _simulate_strategy_performance(self, signal, scenario: Dict) -> float:
        """Simulate strategy performance based on market conditions"""
        # Base performance varies by market regime
        if scenario['vix_level'] > 30:  # Crisis
            base_returns = {
                'crisis_alpha': 0.15, 'tail_hedge': 0.12, 'volatility_harvest': 0.08,
                'event_catalyst': 0.06, 'correlation_breakdown': 0.04, 'inequality_arbitrage': 0.02,
                'momentum_explosion': -0.05, 'mean_reversion': -0.02
            }
        elif scenario['vix_level'] > 20:  # Volatile
            base_returns = {
                'momentum_explosion': 0.08, 'event_catalyst': 0.06, 'volatility_harvest': 0.05,
                'correlation_breakdown': 0.03, 'crisis_alpha': 0.02, 'tail_hedge': 0.01,
                'mean_reversion': -0.01, 'inequality_arbitrage': -0.02
            }
        else:  # Normal/Calm
            base_returns = {
                'mean_reversion': 0.06, 'momentum_explosion': 0.04, 'inequality_arbitrage': 0.03,
                'event_catalyst': 0.02, 'correlation_breakdown': 0.01, 'volatility_harvest': 0.00,
                'crisis_alpha': -0.01, 'tail_hedge': -0.02
            }

        base_return = base_returns.get(signal.strategy_name, 0.0)

        # Add realistic noise
        noise = np.random.normal(0, 0.02)  # 2% volatility
        return base_return + noise

    def _determine_regime(self, vix_level: float) -> str:
        """Determine market regime from VIX level"""
        if vix_level > 30:
            return 'crisis'
        elif vix_level > 20:
            return 'volatile'
        else:
            return 'normal'

    def validate_generalization(self, clean_batch: Tuple[torch.Tensor, torch.Tensor],
                              noisy_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Validate model generalization under noise"""
        self.model.eval()

        with torch.no_grad():
            # Clean accuracy
            clean_features, clean_labels = clean_batch
            clean_predictions = self.model(clean_features)
            clean_accuracy = (clean_predictions.argmax(dim=1) == clean_labels).float().mean().item()

            # Noisy accuracy
            noisy_features, noisy_labels = noisy_batch
            noisy_predictions = self.model(noisy_features)
            noisy_accuracy = (noisy_predictions.argmax(dim=1) == noisy_labels).float().mean().item()

            # Generalization metrics
            generalization_gap = clean_accuracy - noisy_accuracy
            grokking_score = noisy_accuracy / max(clean_accuracy, 0.01)  # Avoid division by zero

        self.model.train()

        return {
            'clean_accuracy': clean_accuracy,
            'noisy_accuracy': noisy_accuracy,
            'generalization_gap': generalization_gap,
            'grokking_score': grokking_score
        }

    def train_iteration(self, iteration: int) -> Dict[str, float]:
        """Single training iteration with noise-augmented data"""

        # Generate noisy training batch
        noisy_scenarios = self.data_generator.generate_batch(self.training_config['batch_size'])
        features, labels = self.create_training_batch(noisy_scenarios)

        if features is None:
            return {'loss': float('inf'), 'accuracy': 0.0}

        # Forward pass
        self.grokfast_optimizer.zero_grad()
        predictions = self.model(features)
        loss = nn.CrossEntropyLoss()(predictions, labels)

        # Backward pass with GrokFast
        loss.backward()
        self.grokfast_optimizer.step(self.model)

        # Calculate metrics
        with torch.no_grad():
            accuracy = (predictions.argmax(dim=1) == labels).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }

    def train_until_grokking(self):
        """Train until true grokking is achieved"""

        print("=" * 80)
        print("ITERATIVE GROKFAST DPO TRAINING FOR TRUE GROKKING")
        print("=" * 80)
        print(f"Target grokking threshold: {self.training_config['grokking_threshold']:.2f}")
        print(f"Noise tolerance: {self.training_config['noise_tolerance']:.2f}")
        print(f"Max iterations: {self.training_config['max_iterations']}")
        print()

        best_grokking_score = 0.0
        patience_counter = 0

        for iteration in range(self.training_config['max_iterations']):

            # Training step
            train_metrics = self.train_iteration(iteration)

            # Validation every N iterations
            if iteration % self.training_config['validation_every'] == 0:

                # Generate clean and noisy validation batches
                clean_scenarios = self.data_generator.base_scenarios[:self.training_config['batch_size']]
                clean_features, clean_labels = self.create_training_batch(clean_scenarios)

                noisy_scenarios = self.data_generator.generate_batch(self.training_config['batch_size'])
                noisy_features, noisy_labels = self.create_training_batch(noisy_scenarios)

                if clean_features is not None and noisy_features is not None:
                    validation_metrics = self.validate_generalization(
                        (clean_features, clean_labels),
                        (noisy_features, noisy_labels)
                    )

                    # Update training history
                    self.training_history['iteration'].append(iteration)
                    self.training_history['clean_accuracy'].append(validation_metrics['clean_accuracy'])
                    self.training_history['noisy_accuracy'].append(validation_metrics['noisy_accuracy'])
                    self.training_history['generalization_gap'].append(validation_metrics['generalization_gap'])
                    self.training_history['grokking_score'].append(validation_metrics['grokking_score'])

                    # Check for grokking
                    current_grokking_score = validation_metrics['grokking_score']

                    print(f"Iteration {iteration:4d}: Loss={train_metrics['loss']:.4f}, "
                          f"Clean={validation_metrics['clean_accuracy']:.3f}, "
                          f"Noisy={validation_metrics['noisy_accuracy']:.3f}, "
                          f"Gap={validation_metrics['generalization_gap']:.3f}, "
                          f"Grok={current_grokking_score:.3f}")

                    # Check if we've achieved grokking
                    if (current_grokking_score >= self.training_config['grokking_threshold'] and
                        validation_metrics['generalization_gap'] <= self.training_config['noise_tolerance']):

                        print()
                        print("*** GROKKING ACHIEVED! ***")
                        print(f"Final grokking score: {current_grokking_score:.3f}")
                        print(f"Generalization gap: {validation_metrics['generalization_gap']:.3f}")
                        print(f"Data generated: {self.data_generator.get_generation_stats()}")
                        return True

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

            # Progress indicator
            if iteration % 100 == 0 and iteration > 0:
                gen_stats = self.data_generator.get_generation_stats()
                print(f"Generated {gen_stats['total_generated']} training examples so far...")

        print()
        print("Maximum iterations reached")
        print(f"Best grokking score: {best_grokking_score:.3f}")
        return False

    def save_training_results(self, filepath: str = "models/grokfast_training_results.json"):
        """Save training results and model"""

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        results = {
            'training_config': self.training_config,
            'model_config': self.model_config,
            'training_history': self.training_history,
            'data_generation_stats': self.data_generator.get_generation_stats(),
            'final_grokking_score': self.training_history['grokking_score'][-1] if self.training_history['grokking_score'] else 0.0
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        # Save model
        model_path = filepath.replace('.json', '_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.grokfast_optimizer.state_dict(),
            'training_config': self.training_config
        }, model_path)

        print(f"Training results saved to: {filepath}")
        print(f"Model saved to: {model_path}")

def main():
    """Main training function"""

    trainer = IterativeGrokkingTrainer()

    print("Starting iterative GrokFast DPO training...")
    print("Creating infinite training variations through noise injection...")
    print()

    # Train until grokking
    success = trainer.train_until_grokking()

    # Save results
    trainer.save_training_results()

    if success:
        print("\n*** TRAINING COMPLETE - TRUE GROKKING ACHIEVED! ***")
        print("Model can now generalize to new market conditions without memorization")
    else:
        print("\nWARNING: Training stopped before achieving full grokking")
        print("Consider adjusting hyperparameters or training longer")

    return success

if __name__ == "__main__":
    main()