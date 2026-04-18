"""
TRM Hyperparameter Oracle for MOO Optimization

Uses GlobalMOO + Pymoo (WovenMOO) to find optimal hyperparameters for TRM model.

Hyperparameters:
- hidden_dim: Model width [256, 512, 1024, 2048]
- T: Recursion cycles [2, 3, 4, 5]
- n: Latent steps per cycle [4, 6, 8, 10]
- learning_rate: [1e-4, 5e-4, 1e-3, 2e-3]
- batch_size: [64, 128, 256, 512]

Objectives (minimize all):
1. neg_val_accuracy: Negative validation accuracy (minimize = maximize accuracy)
2. param_count: Model parameter count (prefer smaller models)
3. overfit_gap: Train accuracy - Val accuracy gap (prefer generalization)
4. training_time: Seconds per epoch (prefer faster training)

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import time
import logging

# Import MOO infrastructure
try:
    from src.optimization.trading_oracle import TradingOracle, ObjectiveSpec, ConstraintSpec, OracleResult
    from src.optimization.pymoo_adapter import PymooAdapter, PymooConfig
    from src.optimization.robust_pipeline import WovenMOOPipeline, WovenConfig
except ImportError:
    from optimization.trading_oracle import TradingOracle, ObjectiveSpec, ConstraintSpec, OracleResult
    from optimization.pymoo_adapter import PymooAdapter, PymooConfig
    from optimization.robust_pipeline import WovenMOOPipeline, WovenConfig

# Import TRM components
try:
    from src.models.trm_model import TinyRecursiveModel
    from src.training.trm_trainer import TRMTrainer
    from src.training.trm_data_loader import TRMDataModule
except ImportError:
    from models.trm_model import TinyRecursiveModel
    from training.trm_trainer import TRMTrainer
    from training.trm_data_loader import TRMDataModule

logger = logging.getLogger(__name__)


@dataclass
class TRMHyperparameters:
    """TRM hyperparameter configuration"""
    hidden_dim: int = 1024
    T: int = 3  # recursion cycles
    n: int = 6  # latent steps
    learning_rate: float = 5e-4
    batch_size: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hidden_dim': self.hidden_dim,
            'T': self.T,
            'n': self.n,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }


class TRMHyperparameterOracle(TradingOracle):
    """
    Oracle for TRM hyperparameter optimization.

    Evaluates hyperparameter configurations by training for a few epochs
    and measuring validation performance, model size, and training efficiency.
    """

    # Hyperparameter bounds (normalized to [0, 1])
    HIDDEN_DIM_OPTIONS = [256, 512, 1024, 2048]
    T_OPTIONS = [2, 3, 4, 5]
    N_OPTIONS = [4, 6, 8, 10]
    LR_OPTIONS = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]
    BATCH_OPTIONS = [64, 128, 256, 512]

    def __init__(
        self,
        data_path: str,
        eval_epochs: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize TRM hyperparameter oracle.

        Args:
            data_path: Path to training data parquet
            eval_epochs: Number of epochs for evaluation (default 3 for speed)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.data_path = Path(data_path)
        self.eval_epochs = eval_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache data module (only load once)
        self._data_module = None
        self._class_weights = None

    def _get_data_module(self) -> TRMDataModule:
        """Lazy load data module"""
        if self._data_module is None:
            self._data_module = TRMDataModule(
                data_path=str(self.data_path),
                random_seed=42
            )
        return self._data_module

    def _get_class_weights(self) -> torch.Tensor:
        """Get class weights for imbalanced data"""
        if self._class_weights is None:
            data_module = self._get_data_module()
            self._class_weights = data_module.compute_class_weights(
                num_classes=8,
                max_weight=10.0
            )
        return self._class_weights

    def _decode_hyperparameters(self, x: np.ndarray) -> TRMHyperparameters:
        """
        Decode normalized [0,1] vector to hyperparameters.

        Args:
            x: Array of shape (5,) with values in [0, 1]
               [hidden_dim_idx, T_idx, n_idx, lr_idx, batch_idx]

        Returns:
            TRMHyperparameters instance
        """
        # Convert continuous [0,1] to discrete indices
        hidden_idx = int(x[0] * (len(self.HIDDEN_DIM_OPTIONS) - 1) + 0.5)
        t_idx = int(x[1] * (len(self.T_OPTIONS) - 1) + 0.5)
        n_idx = int(x[2] * (len(self.N_OPTIONS) - 1) + 0.5)
        lr_idx = int(x[3] * (len(self.LR_OPTIONS) - 1) + 0.5)
        batch_idx = int(x[4] * (len(self.BATCH_OPTIONS) - 1) + 0.5)

        # Clamp indices
        hidden_idx = min(max(hidden_idx, 0), len(self.HIDDEN_DIM_OPTIONS) - 1)
        t_idx = min(max(t_idx, 0), len(self.T_OPTIONS) - 1)
        n_idx = min(max(n_idx, 0), len(self.N_OPTIONS) - 1)
        lr_idx = min(max(lr_idx, 0), len(self.LR_OPTIONS) - 1)
        batch_idx = min(max(batch_idx, 0), len(self.BATCH_OPTIONS) - 1)

        return TRMHyperparameters(
            hidden_dim=self.HIDDEN_DIM_OPTIONS[hidden_idx],
            T=self.T_OPTIONS[t_idx],
            n=self.N_OPTIONS[n_idx],
            learning_rate=self.LR_OPTIONS[lr_idx],
            batch_size=self.BATCH_OPTIONS[batch_idx]
        )

    def get_objectives(self) -> List[ObjectiveSpec]:
        """Define optimization objectives"""
        return [
            ObjectiveSpec(
                name='neg_val_accuracy',
                direction='minimize',  # Minimize negative = maximize accuracy
                weight=1.0,
                bounds=(-100.0, 0.0)
            ),
            ObjectiveSpec(
                name='param_count_millions',
                direction='minimize',  # Prefer smaller models
                weight=0.3,
                bounds=(0.0, 50.0)
            ),
            ObjectiveSpec(
                name='overfit_gap',
                direction='minimize',  # Prefer generalization
                weight=0.5,
                bounds=(0.0, 50.0)
            ),
            ObjectiveSpec(
                name='epoch_time_seconds',
                direction='minimize',  # Prefer faster training
                weight=0.2,
                bounds=(0.0, 60.0)
            )
        ]

    def get_constraints(self) -> List[ConstraintSpec]:
        """Define constraints"""
        return [
            ConstraintSpec(
                name='min_accuracy',
                type='ineq_le',
                expression='neg_val_accuracy <= -10.0'  # At least 10% accuracy
            ),
            ConstraintSpec(
                name='max_params',
                type='ineq_le',
                expression='param_count_millions <= 20.0'  # Max 20M params
            )
        ]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get variable bounds (all [0, 1] for normalized encoding)"""
        n_vars = 5
        lower = np.zeros(n_vars)
        upper = np.ones(n_vars)
        return lower, upper

    def evaluate(self, x: np.ndarray) -> OracleResult:
        """
        Evaluate a hyperparameter configuration.

        Args:
            x: Normalized hyperparameter vector (5,)

        Returns:
            OracleResult with objectives and feasibility
        """
        # Decode hyperparameters
        hp = self._decode_hyperparameters(x)
        logger.info(f"Evaluating: {hp.to_dict()}")

        try:
            # Create model
            model = TinyRecursiveModel(
                input_dim=10,
                hidden_dim=hp.hidden_dim,
                output_dim=8,
                num_latent_steps=hp.n,
                num_recursion_cycles=hp.T
            )

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            param_millions = param_count / 1_000_000

            # Get data loaders
            data_module = self._get_data_module()
            train_loader, val_loader, _ = data_module.create_dataloaders(
                batch_size=hp.batch_size,
                shuffle=True
            )

            # Create trainer
            trainer = TRMTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=hp.learning_rate,
                loss_type='nnc',
                max_grad_norm=1.0,
                class_weights=self._get_class_weights(),
                T=hp.T,
                n=hp.n,
                device=torch.device(self.device)
            )

            # Train for eval_epochs and measure time
            train_accs = []
            val_accs = []
            epoch_times = []

            for epoch in range(self.eval_epochs):
                start_time = time.time()
                train_metrics = trainer.train_epoch()
                epoch_time = time.time() - start_time

                val_metrics = trainer.validate()

                train_accs.append(train_metrics['accuracy'])
                val_accs.append(val_metrics['accuracy'])
                epoch_times.append(epoch_time)

            # Compute objectives
            final_val_acc = val_accs[-1]
            final_train_acc = train_accs[-1]
            avg_epoch_time = np.mean(epoch_times)
            overfit_gap = max(0, final_train_acc - final_val_acc)

            objectives = {
                'neg_val_accuracy': -final_val_acc,
                'param_count_millions': param_millions,
                'overfit_gap': overfit_gap,
                'epoch_time_seconds': avg_epoch_time
            }

            # Check constraints
            feasible = (
                final_val_acc >= 10.0 and  # At least 10% accuracy
                param_millions <= 20.0     # Max 20M params
            )

            logger.info(f"Results: val_acc={final_val_acc:.2f}%, params={param_millions:.2f}M, "
                       f"gap={overfit_gap:.2f}%, time={avg_epoch_time:.2f}s")

            return OracleResult(
                objectives=objectives,
                feasible=feasible,
                metadata={
                    'hyperparameters': hp.to_dict(),
                    'train_history': train_accs,
                    'val_history': val_accs,
                    'epoch_times': epoch_times
                }
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return infeasible result on error
            return OracleResult(
                objectives={
                    'neg_val_accuracy': 0.0,
                    'param_count_millions': 100.0,
                    'overfit_gap': 100.0,
                    'epoch_time_seconds': 100.0
                },
                feasible=False,
                metadata={'error': str(e)}
            )


def run_trm_hyperparameter_optimization(
    data_path: str,
    n_generations: int = 20,
    population_size: int = 20,
    eval_epochs: int = 3,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run MOO hyperparameter optimization for TRM.

    Args:
        data_path: Path to training data
        n_generations: Number of NSGA-II generations
        population_size: Population size per generation
        eval_epochs: Epochs per evaluation (3 = fast, 5 = more accurate)
        output_path: Optional path to save results JSON

    Returns:
        Dictionary with Pareto front and best configurations
    """
    import json
    from datetime import datetime

    print("=" * 70)
    print("TRM HYPERPARAMETER OPTIMIZATION (MOO)")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Generations: {n_generations}, Population: {population_size}")
    print(f"Eval epochs per config: {eval_epochs}")
    print()

    # Create oracle
    oracle = TRMHyperparameterOracle(
        data_path=data_path,
        eval_epochs=eval_epochs
    )

    # Configure Pymoo
    pymoo_config = PymooConfig(
        algorithm='NSGA2',
        population_size=population_size,
        n_generations=n_generations,
        seed=42,
        verbose=True
    )

    # Run optimization
    adapter = PymooAdapter(config=pymoo_config)

    print("Starting optimization...")
    start_time = time.time()
    result = adapter.optimize(oracle)
    total_time = time.time() - start_time

    print(f"\nOptimization complete in {total_time:.1f}s")
    print(f"Pareto front size: {len(result.pareto_front)}")

    # Extract best configurations
    best_configs = []
    for i, (obj, x) in enumerate(zip(result.pareto_front, result.pareto_set)):
        hp = oracle._decode_hyperparameters(x)
        config = {
            'rank': i + 1,
            'hyperparameters': hp.to_dict(),
            'objectives': {
                'val_accuracy': -obj[0],
                'param_count_millions': obj[1],
                'overfit_gap': obj[2],
                'epoch_time_seconds': obj[3]
            }
        }
        best_configs.append(config)

    # Sort by validation accuracy
    best_configs.sort(key=lambda c: c['objectives']['val_accuracy'], reverse=True)

    # Print top 5
    print("\nTop 5 Configurations (by validation accuracy):")
    print("-" * 70)
    for i, config in enumerate(best_configs[:5], 1):
        hp = config['hyperparameters']
        obj = config['objectives']
        print(f"{i}. hidden={hp['hidden_dim']}, T={hp['T']}, n={hp['n']}, "
              f"lr={hp['learning_rate']:.0e}, batch={hp['batch_size']}")
        print(f"   Val Acc: {obj['val_accuracy']:.2f}%, Params: {obj['param_count_millions']:.2f}M, "
              f"Gap: {obj['overfit_gap']:.2f}%, Time: {obj['epoch_time_seconds']:.2f}s")

    # Prepare results
    results = {
        'optimization_config': {
            'n_generations': n_generations,
            'population_size': population_size,
            'eval_epochs': eval_epochs,
            'total_evaluations': result.n_evaluations,
            'total_time_seconds': total_time
        },
        'pareto_front': best_configs,
        'best_config': best_configs[0] if best_configs else None,
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys

    # Default paths
    data_path = Path(__file__).parent.parent.parent / 'data' / 'trm_training' / 'black_swan_labels.parquet'
    output_path = Path(__file__).parent.parent.parent / 'results' / 'trm_hyperparameter_moo.json'

    # Run optimization
    results = run_trm_hyperparameter_optimization(
        data_path=str(data_path),
        n_generations=15,
        population_size=16,
        eval_epochs=3,
        output_path=str(output_path)
    )

    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION:")
    print("=" * 70)
    if results['best_config']:
        best = results['best_config']
        print(f"Hyperparameters: {best['hyperparameters']}")
        print(f"Expected Val Accuracy: {best['objectives']['val_accuracy']:.2f}%")
