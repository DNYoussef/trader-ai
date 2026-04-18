"""
Phase 3: Strategy Generation Training

Trains StrategyGeneratorModel to create optimal allocations using:
1. Blended mode: Learn soft weights over 8 base strategies
2. Direct mode: Learn raw SPY/TLT/Cash allocations

Uses continuous RL (policy gradient with Dirichlet distribution).
Compares performance against Phase 2 discrete strategy selection.
"""
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.strategy_generator import StrategyGeneratorModel, create_strategy_generator
from src.simulation.continuous_simulator import BatchContinuousSimulator, create_continuous_simulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyDataset(Dataset):
    """Dataset providing features and time indices."""

    def __init__(self, features: np.ndarray, time_indices: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.time_indices = torch.tensor(time_indices, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.time_indices[idx]


class ContinuousPolicyTrainer:
    """
    Policy gradient trainer for continuous strategy generation.

    Supports three action modes:
    - discrete: Sample from categorical over 8 strategies
    - blended: Sample strategy weights from Dirichlet
    - direct: Sample allocations from Dirichlet

    Uses REINFORCE with baseline and entropy regularization.
    """

    def __init__(
        self,
        model: StrategyGeneratorModel,
        simulator: BatchContinuousSimulator,
        mode: str = 'blended',
        lr: float = 3e-4,
        baseline_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.simulator = simulator
        self.mode = mode
        self.device = device
        self.entropy_coef = entropy_coef

        # Policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='max', factor=0.5, patience=10
        )

        # Value baseline network
        hidden_dim = model.hidden_dim
        self.baseline = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)

        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(), lr=baseline_lr
        )

        # Metrics tracking
        self.train_history = []
        self.val_history = []

    def compute_baseline(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Compute value baseline from latent state."""
        return self.baseline(latent_state.detach()).squeeze(-1)

    def train_step(
        self,
        features: torch.Tensor,
        time_indices: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        features = features.to(self.device)
        time_indices = time_indices.to(self.device)

        # Sample action from policy
        action, log_prob, info = self.model.sample_action(
            features, mode=self.mode, deterministic=False
        )

        # Get reward from simulator
        reward = self.simulator.get_reward(action, time_indices, mode=self.mode)

        # Compute baseline
        baseline_value = self.compute_baseline(info['latent_state'])

        # Advantage
        advantage = reward - baseline_value.detach()

        # Policy loss (REINFORCE)
        policy_loss = -(log_prob * advantage).mean()

        # Entropy bonus
        entropy = info['entropy'].mean()
        entropy_loss = -self.entropy_coef * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss

        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Update baseline (separate forward pass to avoid graph reuse)
        with torch.no_grad():
            _, _, new_info = self.model.sample_action(
                features, mode=self.mode, deterministic=True
            )
        baseline_pred = self.baseline(new_info['latent_state']).squeeze(-1)
        baseline_loss = F.mse_loss(baseline_pred, reward.detach())

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        # Compute accuracy metrics
        if self.mode == 'discrete':
            optimal = self.simulator.get_optimal_strategy(time_indices)
            accuracy = (action == optimal).float().mean().item()
        elif self.mode == 'blended':
            # For blended: action is strategy_weights (batch, 8)
            _, blended_info = self.simulator.get_blended_reward(action, time_indices)
            accuracy = blended_info.get('beat_best', torch.zeros(1)).mean().item()
        else:  # direct mode
            # For direct: action is allocations (batch, 3)
            _, direct_info = self.simulator.get_direct_reward(action, time_indices)
            accuracy = direct_info.get('beat_best', torch.zeros(1)).mean().item()

        return {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': reward.mean().item(),
            'accuracy': accuracy * 100,
            'mean_advantage': advantage.mean().item(),
            'temperature': self.model.temperature.item(),
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        compare_modes: bool = True
    ) -> Dict[str, float]:
        """Evaluate on held-out data."""
        self.model.eval()

        total_reward = 0
        total_samples = 0
        beat_best_count = 0

        # For comparison
        discrete_rewards = []
        blended_rewards = []
        direct_rewards = []

        with torch.no_grad():
            for features, time_indices in dataloader:
                features = features.to(self.device)
                time_indices = time_indices.to(self.device)
                batch_size = len(features)

                # Current mode (deterministic)
                action, _, info = self.model.sample_action(
                    features, mode=self.mode, deterministic=True
                )
                reward = self.simulator.get_reward(action, time_indices, mode=self.mode)
                total_reward += reward.sum().item()
                total_samples += batch_size

                # Check if beating best discrete
                if self.mode == 'blended':
                    # action is strategy_weights (batch, 8)
                    _, eval_info = self.simulator.get_blended_reward(action, time_indices)
                    beat_best_count += eval_info.get('beat_best', torch.zeros(1)).sum().item()
                elif self.mode == 'direct':
                    # action is allocations (batch, 3)
                    _, eval_info = self.simulator.get_direct_reward(action, time_indices)
                    beat_best_count += eval_info.get('beat_best', torch.zeros(1)).sum().item()

                # Compare all modes
                if compare_modes:
                    # Discrete
                    d_action, _, _ = self.model.sample_action(features, 'discrete', True)
                    d_reward = self.simulator.get_reward(d_action, time_indices, 'discrete')
                    discrete_rewards.append(d_reward.mean().item())

                    # Blended
                    b_action, _, b_info = self.model.sample_action(features, 'blended', True)
                    b_reward = self.simulator.get_reward(b_action, time_indices, 'blended')
                    blended_rewards.append(b_reward.mean().item())

                    # Direct
                    dir_action, _, _ = self.model.sample_action(features, 'direct', True)
                    dir_reward = self.simulator.get_reward(dir_action, time_indices, 'direct')
                    direct_rewards.append(dir_reward.mean().item())

        self.model.train()

        results = {
            'mean_reward': total_reward / total_samples,
            'beat_best_rate': beat_best_count / total_samples * 100,
        }

        if compare_modes:
            results['discrete_reward'] = np.mean(discrete_rewards)
            results['blended_reward'] = np.mean(blended_rewards)
            results['direct_reward'] = np.mean(direct_rewards)

        return results


def load_data_and_simulator(
    data_path: str,
    returns_path: Optional[str] = None,
    horizon_days: int = 5,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, BatchContinuousSimulator]:
    """Load data and create simulator."""
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    features = np.stack(df['features'].values).astype(np.float32)
    time_indices = np.arange(len(features))

    # Load returns
    returns_path = Path(returns_path or 'D:/Projects/trader-ai/data/trm_training/market_returns.parquet')

    if returns_path.exists():
        returns_df = pd.read_parquet(returns_path)
        spy_returns = returns_df['spy_return'].values
        tlt_returns = returns_df['tlt_return'].values
    else:
        logger.info("Downloading return data...")
        import yfinance as yf
        spy = yf.download('SPY', start='2000-01-01', end='2024-12-31', progress=False)
        tlt = yf.download('TLT', start='2000-01-01', end='2024-12-31', progress=False)

        spy_returns = spy['Close'].pct_change().dropna().values
        tlt_returns = tlt['Close'].pct_change().reindex(
            spy['Close'].pct_change().dropna().index
        ).fillna(0).values

        min_len = min(len(spy_returns), len(features))
        spy_returns = spy_returns[-min_len:]
        tlt_returns = tlt_returns[-min_len:]
        time_indices = time_indices[:min_len]
        features = features[:min_len]

    # Create simulator
    simulator = create_continuous_simulator(
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        horizon_days=horizon_days,
        device=device
    )

    logger.info(f"Features: {features.shape}, Simulator samples: {simulator.n_samples}")

    return features, time_indices, simulator


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Strategy Generation Training')
    parser.add_argument('--data', type=str,
                       default='D:/Projects/trader-ai/data/trm_training/labels_110_features.parquet')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to Phase 2 RL checkpoint to initialize from')
    parser.add_argument('--mode', type=str, default='blended',
                       choices=['discrete', 'blended', 'direct'],
                       help='Action space mode')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load data
    features, time_indices, simulator = load_data_and_simulator(
        args.data, horizon_days=args.horizon, device=args.device
    )

    # Normalize features
    norm_path = Path('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json')
    if norm_path.exists():
        with open(norm_path) as f:
            norm_params = json.load(f)
        mean = np.array(norm_params['mean'])
        std = np.array(norm_params['std'])
    else:
        mean = features.mean(axis=0)
        std = features.std(axis=0)

    std[std < 1e-7] = 1.0
    features_norm = (features - mean) / std

    # Split data
    n_samples = len(features_norm)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_dataset = StrategyDataset(features_norm[:n_train], time_indices[:n_train])
    val_dataset = StrategyDataset(features_norm[n_train:n_train+n_val], time_indices[n_train:n_train+n_val])
    test_dataset = StrategyDataset(features_norm[n_train+n_val:], time_indices[n_train+n_val:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    input_dim = features_norm.shape[1]
    model = create_strategy_generator({
        'input_dim': input_dim,
        'hidden_dim': 128,
        'output_mode': 'all'
    })

    # Load Phase 2 weights if available
    if args.checkpoint:
        model.load_from_trm(args.checkpoint)

    # Create trainer
    trainer = ContinuousPolicyTrainer(
        model=model,
        simulator=simulator,
        mode=args.mode,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    # Training
    logger.info("=" * 60)
    logger.info(f"PHASE 3: STRATEGY GENERATION ({args.mode.upper()} MODE)")
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Horizon: {args.horizon} days, LR: {args.lr}")
    logger.info("=" * 60)

    best_val_reward = -float('inf')
    best_epoch = 0
    save_dir = Path('D:/Projects/trader-ai/models/strategy_generator')
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train epoch
        epoch_metrics = {'policy_loss': [], 'entropy': [], 'mean_reward': [], 'accuracy': []}

        for features, time_indices in train_loader:
            metrics = trainer.train_step(features, time_indices)
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])

        # Validate
        val_metrics = trainer.evaluate(val_loader, compare_modes=True)

        # Update scheduler
        trainer.scheduler.step(val_metrics['mean_reward'])

        # Log
        train_rew = np.mean(epoch_metrics['mean_reward'])
        train_ent = np.mean(epoch_metrics['entropy'])
        val_rew = val_metrics['mean_reward']
        beat_rate = val_metrics.get('beat_best_rate', 0)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train: Rew={train_rew:.3f} Ent={train_ent:.3f} | "
            f"Val: Rew={val_rew:.3f} Beat={beat_rate:.1f}% | "
            f"LR={trainer.policy_optimizer.param_groups[0]['lr']:.1e}"
        )

        # Mode comparison
        if 'discrete_reward' in val_metrics:
            logger.info(
                f"         Modes: Discrete={val_metrics['discrete_reward']:.3f} "
                f"Blended={val_metrics['blended_reward']:.3f} "
                f"Direct={val_metrics['direct_reward']:.3f}"
            )

        # Save best
        if val_rew > best_val_reward:
            best_val_reward = val_rew
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'mode': args.mode,
                'val_reward': val_rew,
                'beat_best_rate': beat_rate,
                'norm_mean': mean.tolist(),
                'norm_std': std.tolist(),
            }, save_dir / f'best_model_{args.mode}.pt')
            logger.info(f"  -> New best! Saved.")

        # Early stopping check
        if epoch - best_epoch > 30:
            logger.info(f"Early stopping: no improvement for 30 epochs")
            break

    # Final test
    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)

    # Load best model
    best_ckpt = torch.load(save_dir / f'best_model_{args.mode}.pt', weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = trainer.evaluate(test_loader, compare_modes=True)

    logger.info(f"Test ({args.mode}): Reward={test_metrics['mean_reward']:.3f}")
    logger.info(f"Beat Best Discrete Rate: {test_metrics.get('beat_best_rate', 0):.1f}%")
    logger.info(f"\nMode Comparison:")
    logger.info(f"  Discrete: {test_metrics['discrete_reward']:.3f}")
    logger.info(f"  Blended:  {test_metrics['blended_reward']:.3f}")
    logger.info(f"  Direct:   {test_metrics['direct_reward']:.3f}")
    logger.info(f"\nBest Val Reward: {best_val_reward:.3f} (epoch {best_epoch})")

    # Compare to random baseline
    logger.info("\nBaseline Comparison:")
    logger.info(f"  Random discrete (1/8): ~0.0")
    logger.info(f"  Best mode ({args.mode}): {test_metrics['mean_reward']:.3f}")


if __name__ == '__main__':
    main()
