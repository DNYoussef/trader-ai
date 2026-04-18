"""
RL-Enhanced Strategy Classifier Training

Phase 2: Use backtesting rewards instead of supervised labels.
The simulator tells us which strategy ACTUALLY performed best,
not which one we GUESSED would be best.

Training approach: Policy Gradient (REINFORCE with baseline)
- State: 110 market features
- Action: Choose 1 of 8 strategies
- Reward: How well that strategy performed vs optimal
"""
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.trm_model import TinyRecursiveModel
from src.simulation.strategy_simulator import BatchStrategySimulator, STRATEGY_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class RLStrategyDataset(Dataset):
    """Dataset that provides features and time indices for RL training."""

    def __init__(
        self,
        features: np.ndarray,
        time_indices: np.ndarray,
        transform_fn=None
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.time_indices = torch.tensor(time_indices, dtype=torch.long)
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform_fn:
            x = self.transform_fn(x)
        return x, self.time_indices[idx]


class PolicyGradientTrainer:
    """
    REINFORCE with baseline for strategy selection.

    The model outputs strategy logits, we sample actions,
    get rewards from simulator, update policy.
    """

    def __init__(
        self,
        model: nn.Module,
        simulator: BatchStrategySimulator,
        lr: float = 1e-4,
        baseline_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.simulator = simulator
        self.device = device
        self.entropy_coef = entropy_coef

        # Separate optimizers for policy and baseline
        self.policy_optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

        # Value baseline (reduces variance)
        self.baseline = nn.Sequential(
            nn.Linear(128, 64),  # Takes hidden state
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(), lr=baseline_lr
        )

        # Tracking
        self.episode_rewards = []
        self.episode_losses = []

    def select_action(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions given features.

        Returns:
            actions: Selected strategy indices
            log_probs: Log probability of selected actions
            entropy: Policy entropy
        """
        output = self.model(features)
        logits = output['strategy_logits']

        # Get policy distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs, entropy

    def compute_baseline_value(self, features: torch.Tensor) -> torch.Tensor:
        """Compute baseline value from features."""
        with torch.no_grad():
            output = self.model(features)
        # Use latent state for baseline
        hidden = output.get('latent_state', output['strategy_logits'])
        return self.baseline(hidden).squeeze(-1)

    def train_step(
        self,
        features: torch.Tensor,
        time_indices: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step.

        1. Select actions
        2. Get rewards from simulator
        3. Compute advantage
        4. Update policy and baseline
        """
        features = features.to(self.device)
        time_indices = time_indices.to(self.device)

        # Select actions
        actions, log_probs, entropy = self.select_action(features)

        # Get rewards from simulator
        rewards = self.simulator.get_reward(actions, time_indices)

        # Compute baseline
        baseline_values = self.compute_baseline_value(features)

        # Advantage = reward - baseline
        advantages = rewards - baseline_values.detach()

        # Policy loss (negative because we want to maximize reward)
        policy_loss = -(log_probs * advantages).mean()

        # Entropy bonus (encourages exploration)
        entropy_loss = -self.entropy_coef * entropy.mean()

        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Update baseline (MSE loss)
        baseline_values_new = self.compute_baseline_value(features)
        # Need to recompute with grad
        output = self.model(features)
        hidden = output.get('latent_state', output['strategy_logits'])
        baseline_pred = self.baseline(hidden).squeeze(-1)
        baseline_loss = F.mse_loss(baseline_pred, rewards)

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        # Get optimal strategy for accuracy
        optimal = self.simulator.get_optimal_strategy(time_indices)
        accuracy = (actions == optimal).float().mean().item()

        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'baseline_loss': baseline_loss.item(),
            'mean_reward': rewards.mean().item(),
            'accuracy': accuracy * 100,
            'mean_advantage': advantages.mean().item(),
        }

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate policy on held-out data."""
        self.model.eval()

        total_reward = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, time_indices in dataloader:
                features = features.to(self.device)
                time_indices = time_indices.to(self.device)

                # Deterministic action selection
                actions, _, _ = self.select_action(features, deterministic=True)

                # Get rewards
                rewards = self.simulator.get_reward(actions, time_indices)

                # Get optimal
                optimal = self.simulator.get_optimal_strategy(time_indices)

                total_reward += rewards.sum().item()
                total_correct += (actions == optimal).sum().item()
                total_samples += len(features)

        self.model.train()

        return {
            'mean_reward': total_reward / total_samples,
            'accuracy': total_correct / total_samples * 100,
        }


def load_data_and_simulator(
    data_path: str,
    returns_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and create simulator.

    Returns:
        features, time_indices, spy_returns, tlt_returns
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Extract features
    features = np.stack(df['features'].values).astype(np.float32)

    # Time indices (sequential for now)
    time_indices = np.arange(len(features))

    # Load returns data for simulator
    if returns_path:
        returns_df = pd.read_parquet(returns_path)
        spy_returns = returns_df['spy_return'].values
        tlt_returns = returns_df['tlt_return'].values
    else:
        # Try to extract from feature data or load separately
        returns_path = Path('D:/Projects/trader-ai/data/trm_training/market_returns.parquet')
        if returns_path.exists():
            returns_df = pd.read_parquet(returns_path)
            spy_returns = returns_df['spy_return'].values
            tlt_returns = returns_df['tlt_return'].values
        else:
            # Generate from yfinance
            logger.info("Downloading return data...")
            import yfinance as yf
            spy = yf.download('SPY', start='2000-01-01', end='2024-12-31', progress=False)
            tlt = yf.download('TLT', start='2000-01-01', end='2024-12-31', progress=False)

            spy_returns = spy['Close'].pct_change().dropna().values
            tlt_returns = tlt['Close'].pct_change().reindex(spy['Close'].pct_change().dropna().index).fillna(0).values

            # Align with features
            min_len = min(len(spy_returns), len(features))
            spy_returns = spy_returns[-min_len:]
            tlt_returns = tlt_returns[-min_len:]
            time_indices = time_indices[:min_len]
            features = features[:min_len]

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Returns length: {len(spy_returns)}")

    return features, time_indices, spy_returns, tlt_returns


def main():
    parser = argparse.ArgumentParser(description='RL Strategy Training')
    parser.add_argument('--data', type=str,
                       default='D:/Projects/trader-ai/data/trm_training/labels_110_features.parquet')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--horizon', type=int, default=5,
                       help='Trading horizon for reward calculation')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load data
    features, time_indices, spy_returns, tlt_returns = load_data_and_simulator(args.data)

    # Create simulator
    logger.info("Creating batch simulator...")
    simulator = BatchStrategySimulator(
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        horizon_days=args.horizon,
        device=args.device
    )

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-7] = 1.0
    features_norm = (features - mean) / std

    # Split data
    n_samples = len(features_norm)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_features = features_norm[:n_train]
    train_indices = time_indices[:n_train]
    val_features = features_norm[n_train:n_train+n_val]
    val_indices = time_indices[n_train:n_train+n_val]
    test_features = features_norm[n_train+n_val:]
    test_indices = time_indices[n_train+n_val:]

    # Create datasets
    train_dataset = RLStrategyDataset(train_features, train_indices)
    val_dataset = RLStrategyDataset(val_features, val_indices)
    test_dataset = RLStrategyDataset(test_features, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    input_dim = features_norm.shape[1]
    model = TinyRecursiveModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=8
    )

    # Load pretrained weights if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        logger.info("Loaded pretrained weights")

    # Create trainer
    trainer = PolicyGradientTrainer(
        model=model,
        simulator=simulator,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    # Training loop
    logger.info("="*60)
    logger.info("RL STRATEGY TRAINING")
    logger.info("="*60)
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Horizon: {args.horizon} days")
    logger.info("="*60)

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_metrics = {
            'policy_loss': [],
            'entropy': [],
            'mean_reward': [],
            'accuracy': [],
        }

        for features, time_indices in train_loader:
            metrics = trainer.train_step(features, time_indices)
            for k, v in metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(v)

        # Average metrics
        train_loss = np.mean(epoch_metrics['policy_loss'])
        train_reward = np.mean(epoch_metrics['mean_reward'])
        train_acc = np.mean(epoch_metrics['accuracy'])
        entropy = np.mean(epoch_metrics['entropy'])

        # Validate
        val_metrics = trainer.evaluate(val_loader)
        val_acc = val_metrics['accuracy']
        val_reward = val_metrics['mean_reward']

        # Logging
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train: Loss={train_loss:.4f} Rew={train_reward:.3f} Acc={train_acc:.1f}% | "
            f"Val: Rew={val_reward:.3f} Acc={val_acc:.1f}% | "
            f"Entropy={entropy:.3f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_path = Path('D:/Projects/trader-ai/models/trm_grokking/rl_best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_reward': val_reward,
                'norm_mean': mean,
                'norm_std': std,
            }, save_path)
            logger.info(f"  -> New best! Saved to {save_path}")

    # Final test
    logger.info("="*60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*60)

    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.1f}%")
    logger.info(f"Test Mean Reward: {test_metrics['mean_reward']:.3f}")
    logger.info(f"Best Val Accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")

    # Compare to random baseline
    logger.info("\nBaseline Comparison:")
    random_acc = 12.5  # 1/8
    logger.info(f"  Random: {random_acc:.1f}%")
    logger.info(f"  RL Model: {test_metrics['accuracy']:.1f}%")
    logger.info(f"  Improvement: {test_metrics['accuracy'] - random_acc:.1f}%")


if __name__ == '__main__':
    main()
