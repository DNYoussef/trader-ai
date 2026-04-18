"""
Train Strategy Generator for 80/20 Barbell Portfolio

The AI controls 20% of the portfolio (the risky portion).
Training objectives:
1. Loss averse - avoid big losses (they hurt 2x more)
2. Not paralyzed - penalty for sitting in too much cash
3. Constrained - min 30% SPY, max 40% cash

This creates a model suitable for the "risky" end of a barbell strategy.
"""
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.strategy_generator import create_strategy_generator
from src.simulation.loss_averse_simulator import create_barbell_simulator, BarbellAISimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class StrategyDataset(Dataset):
    def __init__(self, features: np.ndarray, time_indices: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.time_indices = torch.tensor(time_indices, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.time_indices[idx]


class BarbellPolicyTrainer:
    """
    Policy gradient trainer with loss-averse rewards and constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        simulator: BarbellAISimulator,
        lr: float = 1e-4,
        entropy_coef: float = 0.02,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.simulator = simulator
        self.device = device
        self.entropy_coef = entropy_coef

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=15
        )

        # Value baseline
        self.baseline = nn.Sequential(
            nn.Linear(model.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=lr*3)

    def train_step(
        self,
        features: torch.Tensor,
        time_indices: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step with loss-averse rewards."""
        features = features.to(self.device)
        time_indices = time_indices.to(self.device)

        # Get model output
        output = self.model(features, return_all_heads=True)

        # Use blended allocation
        strategy_weights = output['strategy_weights']
        allocations = self.model.get_blended_allocation(strategy_weights)

        # Apply constraints
        constrained_alloc = self.simulator.apply_constraints(allocations)

        # Get loss-averse reward
        reward, reward_info = self.simulator.get_reward(constrained_alloc, time_indices)

        # Compute log prob using Dirichlet
        blend_logits = output['blend_logits']
        concentration = F.softplus(blend_logits) + 0.1
        dist = torch.distributions.Dirichlet(concentration)
        log_prob = dist.log_prob(strategy_weights)
        entropy = dist.entropy()

        # Baseline
        with torch.no_grad():
            baseline_val = self.baseline(output['solution_state'].detach()).squeeze(-1)

        advantage = reward - baseline_val

        # Policy loss
        policy_loss = -(log_prob * advantage.detach()).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update baseline
        with torch.no_grad():
            new_output = self.model(features)
        baseline_pred = self.baseline(new_output['solution_state']).squeeze(-1)
        baseline_loss = F.mse_loss(baseline_pred, reward.detach())

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'mean_reward': reward.mean().item(),
            'mean_return': reward_info['portfolio_return'].mean().item() * 100,
            'loss_rate': reward_info['is_loss'].mean().item() * 100,
            'avg_cash': constrained_alloc[:, 2].mean().item() * 100,
            'avg_spy': constrained_alloc[:, 0].mean().item() * 100,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate with loss-averse metrics."""
        self.model.eval()

        total_reward = 0
        total_return = 0
        total_losses = 0
        total_samples = 0
        all_allocations = []

        with torch.no_grad():
            for features, time_indices in dataloader:
                features = features.to(self.device)
                time_indices = time_indices.to(self.device)

                output = self.model(features)
                strategy_weights = output['strategy_weights']
                allocations = self.model.get_blended_allocation(strategy_weights)
                constrained = self.simulator.apply_constraints(allocations)

                reward, info = self.simulator.get_reward(constrained, time_indices)

                total_reward += reward.sum().item()
                total_return += info['portfolio_return'].sum().item()
                total_losses += info['is_loss'].sum().item()
                total_samples += len(features)
                all_allocations.append(constrained)

        self.model.train()

        all_alloc = torch.cat(all_allocations, dim=0)

        return {
            'mean_reward': total_reward / total_samples,
            'mean_return': total_return / total_samples * 100,
            'loss_rate': total_losses / total_samples * 100,
            'avg_spy': all_alloc[:, 0].mean().item() * 100,
            'avg_tlt': all_alloc[:, 1].mean().item() * 100,
            'avg_cash': all_alloc[:, 2].mean().item() * 100,
        }


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features and returns."""
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    features = np.stack(df['features'].values).astype(np.float32)
    time_indices = np.arange(len(features))

    # Load returns
    returns_path = Path('D:/Projects/trader-ai/data/trm_training/market_returns.parquet')
    if returns_path.exists():
        returns_df = pd.read_parquet(returns_path)
        spy_ret = returns_df['spy_return'].values
        tlt_ret = returns_df['tlt_return'].values
    else:
        logger.info("Downloading returns...")
        import yfinance as yf
        spy = yf.download('SPY', start='2000-01-01', end='2024-12-31', progress=False)
        tlt = yf.download('TLT', start='2000-01-01', end='2024-12-31', progress=False)

        spy_ret = spy['Close'].pct_change().dropna().values
        tlt_ret = tlt['Close'].pct_change().reindex(
            spy['Close'].pct_change().dropna().index
        ).fillna(0).values

        min_len = min(len(spy_ret), len(features))
        spy_ret = spy_ret[-min_len:]
        tlt_ret = tlt_ret[-min_len:]
        features = features[:min_len]
        time_indices = time_indices[:min_len]

    return features, time_indices, spy_ret, tlt_ret


def main():
    parser = argparse.ArgumentParser(description='Train Barbell Strategy')
    parser.add_argument('--data', type=str,
                       default='D:/Projects/trader-ai/data/trm_training/labels_110_features.parquet')
    parser.add_argument('--checkpoint', type=str,
                       default='D:/Projects/trader-ai/models/strategy_generator/best_model_blended.pt',
                       help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--entropy_coef', type=float, default=0.015)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load data
    features, time_indices, spy_ret, tlt_ret = load_data(args.data)

    # Normalize
    norm_path = Path('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json')
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm['mean'])
    std = np.array(norm['std'])
    std[std < 1e-7] = 1.0
    features_norm = (features - mean) / std

    # Create simulator with barbell constraints
    simulator = create_barbell_simulator(spy_ret, tlt_ret, horizon_days=5, device=args.device)

    # Split data
    n = len(features_norm)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_dataset = StrategyDataset(features_norm[:n_train], time_indices[:n_train])
    val_dataset = StrategyDataset(features_norm[n_train:n_train+n_val], time_indices[n_train:n_train+n_val])
    test_dataset = StrategyDataset(features_norm[n_train+n_val:], time_indices[n_train+n_val:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model = create_strategy_generator({'input_dim': 110, 'hidden_dim': 128})

    # Load pretrained weights
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading pretrained: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)

    # Create trainer
    trainer = BarbellPolicyTrainer(
        model=model,
        simulator=simulator,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    # Training
    logger.info("=" * 70)
    logger.info("BARBELL STRATEGY TRAINING (Loss Averse, 20% AI Portion)")
    logger.info("=" * 70)
    logger.info(f"Constraints: min_spy=30%, max_cash=40%")
    logger.info(f"Loss aversion: 1.8x, Drawdown penalty at >3%")
    logger.info("=" * 70)

    save_dir = Path('D:/Projects/trader-ai/models/barbell_strategy')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_reward = -float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        metrics = {'reward': [], 'return': [], 'loss_rate': [], 'spy': [], 'cash': []}

        for features, indices in train_loader:
            m = trainer.train_step(features, indices)
            metrics['reward'].append(m['mean_reward'])
            metrics['return'].append(m['mean_return'])
            metrics['loss_rate'].append(m['loss_rate'])
            metrics['spy'].append(m['avg_spy'])
            metrics['cash'].append(m['avg_cash'])

        # Validate
        val = trainer.evaluate(val_loader)
        trainer.scheduler.step(val['mean_reward'])

        # Log
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train: Rew={np.mean(metrics['reward']):.3f} Ret={np.mean(metrics['return']):+.2f}% Loss={np.mean(metrics['loss_rate']):.0f}% | "
            f"Val: Rew={val['mean_reward']:.3f} Ret={val['mean_return']:+.2f}% | "
            f"Alloc: SPY={val['avg_spy']:.0f}% TLT={val['avg_tlt']:.0f}% Cash={val['avg_cash']:.0f}%"
        )

        # Save best
        if val['mean_reward'] > best_val_reward:
            best_val_reward = val['mean_reward']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_reward': val['mean_reward'],
                'val_return': val['mean_return'],
                'allocation': {'spy': val['avg_spy'], 'tlt': val['avg_tlt'], 'cash': val['avg_cash']},
            }, save_dir / 'best_barbell_model.pt')
            logger.info(f"  -> New best! Saved.")

        if epoch - best_epoch > 30:
            logger.info("Early stopping.")
            break

    # Final test
    logger.info("=" * 70)
    logger.info("FINAL TEST")
    logger.info("=" * 70)

    best_ckpt = torch.load(save_dir / 'best_barbell_model.pt', weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test = trainer.evaluate(test_loader)
    logger.info(f"Test Reward: {test['mean_reward']:.3f}")
    logger.info(f"Test Return: {test['mean_return']:+.2f}%")
    logger.info(f"Loss Rate:   {test['loss_rate']:.1f}%")
    logger.info(f"Allocation:  SPY={test['avg_spy']:.0f}% TLT={test['avg_tlt']:.0f}% Cash={test['avg_cash']:.0f}%")
    logger.info(f"Best epoch:  {best_epoch}")


if __name__ == '__main__':
    main()
