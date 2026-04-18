"""
Train Capital-Aware Strategy Generator

Extends the barbell strategy training to use 125 features:
- 110 market features (VIX, returns, momentum, etc.)
- 15 portfolio context features (milestone, risk tier, capital)

This allows the model to learn capital-aware allocation decisions:
- Be aggressive when small (grow fast)
- Be conservative when large (protect gains)
- Adjust based on milestone progress
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
from src.simulation.loss_averse_simulator import create_barbell_simulator
from src.data.portfolio_context_features import (
    ExtendedFeatureExtractor,
    create_training_dataset_with_context,
    generate_synthetic_capital_trajectory,
    TOTAL_FEATURES,
    print_feature_summary,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class CapitalAwareDataset(Dataset):
    """Dataset with extended features (market + portfolio context)."""

    def __init__(
        self,
        features: np.ndarray,
        time_indices: np.ndarray,
        capitals: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.time_indices = torch.tensor(time_indices, dtype=torch.long)
        self.capitals = torch.tensor(capitals, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.time_indices[idx], self.capitals[idx]


class CapitalAwarePolicyTrainer:
    """
    Policy gradient trainer with capital-aware rewards.

    Key differences from base trainer:
    - Reward scaling based on capital (protect gains at higher levels)
    - Additional loss terms for milestone alignment
    - Dynamic loss aversion based on portfolio state
    """

    def __init__(
        self,
        model: nn.Module,
        simulator,
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

    def _compute_capital_aware_reward(
        self,
        base_reward: torch.Tensor,
        capitals: torch.Tensor,
        allocations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjust reward based on capital level.

        STRONGER SIGNALS for capital differentiation:
        - Higher capital = much more penalty for losses
        - Lower capital = bonus for aggressive growth
        - Strong alignment bonus/penalty for matching target allocation
        """
        # Normalize capital (200-500 range)
        capital_norm = (capitals - 200) / 300  # 0 at $200, 1 at $500
        capital_norm = capital_norm.clamp(0, 1.5)  # Allow for >$500

        # === LOSS AVERSION SCALES WITH CAPITAL (STRONGER) ===
        loss_mask = base_reward < 0
        # 1.0x at $200, 2.5x at $500 (much stronger penalty at high capital)
        capital_loss_aversion = 1.0 + capital_norm * 1.5

        adjusted_reward = torch.where(
            loss_mask,
            base_reward * capital_loss_aversion,
            base_reward
        )

        # === GROWTH BONUS AT LOW CAPITAL ===
        # At low capital, bonus for positive returns (encourage growth)
        growth_bonus = torch.where(
            (base_reward > 0) & (capital_norm < 0.5),
            base_reward * (0.5 - capital_norm) * 0.5,  # Up to 0.25x bonus
            torch.zeros_like(base_reward)
        )
        adjusted_reward = adjusted_reward + growth_bonus

        # === STRONG ALIGNMENT BONUS (3x stronger) ===
        spy_alloc = allocations[:, 0]
        # Target: 90% at $200, 60% at $500
        target_spy = 0.90 - capital_norm * 0.30

        # Stronger alignment signal: 0.15 instead of 0.05
        alignment_bonus = -torch.abs(spy_alloc - target_spy) * 0.15

        # === EXTRA PENALTY FOR BEING TOO CONSERVATIVE AT LOW CAPITAL ===
        too_conservative = (capital_norm < 0.3) & (spy_alloc < 0.70)
        conservative_penalty = torch.where(
            too_conservative,
            torch.tensor(-0.1, device=capitals.device),
            torch.tensor(0.0, device=capitals.device)
        )

        # === EXTRA PENALTY FOR BEING TOO AGGRESSIVE AT HIGH CAPITAL ===
        too_aggressive = (capital_norm > 0.7) & (spy_alloc > 0.75)
        aggressive_penalty = torch.where(
            too_aggressive,
            torch.tensor(-0.1, device=capitals.device),
            torch.tensor(0.0, device=capitals.device)
        )

        return adjusted_reward + alignment_bonus + conservative_penalty + aggressive_penalty

    def train_step(
        self,
        features: torch.Tensor,
        time_indices: torch.Tensor,
        capitals: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step with capital-aware rewards."""
        features = features.to(self.device)
        time_indices = time_indices.to(self.device)
        capitals = capitals.to(self.device)

        # Get model output
        output = self.model(features, return_all_heads=True)

        # Use blended allocation
        strategy_weights = output['strategy_weights']
        allocations = self.model.get_blended_allocation(strategy_weights)

        # Apply constraints
        constrained_alloc = self.simulator.apply_constraints(allocations)

        # Get base reward from simulator
        reward, reward_info = self.simulator.get_reward(constrained_alloc, time_indices)

        # Apply capital-aware adjustment
        reward = self._compute_capital_aware_reward(reward, capitals, constrained_alloc)

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
            'avg_capital': capitals.mean().item(),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate with capital-aware metrics."""
        self.model.eval()

        total_reward = 0
        total_return = 0
        total_losses = 0
        total_samples = 0
        all_allocations = []
        capital_bins = {200: [], 300: [], 400: [], 500: []}

        with torch.no_grad():
            for features, time_indices, capitals in dataloader:
                features = features.to(self.device)
                time_indices = time_indices.to(self.device)
                capitals = capitals.to(self.device)

                output = self.model(features)
                strategy_weights = output['strategy_weights']
                allocations = self.model.get_blended_allocation(strategy_weights)
                constrained = self.simulator.apply_constraints(allocations)

                reward, info = self.simulator.get_reward(constrained, time_indices)
                reward = self._compute_capital_aware_reward(reward, capitals, constrained)

                total_reward += reward.sum().item()
                total_return += info['portfolio_return'].sum().item()
                total_losses += info['is_loss'].sum().item()
                total_samples += len(features)
                all_allocations.append(constrained)

                # Track allocations by capital level
                for i, cap in enumerate(capitals.cpu().numpy()):
                    if cap < 300:
                        capital_bins[200].append(constrained[i, 0].item())
                    elif cap < 400:
                        capital_bins[300].append(constrained[i, 0].item())
                    elif cap < 500:
                        capital_bins[400].append(constrained[i, 0].item())
                    else:
                        capital_bins[500].append(constrained[i, 0].item())

        self.model.train()

        all_alloc = torch.cat(all_allocations, dim=0)

        # Calculate SPY allocation by capital tier
        spy_by_tier = {}
        for tier, spy_list in capital_bins.items():
            if spy_list:
                spy_by_tier[f'spy_at_{tier}'] = np.mean(spy_list) * 100

        return {
            'mean_reward': total_reward / total_samples,
            'mean_return': total_return / total_samples * 100,
            'loss_rate': total_losses / total_samples * 100,
            'avg_spy': all_alloc[:, 0].mean().item() * 100,
            'avg_tlt': all_alloc[:, 1].mean().item() * 100,
            'avg_cash': all_alloc[:, 2].mean().item() * 100,
            **spy_by_tier,
        }


def load_and_extend_data(
    data_path: str,
    n_capital_trajectories: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load market features and extend with portfolio context.

    Generates multiple capital trajectories to augment data.
    """
    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    # Extract market features (110 dim)
    market_features = np.stack(df['features'].values).astype(np.float32)
    time_indices = np.arange(len(market_features))
    n_samples = len(market_features)

    logger.info(f"Loaded {n_samples} samples with {market_features.shape[1]} market features")

    # Load returns for simulator
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

    # Align lengths
    min_len = min(len(spy_ret), len(market_features))
    spy_ret = spy_ret[-min_len:]
    tlt_ret = tlt_ret[-min_len:]
    market_features = market_features[:min_len]
    time_indices = time_indices[:min_len]

    # Generate multiple capital trajectories for data augmentation
    logger.info(f"Generating {n_capital_trajectories} capital trajectories...")

    all_extended = []
    all_time_idx = []
    all_capitals = []

    for traj_idx in range(n_capital_trajectories):
        # Generate DIVERSE capital scenarios covering full range
        if traj_idx == 0:
            # Steady growth to goal
            mean_ret, std_ret, start_cap = 0.003, 0.012, 200.0
        elif traj_idx == 1:
            # Rocket growth (aggressive success story)
            mean_ret, std_ret, start_cap = 0.005, 0.015, 200.0
        elif traj_idx == 2:
            # Start at M1 ($300)
            mean_ret, std_ret, start_cap = 0.002, 0.012, 300.0
        elif traj_idx == 3:
            # Start at M2 ($400)
            mean_ret, std_ret, start_cap = 0.0015, 0.010, 400.0
        elif traj_idx == 4:
            # Start at goal ($500) - protect mode
            mean_ret, std_ret, start_cap = 0.001, 0.008, 500.0
        elif traj_idx == 5:
            # Volatile early stage
            mean_ret, std_ret, start_cap = 0.002, 0.030, 200.0
        elif traj_idx == 6:
            # Volatile mid stage
            mean_ret, std_ret, start_cap = 0.001, 0.025, 350.0
        elif traj_idx == 7:
            # Slow grind from start
            mean_ret, std_ret, start_cap = 0.0008, 0.015, 200.0
        elif traj_idx == 8:
            # Lucky early gains then steady
            mean_ret, std_ret, start_cap = 0.004, 0.018, 220.0
        elif traj_idx == 9:
            # Near goal, need protection
            mean_ret, std_ret, start_cap = 0.0012, 0.012, 450.0
        elif traj_idx < 15:
            # Extra low-capital samples (aggressive zone)
            np.random.seed(42 + traj_idx)
            mean_ret, std_ret, start_cap = 0.002 + np.random.rand() * 0.002, 0.015, 200 + np.random.rand() * 50
        elif traj_idx < 20:
            # Extra high-capital samples (protection zone)
            np.random.seed(42 + traj_idx)
            mean_ret, std_ret, start_cap = 0.001, 0.010, 450 + np.random.rand() * 100
        else:
            # Random mix
            np.random.seed(42 + traj_idx)
            mean_ret = 0.001 + np.random.rand() * 0.003
            std_ret = 0.010 + np.random.rand() * 0.015
            start_cap = 200 + np.random.rand() * 400

        capitals, peak_capitals = generate_synthetic_capital_trajectory(
            n_days=min_len,
            start_capital=start_cap,
            daily_return_mean=mean_ret,
            daily_return_std=std_ret,
            seed=42 + traj_idx,
        )

        # Extend features with portfolio context
        extended = create_training_dataset_with_context(
            market_features=market_features,
            time_indices=time_indices,
            capitals=capitals,
            peak_capitals=peak_capitals,
        )

        all_extended.append(extended)
        all_time_idx.append(time_indices.copy())
        all_capitals.append(capitals)

    # Stack all trajectories
    extended_features = np.concatenate(all_extended, axis=0)
    time_indices = np.concatenate(all_time_idx, axis=0)
    capitals = np.concatenate(all_capitals, axis=0)

    logger.info(f"Extended dataset: {extended_features.shape} ({TOTAL_FEATURES} features)")

    return extended_features, time_indices, capitals, spy_ret, tlt_ret, market_features


def main():
    parser = argparse.ArgumentParser(description='Train Capital-Aware Strategy')
    parser.add_argument('--data', type=str,
                       default='D:/Projects/trader-ai/data/trm_training/labels_110_features.parquet')
    parser.add_argument('--checkpoint', type=str,
                       default='D:/Projects/trader-ai/models/barbell_strategy/best_barbell_model.pt',
                       help='Path to pretrained model (110 features)')
    parser.add_argument('--epochs', type=int, default=150)  # Longer training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-5)  # Lower LR for stability
    parser.add_argument('--entropy_coef', type=float, default=0.02)  # More exploration
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_trajectories', type=int, default=25,  # More diverse data
                       help='Number of capital trajectories for augmentation')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs with lower LR')
    args = parser.parse_args()

    print_feature_summary()

    # Load and extend data
    extended_features, time_indices, capitals, spy_ret, tlt_ret, _ = load_and_extend_data(
        args.data, n_capital_trajectories=args.n_trajectories
    )

    # Normalize features
    norm_path = Path('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json')
    if norm_path.exists():
        with open(norm_path) as f:
            norm = json.load(f)
        # Extend normalization params for new features
        mean = np.array(norm['mean'])
        std = np.array(norm['std'])
        # Add defaults for new features (15)
        mean = np.concatenate([mean, np.zeros(15)])
        std = np.concatenate([std, np.ones(15)])
    else:
        mean = extended_features.mean(axis=0)
        std = extended_features.std(axis=0)

    std[std < 1e-7] = 1.0
    features_norm = (extended_features - mean) / std

    # Save extended normalization
    save_dir = Path('D:/Projects/trader-ai/models/capital_aware_strategy')
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'normalization_params_125.json', 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)

    # Create simulator
    simulator = create_barbell_simulator(spy_ret, tlt_ret, horizon_days=5, device=args.device)

    # Split data
    n = len(features_norm)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_dataset = CapitalAwareDataset(
        features_norm[:n_train], time_indices[:n_train] % len(spy_ret), capitals[:n_train]
    )
    val_dataset = CapitalAwareDataset(
        features_norm[n_train:n_train+n_val],
        time_indices[n_train:n_train+n_val] % len(spy_ret),
        capitals[n_train:n_train+n_val]
    )
    test_dataset = CapitalAwareDataset(
        features_norm[n_train+n_val:],
        time_indices[n_train+n_val:] % len(spy_ret),
        capitals[n_train+n_val:]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model with 125 input features
    model = create_strategy_generator({
        'input_dim': TOTAL_FEATURES,  # 125 instead of 110
        'hidden_dim': 128
    })

    # Try to load pretrained weights (will skip input projection due to size mismatch)
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading pretrained (110-feature) model: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Filter out input_proj weights (size mismatch)
        filtered_state = {k: v for k, v in state_dict.items()
                         if 'input_proj' not in k}

        model.load_state_dict(filtered_state, strict=False)
        logger.info("Loaded pretrained weights (excluding input projection)")

    # Create trainer
    trainer = CapitalAwarePolicyTrainer(
        model=model,
        simulator=simulator,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    # Training
    logger.info("=" * 70)
    logger.info("CAPITAL-AWARE STRATEGY TRAINING v2 (STRONGER SIGNALS)")
    logger.info("=" * 70)
    logger.info(f"Features: 110 market + 15 portfolio context = {TOTAL_FEATURES}")
    logger.info(f"Capital trajectories: {args.n_trajectories}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Target: SPY 90% at $200, 60% at $500")
    logger.info("=" * 70)

    best_val_reward = -float('inf')
    best_epoch = 0
    base_lr = args.lr

    for epoch in range(1, args.epochs + 1):
        # === WARMUP SCHEDULE ===
        if epoch <= args.warmup_epochs:
            warmup_lr = base_lr * (epoch / args.warmup_epochs)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Train
        metrics = {'reward': [], 'return': [], 'loss_rate': [], 'spy': [], 'cash': [], 'capital': []}

        for features, indices, caps in train_loader:
            m = trainer.train_step(features, indices, caps)
            metrics['reward'].append(m['mean_reward'])
            metrics['return'].append(m['mean_return'])
            metrics['loss_rate'].append(m['loss_rate'])
            metrics['spy'].append(m['avg_spy'])
            metrics['cash'].append(m['avg_cash'])
            metrics['capital'].append(m['avg_capital'])

        # Validate
        val = trainer.evaluate(val_loader)
        trainer.scheduler.step(val['mean_reward'])

        # Log
        tier_info = ""
        if 'spy_at_200' in val:
            tier_info = f" | SPY: $200:{val.get('spy_at_200', 0):.0f}% $400:{val.get('spy_at_400', 0):.0f}%"

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train: Rew={np.mean(metrics['reward']):.3f} Ret={np.mean(metrics['return']):+.2f}% | "
            f"Val: Rew={val['mean_reward']:.3f} Ret={val['mean_return']:+.2f}%{tier_info}"
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
                'input_dim': TOTAL_FEATURES,
                'spy_by_tier': {k: v for k, v in val.items() if k.startswith('spy_at')},
            }, save_dir / 'best_capital_aware_model.pt')
            logger.info(f"  -> New best! Saved.")

        if epoch - best_epoch > 40:  # Longer patience for exploration
            logger.info("Early stopping.")
            break

    # Final test
    logger.info("=" * 70)
    logger.info("FINAL TEST")
    logger.info("=" * 70)

    best_ckpt = torch.load(save_dir / 'best_capital_aware_model.pt', weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test = trainer.evaluate(test_loader)
    logger.info(f"Test Reward: {test['mean_reward']:.3f}")
    logger.info(f"Test Return: {test['mean_return']:+.2f}%")
    logger.info(f"Loss Rate:   {test['loss_rate']:.1f}%")
    logger.info(f"Allocation:  SPY={test['avg_spy']:.0f}% TLT={test['avg_tlt']:.0f}% Cash={test['avg_cash']:.0f}%")
    logger.info(f"Best epoch:  {best_epoch}")

    # Show SPY by capital tier
    logger.info("\nSPY Allocation by Capital Tier:")
    for tier in [200, 300, 400, 500]:
        key = f'spy_at_{tier}'
        if key in test:
            logger.info(f"  ${tier}: {test[key]:.1f}% SPY")


if __name__ == '__main__':
    main()
