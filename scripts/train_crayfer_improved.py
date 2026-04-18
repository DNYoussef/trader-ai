"""
Crayfer-Improved Training Script

Integrates all improvements from the Crayfer video analysis:
1. Order book simulation for training data augmentation
2. 15 order book features (140 total features)
3. Hybrid model with price prediction head
4. Unrealized P&L tracking in reward
5. 1-day prediction horizon
6. GPT-style recursive prediction evaluation

Usage:
    python scripts/train_crayfer_improved.py --epochs 100
    python scripts/train_crayfer_improved.py --epochs 100 --use_orderbook
    python scripts/train_crayfer_improved.py --eval_recursive
"""

import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

# Import new modules
from src.models.hybrid_strategy_model import (
    HybridStrategyModel, HybridLoss, RecursivePredictor, create_hybrid_model
)
from src.simulation.orderbook_simulator import (
    OrderBookSimulator, HistoricalOrderBookSimulator, SimulatorConfig
)
from src.simulation.support_resistance import SupportResistanceCalculator
from src.simulation.pnl_tracker import PnLTracker, compute_reward_with_pnl_tracking
from src.simulation.daily_simulator import BatchDailySimulator, DailyDataLoader
from src.data.orderbook_features import (
    OrderBookFeatureExtractor,
    generate_synthetic_orderbook_features,
    add_orderbook_correlation,
    ORDERBOOK_FEATURE_COUNT,
)
from src.data.portfolio_context_features import PortfolioContextExtractor, TOTAL_FEATURES

# Existing modules
from src.portfolio.milestone_tracker import MilestoneTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Constants
MARKET_FEATURES = 110
PORTFOLIO_FEATURES = 15
ORDERBOOK_FEATURES = 15
TOTAL_FEATURES_NEW = MARKET_FEATURES + PORTFOLIO_FEATURES + ORDERBOOK_FEATURES  # 140


class CrayferDataset(Dataset):
    """
    Dataset for Crayfer-improved training.

    Features:
    - 110 market features
    - 15 portfolio context features
    - 15 order book features (real or synthetic)
    """

    def __init__(
        self,
        spy_df: pd.DataFrame,
        tlt_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        use_orderbook: bool = True,
        lookback: int = 60,
    ):
        self.spy = spy_df
        self.tlt = tlt_df
        self.vix = vix_df
        self.use_orderbook = use_orderbook
        self.lookback = lookback

        # Compute returns
        self.spy_returns = self.spy['Close'].pct_change().fillna(0).values
        self.tlt_returns = self.tlt['Close'].pct_change().fillna(0).values

        # Valid indices (need lookback history)
        self.valid_indices = np.arange(lookback, len(spy_df) - 1)

        # Initialize feature extractors
        self.portfolio_extractor = PortfolioContextExtractor()
        self.orderbook_extractor = OrderBookFeatureExtractor()

        # Generate synthetic orderbook features if not using real data
        if use_orderbook:
            n_samples = len(self.valid_indices)
            self.synthetic_ob_features = generate_synthetic_orderbook_features(n_samples)

        logger.info(f"Dataset created with {len(self.valid_indices)} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]

        # Market features (110)
        market_features = self._compute_market_features(data_idx)

        # Portfolio context features (15) - simulate capital trajectory
        capital = 200 + np.random.uniform(0, 300)  # Random capital $200-$500
        portfolio_features = self._compute_portfolio_features(capital)

        # Order book features (15)
        if self.use_orderbook:
            ob_features = self.synthetic_ob_features[idx]
            # Add correlation with market
            ob_features = add_orderbook_correlation(
                market_features.reshape(1, -1),
                ob_features.reshape(1, -1)
            )[0]
        else:
            ob_features = np.zeros(ORDERBOOK_FEATURES, dtype=np.float32)

        # Combine all features
        features = np.concatenate([market_features, portfolio_features, ob_features])

        # Target: next day SPY return
        next_return = self.spy_returns[data_idx + 1]

        # Capital normalization for reward
        capital_norm = (capital - 200) / 300  # 0 at $200, 1 at $500

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'next_return': torch.tensor(next_return, dtype=torch.float32),
            'capital_norm': torch.tensor(capital_norm, dtype=torch.float32),
            'data_idx': data_idx,
        }

    def _compute_market_features(self, idx: int) -> np.ndarray:
        """Compute 110 market features."""
        f = []

        # VIX features
        f.append(self.vix['Close'].iloc[idx] / 100)
        f.append(self.vix['Close'].iloc[idx] / self.vix['Close'].iloc[idx-20:idx].mean() - 1)

        # SPY return features
        spy_ret = self.spy_returns
        f.append(spy_ret[idx])
        f.append(spy_ret[idx-4:idx+1].sum())
        f.append(spy_ret[idx-9:idx+1].sum())
        f.append(spy_ret[idx-19:idx+1].sum())
        f.append(spy_ret[idx-59:idx+1].sum() if idx >= 60 else 0)

        # TLT return features
        tlt_ret = self.tlt_returns
        f.append(tlt_ret[idx])
        f.append(tlt_ret[idx-4:idx+1].sum())
        f.append(tlt_ret[idx-19:idx+1].sum())

        # Volatility features
        f.append(spy_ret[idx-4:idx+1].std() * np.sqrt(252))
        f.append(spy_ret[idx-9:idx+1].std() * np.sqrt(252))
        f.append(spy_ret[idx-19:idx+1].std() * np.sqrt(252))
        f.append(spy_ret[idx-59:idx+1].std() * np.sqrt(252) if idx >= 60 else 0.15)
        f.append(tlt_ret[idx-19:idx+1].std() * np.sqrt(252))

        # Correlation
        spy_20 = spy_ret[idx-19:idx+1]
        tlt_20 = tlt_ret[idx-19:idx+1]
        corr = np.corrcoef(spy_20, tlt_20)[0, 1]
        f.append(corr if not np.isnan(corr) else 0)

        # Moving averages
        spy_close = self.spy['Close'].values
        f.append(spy_close[idx] / spy_close[idx-19:idx+1].mean() - 1)
        f.append(spy_close[idx] / spy_close[idx-49:idx+1].mean() - 1 if idx >= 50 else 0)
        f.append(spy_close[idx] / spy_close[idx-199:idx+1].mean() - 1 if idx >= 200 else 0)

        # RSI
        gains = np.clip(spy_ret[idx-13:idx+1], 0, None).mean()
        losses = np.clip(-spy_ret[idx-13:idx+1], 0, None).mean()
        rsi = gains / (gains + losses + 1e-8)
        f.append(rsi)

        # MACD
        ema12 = pd.Series(spy_close).ewm(span=12).mean().iloc[idx]
        ema26 = pd.Series(spy_close).ewm(span=26).mean().iloc[idx]
        macd = (ema12 - ema26) / spy_close[idx]
        f.append(macd)

        # Bollinger bands
        ma20 = spy_close[idx-19:idx+1].mean()
        std20 = spy_close[idx-19:idx+1].std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        f.append((spy_close[idx] - bb_lower) / (bb_upper - bb_lower + 1e-8))

        # Volume
        vol_ratio = self.spy['Volume'].iloc[idx] / self.spy['Volume'].iloc[idx-19:idx+1].mean()
        f.append(vol_ratio if not np.isnan(vol_ratio) else 1)

        # Momentum
        f.append(1 if spy_ret[idx] > 0 else 0)
        f.append(sum(1 for r in spy_ret[idx-4:idx+1] if r > 0) / 5)
        f.append(sum(1 for r in spy_ret[idx-19:idx+1] if r > 0) / 20)

        # Drawdown
        rolling_max = spy_close[idx-59:idx+1].max() if idx >= 60 else spy_close[idx]
        drawdown = (spy_close[idx] - rolling_max) / rolling_max
        f.append(drawdown)

        # Risk
        f.append(self.vix['Close'].iloc[idx] / 30 - 1)
        f.append(np.percentile(spy_ret[idx-19:idx+1], 5))

        # Trend
        f.append(spy_close[idx-9:idx+1].mean() / spy_close[idx-49:idx+1].mean() - 1 if idx >= 50 else 0)
        f.append(spy_close[idx-19:idx+1].mean() / spy_close[idx-99:idx+1].mean() - 1 if idx >= 100 else 0)

        # Seasonality
        date = self.spy.index[idx]
        if hasattr(date, 'dayofweek'):
            f.append(date.dayofweek / 4)
            f.append(date.day / 31)
            f.append(date.month / 12)
            f.append(1.0 if date.month in [1, 4, 7, 10] else 0)
        else:
            f.extend([0.5, 0.5, 0.5, 0.0])

        # Cross-asset
        f.append(spy_ret[idx] - tlt_ret[idx])
        f.append(spy_ret[idx-19:idx+1].sum() - tlt_ret[idx-19:idx+1].sum())

        # ATR
        high_low = self.spy['High'].iloc[idx] - self.spy['Low'].iloc[idx]
        atr = high_low / spy_close[idx]
        f.append(atr)
        f.append(high_low / (self.spy['High'].iloc[idx-19:idx+1] - self.spy['Low'].iloc[idx-19:idx+1]).mean())

        # Pad to 110
        while len(f) < MARKET_FEATURES:
            f.append(0.0)

        features = np.array(f[:MARKET_FEATURES], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _compute_portfolio_features(self, capital: float) -> np.ndarray:
        """Compute 15 portfolio context features."""
        # Use existing extractor - returns dataclass, convert to array
        context = self.portfolio_extractor.extract(
            capital=capital,
            peak_capital=max(capital, 200),
            days_at_milestone=np.random.randint(0, 30),
            milestones_achieved=max(0, int((capital - 200) / 100)),
        )
        return context.to_array()


class CrayferTrainer:
    """
    Trainer for Crayfer-improved model.

    Key features:
    - Hybrid loss (RL + supervised)
    - Unrealized P&L tracking in rewards
    - 1-day prediction horizon
    - Recursive prediction evaluation
    """

    def __init__(
        self,
        model: HybridStrategyModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-4,
        price_weight: float = 0.5,
        confidence_weight: float = 0.1,
        entropy_weight: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Loss
        self.loss_fn = HybridLoss(
            price_weight=price_weight,
            confidence_weight=confidence_weight,
            entropy_weight=entropy_weight,
        )

        # Simulator for rewards
        self.simulator = BatchDailySimulator(spy_returns, tlt_returns)

        # P&L tracker for unrealized loss penalties
        self.pnl_tracker = PnLTracker(initial_capital=10000)

        # Recursive predictor for evaluation
        self.recursive_predictor = RecursivePredictor(model)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'price_accuracy': [],
            'allocation_sharpe': [],
        }

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            next_returns = batch['next_return'].to(self.device)
            capital_norms = batch['capital_norm'].to(self.device)
            data_indices = batch['data_idx'].numpy()

            # Forward pass
            output = self.model(features)

            # Simulate returns with predicted allocations
            allocations = output['allocations'].detach().cpu().numpy()
            port_ret, spy_ret, tlt_ret = self.simulator.simulate_batch(
                data_indices, allocations
            )

            # Compute rewards (with unrealized P&L tracking)
            rewards = self.simulator.compute_batch_reward(
                port_ret, spy_ret, capital_norms.cpu().numpy()
            )
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            # Compute loss
            losses = self.loss_fn(output, rewards, next_returns)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return {'loss': avg_loss}

    def validate(self) -> Dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        price_errors = []
        all_allocations = []
        all_returns = []
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                next_returns = batch['next_return'].to(self.device)
                capital_norms = batch['capital_norm'].to(self.device)
                data_indices = batch['data_idx'].numpy()

                output = self.model(features)

                # Simulate
                allocations = output['allocations'].cpu().numpy()
                port_ret, spy_ret, tlt_ret = self.simulator.simulate_batch(
                    data_indices, allocations
                )

                rewards = self.simulator.compute_batch_reward(
                    port_ret, spy_ret, capital_norms.cpu().numpy()
                )
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

                losses = self.loss_fn(output, rewards, next_returns)

                total_loss += losses['total_loss'].item()

                # Track price prediction accuracy
                price_pred = output['price_prediction'].squeeze().cpu().numpy()
                price_actual = next_returns.cpu().numpy()
                price_errors.extend(np.abs(price_pred - price_actual))

                all_allocations.extend(allocations)
                all_returns.extend(port_ret)

                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_price_error = np.mean(price_errors)

        # Direction accuracy
        price_direction = np.sign(price_pred - 0.0001)  # Small threshold
        actual_direction = np.sign(price_actual)
        direction_accuracy = np.mean(price_direction == actual_direction)

        return {
            'loss': avg_loss,
            'price_mae': avg_price_error,
            'direction_accuracy': direction_accuracy,
            'avg_return': np.mean(all_returns),
        }

    def evaluate_recursive(self, n_samples: int = 10, n_steps: int = 5) -> Dict:
        """Evaluate recursive prediction."""
        self.model.eval()

        results = []
        for i in range(n_samples):
            # Get random sample
            batch = next(iter(self.val_loader))
            features = batch['features'][0:1].to(self.device)

            # Multi-step prediction
            pred = self.recursive_predictor.predict_multi_step(features, n_steps)
            results.append(pred)

        # Aggregate
        avg_confidence = np.mean([r['avg_confidence'] for r in results])
        avg_cumulative = np.mean([r['cumulative_price_change'] for r in results])

        return {
            'avg_confidence': avg_confidence,
            'avg_cumulative_change': avg_cumulative,
            'n_samples': n_samples,
            'n_steps': n_steps,
        }

    def train(
        self,
        n_epochs: int,
        patience: int = 20,
        save_dir: str = 'D:/Projects/trader-ai/models/crayfer_improved',
    ):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])

            # Log
            logger.info(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Price MAE: {val_metrics['price_mae']:.4f} | "
                f"Direction Acc: {val_metrics['direction_accuracy']:.1%}"
            )

            # Save best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': {
                        'input_dim': self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'n_strategies': self.model.n_strategies,
                    }
                }, save_path / 'best_model.pt')

                logger.info(f"  -> Saved best model (loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        # Final recursive evaluation
        recursive_metrics = self.evaluate_recursive()
        logger.info(f"\nRecursive Prediction Evaluation:")
        logger.info(f"  Avg Confidence: {recursive_metrics['avg_confidence']:.2%}")
        logger.info(f"  Avg Cumulative Change: {recursive_metrics['avg_cumulative_change']:.4f}")

        return self.history


def load_data(start_date: str = '2020-01-01', end_date: str = None) -> Tuple:
    """Load market data."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"Loading data from {start_date} to {end_date}...")

    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    # Flatten multi-level columns if present (yfinance issue)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Align
    common = spy.index.intersection(tlt.index).intersection(vix.index)
    spy = spy.loc[common]
    tlt = tlt.loc[common]
    vix = vix.loc[common]

    logger.info(f"Loaded {len(spy)} days of data")

    return spy, tlt, vix


def main():
    parser = argparse.ArgumentParser(description='Crayfer-Improved Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use_orderbook', action='store_true', help='Use order book features')
    parser.add_argument('--eval_recursive', action='store_true', help='Only evaluate recursive')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Crayfer-Improved Training")
    logger.info("=" * 60)

    # Load data
    spy, tlt, vix = load_data()

    # Create dataset
    dataset = CrayferDataset(spy, tlt, vix, use_orderbook=args.use_orderbook)

    # Split train/val
    n_train = int(len(dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    input_dim = TOTAL_FEATURES_NEW if args.use_orderbook else MARKET_FEATURES + PORTFOLIO_FEATURES
    model = HybridStrategyModel(input_dim=input_dim, hidden_dim=128)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    spy_returns = spy['Close'].pct_change().fillna(0).values
    tlt_returns = tlt['Close'].pct_change().fillna(0).values

    trainer = CrayferTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        device=args.device,
        lr=args.lr,
    )

    if args.eval_recursive:
        # Just evaluate recursive prediction
        metrics = trainer.evaluate_recursive(n_samples=20, n_steps=5)
        print(f"\nRecursive Prediction Results:")
        print(f"  Average Confidence: {metrics['avg_confidence']:.2%}")
        print(f"  Average 5-day Cumulative: {metrics['avg_cumulative_change']:.4f}")
    else:
        # Train
        history = trainer.train(n_epochs=args.epochs, patience=args.patience)

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
