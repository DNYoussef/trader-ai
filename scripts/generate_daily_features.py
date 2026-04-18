"""
Daily Feature Generator

Run this script daily to generate the 125 features for the AI model:
- 110 market features from current market data
- 15 portfolio context features from milestone tracker

Usage:
    python scripts/generate_daily_features.py
    python scripts/generate_daily_features.py --capital 275.50
    python scripts/generate_daily_features.py --inference

Outputs:
    - Saves features to data/daily_features/YYYY-MM-DD.json
    - Prints current allocation recommendation
"""
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.portfolio.milestone_tracker import MilestoneTracker
from src.data.portfolio_context_features import (
    ExtendedFeatureExtractor,
    TOTAL_FEATURES,
)
from src.models.strategy_generator import create_strategy_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class DailyFeatureGenerator:
    """Generate daily features for the capital-aware AI model."""

    def __init__(
        self,
        tracker_data_dir: str = "D:/Projects/trader-ai/data/milestones",
        features_output_dir: str = "D:/Projects/trader-ai/data/daily_features",
        model_path: str = "D:/Projects/trader-ai/models/capital_aware_strategy/best_capital_aware_model.pt",
        norm_path: str = "D:/Projects/trader-ai/models/capital_aware_strategy/normalization_params_125.json",
    ):
        self.tracker = MilestoneTracker(data_dir=tracker_data_dir)
        self.feature_extractor = ExtendedFeatureExtractor(data_dir=tracker_data_dir)
        self.output_dir = Path(features_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path)
        self.norm_path = Path(norm_path)

        # Cache for market data
        self._market_data_cache: Dict = {}

    def fetch_market_data(self, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch SPY, TLT, VIX data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer

        logger.info(f"Fetching market data...")

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

        # Align dates
        common = spy.index.intersection(tlt.index).intersection(vix.index)
        spy = spy.loc[common]
        tlt = tlt.loc[common]
        vix = vix.loc[common]

        logger.info(f"Fetched {len(spy)} days of data")

        return spy, tlt, vix

    def compute_market_features(
        self,
        spy: pd.DataFrame,
        tlt: pd.DataFrame,
        vix: pd.DataFrame,
        idx: int = -1,
    ) -> np.ndarray:
        """
        Compute 110 market features for a given day.

        Same features as used in training.
        """
        f = []

        # Use latest day by default
        if idx < 0:
            idx = len(spy) + idx

        # === VIX Features ===
        f.append(vix['Close'].iloc[idx] / 100)  # VIX level normalized
        f.append(vix['Close'].iloc[idx] / vix['Close'].iloc[idx-20:idx].mean() - 1)  # VIX vs 20d MA

        # === SPY Return Features ===
        spy_ret = spy['Close'].pct_change()
        f.append(spy_ret.iloc[idx])  # 1d return
        f.append(spy_ret.iloc[idx-4:idx+1].sum())  # 5d return
        f.append(spy_ret.iloc[idx-9:idx+1].sum())  # 10d return
        f.append(spy_ret.iloc[idx-19:idx+1].sum())  # 20d return
        f.append(spy_ret.iloc[idx-59:idx+1].sum() if idx >= 60 else 0)  # 60d return

        # === TLT Return Features ===
        tlt_ret = tlt['Close'].pct_change()
        f.append(tlt_ret.iloc[idx])  # 1d
        f.append(tlt_ret.iloc[idx-4:idx+1].sum())  # 5d
        f.append(tlt_ret.iloc[idx-19:idx+1].sum())  # 20d

        # === Volatility Features ===
        f.append(spy_ret.iloc[idx-4:idx+1].std() * np.sqrt(252))  # 5d vol
        f.append(spy_ret.iloc[idx-9:idx+1].std() * np.sqrt(252))  # 10d vol
        f.append(spy_ret.iloc[idx-19:idx+1].std() * np.sqrt(252))  # 20d vol
        f.append(spy_ret.iloc[idx-59:idx+1].std() * np.sqrt(252) if idx >= 60 else 0.15)  # 60d vol

        f.append(tlt_ret.iloc[idx-19:idx+1].std() * np.sqrt(252))  # TLT 20d vol

        # === Correlation Features ===
        spy_20 = spy_ret.iloc[idx-19:idx+1]
        tlt_20 = tlt_ret.iloc[idx-19:idx+1]
        corr = spy_20.corr(tlt_20)
        f.append(corr if not np.isnan(corr) else 0)

        # === Moving Averages ===
        spy_close = spy['Close']
        f.append(spy_close.iloc[idx] / spy_close.iloc[idx-19:idx+1].mean() - 1)  # Price vs 20d MA
        f.append(spy_close.iloc[idx] / spy_close.iloc[idx-49:idx+1].mean() - 1 if idx >= 50 else 0)  # vs 50d MA
        f.append(spy_close.iloc[idx] / spy_close.iloc[idx-199:idx+1].mean() - 1 if idx >= 200 else 0)  # vs 200d MA

        # === RSI Features ===
        gains = spy_ret.iloc[idx-13:idx+1].clip(lower=0).mean()
        losses = (-spy_ret.iloc[idx-13:idx+1].clip(upper=0)).mean()
        rsi = gains / (gains + losses + 1e-8)
        f.append(rsi)

        # === MACD Features ===
        ema12 = spy_close.ewm(span=12).mean()
        ema26 = spy_close.ewm(span=26).mean()
        macd = (ema12.iloc[idx] - ema26.iloc[idx]) / spy_close.iloc[idx]
        f.append(macd)

        # === Bollinger Band Features ===
        ma20 = spy_close.iloc[idx-19:idx+1].mean()
        std20 = spy_close.iloc[idx-19:idx+1].std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        f.append((spy_close.iloc[idx] - bb_lower) / (bb_upper - bb_lower + 1e-8))  # BB position

        # === Volume Features ===
        vol_ratio = spy['Volume'].iloc[idx] / spy['Volume'].iloc[idx-19:idx+1].mean()
        f.append(vol_ratio if not np.isnan(vol_ratio) else 1)

        # === Momentum Features ===
        f.append(1 if spy_ret.iloc[idx] > 0 else 0)  # Up day
        f.append(sum(1 for r in spy_ret.iloc[idx-4:idx+1] if r > 0) / 5)  # 5d win rate
        f.append(sum(1 for r in spy_ret.iloc[idx-19:idx+1] if r > 0) / 20)  # 20d win rate

        # === Drawdown Features ===
        rolling_max = spy_close.iloc[idx-59:idx+1].max() if idx >= 60 else spy_close.iloc[idx]
        drawdown = (spy_close.iloc[idx] - rolling_max) / rolling_max
        f.append(drawdown)

        # === Risk Features ===
        f.append(vix['Close'].iloc[idx] / 30 - 1)  # VIX vs average
        f.append(spy_ret.iloc[idx-19:idx+1].quantile(0.05))  # 5% VaR proxy

        # === Trend Features ===
        f.append(spy_close.iloc[idx-9:idx+1].mean() / spy_close.iloc[idx-49:idx+1].mean() - 1 if idx >= 50 else 0)
        f.append(spy_close.iloc[idx-19:idx+1].mean() / spy_close.iloc[idx-99:idx+1].mean() - 1 if idx >= 100 else 0)

        # === Seasonality Features ===
        date = spy.index[idx]
        f.append(date.dayofweek / 4)  # Day of week
        f.append(date.day / 31)  # Day of month
        f.append(date.month / 12)  # Month of year
        f.append(1.0 if date.month in [1, 4, 7, 10] else 0)  # Quarter start

        # === Cross-Asset Features ===
        f.append(spy_ret.iloc[idx] - tlt_ret.iloc[idx])  # Relative return
        f.append(spy_ret.iloc[idx-19:idx+1].sum() - tlt_ret.iloc[idx-19:idx+1].sum())  # 20d relative

        # === ATR Features ===
        high_low = spy['High'].iloc[idx] - spy['Low'].iloc[idx]
        atr = high_low / spy_close.iloc[idx]
        f.append(atr)
        f.append(high_low / (spy['High'].iloc[idx-19:idx+1] - spy['Low'].iloc[idx-19:idx+1]).mean())

        # Pad to 110 features
        while len(f) < 110:
            f.append(0.0)

        features = np.array(f[:110], dtype=np.float32)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state from milestone tracker."""
        return self.tracker.get_status()

    def update_portfolio(self, capital: float) -> Dict:
        """Update portfolio capital and get new status."""
        return self.tracker.update_capital(capital)

    def generate_features(
        self,
        capital: Optional[float] = None,
        date_str: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate full 125 features for the AI model.

        Args:
            capital: Current capital (uses tracker state if not provided)
            date_str: Date to record (today if not provided)

        Returns:
            features: (125,) array of features
            metadata: Dict with feature breakdown
        """
        # Get current portfolio state
        if capital is not None:
            self.update_portfolio(capital)
        portfolio = self.get_portfolio_state()
        current_capital = portfolio['current_capital']

        # Fetch market data
        spy, tlt, vix = self.fetch_market_data()

        # Compute market features (110)
        market_features = self.compute_market_features(spy, tlt, vix)

        # Extend with portfolio context (15)
        extended_features = self.feature_extractor.extend_features(
            market_features,
            capital=current_capital,
            peak_capital=portfolio['peak_capital'],
            days_at_milestone=portfolio.get('days_trading', 0) % 60,  # Proxy
            milestones_achieved=portfolio['milestones_achieved'],
        )

        # Prepare metadata
        metadata = {
            'date': date_str or datetime.now().strftime('%Y-%m-%d'),
            'capital': current_capital,
            'milestone': portfolio['current_milestone'],
            'milestone_name': portfolio['milestone_name'],
            'allocation_target': portfolio['allocation'],
            'market_features': market_features.tolist(),
            'context_features': extended_features[110:].tolist(),
            'vix': float(vix['Close'].iloc[-1]),
            'spy_price': float(spy['Close'].iloc[-1]),
            'tlt_price': float(tlt['Close'].iloc[-1]),
        }

        return extended_features, metadata

    def save_features(
        self,
        features: np.ndarray,
        metadata: Dict,
    ) -> Path:
        """Save features to JSON file."""
        date_str = metadata['date']
        output_file = self.output_dir / f"{date_str}.json"

        output = {
            'features': features.tolist(),
            'metadata': metadata,
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved features to {output_file}")
        return output_file

    def run_inference(
        self,
        features: np.ndarray,
        device: str = 'cpu',
    ) -> Dict:
        """
        Run the model to get allocation recommendation.

        Returns allocation percentages and strategy info.
        """
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            return {}

        # Load model
        model = create_strategy_generator({'input_dim': TOTAL_FEATURES, 'hidden_dim': 128})
        ckpt = torch.load(self.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Load normalization
        if self.norm_path.exists():
            with open(self.norm_path) as f:
                norm = json.load(f)
            mean = np.array(norm['mean'])
            std = np.array(norm['std'])
        else:
            mean = np.zeros(TOTAL_FEATURES)
            std = np.ones(TOTAL_FEATURES)

        std[std < 1e-7] = 1.0

        # Normalize features
        features_norm = (features - mean) / std
        x = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(x)
            strategy_weights = output['strategy_weights'][0].numpy()
            allocations = model.get_blended_allocation(output['strategy_weights'])[0].numpy()

        # Apply constraints (30% min SPY, 40% max cash)
        spy_pct = max(allocations[0], 0.30)
        cash_pct = min(allocations[2], 0.40)
        tlt_pct = 1 - spy_pct - cash_pct
        if tlt_pct < 0:
            tlt_pct = 0
            spy_pct = 1 - cash_pct

        return {
            'allocation': {
                'spy': spy_pct * 100,
                'tlt': tlt_pct * 100,
                'cash': cash_pct * 100,
            },
            'strategy_weights': strategy_weights.tolist(),
            'raw_allocation': allocations.tolist(),
        }


def print_recommendation(metadata: Dict, inference: Dict):
    """Print formatted recommendation."""
    print()
    print("=" * 60)
    print(f"DAILY RECOMMENDATION - {metadata['date']}")
    print("=" * 60)
    print()

    # Portfolio Status
    print(f"Portfolio Status:")
    print(f"  Capital:     ${metadata['capital']:.2f}")
    print(f"  Milestone:   {metadata['milestone_name']}")
    print(f"  VIX:         {metadata['vix']:.1f}")
    print(f"  SPY:         ${metadata['spy_price']:.2f}")
    print()

    if inference:
        print(f"AI Recommendation:")
        print(f"  SPY:  {inference['allocation']['spy']:.1f}%")
        print(f"  TLT:  {inference['allocation']['tlt']:.1f}%")
        print(f"  Cash: {inference['allocation']['cash']:.1f}%")
        print()

    # Target from milestone
    print(f"Milestone Target:")
    print(f"  SPY:  {metadata['allocation_target']['spy']:.0f}%")
    print(f"  TLT:  {metadata['allocation_target']['tlt']:.0f}%")
    print(f"  Cash: {metadata['allocation_target']['cash']:.0f}%")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate Daily Features')
    parser.add_argument('--capital', type=float, default=None,
                       help='Update capital (uses stored value if not provided)')
    parser.add_argument('--inference', action='store_true',
                       help='Run model inference to get allocation')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save features to file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference')
    args = parser.parse_args()

    generator = DailyFeatureGenerator()

    # Generate features
    features, metadata = generator.generate_features(capital=args.capital)

    # Save features
    if args.save:
        generator.save_features(features, metadata)

    # Run inference if requested
    inference = {}
    if args.inference:
        inference = generator.run_inference(features, device=args.device)
        metadata['inference'] = inference

    # Print recommendation
    print_recommendation(metadata, inference)

    # Return features for programmatic use
    return features, metadata, inference


if __name__ == '__main__':
    main()
