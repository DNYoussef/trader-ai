"""
Backtest Crayfer-Improved Model

Compare performance against:
1. Buy-and-hold SPY
2. 60/40 portfolio
3. Original capital-aware model (if available)

Metrics:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Direction accuracy
"""

import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.models.hybrid_strategy_model import HybridStrategyModel, RecursivePredictor
from src.data.portfolio_context_features import PortfolioContextExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cpu') -> HybridStrategyModel:
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = HybridStrategyModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_strategies=config['n_strategies'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load market data."""
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    # Flatten columns
    for df in [spy, tlt, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Align
    common = spy.index.intersection(tlt.index).intersection(vix.index)
    return spy.loc[common], tlt.loc[common], vix.loc[common]


def compute_features(spy, tlt, vix, idx: int, portfolio_extractor, capital: float = 300) -> np.ndarray:
    """Compute 140 features for a given day."""
    f = []

    spy_ret = spy['Close'].pct_change().fillna(0).values
    tlt_ret = tlt['Close'].pct_change().fillna(0).values
    spy_close = spy['Close'].values

    # VIX features
    f.append(vix['Close'].iloc[idx] / 100)
    f.append(vix['Close'].iloc[idx] / vix['Close'].iloc[max(0,idx-20):idx].mean() - 1 if idx >= 20 else 0)

    # SPY returns
    f.append(spy_ret[idx])
    f.append(spy_ret[max(0,idx-4):idx+1].sum())
    f.append(spy_ret[max(0,idx-9):idx+1].sum())
    f.append(spy_ret[max(0,idx-19):idx+1].sum())
    f.append(spy_ret[max(0,idx-59):idx+1].sum() if idx >= 60 else 0)

    # TLT returns
    f.append(tlt_ret[idx])
    f.append(tlt_ret[max(0,idx-4):idx+1].sum())
    f.append(tlt_ret[max(0,idx-19):idx+1].sum())

    # Volatility
    f.append(spy_ret[max(0,idx-4):idx+1].std() * np.sqrt(252) if idx >= 5 else 0.15)
    f.append(spy_ret[max(0,idx-9):idx+1].std() * np.sqrt(252) if idx >= 10 else 0.15)
    f.append(spy_ret[max(0,idx-19):idx+1].std() * np.sqrt(252) if idx >= 20 else 0.15)
    f.append(spy_ret[max(0,idx-59):idx+1].std() * np.sqrt(252) if idx >= 60 else 0.15)
    f.append(tlt_ret[max(0,idx-19):idx+1].std() * np.sqrt(252) if idx >= 20 else 0.10)

    # Correlation
    if idx >= 20:
        corr = np.corrcoef(spy_ret[idx-19:idx+1], tlt_ret[idx-19:idx+1])[0,1]
        f.append(corr if not np.isnan(corr) else 0)
    else:
        f.append(0)

    # Moving averages
    f.append(spy_close[idx] / spy_close[max(0,idx-19):idx+1].mean() - 1 if idx >= 20 else 0)
    f.append(spy_close[idx] / spy_close[max(0,idx-49):idx+1].mean() - 1 if idx >= 50 else 0)
    f.append(spy_close[idx] / spy_close[max(0,idx-199):idx+1].mean() - 1 if idx >= 200 else 0)

    # RSI
    if idx >= 14:
        gains = np.clip(spy_ret[idx-13:idx+1], 0, None).mean()
        losses = np.clip(-spy_ret[idx-13:idx+1], 0, None).mean()
        rsi = gains / (gains + losses + 1e-8)
    else:
        rsi = 0.5
    f.append(rsi)

    # MACD
    ema12 = pd.Series(spy_close).ewm(span=12).mean().iloc[idx]
    ema26 = pd.Series(spy_close).ewm(span=26).mean().iloc[idx]
    f.append((ema12 - ema26) / spy_close[idx])

    # Bollinger
    if idx >= 20:
        ma20 = spy_close[idx-19:idx+1].mean()
        std20 = spy_close[idx-19:idx+1].std()
        bb = (spy_close[idx] - (ma20 - 2*std20)) / (4*std20 + 1e-8)
    else:
        bb = 0.5
    f.append(bb)

    # Volume
    vol_ratio = spy['Volume'].iloc[idx] / spy['Volume'].iloc[max(0,idx-19):idx].mean() if idx >= 20 else 1
    f.append(vol_ratio if not np.isnan(vol_ratio) else 1)

    # Momentum
    f.append(1 if spy_ret[idx] > 0 else 0)
    f.append(sum(1 for r in spy_ret[max(0,idx-4):idx+1] if r > 0) / 5 if idx >= 5 else 0.5)
    f.append(sum(1 for r in spy_ret[max(0,idx-19):idx+1] if r > 0) / 20 if idx >= 20 else 0.5)

    # Drawdown
    rolling_max = spy_close[max(0,idx-59):idx+1].max() if idx >= 60 else spy_close[idx]
    f.append((spy_close[idx] - rolling_max) / rolling_max)

    # Risk
    f.append(vix['Close'].iloc[idx] / 30 - 1)
    f.append(np.percentile(spy_ret[max(0,idx-19):idx+1], 5) if idx >= 20 else -0.02)

    # Trend
    f.append(spy_close[max(0,idx-9):idx+1].mean() / spy_close[max(0,idx-49):idx+1].mean() - 1 if idx >= 50 else 0)
    f.append(spy_close[max(0,idx-19):idx+1].mean() / spy_close[max(0,idx-99):idx+1].mean() - 1 if idx >= 100 else 0)

    # Seasonality
    date = spy.index[idx]
    if hasattr(date, 'dayofweek'):
        f.append(date.dayofweek / 4)
        f.append(date.day / 31)
        f.append(date.month / 12)
        f.append(1.0 if date.month in [1, 4, 7, 10] else 0)
    else:
        f.extend([0.5, 0.5, 0.5, 0])

    # Cross-asset
    f.append(spy_ret[idx] - tlt_ret[idx])
    f.append(spy_ret[max(0,idx-19):idx+1].sum() - tlt_ret[max(0,idx-19):idx+1].sum() if idx >= 20 else 0)

    # ATR
    high_low = spy['High'].iloc[idx] - spy['Low'].iloc[idx]
    f.append(high_low / spy_close[idx])
    f.append(high_low / (spy['High'].iloc[max(0,idx-19):idx+1] - spy['Low'].iloc[max(0,idx-19):idx+1]).mean() if idx >= 20 else 1)

    # Pad to 110
    while len(f) < 110:
        f.append(0.0)

    market_features = np.array(f[:110], dtype=np.float32)
    market_features = np.nan_to_num(market_features, nan=0.0, posinf=1.0, neginf=-1.0)

    # Portfolio features (15)
    context = portfolio_extractor.extract(
        capital=capital,
        peak_capital=max(capital, 200),
        days_at_milestone=0,
        milestones_achieved=max(0, int((capital - 200) / 100)),
    )
    portfolio_features = context.to_array()

    # Order book features (15) - synthetic
    ob_features = np.zeros(15, dtype=np.float32)
    ob_features[0] = np.random.normal(0, 0.1)  # imbalance
    ob_features[1] = np.abs(np.random.normal(0.03, 0.01))  # spread

    return np.concatenate([market_features, portfolio_features, ob_features])


def backtest(
    model: HybridStrategyModel,
    spy: pd.DataFrame,
    tlt: pd.DataFrame,
    vix: pd.DataFrame,
    initial_capital: float = 200.0,
    start_idx: int = 60,
) -> Dict:
    """Run backtest."""

    portfolio_extractor = PortfolioContextExtractor()
    spy_ret = spy['Close'].pct_change().fillna(0).values
    tlt_ret = tlt['Close'].pct_change().fillna(0).values

    # Track performance
    capital = initial_capital
    peak_capital = initial_capital
    daily_returns = []
    allocations_history = []
    predictions_history = []

    # Baselines
    spy_cumret = 1.0
    tlt_cumret = 1.0
    balanced_cumret = 1.0

    for idx in range(start_idx, len(spy) - 1):
        # Get features
        features = compute_features(spy, tlt, vix, idx, portfolio_extractor, capital)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = model(x)
            alloc = output['allocations'][0].numpy()
            price_pred = output['price_prediction'][0].item()
            confidence = output['confidence'][0].item()

        # Next day returns
        next_spy = spy_ret[idx + 1]
        next_tlt = tlt_ret[idx + 1]

        # Portfolio return
        port_ret = alloc[0] * next_spy + alloc[1] * next_tlt

        # Update capital
        capital *= (1 + port_ret)
        peak_capital = max(peak_capital, capital)

        daily_returns.append(port_ret)
        allocations_history.append(alloc)
        predictions_history.append({
            'pred': price_pred,
            'actual': next_spy,
            'confidence': confidence,
        })

        # Baselines
        spy_cumret *= (1 + next_spy)
        tlt_cumret *= (1 + next_tlt)
        balanced_cumret *= (1 + 0.6 * next_spy + 0.4 * next_tlt)

    # Calculate metrics
    returns = np.array(daily_returns)
    total_return = capital / initial_capital - 1
    annualized = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe = annualized / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = drawdowns.max()

    # Direction accuracy
    preds = np.array([p['pred'] for p in predictions_history])
    actuals = np.array([p['actual'] for p in predictions_history])
    direction_acc = np.mean(np.sign(preds) == np.sign(actuals))

    # Average allocations
    avg_alloc = np.mean(allocations_history, axis=0)

    return {
        'model': {
            'total_return': total_return,
            'annualized': annualized,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'final_capital': capital,
            'direction_accuracy': direction_acc,
            'avg_spy_alloc': avg_alloc[0],
            'avg_tlt_alloc': avg_alloc[1],
            'avg_cash_alloc': avg_alloc[2],
            'n_days': len(returns),
        },
        'spy': {
            'total_return': spy_cumret - 1,
            'annualized': (spy_cumret) ** (252 / len(returns)) - 1,
        },
        'tlt': {
            'total_return': tlt_cumret - 1,
        },
        'balanced_60_40': {
            'total_return': balanced_cumret - 1,
            'annualized': (balanced_cumret) ** (252 / len(returns)) - 1,
        },
    }


def main():
    print("=" * 60)
    print("CRAYFER MODEL BACKTEST")
    print("=" * 60)

    # Load model
    model_path = 'D:/Projects/trader-ai/models/crayfer_improved/best_model.pt'
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Load data - use 2024 as out-of-sample test
    print("\nLoading test data (2024)...")
    spy, tlt, vix = load_data('2024-01-01', '2024-12-31')
    print(f"Test period: {len(spy)} days")

    # Run backtest
    print("\nRunning backtest...")
    results = backtest(model, spy, tlt, vix, initial_capital=200.0, start_idx=30)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (2024)")
    print("=" * 60)

    print(f"\n{'Strategy':<20} {'Total Return':>15} {'Annualized':>12} {'Sharpe':>8}")
    print("-" * 60)

    m = results['model']
    print(f"{'Crayfer Model':<20} {m['total_return']:>14.2%} {m['annualized']:>11.2%} {m['sharpe']:>7.2f}")

    s = results['spy']
    print(f"{'SPY (Buy & Hold)':<20} {s['total_return']:>14.2%} {s['annualized']:>11.2%} {'--':>8}")

    b = results['balanced_60_40']
    print(f"{'60/40 Portfolio':<20} {b['total_return']:>14.2%} {b['annualized']:>11.2%} {'--':>8}")

    print(f"\n{'Additional Metrics'}")
    print("-" * 40)
    print(f"  Max Drawdown:       {m['max_drawdown']:>8.2%}")
    print(f"  Volatility:         {m['volatility']:>8.2%}")
    print(f"  Direction Accuracy: {m['direction_accuracy']:>8.1%}")
    print(f"  Final Capital:      ${m['final_capital']:>7.2f}")
    print(f"  Trading Days:       {m['n_days']:>8}")

    print(f"\n{'Average Allocation'}")
    print("-" * 40)
    print(f"  SPY:  {m['avg_spy_alloc']:>6.1%}")
    print(f"  TLT:  {m['avg_tlt_alloc']:>6.1%}")
    print(f"  Cash: {m['avg_cash_alloc']:>6.1%}")

    # Compare to SPY
    alpha = m['total_return'] - s['total_return']
    print(f"\n{'Alpha vs SPY':>20}: {alpha:>+8.2%}")

    # Save results
    output_path = Path('D:/Projects/trader-ai/models/crayfer_improved/backtest_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
