"""
Strategy Generator Inference Script

Load trained model and generate allocations for current/recent market data.
Shows what the model recommends and compares modes.
"""
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.models.strategy_generator import create_strategy_generator
from src.simulation.strategy_simulator import STRATEGY_ALLOCATIONS, STRATEGY_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def fetch_current_features(lookback_days: int = 252) -> tuple:
    """
    Fetch recent market data and compute features.

    Returns features array and metadata.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 100)

    logger.info(f"Fetching market data from {start_date.date()} to {end_date.date()}")

    # Download data
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    # Align dates
    common_dates = spy.index.intersection(tlt.index).intersection(vix.index)
    spy = spy.loc[common_dates]
    tlt = tlt.loc[common_dates]
    vix = vix.loc[common_dates]

    # Compute returns
    spy_ret = spy['Close'].pct_change()
    tlt_ret = tlt['Close'].pct_change()

    # Build feature vector (simplified - matches training feature structure)
    features = []

    for i in range(20, len(spy)):
        f = []

        # VIX level and changes
        f.append(vix['Close'].iloc[i] / 100)  # Normalized VIX

        # SPY returns at different horizons
        f.append(spy_ret.iloc[i])  # 1-day
        f.append(spy_ret.iloc[i-4:i+1].sum())  # 5-day
        f.append(spy_ret.iloc[i-19:i+1].sum())  # 20-day
        f.append(spy_ret.iloc[i-59:i+1].sum() if i >= 60 else 0)  # 60-day

        # TLT returns
        f.append(tlt_ret.iloc[i])
        f.append(tlt_ret.iloc[i-4:i+1].sum())
        f.append(tlt_ret.iloc[i-19:i+1].sum())

        # Volatility
        f.append(spy_ret.iloc[i-19:i+1].std() * np.sqrt(252))
        f.append(tlt_ret.iloc[i-19:i+1].std() * np.sqrt(252))

        # Correlation
        if i >= 20:
            corr = spy_ret.iloc[i-19:i+1].corr(tlt_ret.iloc[i-19:i+1])
            f.append(corr if not np.isnan(corr) else 0)
        else:
            f.append(0)

        # Volume ratio
        vol_ratio = spy['Volume'].iloc[i] / spy['Volume'].iloc[i-19:i+1].mean()
        f.append(vol_ratio if not np.isnan(vol_ratio) else 1)

        # Price vs moving averages
        f.append(spy['Close'].iloc[i] / spy['Close'].iloc[i-19:i+1].mean() - 1)
        f.append(spy['Close'].iloc[i] / spy['Close'].iloc[i-49:i+1].mean() - 1 if i >= 50 else 0)

        # Momentum indicators
        f.append(1 if spy_ret.iloc[i] > 0 else 0)  # Up day
        f.append(sum(1 for r in spy_ret.iloc[i-4:i+1] if r > 0) / 5)  # % up days

        # RSI approximation
        gains = spy_ret.iloc[i-13:i+1].clip(lower=0).mean()
        losses = (-spy_ret.iloc[i-13:i+1].clip(upper=0)).mean()
        rsi = gains / (gains + losses + 1e-8)
        f.append(rsi)

        # Pad to 110 features (match training data)
        while len(f) < 110:
            f.append(0)

        features.append(f[:110])

    features = np.array(features, dtype=np.float32)
    dates = spy.index[20:].tolist()

    return features, dates, spy_ret.iloc[20:].values, tlt_ret.iloc[20:].values


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load normalization params
    norm_path = Path('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json')
    with open(norm_path) as f:
        norm_params = json.load(f)

    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])
    std[std < 1e-7] = 1.0

    # Create model
    model = create_strategy_generator({
        'input_dim': 110,
        'hidden_dim': 128,
        'output_mode': 'all'
    })

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    return model, mean, std


def run_inference(
    model,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = 'cpu'
) -> dict:
    """Run inference on features."""
    # Normalize
    features_norm = (features - mean) / std
    x = torch.tensor(features_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(x)

        # Get all outputs
        strategy_logits = output['strategy_logits']
        strategy_weights = output['strategy_weights']
        allocations = output['allocations']
        halt_prob = output['halt_probability']

        # Discrete selection
        discrete_idx = strategy_logits.argmax(dim=-1)
        discrete_probs = torch.softmax(strategy_logits, dim=-1)

        # Blended allocation
        blended_alloc = model.get_blended_allocation(strategy_weights)

    return {
        'discrete_idx': discrete_idx.cpu().numpy(),
        'discrete_probs': discrete_probs.cpu().numpy(),
        'strategy_weights': strategy_weights.cpu().numpy(),
        'blended_allocation': blended_alloc.cpu().numpy(),
        'direct_allocation': allocations.cpu().numpy(),
        'confidence': halt_prob.cpu().numpy(),
    }


def print_current_recommendation(results: dict, dates: list):
    """Print the most recent recommendation."""
    idx = -1  # Most recent

    print("\n" + "=" * 70)
    print(f"STRATEGY RECOMMENDATION FOR {dates[idx].strftime('%Y-%m-%d')}")
    print("=" * 70)

    # Discrete
    disc_idx = results['discrete_idx'][idx]
    disc_prob = results['discrete_probs'][idx]
    print(f"\n[DISCRETE MODE]")
    print(f"  Selected: {STRATEGY_NAMES[disc_idx]} (idx={disc_idx})")
    print(f"  Confidence: {disc_prob[disc_idx]*100:.1f}%")
    spy, tlt, cash = STRATEGY_ALLOCATIONS[disc_idx]
    print(f"  Allocation: SPY={spy*100:.0f}% TLT={tlt*100:.0f}% Cash={cash*100:.0f}%")

    # Top 3 strategies
    top3 = disc_prob.argsort()[::-1][:3]
    print(f"  Top 3: ", end="")
    for i, idx in enumerate(top3):
        print(f"{STRATEGY_NAMES[idx]}({disc_prob[idx]*100:.0f}%)", end=" ")
    print()

    # Blended
    weights = results['strategy_weights'][idx]
    blended = results['blended_allocation'][idx]
    print(f"\n[BLENDED MODE]")
    print(f"  Strategy Weights:")
    for i, w in enumerate(weights):
        if w > 0.05:
            print(f"    {STRATEGY_NAMES[i]}: {w*100:.1f}%")
    print(f"  Final Allocation: SPY={blended[0]*100:.1f}% TLT={blended[1]*100:.1f}% Cash={blended[2]*100:.1f}%")

    # Direct
    direct = results['direct_allocation'][idx]
    print(f"\n[DIRECT MODE]")
    print(f"  Allocation: SPY={direct[0]*100:.1f}% TLT={direct[1]*100:.1f}% Cash={direct[2]*100:.1f}%")

    # Confidence
    conf = results['confidence'][idx]
    print(f"\n[MODEL CONFIDENCE]: {conf*100:.1f}%")

    return blended, direct


def evaluate_recent_performance(
    results: dict,
    spy_returns: np.ndarray,
    tlt_returns: np.ndarray,
    dates: list,
    horizon: int = 5,
    n_recent: int = 20
):
    """Evaluate how recommendations would have performed recently."""
    print("\n" + "=" * 70)
    print(f"BACKTEST: LAST {n_recent} TRADING DAYS (horizon={horizon} days)")
    print("=" * 70)

    blended_pnls = []
    direct_pnls = []
    discrete_pnls = []
    optimal_pnls = []

    for i in range(-n_recent - horizon, -horizon):
        blended = results['blended_allocation'][i]
        direct = results['direct_allocation'][i]
        disc_idx = results['discrete_idx'][i]
        disc_alloc = STRATEGY_ALLOCATIONS[disc_idx]

        # Forward returns
        spy_fwd = np.prod(1 + spy_returns[i+1:i+1+horizon]) - 1
        tlt_fwd = np.prod(1 + tlt_returns[i+1:i+1+horizon]) - 1

        # Portfolio returns
        blended_ret = blended[0] * spy_fwd + blended[1] * tlt_fwd
        direct_ret = direct[0] * spy_fwd + direct[1] * tlt_fwd
        discrete_ret = disc_alloc[0] * spy_fwd + disc_alloc[1] * tlt_fwd

        # Optimal (hindsight)
        optimal_ret = max(spy_fwd, tlt_fwd, 0)

        blended_pnls.append(blended_ret)
        direct_pnls.append(direct_ret)
        discrete_pnls.append(discrete_ret)
        optimal_pnls.append(optimal_ret)

    print(f"\nCumulative Returns:")
    print(f"  Blended:  {sum(blended_pnls)*100:+.2f}%")
    print(f"  Direct:   {sum(direct_pnls)*100:+.2f}%")
    print(f"  Discrete: {sum(discrete_pnls)*100:+.2f}%")
    print(f"  Optimal:  {sum(optimal_pnls)*100:+.2f}% (hindsight)")
    print(f"  SPY B&H:  {sum(spy_returns[-n_recent-horizon:-horizon])*100:+.2f}%")

    print(f"\nAverage per-period return:")
    print(f"  Blended:  {np.mean(blended_pnls)*100:+.3f}%")
    print(f"  Direct:   {np.mean(direct_pnls)*100:+.3f}%")
    print(f"  Discrete: {np.mean(discrete_pnls)*100:+.3f}%")

    return {
        'blended_total': sum(blended_pnls),
        'direct_total': sum(direct_pnls),
        'discrete_total': sum(discrete_pnls),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Strategy Generator Inference')
    parser.add_argument('--checkpoint', type=str,
                       default='D:/Projects/trader-ai/models/strategy_generator/best_model_blended.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Fetch current data
    features, dates, spy_ret, tlt_ret = fetch_current_features()
    logger.info(f"Got {len(features)} samples from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Load model
    model, mean, std = load_model(args.checkpoint, args.device)

    # Run inference
    results = run_inference(model, features, mean, std, args.device)

    # Print current recommendation
    blended, direct = print_current_recommendation(results, dates)

    # Evaluate recent performance
    perf = evaluate_recent_performance(results, spy_ret, tlt_ret, dates)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_mode = max(perf, key=perf.get)
    print(f"Best recent performer: {best_mode.replace('_total', '').upper()}")

    if perf['blended_total'] > perf['discrete_total'] * 1.1:
        print("-> Blended mode is working well, USE IT")
    elif perf['discrete_total'] > perf['blended_total'] * 1.1:
        print("-> Discrete mode outperforming, consider ENSEMBLE")
    else:
        print("-> Modes similar, blended provides diversification benefit")


if __name__ == '__main__':
    main()
