"""
Realistic Day-by-Day Backtest Simulation

Simulates $200 invested 1 year ago using the AI strategy.
Tracks daily P&L as the model makes allocation decisions.
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
from src.simulation.loss_averse_simulator import BarbellAISimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def fetch_year_data():
    """Fetch last year of market data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # Extra buffer for features

    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")

    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

    # Align
    common = spy.index.intersection(tlt.index).intersection(vix.index)
    spy = spy.loc[common]
    tlt = tlt.loc[common]
    vix = vix.loc[common]

    return spy, tlt, vix


def compute_features(spy, tlt, vix, idx):
    """Compute 110-dim feature vector for a given day index."""
    f = []

    # VIX
    f.append(vix['Close'].iloc[idx] / 100)

    # SPY returns
    spy_ret = spy['Close'].pct_change()
    f.append(spy_ret.iloc[idx])
    f.append(spy_ret.iloc[idx-4:idx+1].sum())
    f.append(spy_ret.iloc[idx-19:idx+1].sum())
    f.append(spy_ret.iloc[idx-59:idx+1].sum() if idx >= 60 else 0)

    # TLT returns
    tlt_ret = tlt['Close'].pct_change()
    f.append(tlt_ret.iloc[idx])
    f.append(tlt_ret.iloc[idx-4:idx+1].sum())
    f.append(tlt_ret.iloc[idx-19:idx+1].sum())

    # Volatility
    f.append(spy_ret.iloc[idx-19:idx+1].std() * np.sqrt(252))
    f.append(tlt_ret.iloc[idx-19:idx+1].std() * np.sqrt(252))

    # Correlation
    corr = spy_ret.iloc[idx-19:idx+1].corr(tlt_ret.iloc[idx-19:idx+1])
    f.append(corr if not np.isnan(corr) else 0)

    # Volume ratio
    vol_ratio = spy['Volume'].iloc[idx] / spy['Volume'].iloc[idx-19:idx+1].mean()
    f.append(vol_ratio if not np.isnan(vol_ratio) else 1)

    # Price vs MA
    f.append(spy['Close'].iloc[idx] / spy['Close'].iloc[idx-19:idx+1].mean() - 1)
    f.append(spy['Close'].iloc[idx] / spy['Close'].iloc[idx-49:idx+1].mean() - 1 if idx >= 50 else 0)

    # Momentum
    f.append(1 if spy_ret.iloc[idx] > 0 else 0)
    f.append(sum(1 for r in spy_ret.iloc[idx-4:idx+1] if r > 0) / 5)

    # RSI
    gains = spy_ret.iloc[idx-13:idx+1].clip(lower=0).mean()
    losses = (-spy_ret.iloc[idx-13:idx+1].clip(upper=0)).mean()
    f.append(gains / (gains + losses + 1e-8))

    # Pad to 110
    while len(f) < 110:
        f.append(0)

    return np.array(f[:110], dtype=np.float32)


def load_model(checkpoint_path: str):
    """Load trained model."""
    model = create_strategy_generator({'input_dim': 110, 'hidden_dim': 128})

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def run_backtest(
    model,
    spy: pd.DataFrame,
    tlt: pd.DataFrame,
    vix: pd.DataFrame,
    initial_capital: float = 200.0,
    start_offset: int = 252,  # Start 1 year ago
    rebalance_freq: int = 5,  # Rebalance every 5 days
):
    """
    Run day-by-day backtest.

    Args:
        model: Trained strategy model
        spy, tlt, vix: DataFrames with price data
        initial_capital: Starting money
        start_offset: Days from end to start (252 = ~1 year)
        rebalance_freq: How often to rebalance

    Returns:
        DataFrame with daily portfolio values
    """
    # Load normalization
    with open('D:/Projects/trader-ai/models/trm_grokking/normalization_params.json') as f:
        norm = json.load(f)
    mean = np.array(norm['mean'])
    std = np.array(norm['std'])
    std[std < 1e-7] = 1.0

    # Constraints
    min_spy, max_cash = 0.30, 0.40

    def apply_constraints(alloc):
        spy_pct = max(alloc[0], min_spy)
        cash_pct = min(alloc[2], max_cash)
        tlt_pct = 1 - spy_pct - cash_pct
        if tlt_pct < 0:
            tlt_pct = 0
            spy_pct = 1 - cash_pct
        return np.array([spy_pct, tlt_pct, cash_pct])

    # Returns
    spy_ret = spy['Close'].pct_change().fillna(0)
    tlt_ret = tlt['Close'].pct_change().fillna(0)

    # Starting point
    start_idx = len(spy) - start_offset
    if start_idx < 60:
        start_idx = 60

    # Track results
    results = []
    portfolio_value = initial_capital
    current_allocation = np.array([0.66, 0.23, 0.11])  # Start with model's average

    # Holdings
    spy_value = portfolio_value * current_allocation[0]
    tlt_value = portfolio_value * current_allocation[1]
    cash_value = portfolio_value * current_allocation[2]

    logger.info(f"Starting backtest: ${initial_capital:.2f} on {spy.index[start_idx].strftime('%Y-%m-%d')}")

    for i in range(start_idx, len(spy)):
        date = spy.index[i]

        # Apply daily returns
        spy_value *= (1 + spy_ret.iloc[i])
        tlt_value *= (1 + tlt_ret.iloc[i])
        # Cash stays same

        portfolio_value = spy_value + tlt_value + cash_value
        daily_return = (spy_value + tlt_value + cash_value) / (spy_value / (1 + spy_ret.iloc[i]) + tlt_value / (1 + tlt_ret.iloc[i]) + cash_value) - 1

        # Rebalance decision
        rebalanced = False
        if (i - start_idx) % rebalance_freq == 0 and i >= 60:
            # Get model prediction
            features = compute_features(spy, tlt, vix, i)
            features_norm = (features - mean) / std
            x = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(x)
                weights = output['strategy_weights'][0].numpy()
                # Blend to allocation
                from src.simulation.strategy_simulator import STRATEGY_ALLOCATIONS
                alloc = np.zeros(3)
                for j in range(8):
                    alloc += weights[j] * np.array(STRATEGY_ALLOCATIONS[j])

            # Apply constraints
            new_allocation = apply_constraints(alloc)

            # Rebalance
            spy_value = portfolio_value * new_allocation[0]
            tlt_value = portfolio_value * new_allocation[1]
            cash_value = portfolio_value * new_allocation[2]
            current_allocation = new_allocation
            rebalanced = True

        results.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'daily_return': daily_return * 100,
            'spy_pct': spy_value / portfolio_value * 100,
            'tlt_pct': tlt_value / portfolio_value * 100,
            'cash_pct': cash_value / portfolio_value * 100,
            'spy_price': spy['Close'].iloc[i],
            'tlt_price': tlt['Close'].iloc[i],
            'rebalanced': rebalanced,
        })

    return pd.DataFrame(results)


def print_results(df: pd.DataFrame, initial: float):
    """Print backtest results."""
    final = df['portfolio_value'].iloc[-1]
    total_return = (final / initial - 1) * 100
    days = len(df)

    # Comparison benchmarks
    spy_return = (df['spy_price'].iloc[-1] / df['spy_price'].iloc[0] - 1) * 100
    tlt_return = (df['tlt_price'].iloc[-1] / df['tlt_price'].iloc[0] - 1) * 100

    # Stats
    daily_rets = df['daily_return']
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    # Max drawdown
    cummax = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Win rate
    win_days = (daily_rets > 0).sum()
    win_rate = win_days / len(daily_rets) * 100

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: $200 INVESTED 1 YEAR AGO")
    print("=" * 70)
    print(f"Period: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')} ({days} days)")
    print(f"")
    print(f"{'PORTFOLIO PERFORMANCE':^40}")
    print(f"-" * 40)
    print(f"  Starting Capital:  ${initial:.2f}")
    print(f"  Final Value:       ${final:.2f}")
    print(f"  Total Return:      {total_return:+.2f}%")
    print(f"  Profit/Loss:       ${final - initial:+.2f}")
    print(f"")
    print(f"{'RISK METRICS':^40}")
    print(f"-" * 40)
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd:.2f}%")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Avg Daily Return:  {daily_rets.mean():.3f}%")
    print(f"")
    print(f"{'BENCHMARK COMPARISON':^40}")
    print(f"-" * 40)
    print(f"  AI Strategy:       {total_return:+.2f}%  (${final:.2f})")
    print(f"  SPY Buy & Hold:    {spy_return:+.2f}%  (${initial * (1 + spy_return/100):.2f})")
    print(f"  TLT Buy & Hold:    {tlt_return:+.2f}%  (${initial * (1 + tlt_return/100):.2f})")
    print(f"  60/40 Portfolio:   {0.6*spy_return + 0.4*tlt_return:+.2f}%  (${initial * (1 + (0.6*spy_return + 0.4*tlt_return)/100):.2f})")
    print(f"")
    print(f"{'FINAL ALLOCATION':^40}")
    print(f"-" * 40)
    print(f"  SPY:  {df['spy_pct'].iloc[-1]:.1f}%")
    print(f"  TLT:  {df['tlt_pct'].iloc[-1]:.1f}%")
    print(f"  Cash: {df['cash_pct'].iloc[-1]:.1f}%")
    print("=" * 70)

    # Monthly breakdown
    print(f"\n{'MONTHLY PERFORMANCE':^40}")
    print("-" * 40)
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg({
        'portfolio_value': ['first', 'last'],
        'daily_return': 'sum'
    })
    monthly.columns = ['start', 'end', 'return']
    monthly['return'] = (monthly['end'] / monthly['start'] - 1) * 100

    for month, row in monthly.iterrows():
        print(f"  {month}: {row['return']:+.2f}% (${row['end']:.2f})")

    return {
        'final_value': final,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'spy_return': spy_return,
    }


def main():
    # Fetch data
    spy, tlt, vix = fetch_year_data()

    if len(spy) < 300:
        logger.error("Not enough data fetched. Check network connection.")
        return

    # Load model (try barbell first, fall back to blended)
    model_path = Path('D:/Projects/trader-ai/models/barbell_strategy/best_barbell_model.pt')
    if not model_path.exists():
        model_path = Path('D:/Projects/trader-ai/models/strategy_generator/best_model_blended.pt')

    logger.info(f"Loading model: {model_path}")
    model = load_model(str(model_path))

    # Run backtest
    df = run_backtest(
        model, spy, tlt, vix,
        initial_capital=200.0,
        start_offset=252,
        rebalance_freq=5
    )

    # Print results
    results = print_results(df, 200.0)

    # Save detailed results
    output_path = Path('D:/Projects/trader-ai/models/barbell_strategy/backtest_results.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")


if __name__ == '__main__':
    main()
