"""
Generate 110-Feature Training Labels for TRM

Uses EnhancedFeatureExtractor (110 features) instead of basic 10 features.
More features = harder pattern finding = better conditions for grokking.

Feature Categories (110 total):
1. Price & Returns: 15 features
2. Volatility: 12 features
3. Technical Indicators: 20 features (RSI, MACD, Bollinger, etc.)
4. Volume Analysis: 8 features
5. Cross-Asset: 12 features
6. Market Regime: 10 features
7. Risk Metrics: 12 features
8. Momentum: 8 features
9. Breadth: 8 features
10. Calendar/Seasonal: 5 features

Usage:
    python scripts/data/generate_110_feature_labels.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import yfinance as yf
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    DEPS_AVAILABLE = False

# Strategy definitions (same as before)
STRATEGIES = {
    0: {'name': 'ultra_defensive', 'SPY': 0.20, 'TLT': 0.50, 'CASH': 0.30},
    1: {'name': 'defensive', 'SPY': 0.40, 'TLT': 0.30, 'CASH': 0.30},
    2: {'name': 'balanced_safe', 'SPY': 0.60, 'TLT': 0.20, 'CASH': 0.20},
    3: {'name': 'balanced_growth', 'SPY': 0.70, 'TLT': 0.20, 'CASH': 0.10},
    4: {'name': 'growth', 'SPY': 0.80, 'TLT': 0.15, 'CASH': 0.05},
    5: {'name': 'aggressive_growth', 'SPY': 0.90, 'TLT': 0.10, 'CASH': 0.00},
    6: {'name': 'contrarian_long', 'SPY': 0.85, 'TLT': 0.15, 'CASH': 0.00},
    7: {'name': 'tactical_opportunity', 'SPY': 0.75, 'TLT': 0.25, 'CASH': 0.00},
}

# 110 Feature names (from EnhancedFeatureExtractor)
FEATURE_NAMES = [
    # 1. Price & Returns (15)
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d', 'spy_return_60d',
    'tlt_return_1d', 'tlt_return_5d', 'tlt_return_20d',
    'spy_log_return_1d', 'spy_log_return_5d',
    'spy_price_to_ma50', 'spy_price_to_ma200', 'tlt_price_to_ma50',
    'spy_high_low_range', 'spy_close_to_high',
    # 2. Volatility (12)
    'vix_level', 'vix_change_1d', 'vix_change_5d', 'vix_percentile_20d',
    'spy_volatility_5d', 'spy_volatility_10d', 'spy_volatility_20d', 'spy_volatility_60d',
    'volatility_ratio_5_20', 'volatility_ratio_20_60',
    'vix_term_structure', 'realized_vs_implied_vol',
    # 3. Technical Indicators (20)
    'spy_rsi_14', 'spy_rsi_5', 'tlt_rsi_14',
    'spy_macd', 'spy_macd_signal', 'spy_macd_histogram',
    'spy_bb_upper_dist', 'spy_bb_lower_dist', 'spy_bb_width',
    'spy_stochastic_k', 'spy_stochastic_d',
    'spy_atr_14', 'spy_atr_ratio',
    'spy_obv_slope', 'spy_mfi_14',
    'spy_williams_r', 'spy_cci_20',
    'spy_adx_14', 'spy_di_plus', 'spy_di_minus',
    # 4. Volume Analysis (8)
    'spy_volume_ratio_5d', 'spy_volume_ratio_20d',
    'spy_volume_trend', 'spy_volume_volatility',
    'tlt_volume_ratio', 'vix_volume_ratio',
    'up_volume_ratio', 'down_volume_ratio',
    # 5. Cross-Asset (12)
    'spy_tlt_corr_20d', 'spy_tlt_corr_60d',
    'spy_vix_corr_20d', 'spy_gold_corr_20d',
    'spy_tlt_spread', 'spy_tlt_ratio',
    'risk_on_off_indicator', 'flight_to_quality',
    'cross_asset_momentum', 'asset_rotation_signal',
    'correlation_regime', 'diversification_ratio',
    # 6. Market Regime (10)
    'trend_strength_20d', 'trend_strength_60d',
    'regime_volatility', 'regime_momentum',
    'bull_bear_indicator', 'crisis_probability',
    'mean_reversion_signal', 'breakout_signal',
    'consolidation_indicator', 'regime_change_probability',
    # 7. Risk Metrics (12)
    'var_95_1d', 'var_99_1d', 'cvar_95',
    'max_drawdown_20d', 'max_drawdown_60d',
    'sharpe_20d', 'sharpe_60d',
    'sortino_20d', 'calmar_ratio',
    'tail_risk_indicator', 'skewness_20d', 'kurtosis_20d',
    # 8. Momentum (8)
    'momentum_1m', 'momentum_3m', 'momentum_6m',
    'momentum_12m', 'momentum_roc',
    'relative_momentum_spy_tlt', 'dual_momentum_signal',
    'momentum_breadth',
    # 9. Breadth (8)
    'market_breadth', 'advance_decline_ratio',
    'new_highs_lows_ratio', 'percent_above_ma50',
    'percent_above_ma200', 'sector_dispersion',
    'sector_rotation_indicator', 'breadth_thrust',
    # 10. Seasonal/Calendar (5)
    'day_of_week', 'day_of_month', 'month_of_year',
    'is_quarter_end', 'days_to_expiry',
]

NUM_FEATURES = len(FEATURE_NAMES)  # Should be 110


class Enhanced110FeatureGenerator:
    """Generate 110 features from yfinance data."""

    def __init__(self):
        self.data = None

    def download_data(self, start_date: str = '2002-07-01', end_date: str = '2024-12-31') -> pd.DataFrame:
        """Download SPY, TLT, VIX from yfinance."""
        logger.info(f"Downloading data from {start_date} to {end_date}...")

        def get_close(df, ticker):
            if df.empty:
                return pd.Series(dtype=float)
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    return df['Close'].iloc[:, 0]
            return df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

        # Download data
        spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
        tlt_df = yf.download('TLT', start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)

        logger.info(f"Downloaded: SPY={len(spy_df)}, TLT={len(tlt_df)}, VIX={len(vix_df)}")

        # Extract OHLCV
        def extract_ohlcv(df, prefix):
            result = pd.DataFrame()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if isinstance(df.columns, pd.MultiIndex):
                    result[f'{prefix}_{col.lower()}'] = df[col].iloc[:, 0] if col in df.columns.get_level_values(0) else np.nan
                else:
                    result[f'{prefix}_{col.lower()}'] = df[col] if col in df.columns else np.nan
            return result

        spy_data = extract_ohlcv(spy_df, 'SPY')
        tlt_data = extract_ohlcv(tlt_df, 'TLT')

        # VIX only has close
        vix_close = get_close(vix_df, '^VIX')

        # Combine
        data = spy_data.copy()
        data = data.join(tlt_data, how='outer')
        data['VIX_close'] = vix_close

        data.index.name = 'date'
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])

        # Forward fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        self.data = data
        logger.info(f"Combined data: {len(data)} rows")
        return data

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all 110 features."""
        if self.data is None:
            raise ValueError("No data loaded. Call download_data() first.")

        df = self.data.copy()
        features = pd.DataFrame()
        features['date'] = df['date']

        # === 1. PRICE & RETURNS (15) ===
        spy_close = df['SPY_close']
        tlt_close = df['TLT_close']
        spy_ret = spy_close.pct_change()
        tlt_ret = tlt_close.pct_change()

        features['spy_return_1d'] = spy_ret
        features['spy_return_5d'] = spy_ret.rolling(5).sum()
        features['spy_return_10d'] = spy_ret.rolling(10).sum()
        features['spy_return_20d'] = spy_ret.rolling(20).sum()
        features['spy_return_60d'] = spy_ret.rolling(60).sum()
        features['tlt_return_1d'] = tlt_ret
        features['tlt_return_5d'] = tlt_ret.rolling(5).sum()
        features['tlt_return_20d'] = tlt_ret.rolling(20).sum()
        features['spy_log_return_1d'] = np.log(spy_close / spy_close.shift(1))
        features['spy_log_return_5d'] = features['spy_log_return_1d'].rolling(5).sum()

        spy_ma50 = spy_close.rolling(50).mean()
        spy_ma200 = spy_close.rolling(200).mean()
        tlt_ma50 = tlt_close.rolling(50).mean()
        features['spy_price_to_ma50'] = spy_close / spy_ma50 - 1
        features['spy_price_to_ma200'] = spy_close / spy_ma200 - 1
        features['tlt_price_to_ma50'] = tlt_close / tlt_ma50 - 1

        features['spy_high_low_range'] = (df['SPY_high'] - df['SPY_low']) / spy_close
        features['spy_close_to_high'] = (spy_close - df['SPY_low']) / (df['SPY_high'] - df['SPY_low'] + 1e-8)

        # === 2. VOLATILITY (12) ===
        vix = df['VIX_close']
        features['vix_level'] = vix
        features['vix_change_1d'] = vix.pct_change()
        features['vix_change_5d'] = vix.pct_change(5)
        features['vix_percentile_20d'] = vix.rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))

        features['spy_volatility_5d'] = spy_ret.rolling(5).std() * np.sqrt(252)
        features['spy_volatility_10d'] = spy_ret.rolling(10).std() * np.sqrt(252)
        features['spy_volatility_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
        features['spy_volatility_60d'] = spy_ret.rolling(60).std() * np.sqrt(252)
        features['volatility_ratio_5_20'] = features['spy_volatility_5d'] / (features['spy_volatility_20d'] + 1e-8)
        features['volatility_ratio_20_60'] = features['spy_volatility_20d'] / (features['spy_volatility_60d'] + 1e-8)
        features['vix_term_structure'] = vix.rolling(5).mean() / (vix.rolling(20).mean() + 1e-8)
        features['realized_vs_implied_vol'] = features['spy_volatility_20d'] * 100 / (vix + 1e-8)

        # === 3. TECHNICAL INDICATORS (20) ===
        features['spy_rsi_14'] = self._rsi(spy_ret, 14) / 100
        features['spy_rsi_5'] = self._rsi(spy_ret, 5) / 100
        features['tlt_rsi_14'] = self._rsi(tlt_ret, 14) / 100

        ema12 = spy_close.ewm(span=12).mean()
        ema26 = spy_close.ewm(span=26).mean()
        features['spy_macd'] = (ema12 - ema26) / spy_close
        features['spy_macd_signal'] = features['spy_macd'].ewm(span=9).mean()
        features['spy_macd_histogram'] = features['spy_macd'] - features['spy_macd_signal']

        bb_ma = spy_close.rolling(20).mean()
        bb_std = spy_close.rolling(20).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        features['spy_bb_upper_dist'] = (bb_upper - spy_close) / spy_close
        features['spy_bb_lower_dist'] = (spy_close - bb_lower) / spy_close
        features['spy_bb_width'] = (bb_upper - bb_lower) / spy_close

        low_14 = df['SPY_low'].rolling(14).min()
        high_14 = df['SPY_high'].rolling(14).max()
        features['spy_stochastic_k'] = (spy_close - low_14) / (high_14 - low_14 + 1e-8)
        features['spy_stochastic_d'] = features['spy_stochastic_k'].rolling(3).mean()

        tr = pd.concat([
            df['SPY_high'] - df['SPY_low'],
            (df['SPY_high'] - spy_close.shift()).abs(),
            (df['SPY_low'] - spy_close.shift()).abs()
        ], axis=1).max(axis=1)
        features['spy_atr_14'] = tr.rolling(14).mean() / spy_close
        features['spy_atr_ratio'] = tr / (tr.rolling(50).mean() + 1e-8)

        obv = (np.sign(spy_close.diff()) * df['SPY_volume']).cumsum()
        features['spy_obv_slope'] = obv.diff(10) / (df['SPY_volume'].rolling(10).sum() + 1)

        mf = spy_close * df['SPY_volume']
        pos_mf = mf.where(spy_close > spy_close.shift(), 0).rolling(14).sum()
        neg_mf = mf.where(spy_close < spy_close.shift(), 0).rolling(14).sum()
        features['spy_mfi_14'] = pos_mf / (pos_mf + neg_mf + 1e-8)

        features['spy_williams_r'] = (high_14 - spy_close) / (high_14 - low_14 + 1e-8)

        tp = (df['SPY_high'] + df['SPY_low'] + spy_close) / 3
        features['spy_cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8) / 200

        features['spy_adx_14'] = spy_ret.abs().rolling(14).mean() / (spy_ret.rolling(14).std() + 1e-8)
        features['spy_di_plus'] = spy_ret.clip(lower=0).rolling(14).mean()
        features['spy_di_minus'] = (-spy_ret.clip(upper=0)).rolling(14).mean()

        # === 4. VOLUME ANALYSIS (8) ===
        spy_vol = df['SPY_volume']
        features['spy_volume_ratio_5d'] = spy_vol / (spy_vol.rolling(5).mean() + 1)
        features['spy_volume_ratio_20d'] = spy_vol / (spy_vol.rolling(20).mean() + 1)
        features['spy_volume_trend'] = spy_vol.rolling(5).mean() / (spy_vol.rolling(20).mean() + 1) - 1
        features['spy_volume_volatility'] = spy_vol.rolling(20).std() / (spy_vol.rolling(20).mean() + 1)
        features['tlt_volume_ratio'] = df['TLT_volume'] / (df['TLT_volume'].rolling(20).mean() + 1)
        features['vix_volume_ratio'] = 1.0  # VIX doesn't have volume
        up_vol = spy_vol.where(spy_ret > 0, 0).rolling(10).sum()
        down_vol = spy_vol.where(spy_ret < 0, 0).rolling(10).sum()
        features['up_volume_ratio'] = up_vol / (up_vol + down_vol + 1)
        features['down_volume_ratio'] = down_vol / (up_vol + down_vol + 1)

        # === 5. CROSS-ASSET (12) ===
        features['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)
        features['spy_tlt_corr_60d'] = spy_ret.rolling(60).corr(tlt_ret)
        vix_ret = vix.pct_change()
        features['spy_vix_corr_20d'] = spy_ret.rolling(20).corr(vix_ret)
        features['spy_gold_corr_20d'] = 0.0  # Would need GLD data
        features['spy_tlt_spread'] = (spy_close / spy_close.iloc[0]) - (tlt_close / tlt_close.iloc[0])
        features['spy_tlt_ratio'] = spy_close / tlt_close
        features['risk_on_off_indicator'] = -features['spy_tlt_corr_20d']
        features['flight_to_quality'] = features['tlt_return_5d'] - features['spy_return_5d']
        features['cross_asset_momentum'] = features['spy_return_20d'] + features['tlt_return_20d']
        features['asset_rotation_signal'] = features['spy_return_20d'] - features['tlt_return_20d']
        features['correlation_regime'] = (features['spy_tlt_corr_20d'] > 0).astype(float)
        features['diversification_ratio'] = 1.0 - features['spy_tlt_corr_20d'].abs()

        # === 6. MARKET REGIME (10) ===
        features['trend_strength_20d'] = spy_ret.rolling(20).mean() / (spy_ret.rolling(20).std() + 1e-8)
        features['trend_strength_60d'] = spy_ret.rolling(60).mean() / (spy_ret.rolling(60).std() + 1e-8)
        features['regime_volatility'] = spy_ret.rolling(20).std() / (spy_ret.rolling(60).std() + 1e-8)
        features['regime_momentum'] = spy_ret.rolling(20).sum() / (spy_ret.rolling(20).std() * np.sqrt(20) + 1e-8)
        features['bull_bear_indicator'] = (spy_ret.rolling(50).sum() > 0).astype(float)
        high_vol = (spy_ret.rolling(20).std() > spy_ret.rolling(60).std() * 1.5).astype(float)
        neg_ret = (spy_ret.rolling(10).sum() < -0.05).astype(float)
        features['crisis_probability'] = (high_vol + neg_ret) / 2
        z_score = (spy_ret.rolling(5).sum() - spy_ret.rolling(60).mean()) / (spy_ret.rolling(60).std() + 1e-8)
        features['mean_reversion_signal'] = -np.tanh(z_score / 2)
        high_20 = spy_close.rolling(20).max()
        low_20 = spy_close.rolling(20).min()
        features['breakout_signal'] = (spy_close - low_20) / (high_20 - low_20 + 1e-8)
        range_ratio = spy_ret.rolling(5).std() / (spy_ret.rolling(20).std() + 1e-8)
        features['consolidation_indicator'] = 1 - np.clip(range_ratio, 0, 2) / 2
        vol_change = spy_ret.rolling(5).std() / (spy_ret.rolling(20).std() + 1e-8) - 1
        features['regime_change_probability'] = np.clip(vol_change.abs(), 0, 1)

        # === 7. RISK METRICS (12) ===
        features['var_95_1d'] = -spy_ret.rolling(252).quantile(0.05)
        features['var_99_1d'] = -spy_ret.rolling(252).quantile(0.01)
        features['cvar_95'] = -spy_ret.rolling(252).apply(lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else 0)
        roll_max_20 = spy_close.rolling(20).max()
        roll_max_60 = spy_close.rolling(60).max()
        features['max_drawdown_20d'] = (spy_close - roll_max_20) / roll_max_20
        features['max_drawdown_60d'] = (spy_close - roll_max_60) / roll_max_60
        features['sharpe_20d'] = spy_ret.rolling(20).mean() / (spy_ret.rolling(20).std() + 1e-8) * np.sqrt(252)
        features['sharpe_60d'] = spy_ret.rolling(60).mean() / (spy_ret.rolling(60).std() + 1e-8) * np.sqrt(252)
        downside = spy_ret.clip(upper=0)
        features['sortino_20d'] = spy_ret.rolling(20).mean() / (downside.rolling(20).std() + 1e-8) * np.sqrt(252)
        features['calmar_ratio'] = (spy_ret.rolling(252).sum() / (-features['max_drawdown_60d'].clip(upper=-0.01))).clip(-10, 10)
        features['tail_risk_indicator'] = -spy_ret.rolling(20).apply(lambda x: x[x < x.quantile(0.1)].mean() if len(x[x < x.quantile(0.1)]) > 0 else 0)
        features['skewness_20d'] = spy_ret.rolling(20).skew()
        features['kurtosis_20d'] = spy_ret.rolling(20).kurt()

        # === 8. MOMENTUM (8) ===
        features['momentum_1m'] = spy_ret.rolling(21).sum()
        features['momentum_3m'] = spy_ret.rolling(63).sum()
        features['momentum_6m'] = spy_ret.rolling(126).sum()
        features['momentum_12m'] = spy_ret.rolling(252).sum()
        features['momentum_roc'] = spy_close.pct_change(20)
        features['relative_momentum_spy_tlt'] = features['spy_return_20d'] - features['tlt_return_20d']
        features['dual_momentum_signal'] = ((spy_ret.rolling(252).sum() > 0).astype(float) * np.sign(features['relative_momentum_spy_tlt']))
        features['momentum_breadth'] = (features['momentum_1m'] > 0).astype(float)

        # === 9. BREADTH (8) ===
        features['market_breadth'] = 0.5  # Would need multi-stock data
        features['advance_decline_ratio'] = 1.0
        features['new_highs_lows_ratio'] = 0.5
        features['percent_above_ma50'] = (features['spy_price_to_ma50'] > 0).astype(float)
        features['percent_above_ma200'] = (features['spy_price_to_ma200'] > 0).astype(float)
        features['sector_dispersion'] = 0.02
        features['sector_rotation_indicator'] = 0.0
        features['breadth_thrust'] = 0.5

        # === 10. CALENDAR (5) ===
        dates = pd.to_datetime(features['date'])
        features['day_of_week'] = dates.dt.dayofweek / 4.0
        features['day_of_month'] = dates.dt.day / 31.0
        features['month_of_year'] = dates.dt.month / 12.0
        features['is_quarter_end'] = dates.dt.is_quarter_end.astype(float)
        features['days_to_expiry'] = (21 - dates.dt.day % 21) / 21.0

        # Add prices for strategy simulation
        features['spy_close'] = spy_close
        features['tlt_close'] = tlt_close

        # Drop NaN rows (from rolling calculations)
        features = features.dropna()

        logger.info(f"Computed {NUM_FEATURES} features for {len(features)} samples")
        return features

    def _rsi(self, returns: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        gains = returns.clip(lower=0)
        losses = (-returns.clip(upper=0))
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def generate_labels(self, features: pd.DataFrame, lookforward: int = 5) -> pd.DataFrame:
        """Generate regime-based strategy labels."""
        labels = []

        for i in range(len(features) - lookforward):
            row = features.iloc[i]

            vix = row['vix_level']
            ret_5d = row['spy_return_5d']
            ret_20d = row['spy_return_20d']
            vol_20d = row['spy_volatility_20d']
            vix_change = row['vix_change_5d']

            # Regime-based assignment
            if vix > 30 or ret_5d < -0.05:
                strategy_idx = 0  # ultra_defensive
            elif vix > 25 or (vix > 20 and ret_5d < -0.02):
                strategy_idx = 1  # defensive
            elif vix > 20 or vol_20d > 0.25:
                strategy_idx = 2  # balanced_safe
            elif vix_change < -0.15 and ret_5d > 0.01:
                strategy_idx = 6  # contrarian_long
            elif 15 < vix <= 20 and -0.02 < ret_5d < 0.02:
                strategy_idx = 7  # tactical_opportunity
            elif vix <= 15 and ret_5d > 0.025 and ret_20d > 0.04:
                strategy_idx = 5  # aggressive_growth
            elif vix <= 18 and ret_5d > 0.01 and ret_20d > 0.02:
                strategy_idx = 4  # growth
            elif 12 < vix <= 20 and -0.015 < ret_5d < 0.02:
                strategy_idx = 3  # balanced_growth
            else:
                if ret_5d > 0.01:
                    strategy_idx = 4
                elif ret_5d < -0.01:
                    strategy_idx = 1
                else:
                    strategy_idx = 3

            # Calculate forward PnL
            start_spy = row['spy_close']
            start_tlt = row['tlt_close']
            end_row = features.iloc[i + lookforward]
            end_spy = end_row['spy_close']
            end_tlt = end_row['tlt_close']

            spy_ret = (end_spy / start_spy) - 1
            tlt_ret = (end_tlt / start_tlt) - 1

            strat = STRATEGIES[strategy_idx]
            pnl = strat['SPY'] * spy_ret + strat['TLT'] * tlt_ret

            # Extract feature vector
            feature_vec = [row[name] for name in FEATURE_NAMES]

            labels.append({
                'date': row['date'],
                'features': feature_vec,
                'strategy_idx': strategy_idx,
                'pnl': pnl,
            })

        result = pd.DataFrame(labels)
        logger.info(f"Generated {len(result)} labels")

        # Show distribution
        dist = result['strategy_idx'].value_counts().sort_index()
        logger.info(f"Class distribution:")
        for idx, count in dist.items():
            pct = 100 * count / len(result)
            logger.info(f"  {idx} ({STRATEGIES[idx]['name']}): {count} ({pct:.1f}%)")

        return result


def main():
    if not DEPS_AVAILABLE:
        logger.error("Missing dependencies. Install: pip install pandas yfinance")
        return 1

    output_dir = Path('data/trm_training')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate features and labels
    generator = Enhanced110FeatureGenerator()

    logger.info("=" * 70)
    logger.info("STEP 1: Download Historical Data")
    logger.info("=" * 70)
    data = generator.download_data()

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Compute 110 Features")
    logger.info("=" * 70)
    features = generator.compute_all_features()

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Generate Strategy Labels")
    logger.info("=" * 70)
    labels = generator.generate_labels(features)

    # Save
    labels.to_parquet(output_dir / 'labels_110_features.parquet')
    logger.info(f"\nSaved to {output_dir / 'labels_110_features.parquet'}")

    # Verify feature count
    sample_features = labels['features'].iloc[0]
    logger.info(f"\nFeature vector length: {len(sample_features)} (expected {NUM_FEATURES})")

    logger.info("\n" + "=" * 70)
    logger.info("110-FEATURE LABEL GENERATION COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
