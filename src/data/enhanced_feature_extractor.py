"""
Enhanced Market Feature Extractor - 100+ Features for TRM Training

Expands from 10 features to 100+ including:
- Technical indicators (RSI, MACD, Bollinger, etc.)
- Multi-timeframe returns (1d, 5d, 10d, 20d, 60d)
- Volatility metrics (realized, implied, term structure)
- Cross-asset correlations (SPY-TLT, SPY-VIX, etc.)
- Market regime indicators
- Risk metrics (VaR, drawdown, Sharpe)
- Momentum indicators
- Volume analysis
- Breadth indicators

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 4
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """
    Extract 100+ market features from historical database

    Feature Categories:
    1. Price & Returns (15 features)
    2. Volatility (12 features)
    3. Technical Indicators (20 features)
    4. Volume Analysis (8 features)
    5. Cross-Asset (12 features)
    6. Market Regime (10 features)
    7. Risk Metrics (12 features)
    8. Momentum (8 features)
    9. Breadth (8 features)
    10. Seasonal/Calendar (5 features)

    Total: 110 features
    """

    # Feature names for model input (must match order)
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

    NUM_FEATURES = len(FEATURE_NAMES)  # 110

    def __init__(self, historical_manager):
        """
        Initialize enhanced feature extractor

        Args:
            historical_manager: HistoricalDataManager instance
        """
        self.historical_manager = historical_manager
        logger.info(f"EnhancedFeatureExtractor initialized with {self.NUM_FEATURES} features")

    def extract_features(
        self,
        start_date: str,
        end_date: str,
        include_prices: bool = True
    ) -> pd.DataFrame:
        """
        Extract 110 market features for TRM

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_prices: Include raw prices for strategy simulation

        Returns:
            DataFrame with columns: date + 110 features + optional prices
        """
        # Get raw data with buffer for lookback calculations
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')

        df_raw = self.historical_manager.get_training_data(
            start_date=buffer_start,
            end_date=end_date
        )

        if df_raw.empty:
            logger.warning(f"No raw data for {start_date} to {end_date}")
            return pd.DataFrame()

        # Pivot data: one row per date with columns per symbol
        df_pivot = df_raw.pivot_table(
            index='date',
            columns='symbol',
            values=['open', 'high', 'low', 'close', 'volume', 'returns',
                   'log_returns', 'volatility_20d', 'volatility_60d',
                   'rsi_14', 'ma_50', 'ma_200', 'volume_ratio']
        )

        # Flatten multi-level columns
        df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
        df_pivot = df_pivot.reset_index()
        df_pivot['date'] = pd.to_datetime(df_pivot['date'])

        # Extract key symbol columns
        spy_cols = self._get_symbol_cols(df_pivot, 'SPY')
        tlt_cols = self._get_symbol_cols(df_pivot, 'TLT')
        vix_cols = self._get_symbol_cols(df_pivot, '^VIX')

        if not spy_cols.get('close'):
            logger.warning("No SPY data found")
            return pd.DataFrame()

        # Initialize features DataFrame
        features = pd.DataFrame()
        features['date'] = df_pivot['date']

        # Extract all feature categories
        self._extract_price_returns(features, df_pivot, spy_cols, tlt_cols)
        self._extract_volatility(features, df_pivot, spy_cols, vix_cols)
        self._extract_technical(features, df_pivot, spy_cols, tlt_cols)
        self._extract_volume(features, df_pivot, spy_cols, tlt_cols, vix_cols)
        self._extract_cross_asset(features, df_pivot, spy_cols, tlt_cols, vix_cols)
        self._extract_regime(features, df_pivot, spy_cols)
        self._extract_risk(features, df_pivot, spy_cols)
        self._extract_momentum(features, df_pivot, spy_cols, tlt_cols)
        self._extract_breadth(features, df_pivot)
        self._extract_calendar(features)

        # Filter to requested date range
        features = features[features['date'] >= pd.to_datetime(start_date)]

        # Add price columns for strategy simulation
        if include_prices:
            if spy_cols.get('close'):
                features['spy_close'] = df_pivot.loc[features.index, spy_cols['close']].values
            if tlt_cols.get('close'):
                features['tlt_close'] = df_pivot.loc[features.index, tlt_cols['close']].values
            else:
                features['tlt_close'] = 100.0

        # Fill NaN values
        features = self._fill_missing(features)

        logger.info(f"Extracted {len(features)} rows with {self.NUM_FEATURES} features from {start_date} to {end_date}")

        return features

    def _get_symbol_cols(self, df: pd.DataFrame, symbol: str) -> Dict[str, str]:
        """Get column names for a symbol"""
        cols = {}
        for metric in ['open', 'high', 'low', 'close', 'volume', 'returns',
                       'log_returns', 'volatility_20d', 'volatility_60d',
                       'rsi_14', 'ma_50', 'ma_200', 'volume_ratio']:
            col_name = f"{metric}_{symbol}"
            if col_name in df.columns:
                cols[metric] = col_name
        return cols

    def _extract_price_returns(self, features: pd.DataFrame, df: pd.DataFrame,
                               spy_cols: Dict, tlt_cols: Dict):
        """Extract price and return features (15)"""
        # SPY returns at multiple horizons
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]
            features['spy_return_1d'] = ret
            features['spy_return_5d'] = ret.rolling(5).sum()
            features['spy_return_10d'] = ret.rolling(10).sum()
            features['spy_return_20d'] = ret.rolling(20).sum()
            features['spy_return_60d'] = ret.rolling(60).sum()

        # TLT returns
        if tlt_cols.get('returns'):
            tlt_ret = df[tlt_cols['returns']]
            features['tlt_return_1d'] = tlt_ret
            features['tlt_return_5d'] = tlt_ret.rolling(5).sum()
            features['tlt_return_20d'] = tlt_ret.rolling(20).sum()

        # Log returns
        if spy_cols.get('log_returns'):
            log_ret = df[spy_cols['log_returns']]
            features['spy_log_return_1d'] = log_ret
            features['spy_log_return_5d'] = log_ret.rolling(5).sum()

        # Price to moving averages
        if spy_cols.get('close') and spy_cols.get('ma_50'):
            features['spy_price_to_ma50'] = df[spy_cols['close']] / df[spy_cols['ma_50']] - 1
        if spy_cols.get('close') and spy_cols.get('ma_200'):
            features['spy_price_to_ma200'] = df[spy_cols['close']] / df[spy_cols['ma_200']] - 1
        if tlt_cols.get('close') and tlt_cols.get('ma_50'):
            features['tlt_price_to_ma50'] = df[tlt_cols['close']] / df[tlt_cols['ma_50']] - 1

        # High-low range
        if spy_cols.get('high') and spy_cols.get('low'):
            features['spy_high_low_range'] = (df[spy_cols['high']] - df[spy_cols['low']]) / df[spy_cols['close']]
            features['spy_close_to_high'] = (df[spy_cols['close']] - df[spy_cols['low']]) / (df[spy_cols['high']] - df[spy_cols['low']] + 1e-8)

    def _extract_volatility(self, features: pd.DataFrame, df: pd.DataFrame,
                            spy_cols: Dict, vix_cols: Dict):
        """Extract volatility features (12)"""
        # VIX metrics
        if vix_cols.get('close'):
            vix = df[vix_cols['close']]
            features['vix_level'] = vix
            features['vix_change_1d'] = vix.pct_change()
            features['vix_change_5d'] = vix.pct_change(5)
            features['vix_percentile_20d'] = vix.rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
        else:
            features['vix_level'] = 20.0

        # Realized volatility at different horizons
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]
            features['spy_volatility_5d'] = ret.rolling(5).std() * np.sqrt(252)
            features['spy_volatility_10d'] = ret.rolling(10).std() * np.sqrt(252)
            features['spy_volatility_20d'] = ret.rolling(20).std() * np.sqrt(252)
            features['spy_volatility_60d'] = ret.rolling(60).std() * np.sqrt(252)

            # Volatility ratios (term structure)
            features['volatility_ratio_5_20'] = features['spy_volatility_5d'] / (features['spy_volatility_20d'] + 1e-8)
            features['volatility_ratio_20_60'] = features['spy_volatility_20d'] / (features['spy_volatility_60d'] + 1e-8)

        # VIX term structure proxy
        if vix_cols.get('close'):
            features['vix_term_structure'] = vix.rolling(5).mean() / (vix.rolling(20).mean() + 1e-8)

        # Realized vs implied
        if vix_cols.get('close') and 'spy_volatility_20d' in features.columns:
            features['realized_vs_implied_vol'] = features['spy_volatility_20d'] * 100 / (features['vix_level'] + 1e-8)

    def _extract_technical(self, features: pd.DataFrame, df: pd.DataFrame,
                           spy_cols: Dict, tlt_cols: Dict):
        """Extract technical indicator features (20)"""
        # RSI
        if spy_cols.get('rsi_14'):
            features['spy_rsi_14'] = df[spy_cols['rsi_14']] / 100.0  # Normalize to 0-1
        if spy_cols.get('returns'):
            features['spy_rsi_5'] = self._calculate_rsi(df[spy_cols['returns']], 5) / 100.0
        if tlt_cols.get('rsi_14'):
            features['tlt_rsi_14'] = df[tlt_cols['rsi_14']] / 100.0

        # MACD (calculate from price)
        if spy_cols.get('close'):
            close = df[spy_cols['close']]
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            features['spy_macd'] = (ema12 - ema26) / close
            features['spy_macd_signal'] = features['spy_macd'].ewm(span=9).mean()
            features['spy_macd_histogram'] = features['spy_macd'] - features['spy_macd_signal']

        # Bollinger Bands
        if spy_cols.get('close'):
            close = df[spy_cols['close']]
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = ma20 + 2 * std20
            bb_lower = ma20 - 2 * std20
            features['spy_bb_upper_dist'] = (bb_upper - close) / close
            features['spy_bb_lower_dist'] = (close - bb_lower) / close
            features['spy_bb_width'] = (bb_upper - bb_lower) / close

        # Stochastic
        if spy_cols.get('high') and spy_cols.get('low') and spy_cols.get('close'):
            high = df[spy_cols['high']]
            low = df[spy_cols['low']]
            close = df[spy_cols['close']]
            lowest_14 = low.rolling(14).min()
            highest_14 = high.rolling(14).max()
            features['spy_stochastic_k'] = (close - lowest_14) / (highest_14 - lowest_14 + 1e-8)
            features['spy_stochastic_d'] = features['spy_stochastic_k'].rolling(3).mean()

        # ATR
        if spy_cols.get('high') and spy_cols.get('low') and spy_cols.get('close'):
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            features['spy_atr_14'] = tr.rolling(14).mean() / close
            features['spy_atr_ratio'] = tr / (tr.rolling(50).mean() + 1e-8)

        # Other indicators
        if spy_cols.get('volume') and spy_cols.get('close'):
            # OBV slope
            vol = df[spy_cols['volume']]
            close = df[spy_cols['close']]
            obv = (np.sign(close.diff()) * vol).cumsum()
            features['spy_obv_slope'] = obv.diff(10) / (vol.rolling(10).sum() + 1)

        # MFI (simplified)
        if spy_cols.get('close') and spy_cols.get('volume'):
            typical_price = close
            mf = typical_price * vol
            pos_mf = mf.where(close > close.shift(), 0).rolling(14).sum()
            neg_mf = mf.where(close < close.shift(), 0).rolling(14).sum()
            features['spy_mfi_14'] = pos_mf / (pos_mf + neg_mf + 1e-8)

        # Williams %R
        if spy_cols.get('high') and spy_cols.get('low') and spy_cols.get('close'):
            features['spy_williams_r'] = (highest_14 - close) / (highest_14 - lowest_14 + 1e-8)

        # CCI
        if spy_cols.get('high') and spy_cols.get('low') and spy_cols.get('close'):
            tp = (high + low + close) / 3
            features['spy_cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)
            features['spy_cci_20'] = features['spy_cci_20'] / 200  # Normalize

        # ADX (simplified)
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]
            features['spy_adx_14'] = ret.abs().rolling(14).mean() / (ret.rolling(14).std() + 1e-8)
            features['spy_di_plus'] = ret.clip(lower=0).rolling(14).mean()
            features['spy_di_minus'] = (-ret.clip(upper=0)).rolling(14).mean()

    def _extract_volume(self, features: pd.DataFrame, df: pd.DataFrame,
                        spy_cols: Dict, tlt_cols: Dict, vix_cols: Dict):
        """Extract volume analysis features (8)"""
        if spy_cols.get('volume'):
            vol = df[spy_cols['volume']]
            features['spy_volume_ratio_5d'] = vol / (vol.rolling(5).mean() + 1)
            features['spy_volume_ratio_20d'] = vol / (vol.rolling(20).mean() + 1)
            features['spy_volume_trend'] = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1) - 1
            features['spy_volume_volatility'] = vol.rolling(20).std() / (vol.rolling(20).mean() + 1)

        if tlt_cols.get('volume_ratio'):
            features['tlt_volume_ratio'] = df[tlt_cols['volume_ratio']]

        if vix_cols.get('volume_ratio'):
            features['vix_volume_ratio'] = df[vix_cols['volume_ratio']]

        # Up/down volume (proxy using returns)
        if spy_cols.get('returns') and spy_cols.get('volume'):
            ret = df[spy_cols['returns']]
            vol = df[spy_cols['volume']]
            up_vol = vol.where(ret > 0, 0).rolling(10).sum()
            down_vol = vol.where(ret < 0, 0).rolling(10).sum()
            total_vol = up_vol + down_vol + 1
            features['up_volume_ratio'] = up_vol / total_vol
            features['down_volume_ratio'] = down_vol / total_vol

    def _extract_cross_asset(self, features: pd.DataFrame, df: pd.DataFrame,
                             spy_cols: Dict, tlt_cols: Dict, vix_cols: Dict):
        """Extract cross-asset features (12)"""
        # Correlations
        if spy_cols.get('returns') and tlt_cols.get('returns'):
            spy_ret = df[spy_cols['returns']]
            tlt_ret = df[tlt_cols['returns']]
            features['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)
            features['spy_tlt_corr_60d'] = spy_ret.rolling(60).corr(tlt_ret)

        if spy_cols.get('returns') and vix_cols.get('returns'):
            vix_ret = df[vix_cols['returns']]
            features['spy_vix_corr_20d'] = spy_ret.rolling(20).corr(vix_ret)

        # Gold correlation (proxy)
        features['spy_gold_corr_20d'] = 0.0  # Would need GLD data

        # SPY-TLT spread and ratio
        if spy_cols.get('close') and tlt_cols.get('close'):
            spy_close = df[spy_cols['close']]
            tlt_close = df[tlt_cols['close']]
            features['spy_tlt_spread'] = (spy_close / spy_close.iloc[0]) - (tlt_close / tlt_close.iloc[0])
            features['spy_tlt_ratio'] = spy_close / tlt_close

        # Risk on/off indicator
        if 'spy_tlt_corr_20d' in features.columns:
            features['risk_on_off_indicator'] = -features['spy_tlt_corr_20d']  # Negative corr = risk on

        # Flight to quality
        if spy_cols.get('returns') and tlt_cols.get('returns'):
            features['flight_to_quality'] = tlt_ret.rolling(5).sum() - spy_ret.rolling(5).sum()

        # Cross-asset momentum
        if spy_cols.get('returns') and tlt_cols.get('returns'):
            features['cross_asset_momentum'] = spy_ret.rolling(20).sum() + tlt_ret.rolling(20).sum()

        # Asset rotation signal
        features['asset_rotation_signal'] = features.get('spy_return_20d', 0) - features.get('tlt_return_20d', 0)

        # Correlation regime
        if 'spy_tlt_corr_20d' in features.columns:
            features['correlation_regime'] = (features['spy_tlt_corr_20d'] > 0).astype(float)

        # Diversification ratio (proxy)
        features['diversification_ratio'] = 1.0 - features.get('spy_tlt_corr_20d', 0).abs()

    def _extract_regime(self, features: pd.DataFrame, df: pd.DataFrame, spy_cols: Dict):
        """Extract market regime features (10)"""
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]

            # Trend strength
            features['trend_strength_20d'] = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
            features['trend_strength_60d'] = ret.rolling(60).mean() / (ret.rolling(60).std() + 1e-8)

            # Regime indicators
            features['regime_volatility'] = ret.rolling(20).std() / (ret.rolling(60).std() + 1e-8)
            features['regime_momentum'] = ret.rolling(20).sum() / (ret.rolling(20).std() * np.sqrt(20) + 1e-8)

            # Bull/bear
            features['bull_bear_indicator'] = (ret.rolling(50).sum() > 0).astype(float)

            # Crisis probability (high vol + negative returns)
            high_vol = (ret.rolling(20).std() > ret.rolling(60).std() * 1.5).astype(float)
            neg_ret = (ret.rolling(10).sum() < -0.05).astype(float)
            features['crisis_probability'] = (high_vol + neg_ret) / 2

            # Mean reversion signal
            z_score = (ret.rolling(5).sum() - ret.rolling(60).mean()) / (ret.rolling(60).std() + 1e-8)
            features['mean_reversion_signal'] = -np.tanh(z_score / 2)

            # Breakout signal
            if spy_cols.get('close'):
                close = df[spy_cols['close']]
                high_20 = close.rolling(20).max()
                low_20 = close.rolling(20).min()
                features['breakout_signal'] = (close - low_20) / (high_20 - low_20 + 1e-8)

            # Consolidation
            range_ratio = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-8)
            features['consolidation_indicator'] = 1 - np.clip(range_ratio, 0, 2) / 2

            # Regime change probability
            vol_change = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-8) - 1
            features['regime_change_probability'] = np.clip(vol_change.abs(), 0, 1)

    def _extract_risk(self, features: pd.DataFrame, df: pd.DataFrame, spy_cols: Dict):
        """Extract risk metrics (12)"""
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]

            # VaR and CVaR
            features['var_95_1d'] = -ret.rolling(252).quantile(0.05)
            features['var_99_1d'] = -ret.rolling(252).quantile(0.01)
            features['cvar_95'] = -ret.rolling(252).apply(lambda x: x[x <= x.quantile(0.05)].mean())

            # Drawdown
            if spy_cols.get('close'):
                close = df[spy_cols['close']]
                roll_max_20 = close.rolling(20).max()
                roll_max_60 = close.rolling(60).max()
                features['max_drawdown_20d'] = (close - roll_max_20) / roll_max_20
                features['max_drawdown_60d'] = (close - roll_max_60) / roll_max_60

            # Sharpe ratio
            rf = 0.0  # Simplified
            features['sharpe_20d'] = (ret.rolling(20).mean() - rf/252) / (ret.rolling(20).std() + 1e-8) * np.sqrt(252)
            features['sharpe_60d'] = (ret.rolling(60).mean() - rf/252) / (ret.rolling(60).std() + 1e-8) * np.sqrt(252)

            # Sortino
            downside = ret.clip(upper=0)
            features['sortino_20d'] = (ret.rolling(20).mean() - rf/252) / (downside.rolling(20).std() + 1e-8) * np.sqrt(252)

            # Calmar
            if 'max_drawdown_60d' in features.columns:
                features['calmar_ratio'] = ret.rolling(252).sum() / (-features['max_drawdown_60d'].clip(upper=-0.01))
                features['calmar_ratio'] = features['calmar_ratio'].clip(-10, 10)

            # Higher moments
            features['tail_risk_indicator'] = -ret.rolling(20).apply(lambda x: x[x < x.quantile(0.1)].mean())
            features['skewness_20d'] = ret.rolling(20).skew()
            features['kurtosis_20d'] = ret.rolling(20).kurt()

    def _extract_momentum(self, features: pd.DataFrame, df: pd.DataFrame,
                          spy_cols: Dict, tlt_cols: Dict):
        """Extract momentum features (8)"""
        if spy_cols.get('returns'):
            ret = df[spy_cols['returns']]

            # Multi-horizon momentum
            features['momentum_1m'] = ret.rolling(21).sum()
            features['momentum_3m'] = ret.rolling(63).sum()
            features['momentum_6m'] = ret.rolling(126).sum()
            features['momentum_12m'] = ret.rolling(252).sum()

            # Rate of change
            if spy_cols.get('close'):
                close = df[spy_cols['close']]
                features['momentum_roc'] = close.pct_change(20)

            # Relative momentum SPY vs TLT
            if tlt_cols.get('returns'):
                tlt_ret = df[tlt_cols['returns']]
                features['relative_momentum_spy_tlt'] = ret.rolling(20).sum() - tlt_ret.rolling(20).sum()

            # Dual momentum signal (Antonacci)
            spy_12m = ret.rolling(252).sum()
            features['dual_momentum_signal'] = (spy_12m > 0).astype(float) * np.sign(features.get('relative_momentum_spy_tlt', 0))

            # Momentum breadth
            features['momentum_breadth'] = (features.get('momentum_1m', 0) > 0).astype(float)

    def _extract_breadth(self, features: pd.DataFrame, df: pd.DataFrame):
        """Extract breadth features (8)"""
        returns_cols = [c for c in df.columns if c.startswith('returns_') and 'SPY' not in c]

        if returns_cols:
            returns_df = df[returns_cols]

            # Market breadth
            features['market_breadth'] = (returns_df > 0).mean(axis=1)

            # Advance/decline
            advances = (returns_df > 0).sum(axis=1)
            declines = (returns_df < 0).sum(axis=1)
            features['advance_decline_ratio'] = advances / (declines + 1)

            # New highs/lows (proxy)
            features['new_highs_lows_ratio'] = (returns_df > returns_df.rolling(20).max().shift(1)).mean(axis=1)
        else:
            features['market_breadth'] = 0.5
            features['advance_decline_ratio'] = 1.0
            features['new_highs_lows_ratio'] = 0.5

        # Percent above MAs (proxy)
        features['percent_above_ma50'] = features.get('spy_price_to_ma50', 0) > 0
        features['percent_above_ma200'] = features.get('spy_price_to_ma200', 0) > 0

        # Sector dispersion
        if returns_cols:
            features['sector_dispersion'] = df[returns_cols].std(axis=1)
        else:
            features['sector_dispersion'] = 0.02

        # Sector rotation (proxy)
        features['sector_rotation_indicator'] = features.get('sector_dispersion', 0.02) / 0.02 - 1

        # Breadth thrust (simplified)
        if 'market_breadth' in features.columns:
            features['breadth_thrust'] = features['market_breadth'].rolling(10).apply(
                lambda x: (x > 0.6).sum() / len(x)
            )

    def _extract_calendar(self, features: pd.DataFrame):
        """Extract calendar/seasonal features (5)"""
        dates = pd.to_datetime(features['date'])

        # Cyclical encoding for day of week (0-4 -> sin/cos would be better)
        features['day_of_week'] = dates.dt.dayofweek / 4.0  # Normalize 0-1

        # Day of month
        features['day_of_month'] = dates.dt.day / 31.0  # Normalize 0-1

        # Month of year
        features['month_of_year'] = dates.dt.month / 12.0  # Normalize 0-1

        # Quarter end
        features['is_quarter_end'] = dates.dt.is_quarter_end.astype(float)

        # Days to options expiry (3rd Friday proxy)
        features['days_to_expiry'] = (21 - dates.dt.day % 21) / 21.0

    def _calculate_rsi(self, returns: pd.Series, period: int) -> pd.Series:
        """Calculate RSI from returns"""
        gains = returns.clip(lower=0)
        losses = (-returns.clip(upper=0))

        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _fill_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults"""
        # Forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')

        # Default values for remaining NaN
        defaults = {
            'vix_level': 20.0,
            'spy_return_1d': 0.0, 'spy_return_5d': 0.0, 'spy_return_10d': 0.0,
            'spy_return_20d': 0.0, 'spy_return_60d': 0.0,
            'tlt_return_1d': 0.0, 'tlt_return_5d': 0.0, 'tlt_return_20d': 0.0,
            'spy_volatility_20d': 0.15, 'spy_volatility_60d': 0.15,
            'spy_rsi_14': 0.5, 'tlt_rsi_14': 0.5,
            'market_breadth': 0.5, 'correlation_regime': 0.5,
            'bull_bear_indicator': 0.5, 'crisis_probability': 0.1,
        }

        for col, default in defaults.items():
            if col in features.columns:
                features[col] = features[col].fillna(default)

        # Fill any remaining NaN with 0
        features = features.fillna(0)

        return features

    def get_feature_vector(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature vector in correct order for model input

        Args:
            features_df: DataFrame with all features

        Returns:
            numpy array of shape (n_samples, 110)
        """
        # Ensure all features exist
        for name in self.FEATURE_NAMES:
            if name not in features_df.columns:
                features_df[name] = 0.0

        return features_df[self.FEATURE_NAMES].values


if __name__ == "__main__":
    # Test enhanced feature extractor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("Testing Enhanced Feature Extractor...")
    print("=" * 80)
    print(f"Total features: {EnhancedFeatureExtractor.NUM_FEATURES}")
    print(f"Feature names: {EnhancedFeatureExtractor.FEATURE_NAMES[:10]}...")

    # Would need to initialize with historical manager
    print("\nFeature categories:")
    print("  1. Price & Returns: 15 features")
    print("  2. Volatility: 12 features")
    print("  3. Technical: 20 features")
    print("  4. Volume: 8 features")
    print("  5. Cross-Asset: 12 features")
    print("  6. Regime: 10 features")
    print("  7. Risk: 12 features")
    print("  8. Momentum: 8 features")
    print("  9. Breadth: 8 features")
    print("  10. Calendar: 5 features")
    print(f"\nTotal: {EnhancedFeatureExtractor.NUM_FEATURES} features")
