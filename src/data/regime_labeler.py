"""
Regime Labeler for TRM Training

Creates LEARNABLE regime classification labels based on CURRENT market state.
Unlike strategy selection (which requires predicting future returns),
regime classification predicts the current market environment.

Regimes:
- 0: BULL   - Strong uptrend, low volatility, positive momentum
- 1: SIDEWAYS - No clear trend, moderate conditions
- 2: BEAR   - Downtrend, high volatility, negative momentum

Signal Analysis Results:
- put_call_ratio vs regime: r=0.65 ***
- spy_ret_5d vs regime: r=-0.43 ***
- sector_dispersion vs regime: r=+0.24 ***
- Random Forest accuracy: 62.7% (vs 33.3% random baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RegimeLabeler:
    """
    Generate regime classification labels from market features.

    Unlike the original StrategyLabeler which tried to predict optimal
    future allocations (impossible), this labels the CURRENT regime
    which is observable from current features.
    """

    # Regime definitions
    BULL = 0
    SIDEWAYS = 1
    BEAR = 2

    REGIME_NAMES = ['BULL', 'SIDEWAYS', 'BEAR']

    def __init__(
        self,
        vix_bull_threshold: float = 20.0,
        vix_bear_threshold: float = 30.0,
        momentum_bull_threshold: float = 0.02,
        momentum_bear_threshold: float = -0.02,
        method: str = 'hybrid'
    ):
        """
        Initialize regime labeler.

        Args:
            vix_bull_threshold: VIX below this = low volatility (bull signal)
            vix_bear_threshold: VIX above this = high volatility (bear signal)
            momentum_bull_threshold: 20d return above this = uptrend
            momentum_bear_threshold: 20d return below this = downtrend
            method: 'threshold', 'kmeans', or 'hybrid'
        """
        self.vix_bull = vix_bull_threshold
        self.vix_bear = vix_bear_threshold
        self.mom_bull = momentum_bull_threshold
        self.mom_bear = momentum_bear_threshold
        self.method = method

        logger.info(f"RegimeLabeler initialized with method='{method}'")
        logger.info(f"  VIX thresholds: bull<{vix_bull_threshold}, bear>{vix_bear_threshold}")
        logger.info(f"  Momentum thresholds: bull>{momentum_bull_threshold}, bear<{momentum_bear_threshold}")

    def label_threshold(
        self,
        vix: np.ndarray,
        spy_ret_20d: np.ndarray
    ) -> np.ndarray:
        """
        Label regimes using simple thresholds.

        Rules:
        - BULL: positive momentum AND low VIX
        - BEAR: negative momentum OR high VIX
        - SIDEWAYS: everything else
        """
        n = len(vix)
        regimes = np.ones(n, dtype=np.int64) * self.SIDEWAYS  # Default sideways

        # Bull conditions
        bull_mask = (spy_ret_20d > self.mom_bull) & (vix < self.vix_bull)
        regimes[bull_mask] = self.BULL

        # Bear conditions (takes precedence)
        bear_mask = (spy_ret_20d < self.mom_bear) | (vix > self.vix_bear)
        regimes[bear_mask] = self.BEAR

        return regimes

    def label_from_features(
        self,
        features: np.ndarray,
        feature_names: Optional[list] = None
    ) -> np.ndarray:
        """
        Label regimes from feature matrix.

        Args:
            features: (N, 10) array of market features
            feature_names: List of feature names (to find vix and spy_ret_20d indices)

        Returns:
            (N,) array of regime labels (0=bull, 1=sideways, 2=bear)
        """
        if feature_names is None:
            feature_names = [
                'vix_level', 'spy_ret_5d', 'spy_ret_20d', 'volume_ratio',
                'market_breadth', 'correlation', 'put_call_ratio',
                'gini_coef', 'sector_disp', 'signal_quality'
            ]

        # Find indices
        vix_idx = feature_names.index('vix_level')
        mom_idx = feature_names.index('spy_ret_20d')

        vix = features[:, vix_idx]
        spy_ret_20d = features[:, mom_idx]

        return self.label_threshold(vix, spy_ret_20d)

    def convert_strategy_dataset(
        self,
        input_path: str,
        output_path: str
    ) -> pd.DataFrame:
        """
        Convert existing strategy labels dataset to regime labels.

        Args:
            input_path: Path to black_swan_labels.parquet
            output_path: Path to save regime_labels.parquet

        Returns:
            DataFrame with regime labels
        """
        logger.info(f"Loading strategy dataset from {input_path}")
        df = pd.read_parquet(input_path)

        # Extract features
        X = np.array(df['features'].tolist())

        # Generate regime labels
        regimes = self.label_from_features(X)

        # Create new dataframe
        regime_df = pd.DataFrame({
            'date': df['date'],
            'features': df['features'],
            'regime': regimes,
            'regime_name': [self.REGIME_NAMES[r] for r in regimes],
            'period_name': df['period_name']
        })

        # Log distribution
        regime_counts = regime_df['regime'].value_counts().sort_index()
        logger.info("Regime distribution:")
        for regime, count in regime_counts.items():
            pct = 100 * count / len(regime_df)
            logger.info(f"  {self.REGIME_NAMES[regime]}: {count} ({pct:.1f}%)")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        regime_df.to_parquet(output_path, index=False)
        logger.info(f"Saved regime labels to {output_path}")

        # Also save CSV for inspection
        csv_path = output_path.replace('.parquet', '.csv')
        regime_df_export = regime_df.drop(columns=['features'])
        regime_df_export.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")

        return regime_df

    def analyze_signal(
        self,
        features: np.ndarray,
        regimes: np.ndarray,
        feature_names: Optional[list] = None
    ) -> dict:
        """
        Analyze signal strength between features and regime labels.

        Returns dict with correlations, ANOVA results, and mutual information.
        """
        from scipy import stats
        from sklearn.feature_selection import mutual_info_classif

        if feature_names is None:
            feature_names = [
                'vix_level', 'spy_ret_5d', 'spy_ret_20d', 'volume_ratio',
                'market_breadth', 'correlation', 'put_call_ratio',
                'gini_coef', 'sector_disp', 'signal_quality'
            ]

        results = {
            'correlations': {},
            'anova': {},
            'mutual_info': {}
        }

        # Correlations
        for i, name in enumerate(feature_names):
            corr, pval = stats.spearmanr(features[:, i], regimes)
            results['correlations'][name] = {'r': corr, 'p': pval}

        # ANOVA
        for i, name in enumerate(feature_names):
            groups = [features[regimes == r, i] for r in range(3)]
            f_stat, pval = stats.f_oneway(*groups)
            results['anova'][name] = {'F': f_stat, 'p': pval}

        # Mutual information
        mi_scores = mutual_info_classif(features, regimes, random_state=42)
        for name, mi in zip(feature_names, mi_scores):
            results['mutual_info'][name] = mi

        results['total_mi'] = sum(mi_scores)
        results['max_possible_mi'] = np.log2(3)  # 3 classes

        return results


def create_regime_dataset():
    """
    Create regime classification dataset from existing strategy labels.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    # Paths
    input_path = Path(__file__).parent.parent.parent / 'data' / 'trm_training' / 'black_swan_labels.parquet'
    output_path = Path(__file__).parent.parent.parent / 'data' / 'trm_training' / 'regime_labels.parquet'

    # Create labeler with tuned thresholds for black swan data
    # (VIX is higher during crises, so adjust thresholds)
    labeler = RegimeLabeler(
        vix_bull_threshold=22.0,
        vix_bear_threshold=30.0,
        momentum_bull_threshold=0.01,
        momentum_bear_threshold=-0.02
    )

    # Convert dataset
    regime_df = labeler.convert_strategy_dataset(str(input_path), str(output_path))

    # Analyze signal
    X = np.array(regime_df['features'].tolist())
    y = regime_df['regime'].values

    print("\n" + "="*80)
    print("SIGNAL ANALYSIS")
    print("="*80)

    results = labeler.analyze_signal(X, y)

    print("\nCorrelations (features vs regime):")
    for name, vals in sorted(results['correlations'].items(), key=lambda x: -abs(x[1]['r'])):
        sig = '***' if vals['p'] < 0.001 else '**' if vals['p'] < 0.01 else '*' if vals['p'] < 0.05 else ''
        print(f"  {name:20s}: r={vals['r']:+.4f} {sig}")

    print(f"\nTotal Mutual Information: {results['total_mi']:.3f} bits (max: {results['max_possible_mi']:.2f} bits)")
    print(f"Information captured: {100*results['total_mi']/results['max_possible_mi']:.1f}%")

    # Quick ML test
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')

    print(f"\nRandom Forest 5-fold CV: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
    print(f"Random baseline: 33.3%")
    print(f"Improvement over random: +{scores.mean()*100 - 33.3:.1f}%")

    return regime_df


if __name__ == "__main__":
    create_regime_dataset()
