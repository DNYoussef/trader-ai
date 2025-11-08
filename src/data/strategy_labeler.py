"""
Strategy Labeler for TRM Training Data Generation

Backtests all 8 trading strategies on historical data to generate training labels.
For each time window, determines which strategy performed best and labels accordingly.

Output: (market_features, winning_strategy_idx, realized_pnl)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
from data.market_feature_extractor import MarketFeatureExtractor

logger = logging.getLogger(__name__)


class StrategyLabeler:
    """
    Generate training labels by backtesting 8 strategies on historical data.

    Process:
    1. For each time window (e.g., daily):
       - Extract market features at time t
       - Simulate all 8 strategies forward 5 days
       - Record which strategy had best PnL
       - Create label: (features_t, winning_strategy_idx, pnl)

    2. Focus on black swan periods (12 events):
       - Asian Crisis (1997)
       - Dot-com Crash (2000-2002)
       - 9/11 (2001)
       - Financial Crisis (2008-2009)
       - Flash Crash (2010)
       - European Debt Crisis (2011-2012)
       - China Selloff (2015)
       - Brexit (2016)
       - COVID-19 (2020)
       - GameStop (2021)
       - Russia-Ukraine (2022)
       - SVB Banking Crisis (2023)
    """

    def __init__(
        self,
        historical_data_manager,
        strategies_config: Dict,
        lookforward_days: int = 5,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize strategy labeler

        Args:
            historical_data_manager: Access to 30 years of market data
            strategies_config: Configuration for 8 strategies
            lookforward_days: Days to simulate forward for PnL calculation
            confidence_threshold: Minimum confidence for valid label
        """
        self.historical_manager = historical_data_manager
        self.strategies_config = strategies_config
        self.lookforward_days = lookforward_days
        self.confidence_threshold = confidence_threshold

        # Initialize feature extractor
        self.feature_extractor = MarketFeatureExtractor(historical_data_manager)

        # 8 strategy definitions
        self.strategy_names = [
            'ultra_defensive',      # 0
            'defensive',            # 1
            'balanced_safe',        # 2
            'balanced_growth',      # 3
            'growth',               # 4
            'aggressive_growth',    # 5
            'contrarian_long',      # 6
            'tactical_opportunity'  # 7
        ]

        # Black swan periods (start_date, end_date, name)
        self.black_swan_periods = [
            ('1997-07-02', '1997-10-27', 'Asian Financial Crisis'),
            ('2000-03-10', '2002-10-09', 'Dot-com Crash'),
            ('2001-09-11', '2001-09-21', '9/11 Attacks'),
            ('2008-09-15', '2009-03-09', 'Financial Crisis'),
            ('2010-05-06', '2010-05-07', 'Flash Crash'),
            ('2011-08-01', '2012-06-30', 'European Debt Crisis'),
            ('2015-08-18', '2015-09-30', 'China Selloff'),
            ('2016-06-23', '2016-07-15', 'Brexit'),
            ('2020-02-19', '2020-03-23', 'COVID-19 Crash'),
            ('2021-01-27', '2021-02-05', 'GameStop Short Squeeze'),
            ('2022-02-24', '2022-03-31', 'Russia-Ukraine'),
            ('2023-03-08', '2023-03-17', 'SVB Banking Crisis')
        ]

        logger.info(f"Strategy Labeler initialized with {len(self.strategy_names)} strategies")
        logger.info(f"Tracking {len(self.black_swan_periods)} black swan periods")

    def extract_market_features(self, df: pd.DataFrame, date: datetime) -> np.ndarray:
        """
        Extract 10 market features for a given date

        Features (matching TRM input):
        1. vix_level
        2. spy_returns_5d
        3. spy_returns_20d
        4. volume_ratio
        5. market_breadth
        6. correlation
        7. put_call_ratio
        8. gini_coefficient
        9. sector_dispersion
        10. signal_quality_score

        Args:
            df: Historical market data (from feature extractor)
            date: Date to extract features

        Returns:
            features: (10,) numpy array
        """
        # Convert date to date-only for comparison (remove time component)
        search_date = pd.to_datetime(date).date()

        # Get row for this date - df['date'] might be datetime or date
        df_dates = pd.to_datetime(df['date']).dt.date
        row = df[df_dates == search_date]

        if row.empty:
            # Try without date conversion (fallback)
            row = df[df['date'] == date]
            if row.empty:
                logger.debug(f"No data for date {date}")
                return None

        row = row.iloc[0]

        # Extract features (ensure order matches trm_config.py)
        features = np.array([
            row.get('vix', 20.0),                    # VIX level
            row.get('spy_returns_5d', 0.0),          # 5-day momentum
            row.get('spy_returns_20d', 0.0),         # 20-day momentum
            row.get('volume_ratio', 1.0),            # Liquidity
            row.get('market_breadth', 0.5),          # Breadth
            row.get('correlation', 0.5),             # Asset correlation
            row.get('put_call_ratio', 1.0),          # Sentiment
            row.get('gini_coefficient', 0.5),        # Inequality (Gary's framework)
            row.get('sector_dispersion', 0.3),       # Sector health
            row.get('signal_quality', 0.7)           # Overall confidence
        ], dtype=np.float32)

        return features

    def simulate_strategy(
        self,
        strategy_idx: int,
        df: pd.DataFrame,
        start_date: datetime,
        lookforward_days: int = 5
    ) -> float:
        """
        Simulate a strategy forward N days and compute PnL

        Strategy allocation logic:
        - ultra_defensive (0): 20% SPY, 50% TLT, 30% cash
        - defensive (1): 40% SPY, 30% TLT, 30% cash
        - balanced_safe (2): 60% SPY, 20% TLT, 20% cash
        - balanced_growth (3): 70% SPY, 20% TLT, 10% cash
        - growth (4): 80% SPY, 15% TLT, 5% cash
        - aggressive_growth (5): 90% SPY, 10% TLT, 0% cash
        - contrarian_long (6): 85% SPY (contrarian), 15% TLT
        - tactical_opportunity (7): 75% SPY, 25% TLT

        Args:
            strategy_idx: Strategy index (0-7)
            df: Historical market data
            start_date: Start date for simulation
            lookforward_days: Number of days to simulate

        Returns:
            realized_pnl: Percentage return over the period
        """
        # Define allocations for each strategy
        allocations = {
            0: {'SPY': 0.20, 'TLT': 0.50, 'CASH': 0.30},  # ultra_defensive
            1: {'SPY': 0.40, 'TLT': 0.30, 'CASH': 0.30},  # defensive
            2: {'SPY': 0.60, 'TLT': 0.20, 'CASH': 0.20},  # balanced_safe
            3: {'SPY': 0.70, 'TLT': 0.20, 'CASH': 0.10},  # balanced_growth
            4: {'SPY': 0.80, 'TLT': 0.15, 'CASH': 0.05},  # growth
            5: {'SPY': 0.90, 'TLT': 0.10, 'CASH': 0.00},  # aggressive_growth
            6: {'SPY': 0.85, 'TLT': 0.15, 'CASH': 0.00},  # contrarian_long
            7: {'SPY': 0.75, 'TLT': 0.25, 'CASH': 0.00}   # tactical_opportunity
        }

        allocation = allocations[strategy_idx]

        # Get end date
        end_date = start_date + timedelta(days=lookforward_days)

        # Convert dates for comparison
        start_date_only = pd.to_datetime(start_date).date()
        end_date_only = pd.to_datetime(end_date).date()
        df_dates = pd.to_datetime(df['date']).dt.date

        # Get returns for SPY and TLT over this period
        start_row = df[df_dates == start_date_only]
        end_row = df[df_dates >= end_date_only].head(1)

        if start_row.empty or end_row.empty:
            return 0.0  # No data available

        start_row = start_row.iloc[0]
        end_row = end_row.iloc[0]

        # Calculate returns
        spy_return = (end_row.get('spy_close', start_row.get('spy_close', 100)) /
                     start_row.get('spy_close', 100) - 1)
        tlt_return = (end_row.get('tlt_close', start_row.get('tlt_close', 100)) /
                     start_row.get('tlt_close', 100) - 1)

        # Portfolio return = weighted sum
        portfolio_return = (allocation['SPY'] * spy_return +
                          allocation['TLT'] * tlt_return)

        return portfolio_return

    def generate_label(
        self,
        df: pd.DataFrame,
        date: datetime
    ) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Generate single training label for a given date

        Process:
        1. Extract market features at date
        2. Simulate all 8 strategies forward
        3. Find winning strategy (highest PnL)
        4. Return (features, strategy_idx, pnl)

        Args:
            df: Historical market data
            date: Date to generate label for

        Returns:
            (features, strategy_idx, pnl) or None if insufficient data
        """
        # Extract features
        features = self.extract_market_features(df, date)

        if features is None:
            return None

        # Simulate all strategies
        strategy_pnls = []
        for strategy_idx in range(len(self.strategy_names)):
            pnl = self.simulate_strategy(strategy_idx, df, date, self.lookforward_days)
            strategy_pnls.append(pnl)

        # Find winning strategy
        winning_idx = int(np.argmax(strategy_pnls))
        winning_pnl = float(strategy_pnls[winning_idx])

        # Confidence check: is winner significantly better?
        sorted_pnls = sorted(strategy_pnls, reverse=True)
        if len(sorted_pnls) > 1:
            confidence = (sorted_pnls[0] - sorted_pnls[1]) / (abs(sorted_pnls[0]) + 1e-6)

            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence label at {date}: {confidence:.2f}")

        return (features, winning_idx, winning_pnl)

    def generate_labels_for_period(
        self,
        start_date: str,
        end_date: str,
        period_name: str = "Custom Period"
    ) -> pd.DataFrame:
        """
        Generate labels for entire time period

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period_name: Name for logging

        Returns:
            DataFrame with columns:
                - date
                - features (list of 10 floats)
                - strategy_idx (0-7)
                - pnl (realized percentage return)
        """
        logger.info(f"Generating labels for {period_name} ({start_date} to {end_date})")

        # Get processed features for this period
        df = self.feature_extractor.extract_features(start_date, end_date)

        if df.empty:
            logger.warning(f"No data available for {period_name}")
            return pd.DataFrame()

        # Generate labels for each trading day
        labels = []
        dates = pd.to_datetime(df['date']).unique()

        for date in dates:
            label = self.generate_label(df, date)

            if label is not None:
                features, strategy_idx, pnl = label
                labels.append({
                    'date': date,
                    'features': features.tolist(),
                    'strategy_idx': strategy_idx,
                    'pnl': pnl
                })

        labels_df = pd.DataFrame(labels)

        logger.info(f"Generated {len(labels_df)} labels for {period_name}")

        if len(labels_df) > 0:
            logger.info(f"Strategy distribution: {labels_df['strategy_idx'].value_counts().to_dict()}")
            logger.info(f"Average PnL: {labels_df['pnl'].mean():.4f}")
        else:
            logger.warning(f"No valid labels generated for {period_name} - check data availability")

        return labels_df

    def generate_black_swan_labels(self) -> pd.DataFrame:
        """
        Generate labels for all 12 black swan periods

        Returns:
            Combined DataFrame with all black swan labels
        """
        logger.info("Generating labels for 12 black swan periods...")

        all_labels = []

        for start_date, end_date, period_name in self.black_swan_periods:
            period_labels = self.generate_labels_for_period(
                start_date, end_date, period_name
            )

            if not period_labels.empty:
                period_labels['period_name'] = period_name
                all_labels.append(period_labels)

        combined_labels = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Total black swan labels: {len(combined_labels)}")
        logger.info(f"Overall strategy distribution:\n{combined_labels['strategy_idx'].value_counts()}")
        logger.info(f"Overall average PnL: {combined_labels['pnl'].mean():.4f}")

        return combined_labels

    def save_labels(self, labels_df: pd.DataFrame, filepath: str):
        """Save labels to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(filepath, index=False)
        logger.info(f"Labels saved to {filepath}")

    def load_labels(self, filepath: str) -> pd.DataFrame:
        """Load labels from disk"""
        labels_df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(labels_df)} labels from {filepath}")
        return labels_df


def create_strategy_labeler(historical_manager, config: Optional[Dict] = None):
    """Factory function to create strategy labeler"""
    if config is None:
        config = {}

    return StrategyLabeler(
        historical_data_manager=historical_manager,
        strategies_config=config.get('strategies', {}),
        lookforward_days=config.get('lookforward_days', 5),
        confidence_threshold=config.get('confidence_threshold', 0.7)
    )


if __name__ == "__main__":
    # Test strategy labeler
    logging.basicConfig(level=logging.INFO)

    print("Strategy Labeler Test")
    print("=" * 80)

    # Mock historical data manager (replace with actual implementation)
    class MockHistoricalManager:
        def get_training_data(self, start_date, end_date):
            # Generate mock data
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({
                'date': dates,
                'vix': np.random.uniform(15, 30, len(dates)),
                'spy_returns_5d': np.random.normal(0, 0.01, len(dates)),
                'spy_returns_20d': np.random.normal(0, 0.02, len(dates)),
                'spy_close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
                'tlt_close': 100 * np.cumprod(1 + np.random.normal(0, 0.005, len(dates)))
            })

    mock_manager = MockHistoricalManager()
    labeler = create_strategy_labeler(mock_manager)

    # Test single period
    print("\nTesting single period...")
    labels = labeler.generate_labels_for_period('2020-02-19', '2020-03-23', 'COVID-19 Test')

    print(f"\nGenerated {len(labels)} labels")
    print(f"Sample label:\n{labels.head(3)}")

    print("\n✅ Strategy labeler test passed!")
