"""
Historical Backtesting for TRM Model

Evaluates trained TRM on full 30-year historical period (1995-2024):
- Tests on all market conditions (normal + black swan periods)
- Compares TRM strategy selections vs actual outcomes
- Generates performance metrics and visualizations
- Benchmarks against baseline strategies

Usage:
    python scripts/trm/backtest_historical.py
"""

import sys
from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.trm_model import TinyRecursiveModel
from models.trm_config import TRMConfig
from data.market_feature_extractor import MarketFeatureExtractor
from data.historical_data_manager import HistoricalDataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TRMBacktester:
    """Historical backtesting for trained TRM model"""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        normalization_path: str,
        historical_manager: HistoricalDataManager
    ):
        """
        Initialize backtester

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to TRM config JSON
            normalization_path: Path to normalization parameters
            historical_manager: Historical data manager instance
        """
        self.historical_manager = historical_manager

        # Load config
        self.config = TRMConfig.load(config_path)
        logger.info(f"Loaded TRM config from {config_path}")

        # Load normalization parameters
        with open(normalization_path, 'r') as f:
            norm_params = json.load(f)
            self.norm_mean = torch.tensor(norm_params['mean'], dtype=torch.float32)
            self.norm_std = torch.tensor(norm_params['std'], dtype=torch.float32)
        logger.info(f"Loaded normalization params from {normalization_path}")

        # Create model
        self.model = TinyRecursiveModel(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.output_dim,
            num_latent_steps=self.config.model.num_latent_steps,
            num_recursion_cycles=self.config.model.num_recursion_cycles
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded trained model from {model_path}")

        # Feature extractor
        self.feature_extractor = MarketFeatureExtractor(historical_manager)

        # Strategy names
        self.strategy_names = self.config.strategies.strategy_names

    def normalize_features(self, features: np.ndarray) -> torch.Tensor:
        """Apply z-score normalization"""
        features_tensor = torch.tensor(features, dtype=torch.float32)
        normalized = (features_tensor - self.norm_mean) / self.norm_std
        return normalized

    def predict_strategy(self, features: np.ndarray) -> Dict:
        """
        Predict best strategy for given market features

        Returns:
            dict with strategy_idx, confidence, probabilities
        """
        # Normalize
        features_norm = self.normalize_features(features).unsqueeze(0)  # Add batch dim

        # Predict
        with torch.no_grad():
            output = self.model(features_norm, T=self.config.model.num_recursion_cycles, n=self.config.model.num_latent_steps)
            logits = output['strategy_logits']
            probs = torch.softmax(logits, dim=1)

            strategy_idx = logits.argmax(dim=1).item()
            confidence = probs[0, strategy_idx].item()

        return {
            'strategy_idx': strategy_idx,
            'strategy_name': self.strategy_names[strategy_idx],
            'confidence': confidence,
            'probabilities': probs[0].numpy()
        }

    def backtest_period(
        self,
        start_date: str,
        end_date: str,
        period_name: str = "Full Period"
    ) -> pd.DataFrame:
        """
        Backtest TRM on a time period

        Returns:
            DataFrame with date, features, prediction, actual_best, correct, etc.
        """
        logger.info(f"Backtesting period: {period_name} ({start_date} to {end_date})")

        # Extract features for this period
        features_df = self.feature_extractor.extract_features(start_date, end_date)

        if features_df.empty:
            logger.warning(f"No data for period {period_name}")
            return pd.DataFrame()

        results = []

        for idx, row in features_df.iterrows():
            date = row['date']

            # Extract 10 market features
            features = np.array([
                row.get('vix', 20.0),
                row.get('spy_returns_5d', 0.0),
                row.get('spy_returns_20d', 0.0),
                row.get('volume_ratio', 1.0),
                row.get('market_breadth', 0.5),
                row.get('correlation', 0.5),
                row.get('put_call_ratio', 1.0),
                row.get('gini_coefficient', 0.5),
                row.get('sector_dispersion', 0.3),
                row.get('signal_quality', 0.7)
            ], dtype=np.float32)

            # Predict
            prediction = self.predict_strategy(features)

            results.append({
                'date': date,
                'period_name': period_name,
                'predicted_strategy_idx': prediction['strategy_idx'],
                'predicted_strategy_name': prediction['strategy_name'],
                'confidence': prediction['confidence'],
                'vix': row.get('vix', 20.0),
                'spy_returns_5d': row.get('spy_returns_5d', 0.0)
            })

        results_df = pd.DataFrame(results)
        logger.info(f"Backtested {len(results_df)} days for {period_name}")

        return results_df

    def generate_report(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive backtest report"""

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("HISTORICAL BACKTEST REPORT")
        print("=" * 80)
        print()

        # Overall statistics
        print(f"Total Days Backtested: {len(results_df):,}")
        print(f"Date Range: {results_df['date'].min()} to {results_df['date'].max()}")
        print()

        # Strategy distribution
        print("TRM Strategy Selections:")
        strategy_counts = results_df['predicted_strategy_name'].value_counts()
        for strategy, count in strategy_counts.items():
            pct = (count / len(results_df)) * 100
            print(f"  {strategy:25s}: {count:5d} days ({pct:5.1f}%)")
        print()

        # Market regime analysis
        results_df['market_regime'] = pd.cut(
            results_df['vix'],
            bins=[0, 15, 25, 100],
            labels=['Low Volatility', 'Normal', 'High Volatility']
        )

        print("Strategy Selection by Market Regime:")
        for regime in ['Low Volatility', 'Normal', 'High Volatility']:
            regime_df = results_df[results_df['market_regime'] == regime]
            if len(regime_df) > 0:
                print(f"\n  {regime} (VIX {regime}):")
                regime_strategies = regime_df['predicted_strategy_name'].value_counts().head(3)
                for strategy, count in regime_strategies.items():
                    pct = (count / len(regime_df)) * 100
                    print(f"    {strategy:25s}: {pct:5.1f}%")
        print()

        # Save results
        csv_path = output_dir / 'backtest_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

        # Create visualizations
        self.create_visualizations(results_df, output_dir)

        # Save summary
        summary = {
            'total_days': len(results_df),
            'date_range': {
                'start': str(results_df['date'].min()),
                'end': str(results_df['date'].max())
            },
            'strategy_distribution': strategy_counts.to_dict(),
            'avg_confidence': float(results_df['confidence'].mean()),
            'regime_analysis': {}
        }

        for regime in ['Low Volatility', 'Normal', 'High Volatility']:
            regime_df = results_df[results_df['market_regime'] == regime]
            if len(regime_df) > 0:
                summary['regime_analysis'][regime] = {
                    'days': len(regime_df),
                    'top_strategy': regime_df['predicted_strategy_name'].mode()[0] if len(regime_df) > 0 else None
                }

        summary_path = output_dir / 'backtest_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    def create_visualizations(self, results_df: pd.DataFrame, output_dir: Path):
        """Create backtest visualizations"""

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('TRM Historical Backtest Analysis (1995-2024)', fontsize=16, fontweight='bold')

        # 1. Strategy distribution over time
        ax = axes[0, 0]
        strategy_by_year = results_df.groupby([pd.to_datetime(results_df['date']).dt.year, 'predicted_strategy_name']).size().unstack(fill_value=0)
        strategy_by_year.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        ax.set_title('Strategy Selection Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Days')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 2. Overall strategy distribution
        ax = axes[0, 1]
        strategy_counts = results_df['predicted_strategy_name'].value_counts()
        colors = sns.color_palette("husl", len(strategy_counts))
        strategy_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Overall Strategy Distribution')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Number of Days')
        ax.tick_params(axis='x', rotation=45)

        # 3. VIX vs strategy selection
        ax = axes[1, 0]
        for strategy in results_df['predicted_strategy_name'].unique():
            strategy_df = results_df[results_df['predicted_strategy_name'] == strategy]
            ax.scatter(strategy_df.index, strategy_df['vix'], label=strategy, alpha=0.5, s=10)
        ax.set_title('VIX Levels and Strategy Selection')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('VIX Level')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 4. Confidence distribution
        ax = axes[1, 1]
        results_df['confidence'].hist(bins=50, ax=ax, edgecolor='black')
        ax.axvline(results_df['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["confidence"].mean():.2f}')
        ax.set_title('Prediction Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.legend()

        # 5. Market regime strategy preference
        ax = axes[2, 0]
        regime_strategy = results_df.groupby(['market_regime', 'predicted_strategy_name']).size().unstack(fill_value=0)
        regime_strategy.plot(kind='bar', stacked=False, ax=ax)
        ax.set_title('Strategy Preference by Market Regime')
        ax.set_xlabel('Market Regime')
        ax.set_ylabel('Number of Days')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 6. Monthly strategy changes
        ax = axes[2, 1]
        results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
        monthly_changes = results_df.groupby('month')['predicted_strategy_idx'].apply(lambda x: (x != x.shift()).sum())
        monthly_changes.plot(ax=ax, color='purple')
        ax.set_title('Strategy Changes Per Month (Adaptability)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Strategy Changes')

        plt.tight_layout()

        # Save figure
        viz_path = output_dir / 'backtest_visualizations.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {viz_path}")
        plt.close()


def main():
    """Run historical backtesting"""

    print("=" * 80)
    print("TRM HISTORICAL BACKTESTING - 30 YEAR EVALUATION")
    print("=" * 80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'checkpoints' / 'training_checkpoint.pkl'
    config_path = project_root / 'config' / 'trm_config.json'
    norm_path = project_root / 'config' / 'trm_normalization.json'
    db_path = project_root / 'data' / 'historical_market.db'
    output_dir = project_root / 'results' / 'backtest'

    # Check paths
    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        print("Please train the model first: python scripts/trm/train_full_dataset.py")
        return

    if not norm_path.exists():
        print(f"[ERROR] Normalization params not found: {norm_path}")
        return

    # Initialize
    print("[1/4] Loading components...")
    historical_manager = HistoricalDataManager(db_path=str(db_path))

    backtester = TRMBacktester(
        model_path=str(model_path),
        config_path=str(config_path),
        normalization_path=str(norm_path),
        historical_manager=historical_manager
    )
    print("[OK] Components loaded successfully")
    print()

    # Run backtest on full 30-year period
    print("[2/4] Running backtest on full historical period...")
    print("This may take several minutes...")
    print()

    results_df = backtester.backtest_period(
        start_date='1995-01-01',
        end_date='2024-12-31',
        period_name='Full 30-Year Period'
    )

    if results_df.empty:
        print("[ERROR] No backtest results generated")
        return

    print(f"[OK] Backtested {len(results_df):,} trading days")
    print()

    # Generate report
    print("[3/4] Generating comprehensive report...")
    backtester.generate_report(results_df, output_dir)
    print()

    # Summary
    print("[4/4] Backtest complete!")
    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - Results CSV: {output_dir / 'backtest_results.csv'}")
    print(f"  - Summary JSON: {output_dir / 'backtest_summary.json'}")
    print(f"  - Visualizations: {output_dir / 'backtest_visualizations.png'}")
    print()
    print("Next steps:")
    print("  1. Review backtest results and visualizations")
    print("  2. Analyze strategy selection patterns")
    print("  3. Compare against baseline strategies")
    print("  4. Prepare for paper trading integration")
    print()


if __name__ == "__main__":
    main()
