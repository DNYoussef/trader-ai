"""
Generate TRM Training Labels for Black Swan Periods

This script:
1. Initializes the historical data manager
2. Creates the strategy labeler
3. Generates labels for 12 black swan periods
4. Saves labels to disk for TRM training

Usage:
    python scripts/trm/generate_black_swan_labels.py
"""

import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.historical_data_manager import HistoricalDataManager
from data.strategy_labeler import StrategyLabeler
from models.trm_config import TRMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate training labels for TRM"""

    logger.info("=" * 80)
    logger.info("TRM Training Label Generation for Black Swan Periods")
    logger.info("=" * 80)

    # Load TRM configuration
    logger.info("\n[1/5] Loading TRM configuration...")
    config_path = Path(__file__).parent.parent.parent / 'config' / 'trm_config.json'

    if config_path.exists():
        config = TRMConfig.load(str(config_path))
        logger.info(f"Configuration loaded from {config_path}")
    else:
        logger.info("Using default configuration")
        config = TRMConfig()

    # Initialize historical data manager
    logger.info("\n[2/5] Initializing historical data manager...")
    db_path = Path(__file__).parent.parent.parent / 'data' / 'historical_market.db'
    historical_manager = HistoricalDataManager(db_path=str(db_path))
    logger.info(f"Historical data manager initialized")

    # Create strategy labeler
    logger.info("\n[3/5] Creating strategy labeler...")
    labeler = StrategyLabeler(
        historical_data_manager=historical_manager,
        strategies_config=config.strategies.to_dict(),
        lookforward_days=5,
        confidence_threshold=0.7
    )
    logger.info(f"Strategy labeler created with {len(labeler.strategy_names)} strategies")

    # Generate labels for black swan periods
    logger.info("\n[4/5] Generating labels for 12 black swan periods...")
    logger.info("This may take several minutes depending on data availability...")

    try:
        black_swan_labels = labeler.generate_black_swan_labels()

        logger.info("\n" + "=" * 80)
        logger.info("BLACK SWAN LABEL GENERATION SUMMARY")
        logger.info("=" * 80)

        if len(black_swan_labels) > 0:
            logger.info(f"Total labels generated: {len(black_swan_labels):,}")
            logger.info(f"\nStrategy distribution:")
            for strategy_idx, count in sorted(black_swan_labels['strategy_idx'].value_counts().items()):
                strategy_name = labeler.strategy_names[strategy_idx]
                pct = (count / len(black_swan_labels)) * 100
                logger.info(f"  {strategy_idx} ({strategy_name}): {count:,} labels ({pct:.1f}%)")

            logger.info(f"\nPnL statistics:")
            logger.info(f"  Mean PnL: {black_swan_labels['pnl'].mean():.4f}")
            logger.info(f"  Median PnL: {black_swan_labels['pnl'].median():.4f}")
            logger.info(f"  Std Dev: {black_swan_labels['pnl'].std():.4f}")
            logger.info(f"  Min PnL: {black_swan_labels['pnl'].min():.4f}")
            logger.info(f"  Max PnL: {black_swan_labels['pnl'].max():.4f}")

            logger.info(f"\nPeriod breakdown:")
            for period_name, period_df in black_swan_labels.groupby('period_name'):
                logger.info(f"  {period_name}: {len(period_df):,} labels")

        else:
            logger.warning("No labels generated! Check historical data availability.")
            logger.warning("You may need to populate the historical database first.")
            return

        # Save labels to disk
        logger.info("\n[5/5] Saving labels to disk...")
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'trm_training'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet (efficient storage)
        parquet_path = output_dir / 'black_swan_labels.parquet'
        labeler.save_labels(black_swan_labels, str(parquet_path))
        logger.info(f"Labels saved to {parquet_path}")

        # Also save as CSV for easy inspection
        csv_path = output_dir / 'black_swan_labels.csv'
        # Expand features list into columns
        features_df = pd.DataFrame(
            black_swan_labels['features'].tolist(),
            columns=config.features.feature_names
        )
        export_df = pd.concat([
            black_swan_labels[['date', 'strategy_idx', 'pnl', 'period_name']],
            features_df
        ], axis=1)
        export_df.to_csv(csv_path, index=False)
        logger.info(f"CSV export saved to {csv_path}")

        # Save summary statistics
        summary_path = output_dir / 'label_generation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("TRM BLACK SWAN LABEL GENERATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total labels: {len(black_swan_labels):,}\n")
            f.write(f"Date range: {black_swan_labels['date'].min()} to {black_swan_labels['date'].max()}\n\n")

            f.write("Strategy Distribution:\n")
            for strategy_idx, count in sorted(black_swan_labels['strategy_idx'].value_counts().items()):
                strategy_name = labeler.strategy_names[strategy_idx]
                pct = (count / len(black_swan_labels)) * 100
                f.write(f"  {strategy_idx} ({strategy_name}): {count:,} ({pct:.1f}%)\n")

            f.write(f"\nPnL Statistics:\n")
            f.write(f"  Mean: {black_swan_labels['pnl'].mean():.4f}\n")
            f.write(f"  Median: {black_swan_labels['pnl'].median():.4f}\n")
            f.write(f"  Std Dev: {black_swan_labels['pnl'].std():.4f}\n")
            f.write(f"  Min: {black_swan_labels['pnl'].min():.4f}\n")
            f.write(f"  Max: {black_swan_labels['pnl'].max():.4f}\n")

        logger.info(f"Summary saved to {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("LABEL GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review labels in: {csv_path}")
        logger.info(f"  2. Check summary: {summary_path}")
        logger.info(f"  3. Proceed to Phase 2: TRM training pipeline")
        logger.info(f"\nReady for TRM training!")

    except Exception as e:
        logger.error(f"Error generating labels: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check that historical database exists and has data")
        logger.error("  2. Verify market data for 1997-2024 period")
        logger.error("  3. Run: python src/data/historical_data_manager.py --populate")
        return


if __name__ == "__main__":
    main()
