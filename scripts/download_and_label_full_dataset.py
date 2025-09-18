"""
Download and Label Full 30-Year Dataset for Black Swan Hunting
Downloads 1995-2024 data and labels all black swan events
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.data.historical_data_manager import HistoricalDataManager
from src.data.black_swan_labeler import BlackSwanLabeler

# Extended list of symbols for comprehensive market coverage
SYMBOLS = [
    # Major Indices
    'SPY', 'QQQ', 'DIA', 'IWM', 'MDY',

    # Sectors
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLB', 'XLP', 'XLY', 'XLRE',

    # Volatility
    '^VIX', 'VIXY', 'VXX', 'UVXY',

    # Bonds
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'AGG',

    # Commodities
    'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC',

    # Currencies
    'UUP', 'FXE', 'FXY', 'FXB',

    # International
    'EEM', 'EFA', 'FXI', 'EWJ', 'EWZ', 'RSX',

    # Safe Haven Assets
    'TIP', 'MINT', 'SHV',

    # Individual Stocks (Mega Cap)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'MA', 'HD',

    # Meme Stocks (for recent events)
    'GME', 'AMC', 'BB', 'NOK',

    # Crypto Proxies
    'MSTR', 'RIOT', 'MARA', 'SQ', 'PYPL',

    # Banks (for 2008 crisis)
    'BAC', 'C', 'WFC', 'GS', 'MS',

    # Tech Bubble Stocks
    'CSCO', 'INTC', 'ORCL', 'IBM'
]

def download_full_dataset():
    """Download complete 30-year dataset"""
    logger.info("=" * 60)
    logger.info("Downloading Full 30-Year Dataset (1995-2024)")
    logger.info("=" * 60)

    manager = HistoricalDataManager()

    # Download in yearly chunks to avoid timeouts
    start_year = 1995
    end_year = 2024

    all_data = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        logger.info(f"\nDownloading data for {year}...")

        try:
            # Download data for this year
            success = manager.download_historical_data(
                symbols=SYMBOLS,
                start_date=start_date,
                end_date=end_date
            )

            if success:
                # Get the downloaded data
                df = manager.get_training_data(
                    start_date=start_date,
                    end_date=end_date,
                    symbols=SYMBOLS
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Downloaded {len(df)} records for {year}")
                else:
                    logger.warning(f"No data retrieved for {year}")

        except Exception as e:
            logger.error(f"Error downloading {year}: {e}")
            continue

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\nTotal records downloaded: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"Unique symbols: {combined_df['symbol'].nunique()}")

        return combined_df
    else:
        logger.error("No data downloaded")
        return pd.DataFrame()

def label_black_swan_events(df):
    """Label the dataset with black swan events"""
    logger.info("\n" + "=" * 60)
    logger.info("Labeling Black Swan Events")
    logger.info("=" * 60)

    if df.empty:
        logger.error("No data to label")
        return df

    labeler = BlackSwanLabeler()

    # Label events for each symbol
    labeled_data = []
    symbols = df['symbol'].unique()

    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()

        if len(symbol_df) > 20:  # Need enough data for calculations
            try:
                labeled_df = labeler.label_tail_events(symbol_df)
                labeled_data.append(labeled_df)

                # Report black swans for this symbol
                black_swans = labeled_df['is_black_swan'].sum()
                if black_swans > 0:
                    logger.info(f"{symbol}: Found {black_swans} black swan events")

                    # Show dates of major events
                    swan_dates = labeled_df[labeled_df['is_black_swan']]['date'].tolist()
                    for date in swan_dates[:5]:  # Show first 5
                        event_row = labeled_df[labeled_df['date'] == date].iloc[0]
                        logger.info(f"  {date}: {event_row['returns']:.2%} return, "
                                  f"Z-score: {event_row.get('z_score', 0):.1f}")

            except Exception as e:
                logger.warning(f"Error labeling {symbol}: {e}")
                continue

    if labeled_data:
        combined_labeled = pd.concat(labeled_data, ignore_index=True)

        # Overall statistics
        total_observations = len(combined_labeled)
        total_black_swans = combined_labeled['is_black_swan'].sum()
        black_swan_rate = total_black_swans / total_observations if total_observations > 0 else 0

        logger.info("\n" + "=" * 60)
        logger.info("Labeling Complete")
        logger.info(f"Total observations: {total_observations:,}")
        logger.info(f"Black swan events: {total_black_swans:,}")
        logger.info(f"Black swan frequency: {black_swan_rate:.2%}")

        # Calculate convexity statistics
        if 'convexity_score' in combined_labeled.columns:
            avg_convexity = combined_labeled[combined_labeled['is_black_swan']]['convexity_score'].mean()
            logger.info(f"Average convexity during black swans: {avg_convexity:.2f}")

        return combined_labeled
    else:
        logger.error("No data labeled")
        return pd.DataFrame()

def save_labeled_dataset(df):
    """Save the labeled dataset to database and CSV"""
    logger.info("\n" + "=" * 60)
    logger.info("Saving Labeled Dataset")
    logger.info("=" * 60)

    if df.empty:
        logger.error("No data to save")
        return

    # Save to database
    import sqlite3
    db_path = Path("data/black_swan_training.db")

    with sqlite3.connect(db_path) as conn:
        # Save to a new table for labeled data
        df.to_sql('labeled_market_data', conn, if_exists='replace', index=False)

        # Create index for performance
        cursor = conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_labeled_date ON labeled_market_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_labeled_symbol ON labeled_market_data(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_labeled_black_swan ON labeled_market_data(is_black_swan)')

        conn.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM labeled_market_data")
        count = cursor.fetchone()[0]
        logger.info(f"Saved {count} records to database")

        # Count black swan events
        cursor.execute("SELECT COUNT(*) FROM labeled_market_data WHERE is_black_swan = 1")
        swan_count = cursor.fetchone()[0]
        logger.info(f"Black swan events in database: {swan_count}")

    # Also save to CSV for easy analysis
    csv_path = Path("data/labeled_black_swan_dataset.csv")
    csv_path.parent.mkdir(exist_ok=True)

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset to {csv_path}")

    # Save black swan events separately
    black_swans = df[df['is_black_swan'] == True]
    if not black_swans.empty:
        swan_csv = Path("data/black_swan_events_only.csv")
        black_swans.to_csv(swan_csv, index=False)
        logger.info(f"Saved {len(black_swans)} black swan events to {swan_csv}")

def analyze_dataset_quality(df):
    """Analyze the quality and coverage of the dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Quality Analysis")
    logger.info("=" * 60)

    if df.empty:
        logger.error("No data to analyze")
        return

    # Date coverage
    df['date'] = pd.to_datetime(df['date'])
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    logger.info(f"Date range: {date_range}")

    # Symbol coverage
    symbols = df['symbol'].unique()
    logger.info(f"Symbols covered: {len(symbols)}")

    # Data completeness by year
    df['year'] = df['date'].dt.year
    yearly_counts = df.groupby('year').size()
    logger.info("\nData points by year:")
    for year, count in yearly_counts.items():
        logger.info(f"  {year}: {count:,} records")

    # Missing data analysis
    missing_cols = df.isnull().sum()
    if missing_cols.sum() > 0:
        logger.info("\nMissing data:")
        for col, missing in missing_cols[missing_cols > 0].items():
            pct = (missing / len(df)) * 100
            logger.info(f"  {col}: {missing:,} ({pct:.1f}%)")

    # Black swan event distribution
    if 'is_black_swan' in df.columns:
        swan_by_year = df[df['is_black_swan'] == True].groupby('year').size()
        logger.info("\nBlack swan events by year:")
        for year, count in swan_by_year.items():
            logger.info(f"  {year}: {count} events")

    # Return distribution
    if 'returns' in df.columns:
        returns = df['returns'].dropna()
        logger.info("\nReturn distribution:")
        logger.info(f"  Mean: {returns.mean():.4f}")
        logger.info(f"  Std: {returns.std():.4f}")
        logger.info(f"  Skew: {returns.skew():.2f}")
        logger.info(f"  Kurtosis: {returns.kurtosis():.2f}")
        logger.info(f"  Min: {returns.min():.2%}")
        logger.info(f"  Max: {returns.max():.2%}")

        # Tail events
        extreme_negative = (returns < -0.05).sum()
        extreme_positive = (returns > 0.05).sum()
        logger.info(f"\nExtreme events (>5% move):")
        logger.info(f"  Negative: {extreme_negative:,}")
        logger.info(f"  Positive: {extreme_positive:,}")

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("BLACK SWAN DATASET CREATION PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 1: Download full dataset
        logger.info("\nStep 1: Downloading full 30-year dataset...")
        df = download_full_dataset()

        if df.empty:
            logger.error("Failed to download dataset")
            return False

        # Step 2: Label black swan events
        logger.info("\nStep 2: Labeling black swan events...")
        labeled_df = label_black_swan_events(df)

        if labeled_df.empty:
            logger.error("Failed to label dataset")
            return False

        # Step 3: Save labeled dataset
        logger.info("\nStep 3: Saving labeled dataset...")
        save_labeled_dataset(labeled_df)

        # Step 4: Analyze dataset quality
        logger.info("\nStep 4: Analyzing dataset quality...")
        analyze_dataset_quality(labeled_df)

        logger.info("\n" + "=" * 60)
        logger.info("DATASET CREATION COMPLETE!")
        logger.info("=" * 60)
        logger.info("\nDataset is ready for training the Black Swan Hunting AI")
        logger.info("Files created:")
        logger.info("  - data/black_swan_training.db (labeled_market_data table)")
        logger.info("  - data/labeled_black_swan_dataset.csv")
        logger.info("  - data/black_swan_events_only.csv")

        return True

    except Exception as e:
        logger.error(f"Fatal error in dataset creation: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)