"""
Initialize Black Swan Database
Creates and populates the database with initial historical data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_database():
    """Create the black swan training database with all required tables"""

    db_path = Path("data/black_swan_training.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating database at: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                returns REAL,
                volatility_20d REAL,
                vix_level REAL,
                UNIQUE(date, symbol)
            )
        ''')

        # Black swan events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS black_swan_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                event_name TEXT NOT NULL,
                severity REAL,
                affected_sectors TEXT,
                recovery_days INTEGER,
                vix_peak REAL,
                description TEXT
            )
        ''')

        # Strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                market_regime TEXT,
                returns REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                black_swan_capture BOOLEAN,
                convexity_ratio REAL,
                metadata TEXT
            )
        ''')

        # Strategy weights table for optimization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                weight REAL,
                performance_30d REAL,
                performance_90d REAL,
                black_swan_success_rate REAL,
                optimization_metric REAL
            )
        ''')

        # Training data table for LLM fine-tuning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                market_state TEXT NOT NULL,
                selected_strategy TEXT NOT NULL,
                outcome REAL,
                reward REAL,
                convexity_achieved REAL,
                metadata TEXT
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_data(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_date_symbol ON market_data(date, symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_date ON strategy_performance(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_name ON strategy_performance(strategy_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_date ON training_examples(date)')

        conn.commit()

        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        logger.info(f"Created {len(tables)} tables: {[t[0] for t in tables]}")

        # Verify indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        logger.info(f"Created {len(indexes)} indexes")

        return db_path

def populate_initial_black_swan_events():
    """Populate the database with known black swan events"""

    db_path = Path("data/black_swan_training.db")

    events = [
        ("1997-10-27", "Asian Financial Crisis", -0.07, "Finance,Technology", 45, 48.0,
         "Currency crisis spreads from Thailand to global markets"),
        ("1998-08-31", "LTCM Collapse", -0.19, "Finance,All", 60, 45.7,
         "Long-Term Capital Management hedge fund collapse threatens system"),
        ("2000-03-10", "Dotcom Bubble Peak", -0.78, "Technology", 900, 37.5,
         "Tech bubble bursts, NASDAQ loses 78% over 2.5 years"),
        ("2001-09-17", "9/11 Market Reopening", -0.12, "All", 45, 49.4,
         "Markets reopen after terrorist attacks with massive selloff"),
        ("2008-09-15", "Lehman Brothers Collapse", -0.54, "Finance,Real Estate", 400, 89.5,
         "Global Financial Crisis reaches peak with Lehman bankruptcy"),
        ("2010-05-06", "Flash Crash", -0.09, "All", 1, 45.8,
         "Dow drops 1000 points in minutes due to algorithmic trading"),
        ("2011-08-08", "US Credit Downgrade", -0.17, "Finance,All", 60, 48.0,
         "S&P downgrades US credit rating from AAA"),
        ("2015-08-24", "China Devaluation", -0.11, "All", 45, 53.3,
         "China devalues yuan triggering global selloff"),
        ("2018-02-05", "Volmageddon", -0.10, "Finance", 30, 50.3,
         "XIV ETN collapses, VIX spikes destroying short volatility trades"),
        ("2020-03-23", "COVID-19 Crash", -0.34, "All", 120, 85.5,
         "Pandemic lockdowns trigger fastest bear market in history"),
        ("2021-01-27", "GameStop Squeeze", 0.50, "Consumer,Finance", 30, 37.2,
         "Retail traders cause massive short squeeze in meme stocks"),
        ("2022-09-23", "UK Gilt Crisis", -0.08, "Finance", 20, 31.6,
         "UK pension fund margin calls threaten financial stability"),
        ("2023-03-10", "SVB Collapse", -0.05, "Finance,Technology", 10, 29.8,
         "Silicon Valley Bank fails causing regional banking crisis")
    ]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for event in events:
            cursor.execute('''
                INSERT OR REPLACE INTO black_swan_events
                (date, event_name, severity, affected_sectors, recovery_days, vix_peak, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', event)

        conn.commit()

        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM black_swan_events")
        count = cursor.fetchone()[0]
        logger.info(f"Populated {count} black swan events")

def download_initial_market_data():
    """Download initial batch of market data for key symbols"""

    from src.data.historical_data_manager import HistoricalDataManager

    # Key symbols for initial testing
    symbols = [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'IWM',   # Russell 2000
        'DIA',   # Dow Jones
        'VIX',   # Volatility Index (if available)
        'TLT',   # 20+ Year Treasury
        'GLD',   # Gold
        'SLV',   # Silver
        'USO',   # Oil
        'UUP',   # US Dollar
        'EEM',   # Emerging Markets
        'HYG',   # High Yield Bonds
        'XLF',   # Financials
        'XLK',   # Technology
        'XLE',   # Energy
    ]

    manager = HistoricalDataManager()

    logger.info(f"Downloading historical data for {len(symbols)} symbols...")
    logger.info("This may take a few minutes...")

    # Download last 5 years for quick start (can expand later)
    start_date = "2019-01-01"

    success = manager.download_historical_data(
        symbols=symbols,
        start_date=start_date
    )

    if success:
        logger.info("Initial data download complete")

        # Get some statistics
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as records FROM market_data")
            stats = cursor.fetchone()
            logger.info(f"Database contains {stats[0]} symbols with {stats[1]} total records")
    else:
        logger.warning("Some symbols failed to download")

    return success

def verify_database():
    """Verify database integrity and contents"""

    db_path = Path("data/black_swan_training.db")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        expected_tables = [
            'market_data',
            'black_swan_events',
            'strategy_performance',
            'strategy_weights',
            'training_examples'
        ]

        for expected in expected_tables:
            if expected not in [t[0] for t in tables]:
                logger.error(f"Missing table: {expected}")
                return False

        logger.info("All required tables present")

        # Check data
        cursor.execute("SELECT COUNT(*) FROM black_swan_events")
        event_count = cursor.fetchone()[0]
        logger.info(f"Black swan events: {event_count}")

        cursor.execute("SELECT COUNT(*) FROM market_data")
        market_count = cursor.fetchone()[0]
        logger.info(f"Market data records: {market_count}")

        if event_count == 0:
            logger.warning("No black swan events in database")

        if market_count == 0:
            logger.warning("No market data in database")

        return True

def main():
    """Main initialization function"""

    logger.info("=" * 60)
    logger.info("Black Swan Database Initialization")
    logger.info("=" * 60)

    try:
        # Step 1: Create database
        logger.info("\nStep 1: Creating database...")
        db_path = create_database()
        logger.info(f"Database created at: {db_path}")

        # Step 2: Populate black swan events
        logger.info("\nStep 2: Populating black swan events...")
        populate_initial_black_swan_events()

        # Step 3: Download initial market data
        logger.info("\nStep 3: Downloading initial market data...")
        logger.info("Note: This will download 5 years of data for 15 key symbols")
        logger.info("You can expand this later by running the historical_data_manager")

        success = download_initial_market_data()

        # Step 4: Verify
        logger.info("\nStep 4: Verifying database...")
        if verify_database():
            logger.info("\n" + "=" * 60)
            logger.info("Database initialization SUCCESSFUL!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Run scripts/training/train_black_swan_ai.py to train the AI")
            logger.info("2. Install Ollama and pull Mistral model")
            logger.info("3. Test strategies with backtesting framework")
        else:
            logger.error("Database verification failed")
            return False

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)