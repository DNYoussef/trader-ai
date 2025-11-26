"""
Historical Data Manager for Black Swan Hunting AI System
Manages 30 years of historical market data (1995-2024) with chunked loading
to prevent memory overflow and enable efficient backtesting.
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BlackSwanEvent:
    """Represents a historical black swan event for training"""
    date: str
    event_name: str
    severity: float  # Negative for crashes, positive for melt-ups
    recovery_days: int
    affected_sectors: List[str]
    vix_peak: float
    description: str

@dataclass
class MarketRegime:
    """Market regime classification for context"""
    start_date: str
    end_date: str
    regime_type: str  # 'bull', 'bear', 'volatile', 'calm'
    avg_volatility: float
    avg_correlation: float
    dominant_factor: str

class HistoricalDataManager:
    """
    Manages 30 years of historical data (1995-2024)
    Uses chunked loading to prevent memory overflow
    Implements caching and efficient data retrieval
    """

    def __init__(self, db_path: str = None, cache_dir: str = None):
        """
        Initialize the historical data manager

        Args:
            db_path: Path to SQLite database
            cache_dir: Directory for cached data files
        """
        self.db_path = Path(db_path or "data/historical_market.db")
        self.cache_dir = Path(cache_dir or "data/cache/historical")

        # Create directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Chunk parameters for memory management
        self.chunk_size = 100  # Process 100 symbols at a time
        self.date_chunk_years = 5  # Process 5 years at a time

        # S&P 500 sectors for analysis
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'CVS', 'ABBV'],
            'Consumer': ['AMZN', 'TSLA', 'WMT', 'HD', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON'],
            'Real Estate': ['PLD', 'AMT', 'CCI', 'SPG', 'PSA'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM']
        }

        logger.info(f"HistoricalDataManager initialized - DB: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Main market data table
            conn.execute('''
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
                    log_returns REAL,
                    volatility_20d REAL,
                    volatility_60d REAL,
                    rsi_14 REAL,
                    ma_50 REAL,
                    ma_200 REAL,
                    volume_ratio REAL,
                    UNIQUE(date, symbol)
                )
            ''')

            # Black swan events table
            conn.execute('''
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

            # Market regimes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    regime_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    avg_volatility REAL,
                    avg_correlation REAL,
                    dominant_factor TEXT
                )
            ''')

            # Strategy performance tracking
            conn.execute('''
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

            # Create indexes for faster queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_market_data_date_symbol ON market_data(date, symbol)')

            conn.commit()

    def get_black_swan_events(self) -> List[BlackSwanEvent]:
        """
        Returns labeled crisis events for training
        These are the major market dislocations from 1995-2024
        """
        return [
            BlackSwanEvent(
                date="1997-10-27",
                event_name="Asian Financial Crisis",
                severity=-0.07,  # Single day drop
                recovery_days=45,
                affected_sectors=["Finance", "Technology"],
                vix_peak=48.0,
                description="Mini-crash triggered by Asian currency crisis"
            ),
            BlackSwanEvent(
                date="1998-08-31",
                event_name="LTCM Collapse",
                severity=-0.19,  # Monthly drop
                recovery_days=60,
                affected_sectors=["Finance", "Technology"],
                vix_peak=45.7,
                description="Long-Term Capital Management hedge fund collapse"
            ),
            BlackSwanEvent(
                date="2000-03-10",
                event_name="Dotcom Bubble Peak",
                severity=-0.78,  # Total drawdown over 2.5 years
                recovery_days=900,
                affected_sectors=["Technology"],
                vix_peak=37.5,
                description="Tech bubble burst, NASDAQ down 78%"
            ),
            BlackSwanEvent(
                date="2001-09-17",
                event_name="9/11 Attacks",
                severity=-0.12,  # Week drop
                recovery_days=45,
                affected_sectors=["All"],
                vix_peak=49.4,
                description="Market reopening after terrorist attacks"
            ),
            BlackSwanEvent(
                date="2008-09-15",
                event_name="Lehman Brothers Collapse",
                severity=-0.54,  # Total GFC drawdown
                recovery_days=400,
                affected_sectors=["Finance", "Real Estate"],
                vix_peak=89.5,
                description="Global Financial Crisis peak"
            ),
            BlackSwanEvent(
                date="2010-05-06",
                event_name="Flash Crash",
                severity=-0.09,  # Intraday
                recovery_days=1,
                affected_sectors=["All"],
                vix_peak=45.8,
                description="1000 point intraday drop in minutes"
            ),
            BlackSwanEvent(
                date="2011-08-08",
                event_name="US Downgrade",
                severity=-0.17,  # Monthly
                recovery_days=60,
                affected_sectors=["Finance"],
                vix_peak=48.0,
                description="S&P downgrades US credit rating"
            ),
            BlackSwanEvent(
                date="2015-08-24",
                event_name="China Devaluation",
                severity=-0.11,  # Weekly
                recovery_days=45,
                affected_sectors=["All"],
                vix_peak=53.3,
                description="China yuan devaluation triggers selloff"
            ),
            BlackSwanEvent(
                date="2018-02-05",
                event_name="Volmageddon",
                severity=-0.10,  # Weekly
                recovery_days=30,
                affected_sectors=["Finance"],
                vix_peak=50.3,
                description="XIV implosion, volatility spike"
            ),
            BlackSwanEvent(
                date="2020-03-23",
                event_name="COVID-19 Crash",
                severity=-0.34,  # Monthly
                recovery_days=120,
                affected_sectors=["All"],
                vix_peak=85.5,
                description="Pandemic-induced fastest bear market"
            ),
            BlackSwanEvent(
                date="2021-01-27",
                event_name="GameStop Squeeze",
                severity=0.50,  # Positive black swan for GME
                recovery_days=30,
                affected_sectors=["Consumer", "Finance"],
                vix_peak=37.2,
                description="Retail-driven short squeeze"
            ),
            BlackSwanEvent(
                date="2022-09-23",
                event_name="UK Gilt Crisis",
                severity=-0.08,
                recovery_days=20,
                affected_sectors=["Finance"],
                vix_peak=31.6,
                description="UK pension fund crisis, pound collapse"
            )
        ]

    def download_historical_data(self,
                                symbols: List[str] = None,
                                start_date: str = "1995-01-01",
                                end_date: str = None) -> bool:
        """
        Download historical data in chunks and store in SQLite

        Args:
            symbols: List of symbols to download (default: S&P 500)
            start_date: Start date for historical data
            end_date: End date (default: today)

        Returns:
            bool: Success status
        """
        if symbols is None:
            # Get all symbols from sectors
            symbols = []
            for sector_symbols in self.sectors.values():
                symbols.extend(sector_symbols)

            # Add market indices and volatility
            symbols.extend(['SPY', 'QQQ', 'IWM', 'DIA', 'VIX', 'TLT', 'GLD'])

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")

        # Process in chunks to avoid memory issues
        failed_symbols = []

        for i in range(0, len(symbols), self.chunk_size):
            chunk = symbols[i:i + self.chunk_size]
            logger.info(f"Processing chunk {i//self.chunk_size + 1}/{(len(symbols)-1)//self.chunk_size + 1}")

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._download_symbol_data, symbol, start_date, end_date): symbol
                    for symbol in chunk
                }

                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            failed_symbols.append(symbol)
                    except Exception as e:
                        logger.error(f"Error downloading {symbol}: {e}")
                        failed_symbols.append(symbol)

            # Rate limiting
            time.sleep(1)

        if failed_symbols:
            logger.warning(f"Failed to download: {failed_symbols}")

        # Store black swan events
        self._store_black_swan_events()

        return len(failed_symbols) == 0

    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Download data for a single symbol

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            bool: Success status
        """
        try:
            # Download from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return False

            # Calculate technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility_20d'] = df['Returns'].rolling(20).std()
            df['Volatility_60d'] = df['Returns'].rolling(60).std()
            df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['MA_200'] = df['Close'].rolling(200).mean()
            df['VolumeRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            # Store in database
            self._store_symbol_data(symbol, df)

            return True

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return False

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _store_symbol_data(self, symbol: str, df: pd.DataFrame):
        """Store symbol data in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            for index, row in df.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO market_data
                        (date, symbol, open, high, low, close, volume, returns, log_returns,
                         volatility_20d, volatility_60d, rsi_14, ma_50, ma_200, volume_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        index.strftime('%Y-%m-%d'),
                        symbol,
                        row.get('Open'),
                        row.get('High'),
                        row.get('Low'),
                        row.get('Close'),
                        row.get('Volume'),
                        row.get('Returns'),
                        row.get('LogReturns'),
                        row.get('Volatility_20d'),
                        row.get('Volatility_60d'),
                        row.get('RSI_14'),
                        row.get('MA_50'),
                        row.get('MA_200'),
                        row.get('VolumeRatio')
                    ))
                except Exception as e:
                    logger.error(f"Error storing data for {symbol} on {index}: {e}")

            conn.commit()

    def _store_black_swan_events(self):
        """Store black swan events in database"""
        events = self.get_black_swan_events()

        with sqlite3.connect(self.db_path) as conn:
            for event in events:
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO black_swan_events
                        (date, event_name, severity, affected_sectors, recovery_days, vix_peak, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.date,
                        event.event_name,
                        event.severity,
                        json.dumps(event.affected_sectors),
                        event.recovery_days,
                        event.vix_peak,
                        event.description
                    ))
                except Exception as e:
                    logger.error(f"Error storing black swan event {event.event_name}: {e}")

            conn.commit()

    def get_training_data(self,
                         start_date: str = None,
                         end_date: str = None,
                         symbols: List[str] = None) -> pd.DataFrame:
        """
        Get training data from database

        Args:
            start_date: Start date for data
            end_date: End date for data
            symbols: List of symbols to retrieve

        Returns:
            DataFrame with training data
        """
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        query += " ORDER BY date, symbol"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_black_swan_periods(self) -> List[Tuple[str, str, str]]:
        """
        Get date ranges around black swan events for focused analysis

        Returns:
            List of (start_date, end_date, event_name) tuples
        """
        periods = []
        events = self.get_black_swan_events()

        for event in events:
            # Get 60 days before and recovery days after
            event_date = datetime.strptime(event.date, "%Y-%m-%d")
            start = (event_date - timedelta(days=60)).strftime("%Y-%m-%d")
            end = (event_date + timedelta(days=event.recovery_days)).strftime("%Y-%m-%d")
            periods.append((start, end, event.event_name))

        return periods

    def calculate_market_regimes(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Calculate market regime classifications based on volatility and trends

        Args:
            lookback_days: Days to look back for regime calculation

        Returns:
            DataFrame with regime classifications
        """
        # Get SPY data as market proxy
        query = """
            SELECT date, close, returns, volatility_20d
            FROM market_data
            WHERE symbol = 'SPY'
            ORDER BY date
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)

        if df.empty:
            logger.warning("No SPY data found for regime calculation")
            return pd.DataFrame()

        # Calculate rolling metrics
        df['trend'] = df['close'].pct_change(lookback_days)
        df['vol_percentile'] = df['volatility_20d'].rolling(252).rank(pct=True)

        # Classify regimes
        def classify_regime(row):
            if pd.isna(row['trend']) or pd.isna(row['vol_percentile']):
                return 'unknown'

            if row['vol_percentile'] > 0.8:
                return 'crisis' if row['trend'] < -0.1 else 'volatile_bull'
            elif row['vol_percentile'] < 0.2:
                return 'calm_bull' if row['trend'] > 0 else 'calm_bear'
            else:
                return 'normal_bull' if row['trend'] > 0 else 'normal_bear'

        df['regime'] = df.apply(classify_regime, axis=1)

        return df[['date', 'regime', 'volatility_20d', 'trend']]


if __name__ == "__main__":
    # Example usage
    manager = HistoricalDataManager()

    # Download initial batch of data
    symbols = ['SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD']
    success = manager.download_historical_data(symbols, start_date="2020-01-01")

    if success:
        logger.info("Data download successful")

        # Get black swan events
        events = manager.get_black_swan_events()
        logger.info(f"Loaded {len(events)} black swan events")

        # Get training data
        df = manager.get_training_data(start_date="2020-01-01", symbols=['SPY'])
        logger.info(f"Retrieved {len(df)} rows of training data")