"""
Trading Database Schema for Railway PostgreSQL
Paper Trading Persistence Layer for Gary x Taleb System
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Get DATABASE_URL from environment
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///local_trading.db')

# Handle Railway's postgres:// vs postgresql:// (SQLAlchemy 1.4+ requires postgresql://)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create engine with connection pooling for Railway
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=False           # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Trade(Base):
    """Individual trade records"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(50), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), index=True)
    direction = Column(String(10))  # 'long' or 'short'
    position_size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    return_pct = Column(Float, nullable=True)

    # Phase 5 Enhancement Metrics
    ng_score = Column(Float)  # Narrative Gap score
    ng_multiplier = Column(Float)  # Position multiplier from NG
    brier_adjustment = Column(Float)  # Brier score risk adjustment
    dpi_enhancement = Column(Float)  # DPI regime enhancement

    # Metadata
    is_simulation = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'direction': self.direction,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'ng_score': self.ng_score,
            'ng_multiplier': self.ng_multiplier,
            'brier_adjustment': self.brier_adjustment,
            'dpi_enhancement': self.dpi_enhancement,
            'is_simulation': self.is_simulation
        }


class PortfolioState(Base):
    """Portfolio state snapshots for recovery"""
    __tablename__ = 'portfolio_state'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    capital = Column(Float)
    initial_capital = Column(Float)
    positions_json = Column(Text)  # JSON serialized positions
    total_pnl = Column(Float)
    trade_count = Column(Integer)
    winning_trades = Column(Integer)

    # Session info
    session_id = Column(String(50), index=True)
    is_active = Column(Boolean, default=True)

    def get_positions(self) -> Dict:
        if self.positions_json:
            return json.loads(self.positions_json)
        return {}

    def set_positions(self, positions: Dict):
        self.positions_json = json.dumps(positions)


class Phase5Metrics(Base):
    """Phase 5 component metrics for monitoring"""
    __tablename__ = 'phase5_metrics'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Brier Score Tracking
    brier_score = Column(Float)
    prediction_count = Column(Integer)
    recent_accuracy = Column(Float)

    # Narrative Gap Stats
    ng_signal_count = Column(Integer)
    avg_ng_score = Column(Float)
    high_ng_trades = Column(Integer)  # Trades with |NG| > 0.03

    # DPI Stats
    dpi_regime = Column(String(20))
    avg_dpi_enhancement = Column(Float)

    # Overall Health
    health_status = Column(String(50))
    session_id = Column(String(50), index=True)


class TradingSession(Base):
    """Trading session tracking"""
    __tablename__ = 'trading_sessions'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Configuration
    initial_capital = Column(Float)
    base_position_size = Column(Float)
    max_position_size = Column(Float)

    # Results
    final_capital = Column(Float, nullable=True)
    total_return_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)

    # Status
    status = Column(String(20), default='active')  # active, completed, error


def init_db():
    """Create all tables if they don't exist"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_db():
    """Get database session (for dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> SessionLocal:
    """Get a new database session"""
    return SessionLocal()


# Test connection on import
def test_connection() -> bool:
    """Test database connection"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the schema
    logging.basicConfig(level=logging.INFO)
    print(f"DATABASE_URL: {DATABASE_URL[:50]}...")

    if test_connection():
        print("Connection OK")
        if init_db():
            print("Tables created successfully")
        else:
            print("Table creation failed")
    else:
        print("Connection failed")
