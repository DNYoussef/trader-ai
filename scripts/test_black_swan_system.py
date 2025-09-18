"""
Test Script for Black Swan Hunting AI System
Verifies all components are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_manager():
    """Test the historical data manager"""
    logger.info("Testing Historical Data Manager...")

    from src.data.historical_data_manager import HistoricalDataManager

    manager = HistoricalDataManager()

    # Get black swan events
    events = manager.get_black_swan_events()
    logger.info(f"Found {len(events)} black swan events")

    # Get some training data
    df = manager.get_training_data(
        start_date="2020-01-01",
        end_date="2020-12-31",
        symbols=['SPY']
    )

    if not df.empty:
        logger.info(f"Retrieved {len(df)} rows of training data")
        logger.info(f"Columns: {df.columns.tolist()}")
        return True
    else:
        logger.warning("No training data found")
        return False

def test_black_swan_labeler():
    """Test the black swan labeler"""
    logger.info("Testing Black Swan Labeler...")

    from src.data.black_swan_labeler import BlackSwanLabeler

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')

    # Simulate returns with some extreme events
    returns = np.random.normal(0.0005, 0.02, len(dates))
    # Add some black swan events
    returns[100] = -0.10  # 10% crash
    returns[200] = 0.15   # 15% melt-up

    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'returns': returns
    })

    labeler = BlackSwanLabeler()
    labeled_df = labeler.label_tail_events(df)

    # Check results
    black_swans = labeled_df['is_black_swan'].sum()
    logger.info(f"Detected {black_swans} black swan events")

    stats = labeler.get_black_swan_statistics(labeled_df)
    logger.info(f"Black swan frequency: {stats.get('black_swan_frequency', 0):.2%}")

    return black_swans > 0

def test_strategy_toolbox():
    """Test the black swan strategy toolbox"""
    logger.info("Testing Strategy Toolbox...")

    from src.strategies.black_swan_strategies import (
        BlackSwanStrategyToolbox,
        MarketState
    )

    # Create market state
    market_state = MarketState(
        timestamp=datetime.now(),
        vix_level=25.0,  # Elevated VIX
        vix_percentile=0.7,
        spy_returns_5d=-0.05,
        spy_returns_20d=-0.02,
        put_call_ratio=1.5,
        market_breadth=0.35,
        correlation=0.55,
        volume_ratio=1.8,
        regime='volatile'
    )

    # Create toolbox
    toolbox = BlackSwanStrategyToolbox()

    # Get empty historical data for now
    historical_data = pd.DataFrame()

    # Analyze strategies
    signals = toolbox.analyze_all_strategies(market_state, historical_data)
    logger.info(f"Generated {len(signals)} strategy signals")

    if signals:
        best_signals = toolbox.select_best_strategies(signals, max_strategies=3)
        for signal in best_signals:
            logger.info(f"  {signal.strategy_name}: {signal.action} {signal.symbol} "
                       f"(convexity={signal.convexity_ratio:.1f}x)")

    return True

def test_convex_reward():
    """Test the convex reward function"""
    logger.info("Testing Convex Reward Function...")

    from src.strategies.convex_reward_function import (
        ConvexRewardFunction,
        TradeOutcome
    )

    reward_func = ConvexRewardFunction()

    # Test black swan capture
    black_swan_outcome = TradeOutcome(
        strategy_name='tail_hedge',
        entry_date=datetime.now(),
        exit_date=datetime.now(),
        symbol='SPY_PUT',
        returns=1.5,  # 150% return
        max_drawdown=-0.03,
        holding_period_days=5,
        volatility_during_trade=0.08,
        is_black_swan_period=True,
        black_swan_captured=True,
        convexity_achieved=50.0
    )

    metrics = reward_func.calculate_reward(black_swan_outcome)
    logger.info(f"Black Swan Capture Reward: {metrics.final_reward:.2f} "
               f"(multiplier: {metrics.black_swan_multiplier:.1f}x)")

    # Test small loss
    small_loss_outcome = TradeOutcome(
        strategy_name='mean_reversion',
        entry_date=datetime.now(),
        exit_date=datetime.now(),
        symbol='AAPL',
        returns=-0.08,  # 8% loss
        max_drawdown=-0.08,
        holding_period_days=3,
        volatility_during_trade=0.02,
        is_black_swan_period=False,
        black_swan_captured=False,
        convexity_achieved=1.0
    )

    metrics = reward_func.calculate_reward(small_loss_outcome)
    logger.info(f"Small Loss Reward: {metrics.final_reward:.2f}")

    # Get statistics
    stats = reward_func.get_reward_statistics()
    logger.info(f"Reward Statistics: avg={stats['avg_reward']:.2f}, "
               f"black_swans={stats['black_swan_captures']}")

    return True

def test_database():
    """Test database connectivity"""
    logger.info("Testing Database...")

    import sqlite3
    from pathlib import Path

    db_path = Path("data/black_swan_training.db")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Database tables: {[t[0] for t in tables]}")

        # Check black swan events
        cursor.execute("SELECT COUNT(*) FROM black_swan_events")
        event_count = cursor.fetchone()[0]
        logger.info(f"Black swan events in database: {event_count}")

        # Check market data
        cursor.execute("SELECT COUNT(*) FROM market_data")
        data_count = cursor.fetchone()[0]
        logger.info(f"Market data records: {data_count}")

    return event_count > 0

def test_ollama():
    """Test Ollama connectivity"""
    logger.info("Testing Ollama...")

    import subprocess

    try:
        # Check if Ollama is running
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True,
                              text=True,
                              timeout=5)

        if result.returncode == 0:
            models = result.stdout.strip().split('\n')
            logger.info(f"Ollama is running. Models: {len(models)-1}")  # Subtract header

            # Check for Mistral
            if 'mistral' in result.stdout.lower():
                logger.info("Mistral model is available")
                return True
            else:
                logger.warning("Mistral model not yet installed")
                return False
        else:
            logger.error("Ollama command failed")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Ollama command timed out - may be starting up")
        return False
    except FileNotFoundError:
        logger.error("Ollama not found in PATH")
        return False
    except Exception as e:
        logger.error(f"Error testing Ollama: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Black Swan Hunting AI System - Component Test")
    logger.info("=" * 60)

    test_results = {}

    # Run tests
    tests = [
        ("Database", test_database),
        ("Data Manager", test_data_manager),
        ("Black Swan Labeler", test_black_swan_labeler),
        ("Strategy Toolbox", test_strategy_toolbox),
        ("Convex Reward Function", test_convex_reward),
        ("Ollama LLM", test_ollama)
    ]

    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Testing {test_name} ---")
            result = test_func()
            test_results[test_name] = "✅ PASS" if result else "⚠️ WARN"
        except Exception as e:
            logger.error(f"Test failed: {e}")
            test_results[test_name] = "❌ FAIL"

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, result in test_results.items():
        logger.info(f"{test_name}: {result}")

    # Overall status
    if all("PASS" in r for r in test_results.values()):
        logger.info("\n🎉 All systems operational! Ready for black swan hunting!")
        return 0
    elif any("FAIL" in r for r in test_results.values()):
        logger.error("\n❌ Some components failed. Please check the logs.")
        return 1
    else:
        logger.warning("\n⚠️ System partially operational. Ollama model may still be downloading.")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)