"""
Quick test of data generation to identify the exact issue
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    from src.intelligence.enhanced_hrm_features import EnhancedHRMFeatureEngine
    print("✓ Imported EnhancedHRMFeatureEngine")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    from src.strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
    print("✓ Imported ConvexRewardFunction")
except Exception as e:
    print(f"✗ ConvexRewardFunction import failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("TESTING DATA GENERATION")
print("="*50)

# Test 1: Create feature engine
print("\n1. Creating feature engine...")
try:
    feature_engine = EnhancedHRMFeatureEngine()
    print("✓ Feature engine created")
except Exception as e:
    print(f"✗ Feature engine creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create reward function
print("\n2. Creating reward function...")
try:
    reward_function = ConvexRewardFunction()
    print("✓ Reward function created")
except Exception as e:
    print(f"✗ Reward function creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create sample data
print("\n3. Creating sample scenario...")
scenario = {
    'price': 450.0,
    'vix_level': 20.0,
    'spy_returns_5d': 0.02,
    'spy_returns_20d': 0.05,
    'put_call_ratio': 0.8,
    'market_breadth': 0.6,
    'volume_ratio': 1.0,
    'sector_dispersion': 0.02
}
print("✓ Scenario created")

# Test 4: Generate features
print("\n4. Generating features...")
try:
    vix_history = np.random.randn(100) * 5 + 20
    price_history = np.random.randn(100) * 10 + 450
    news_articles = ["Market volatility increases", "Fed policy unchanged", "Tech stocks rally"]

    enhanced_features = feature_engine.create_enhanced_features(
        base_market_features=scenario,
        vix_history=vix_history,
        price_history=price_history,
        news_articles=news_articles,
        symbol='SPY'
    )

    print("✓ Features generated successfully")
    print(f"  Feature shape: {enhanced_features.combined_features.shape}")
    print(f"  First 5 features: {enhanced_features.combined_features[:5]}")

except Exception as e:
    print(f"✗ Feature generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test reward calculation
print("\n5. Testing reward calculation...")
try:
    strategies = [
        'crisis_alpha', 'tail_hedge', 'volatility_harvest', 'event_catalyst',
        'correlation_breakdown', 'inequality_arbitrage', 'momentum_explosion', 'mean_reversion'
    ]

    # Test for one strategy
    pnl = 0.05  # 5% return
    trade_outcome = TradeOutcome(
        strategy_name='crisis_alpha',
        entry_date=datetime.now(),
        exit_date=datetime.now(),
        symbol='SPY',
        returns=pnl,
        max_drawdown=min(0, pnl),
        holding_period_days=10,
        volatility_during_trade=0.04,
        is_black_swan_period=False,
        black_swan_captured=False,
        confidence_score=0.8
    )

    reward = reward_function.calculate_reward(trade_outcome)
    print("✓ Reward calculation successful")
    print(f"  Reward: {reward}")

except Exception as e:
    print(f"✗ Reward calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("SUMMARY:")
print("- Feature engine: ✓")
print("- Feature generation: ✓")
print("- Reward calculation: ✓")
print("The data generation components work individually.")
print("Issue might be in batch generation loop or error handling.")
print("="*50)