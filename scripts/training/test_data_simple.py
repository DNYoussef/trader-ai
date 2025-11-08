"""
Simple ASCII test of data generation
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
    print("[OK] Imported EnhancedHRMFeatureEngine")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

try:
    from src.strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome
    print("[OK] Imported ConvexRewardFunction")
except Exception as e:
    print(f"[ERROR] ConvexRewardFunction import failed: {e}")
    sys.exit(1)

print("\nTesting data generation...")

# Create feature engine
try:
    feature_engine = EnhancedHRMFeatureEngine()
    print("[OK] Feature engine created")
except Exception as e:
    print(f"[ERROR] Feature engine creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create sample data
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

# Test feature generation
try:
    vix_history = np.random.randn(100) * 5 + 20
    price_history = np.random.randn(100) * 10 + 450
    news_articles = ["Market up", "Fed unchanged", "Tech rally"]

    enhanced_features = feature_engine.create_enhanced_features(
        base_market_features=scenario,
        vix_history=vix_history,
        price_history=price_history,
        news_articles=news_articles,
        symbol='SPY'
    )

    print("[OK] Features generated")
    print(f"Shape: {enhanced_features.combined_features.shape}")
    print(f"Sample: {enhanced_features.combined_features[:3]}")

except Exception as e:
    print(f"[ERROR] Feature generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete")