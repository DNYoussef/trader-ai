"""Test enhanced batch generation to debug training issue"""
import sys
from pathlib import Path
import numpy as np

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.intelligence.enhanced_hrm_features import EnhancedHRMFeatureEngine

def test_batch_generation():
    print("Testing enhanced batch generation...")

    # Initialize feature engine
    feature_engine = EnhancedHRMFeatureEngine()

    # Test scenario
    scenario = {
        'name': 'Test',
        'vix_level': 20,
        'spy_returns_5d': 0.01,
        'spy_returns_20d': 0.02,
        'put_call_ratio': 1.0,
        'market_breadth': 0.6,
        'volume_ratio': 1.2,
        'correlation': 0.5,
        'gini_coefficient': 0.42,
        'sector_dispersion': 0.008,
        'signal_quality_score': 0.5
    }

    # Test data
    vix_history = np.random.normal(22, 8, 500)
    price_history = np.random.normal(400, 50, 500)
    news_articles = ["Market shows positive momentum", "Economic data beats expectations"]

    try:
        print("Calling create_enhanced_features...")
        enhanced_features = feature_engine.create_enhanced_features(
            base_market_features=scenario,
            vix_history=vix_history,
            price_history=price_history,
            news_articles=news_articles,
            symbol='SPY'
        )

        print(f"SUCCESS! Features shape: {enhanced_features.combined_features.shape}")
        print(f"Features: {enhanced_features.combined_features[:10]}")  # First 10

    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_generation()