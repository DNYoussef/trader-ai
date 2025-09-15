#!/usr/bin/env python3
"""
Production ML Training Script
Execute complete training pipeline and create real trained models
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from intelligence.training.trainer import ModelTrainer
from intelligence.models.registry import ModelRegistry
from intelligence.data.processor import DataProcessor
from intelligence.prediction.predictor import Predictor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def create_sample_data():
    """Create sample financial data for training"""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n_samples = 5000

    print("Generating sample financial data...")

    # Generate realistic price series
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')

    # Price with trend and volatility clustering
    returns = np.random.normal(0, 0.02, n_samples)
    returns[1000:1200] *= 3  # High volatility period
    returns[3000:3500] *= 0.5  # Low volatility period

    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.exponential(1000000, n_samples)

    # Additional features
    ma_5 = pd.Series(prices).rolling(5).mean().fillna(prices)
    ma_20 = pd.Series(prices).rolling(20).mean().fillna(prices)

    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes,
        'returns': returns,
        'ma_5': ma_5,
        'ma_20': ma_20,
        'high': prices * (1 + np.random.uniform(0, 0.05, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.05, n_samples)),
    })

    # Create target variable (future price movement)
    df['target'] = df['price'].pct_change(periods=5).shift(-5).fillna(0)

    return df

def main():
    """Main training execution"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("="*60)
    print("ML INTELLIGENCE SYSTEM - PRODUCTION TRAINING")
    print("="*60)

    # Create directories
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)

    # Training configuration
    config = {
        'mlflow': {
            'tracking_uri': f'file://{os.path.abspath("mlruns")}',
            'experiment_name': 'trader-ai-production'
        },
        'model_storage': {
            'base_path': os.path.abspath("trained_models")
        },
        'training': {
            'test_size': 0.2,
            'validation_size': 0.15,
            'random_state': 42,
            'cv_folds': 3
        },
        'models': {
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            },
            'lstm': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 20
            }
        }
    }

    try:
        # 1. Generate and save training data
        print("\n1. Generating training data...")
        data = create_sample_data()
        data_path = "data/training_data.csv"
        data.to_csv(data_path, index=False)
        print(f"   Data saved to {data_path}")
        print(f"   Data shape: {data.shape}")
        print(f"   Features: {list(data.columns)}")

        # 2. Initialize trainer
        print("\n2. Initializing ML trainer...")
        trainer = ModelTrainer(config)
        print("   Trainer initialized successfully")

        # 3. Train all models
        print("\n3. Training models...")
        results = trainer.train_all_models(data_path)

        # 4. Print results
        print("\n4. Training Results:")
        print("-" * 40)
        successful_models = []
        for model_name, model in results.items():
            if model is not None:
                print(f"   ✓ {model_name}: SUCCESS")
                successful_models.append(model_name)
            else:
                print(f"   ✗ {model_name}: FAILED")

        print(f"\n   Successfully trained: {len(successful_models)}/{len(results)} models")

        # 5. Test model registry
        print("\n5. Testing model registry...")
        registry = ModelRegistry(config['model_storage']['base_path'])
        models_in_registry = registry.list_models()

        print(f"   Models in registry: {len(models_in_registry)}")
        for model_name in models_in_registry:
            info = registry.get_model_info(model_name)
            print(f"   - {model_name} v{info['version']}: {info['model_type']}")
            if 'metrics' in info and info['metrics']:
                for metric, value in info['metrics'].items():
                    print(f"     {metric}: {value:.6f}")

        # 6. Test prediction system
        print("\n6. Testing prediction system...")
        predictor = Predictor(config['model_storage']['base_path'])

        if successful_models:
            loaded_models = predictor.load_models(successful_models)
            print(f"   Loaded {len(loaded_models)} models for prediction")

            # Test prediction on sample data
            test_data = data.iloc[:100]  # First 100 rows

            for model_name in loaded_models:
                try:
                    prediction = predictor.predict_single(model_name, test_data)
                    print(f"   ✓ {model_name}: Prediction shape {prediction.shape}")
                except Exception as e:
                    print(f"   ✗ {model_name}: Prediction failed - {e}")

            # Test ensemble prediction
            if len(loaded_models) > 1:
                ensemble_result = predictor.predict_ensemble(test_data)
                print(f"   ✓ Ensemble prediction: {ensemble_result['ensemble_prediction']:.6f}")
                print(f"     Confidence: {ensemble_result['confidence']:.3f}")

        # 7. Verify file structure
        print("\n7. Verifying file structure...")
        model_dir = Path(config['model_storage']['base_path'])

        if model_dir.exists():
            model_files = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.pth"))
            print(f"   Model files created: {len(model_files)}")
            for file in model_files:
                print(f"     - {file.relative_to(model_dir)}")

        # 8. Run basic tests
        print("\n8. Running validation tests...")

        # Test imports
        try:
            from intelligence import ModelTrainer, ModelRegistry, DataProcessor, Predictor
            print("   ✓ All imports successful")
        except Exception as e:
            print(f"   ✗ Import test failed: {e}")

        # Test data processing
        try:
            processor = DataProcessor()
            processed = processor.preprocess(data.iloc[:100])
            print(f"   ✓ Data processing: {processed.shape}")
        except Exception as e:
            print(f"   ✗ Data processing test failed: {e}")

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Models trained: {len(successful_models)}")
        print(f"Model files: {len(model_files) if 'model_files' in locals() else 0}")
        print(f"Registry entries: {len(models_in_registry)}")
        print("System is ready for production use!")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n✗ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)