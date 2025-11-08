"""
TRM Streaming Integration
Connects trained TRM model to real-time data streams for live strategy predictions
"""

import asyncio
import logging
import json
import numpy as np
import torch
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Import TRM model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.trm_model import TinyRecursiveModel
from data.market_feature_extractor import MarketFeatureExtractor
from data.historical_data_manager import HistoricalDataManager

logger = logging.getLogger(__name__)


class TRMStreamingPredictor:
    """
    Real-time TRM predictions for streaming data

    Extracts 10 market features and predicts optimal strategy (0-7)
    Updates at configurable intervals for WebSocket broadcasting
    """

    def __init__(self,
                 model_path: str = "C:/Users/17175/Desktop/trader-ai/checkpoints/training_checkpoint.pkl",
                 normalization_path: str = "C:/Users/17175/Desktop/trader-ai/config/trm_normalization.json",
                 update_interval: int = 60):
        """
        Initialize TRM streaming predictor

        Args:
            model_path: Path to trained TRM checkpoint
            normalization_path: Path to normalization parameters
            update_interval: Prediction update interval in seconds
        """
        self.model_path = model_path
        self.normalization_path = normalization_path
        self.update_interval = update_interval

        # Strategy names
        self.strategy_names = [
            "ultra_defensive",
            "defensive",
            "balanced_safe",
            "balanced_growth",
            "balanced_aggressive",
            "aggressive_growth",
            "max_growth",
            "tactical_opportunity"
        ]

        # Model state
        self.model: Optional[TinyRecursiveModel] = None
        self.normalization_params: Optional[Dict] = None

        # Initialize data managers
        self.historical_manager = HistoricalDataManager(
            db_path="C:/Users/17175/Desktop/trader-ai/data/historical_data.db"
        )
        self.feature_extractor = MarketFeatureExtractor(self.historical_manager)

        # Prediction state
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.prediction_history = []
        self.is_running = False

        # Load model and normalization
        self._load_model()
        self._load_normalization()

    def _load_model(self):
        """Load trained TRM model from checkpoint"""
        try:
            logger.info(f"Loading TRM model from {self.model_path}")

            # Create model with same architecture as training
            self.model = TinyRecursiveModel(
                input_dim=10,
                hidden_dim=512,
                output_dim=8,
                num_latent_steps=6,
                num_recursion_cycles=3
            )

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Handle both direct state_dict and wrapped checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()  # Set to evaluation mode
            logger.info("TRM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TRM model: {e}")
            raise

    def _load_normalization(self):
        """Load feature normalization parameters"""
        try:
            logger.info(f"Loading normalization from {self.normalization_path}")

            with open(self.normalization_path, 'r') as f:
                self.normalization_params = json.load(f)

            logger.info("Normalization parameters loaded")
            logger.info(f"Features: {self.normalization_params.get('feature_names', [])}")

        except Exception as e:
            logger.error(f"Failed to load normalization: {e}")
            raise

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization using training statistics

        Args:
            features: Raw feature array (10,)

        Returns:
            Normalized features (10,)
        """
        mean = np.array(self.normalization_params['mean'])
        std = np.array(self.normalization_params['std'])

        # Z-score normalization
        normalized = (features - mean) / (std + 1e-8)

        return normalized

    async def extract_features(self) -> Optional[np.ndarray]:
        """
        Extract 10 market features for TRM prediction

        Returns:
            Feature array (10,) or None if extraction fails
        """
        try:
            # Extract features using MarketFeatureExtractor
            # This queries the database for latest market data
            features_df = await asyncio.to_thread(
                self.feature_extractor.extract_features,
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if features_df is None or len(features_df) == 0:
                logger.warning("No features extracted from market data")
                return None

            # Get most recent row (latest market state)
            latest = features_df.iloc[-1]

            # Extract 10 features in correct order
            feature_names = self.normalization_params['feature_names']
            features = np.array([latest[name] for name in feature_names])

            logger.debug(f"Extracted features: {dict(zip(feature_names, features))}")

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make TRM prediction from features

        Args:
            features: Raw features (10,)

        Returns:
            Prediction dictionary with strategy, probabilities, confidence
        """
        try:
            # Normalize features
            normalized = self._normalize_features(features)

            # Convert to PyTorch tensor (batch size 1)
            features_tensor = torch.FloatTensor(normalized).unsqueeze(0)  # (1, 10)

            # Forward pass through TRM with recursion
            with torch.no_grad():
                output = self.model(features_tensor, T=3, n=6)

            # Extract logits and compute probabilities
            logits = output['strategy_logits']  # (1, 8)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()  # (8,)

            # Get predicted strategy (argmax)
            predicted_idx = int(np.argmax(probabilities))
            predicted_strategy = self.strategy_names[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            # Build prediction result
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'strategy_id': predicted_idx,
                'strategy_name': predicted_strategy,
                'confidence': confidence,
                'probabilities': {
                    name: float(prob)
                    for name, prob in zip(self.strategy_names, probabilities)
                },
                'raw_features': features.tolist(),
                'normalized_features': normalized.tolist(),
                'halt_probability': float(output['halt_probability'].item()),
                'model_metadata': {
                    'recursion_cycles': 3,
                    'latent_steps': 6,
                    'effective_depth': 42
                }
            }

            return prediction

        except Exception as e:
            logger.error(f"TRM prediction failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'strategy_id': None,
                'strategy_name': 'ERROR',
                'confidence': 0.0
            }

    async def predict_streaming(self) -> Optional[Dict[str, Any]]:
        """
        Extract features and make prediction for streaming

        Returns:
            Prediction dictionary or None if extraction failed
        """
        # Extract features
        features = await self.extract_features()

        if features is None:
            return None

        # Make prediction
        prediction = self.predict(features)

        # Store in history
        self.last_prediction = prediction
        self.prediction_history.append(prediction)

        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]

        return prediction

    async def stream_predictions(self, callback=None):
        """
        Continuous streaming prediction loop

        Args:
            callback: Optional callback function(prediction: Dict)
        """
        self.is_running = True
        logger.info(f"Starting TRM streaming predictions (interval={self.update_interval}s)")

        while self.is_running:
            try:
                # Make prediction
                prediction = await self.predict_streaming()

                if prediction and callback:
                    await callback(prediction)

                # Log prediction
                if prediction and 'strategy_name' in prediction:
                    logger.info(
                        f"TRM Prediction: {prediction['strategy_name']} "
                        f"(confidence={prediction['confidence']:.2%})"
                    )

            except Exception as e:
                logger.error(f"Streaming prediction error: {e}")

            # Wait for next interval
            await asyncio.sleep(self.update_interval)

    def stop_streaming(self):
        """Stop streaming predictions"""
        self.is_running = False
        logger.info("Stopped TRM streaming predictions")

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return {'status': 'no_predictions'}

        # Analyze last 10 predictions
        recent = self.prediction_history[-10:]

        strategy_counts = {}
        for pred in recent:
            if 'strategy_name' in pred:
                name = pred['strategy_name']
                strategy_counts[name] = strategy_counts.get(name, 0) + 1

        avg_confidence = np.mean([
            p['confidence'] for p in recent
            if 'confidence' in p and p['confidence'] > 0
        ])

        return {
            'total_predictions': len(self.prediction_history),
            'recent_count': len(recent),
            'strategy_distribution': strategy_counts,
            'average_confidence': float(avg_confidence),
            'last_prediction': self.last_prediction
        }


# Integration with WebSocket server
async def broadcast_trm_predictions(predictor: TRMStreamingPredictor, websocket_manager):
    """
    Callback for broadcasting TRM predictions via WebSocket

    Args:
        predictor: TRM streaming predictor instance
        websocket_manager: WebSocket connection manager from websocket_server.py
    """
    async def callback(prediction: Dict[str, Any]):
        # Format for WebSocket broadcast
        message = {
            'type': 'trm_prediction',
            'data': prediction,
            'timestamp': datetime.now().isoformat()
        }

        # Broadcast to all connected clients
        await websocket_manager.broadcast(message)

    # Start streaming with broadcast callback
    await predictor.stream_predictions(callback=callback)


# Standalone test function
async def test_streaming():
    """Test TRM streaming predictions"""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

    # Create predictor
    predictor = TRMStreamingPredictor(update_interval=5)  # 5 second intervals for testing

    # Run for 30 seconds
    async def test_callback(prediction):
        print(f"\n{'='*60}")
        print(f"Strategy: {prediction['strategy_name']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Top 3 Probabilities:")
        probs = prediction['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, prob in sorted_probs:
            print(f"  {name:25s}: {prob:.2%}")
        print(f"{'='*60}\n")

    try:
        # Run streaming for 30 seconds
        stream_task = asyncio.create_task(
            predictor.stream_predictions(callback=test_callback)
        )

        await asyncio.sleep(30)
        predictor.stop_streaming()

        # Show summary
        summary = predictor.get_prediction_summary()
        print(f"\nPrediction Summary:")
        print(json.dumps(summary, indent=2))

    except KeyboardInterrupt:
        predictor.stop_streaming()
        print("\nStreaming stopped by user")


if __name__ == "__main__":
    asyncio.run(test_streaming())
