"""
AI Model Service for HRM 32D Model Inference
Loads trained model and provides strategy predictions
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Black Swan Strategy Names (output of the model)
STRATEGY_NAMES = [
    "Inequality Mispricing Exploit",
    "Volatility Arbitrage",
    "Narrative Gap Trade",
    "Correlation Breakdown",
    "Barbell Position",
    "Antifragile Convex",
    "Black Swan Hunt",
    "Risk Parity Rebalance"
]

@dataclass
class StrategyPrediction:
    """Strategy prediction from AI model"""
    strategy_index: int
    strategy_name: str
    confidence: float
    probabilities: List[float]
    features_used: List[float]
    timestamp: datetime
    metadata: Dict

class OptimizedHRM32D(nn.Module):
    """Optimized HRM Model Architecture (must match training)"""

    def __init__(self, input_dim=32, hidden_dim=512, output_dim=8):
        super().__init__()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformer-like processing layers
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(4)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """Forward pass"""
        # Input projection
        x = self.input_projection(x)

        # Process through layers with residual connections
        for layer in self.processing_layers:
            x = x + layer(x)

        # Output
        logits = self.output_head(x)
        return logits

class AIModelService:
    """Service for loading and using the trained HRM model"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize AI model service

        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_metadata = {}

        # Default model path
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            # Try to find the best available model
            potential_models = [
                project_root / 'models' / 'enhanced_hrm_32d_checkpoint_20000.pth',
                project_root / 'models' / 'optimized_hrm_32d_grokfast.pth',
                project_root / 'models' / 'enhanced_hrm_32d_grokfast.pth',
            ]

            for path in potential_models:
                if path.exists():
                    model_path = str(path)
                    logger.info(f"Found model at: {path}")
                    break

        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No trained model found, using random initialization")
            self._initialize_random_model()

    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from checkpoint

        Args:
            model_path: Path to model checkpoint

        Returns:
            Success status
        """
        try:
            # Initialize model architecture
            self.model = OptimizedHRM32D(input_dim=32, hidden_dim=512, output_dim=8)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict is the state dict
                    self.model.load_state_dict(checkpoint)

                # Extract metadata if available
                self.model_metadata = {
                    'iteration': checkpoint.get('iteration', 'unknown'),
                    'clean_accuracy': checkpoint.get('clean_accuracy', 0.0),
                    'noisy_accuracy': checkpoint.get('noisy_accuracy', 0.0),
                    'loaded_from': model_path,
                    'loaded_at': datetime.now().isoformat()
                }
            else:
                # Direct model object
                self.model = checkpoint
                self.model_metadata = {
                    'loaded_from': model_path,
                    'loaded_at': datetime.now().isoformat()
                }

            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info(f"Successfully loaded model from {model_path}")
            logger.info(f"Model metadata: {self.model_metadata}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self._initialize_random_model()
            return False

    def _initialize_random_model(self):
        """Initialize model with random weights (fallback)"""
        self.model = OptimizedHRM32D(input_dim=32, hidden_dim=512, output_dim=8)
        self.model.to(self.device)
        self.model.eval()
        self.model_metadata = {
            'type': 'random_initialization',
            'warning': 'Using untrained model - predictions will be random'
        }
        logger.warning("Using randomly initialized model - predictions will not be meaningful")

    def predict(self, features: List[float]) -> StrategyPrediction:
        """
        Predict trading strategy from 32 features

        Args:
            features: List of 32 feature values

        Returns:
            StrategyPrediction with recommended strategy
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        if len(features) != 32:
            raise ValueError(f"Expected 32 features, got {len(features)}")

        try:
            # Convert to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # Get best strategy
            strategy_index = int(np.argmax(probabilities))
            confidence = float(probabilities[strategy_index])

            # Create prediction object
            prediction = StrategyPrediction(
                strategy_index=strategy_index,
                strategy_name=STRATEGY_NAMES[strategy_index],
                confidence=confidence,
                probabilities=probabilities.tolist(),
                features_used=features,
                timestamp=datetime.now(),
                metadata={
                    'model_metadata': self.model_metadata,
                    'device': str(self.device),
                    'top_3_strategies': self._get_top_strategies(probabilities, 3)
                }
            )

            logger.info(f"Predicted strategy: {prediction.strategy_name} (confidence: {confidence:.2%})")

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback prediction
            return StrategyPrediction(
                strategy_index=0,
                strategy_name=STRATEGY_NAMES[0],
                confidence=0.125,  # 1/8 uniform probability
                probabilities=[0.125] * 8,
                features_used=features,
                timestamp=datetime.now(),
                metadata={'error': str(e), 'fallback': True}
            )

    def _get_top_strategies(self, probabilities: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Get top K strategies with their probabilities"""
        indices = np.argsort(probabilities)[::-1][:top_k]

        return [
            {
                'index': int(idx),
                'name': STRATEGY_NAMES[idx],
                'probability': float(probabilities[idx])
            }
            for idx in indices
        ]

    def predict_batch(self, feature_batch: List[List[float]]) -> List[StrategyPrediction]:
        """
        Predict strategies for multiple feature sets

        Args:
            feature_batch: List of feature vectors (each with 32 features)

        Returns:
            List of StrategyPredictions
        """
        predictions = []
        for features in feature_batch:
            predictions.append(self.predict(features))
        return predictions

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'metadata': self.model_metadata,
            'strategies': STRATEGY_NAMES,
            'input_features': 32,
            'output_strategies': 8
        }

    def explain_prediction(self, prediction: StrategyPrediction) -> Dict:
        """
        Provide explanation for a prediction

        Args:
            prediction: StrategyPrediction to explain

        Returns:
            Dict with explanation details
        """
        # Feature importance (simplified - could use SHAP or similar)
        feature_values = np.array(prediction.features_used)
        feature_importance = np.abs(feature_values) / (np.sum(np.abs(feature_values)) + 1e-8)

        # Get top important features
        top_indices = np.argsort(feature_importance)[::-1][:5]

        feature_names = [
            "VIX Level", "VIX Percentile", "SPY 5D Returns", "SPY 20D Returns",
            "Put/Call Ratio", "Market Breadth", "Correlation", "Volume Ratio",
            "Gini Coefficient", "Top 1% Wealth", "Real Wage Growth", "Luxury/Discount",
            "Wealth Velocity", "Wealth Concentration", "Inequality Accel", "Sector Dispersion",
            "Correlation Breakdown", "VIX Term Structure", "Risk-On Sentiment", "Days to FOMC",
            "Days to CPI", "Days to Earnings", "Signal Quality", "AI Confidence",
            "VIX 1H Forecast", "VIX 6H Forecast", "VIX 24H Forecast", "Price Forecast",
            "Forecast Uncertainty", "Sentiment Score", "Sentiment Vol", "Price Prob"
        ]

        explanation = {
            'strategy': prediction.strategy_name,
            'confidence': prediction.confidence,
            'reasoning': self._generate_reasoning(prediction),
            'top_features': [
                {
                    'name': feature_names[idx] if idx < len(feature_names) else f"Feature {idx+1}",
                    'value': float(feature_values[idx]),
                    'importance': float(feature_importance[idx])
                }
                for idx in top_indices
            ],
            'alternative_strategies': [
                {
                    'name': STRATEGY_NAMES[i],
                    'probability': prediction.probabilities[i]
                }
                for i in range(len(STRATEGY_NAMES))
                if i != prediction.strategy_index
            ][:3],  # Top 3 alternatives
            'risk_level': self._assess_risk_level(prediction)
        }

        return explanation

    def _generate_reasoning(self, prediction: StrategyPrediction) -> str:
        """Generate human-readable reasoning for prediction"""
        strategy_reasonings = {
            "Inequality Mispricing Exploit": "High wealth concentration and inequality metrics suggest mispricing opportunities in affected sectors.",
            "Volatility Arbitrage": "VIX term structure and forecasts indicate profitable volatility trading opportunities.",
            "Narrative Gap Trade": "Significant divergence between market narrative and underlying fundamentals detected.",
            "Correlation Breakdown": "Inter-market correlations showing unusual patterns, suggesting regime change.",
            "Barbell Position": "Risk metrics favor extreme risk barbell strategy (80% safe, 20% high-risk).",
            "Antifragile Convex": "Market conditions favor positions that gain from disorder and volatility.",
            "Black Swan Hunt": "Tail risk indicators elevated, suggesting preparation for extreme events.",
            "Risk Parity Rebalance": "Portfolio risk distribution suboptimal, rebalancing recommended."
        }

        base_reasoning = strategy_reasonings.get(
            prediction.strategy_name,
            "Market conditions analyzed across all 32 features."
        )

        confidence_modifier = ""
        if prediction.confidence > 0.8:
            confidence_modifier = " Strong signal detected."
        elif prediction.confidence > 0.6:
            confidence_modifier = " Moderate confidence in this approach."
        elif prediction.confidence > 0.4:
            confidence_modifier = " Weak signal, consider alternatives."
        else:
            confidence_modifier = " Low confidence, high uncertainty."

        return base_reasoning + confidence_modifier

    def _assess_risk_level(self, prediction: StrategyPrediction) -> str:
        """Assess risk level of predicted strategy"""
        high_risk_strategies = ["Black Swan Hunt", "Antifragile Convex", "Volatility Arbitrage"]
        medium_risk_strategies = ["Inequality Mispricing Exploit", "Narrative Gap Trade", "Correlation Breakdown"]
        low_risk_strategies = ["Barbell Position", "Risk Parity Rebalance"]

        if prediction.strategy_name in high_risk_strategies:
            return "HIGH"
        elif prediction.strategy_name in medium_risk_strategies:
            return "MEDIUM"
        else:
            return "LOW"

# Singleton instance
_model_service = None

def get_model_service() -> AIModelService:
    """Get or create model service singleton"""
    global _model_service
    if _model_service is None:
        _model_service = AIModelService()
    return _model_service

if __name__ == "__main__":
    # Test the model service
    service = get_model_service()
    print(f"Model info: {service.get_model_info()}")

    # Test prediction with random features
    test_features = np.random.randn(32).tolist()
    prediction = service.predict(test_features)
    print(f"Prediction: {prediction.strategy_name} ({prediction.confidence:.2%})")

    # Get explanation
    explanation = service.explain_prediction(prediction)
    print(f"Explanation: {json.dumps(explanation, indent=2, default=str)}")