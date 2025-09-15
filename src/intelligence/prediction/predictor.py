"""
Production prediction system
Real-time inference with model ensemble
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Union
import logging
from ..models.registry import ModelRegistry
from ..data.processor import DataProcessor

logger = logging.getLogger(__name__)

class Predictor:
    """
    Production prediction system with ensemble capability
    """

    def __init__(self, registry_path: str = "/c/Users/17175/Desktop/trader-ai/trained_models"):
        self.registry = ModelRegistry(registry_path)
        self.data_processor = DataProcessor()
        self.models = {}
        self.ensemble_weights = {}

    def load_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Load specified models for prediction"""
        loaded_models = {}

        for model_name in model_names:
            try:
                model = self.registry.load_model(model_name)
                loaded_models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

        self.models = loaded_models
        return loaded_models

    def predict_single(self, model_name: str, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Single model prediction

        Args:
            model_name: Name of the model to use
            data: Input data for prediction

        Returns:
            Prediction array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Preprocess data if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            processed_data = self.data_processor.transform_new_data(data)
            # Get numeric features only
            feature_cols = [col for col in processed_data.columns
                          if col not in ['timestamp', 'date', 'target']]
            X = processed_data[feature_cols].values
        else:
            X = data

        # Make prediction based on model type
        if hasattr(model, 'predict'):  # Scikit-learn models
            predictions = model.predict(X)
        elif hasattr(model, 'forward'):  # PyTorch models
            model.eval()
            with torch.no_grad():
                if model_name == 'lstm':
                    # LSTM expects sequences
                    if len(X.shape) == 2:
                        # Convert to sequences (use last 60 points)
                        sequence_length = min(60, X.shape[0])
                        if X.shape[0] >= sequence_length:
                            X_seq = X[-sequence_length:].reshape(1, sequence_length, -1)
                        else:
                            # Pad with zeros if not enough data
                            padding = np.zeros((sequence_length - X.shape[0], X.shape[1]))
                            X_padded = np.vstack([padding, X])
                            X_seq = X_padded.reshape(1, sequence_length, -1)
                        X_tensor = torch.FloatTensor(X_seq)
                    else:
                        X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = torch.FloatTensor(X)

                predictions = model(X_tensor).squeeze().numpy()
                if predictions.ndim == 0:
                    predictions = np.array([predictions])
        else:
            raise ValueError(f"Unsupported model type for {model_name}")

        return predictions

    def predict_ensemble(self, data: Union[pd.DataFrame, np.ndarray],
                        models: Optional[List[str]] = None,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Ensemble prediction using multiple models

        Args:
            data: Input data for prediction
            models: List of model names to use (all loaded models if None)
            weights: Model weights for ensemble (equal weights if None)

        Returns:
            Dictionary with individual predictions and ensemble result
        """
        if models is None:
            models = list(self.models.keys())

        if weights is None:
            weights = {model: 1.0 / len(models) for model in models}

        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            weights = {k: v / total_weight for k, v in weights.items()}

        predictions = {}
        valid_predictions = []
        valid_weights = []

        for model_name in models:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not loaded, skipping")
                continue

            try:
                pred = self.predict_single(model_name, data)
                predictions[model_name] = pred

                # For ensemble, use the last prediction or mean if multiple
                if len(pred) > 0:
                    ensemble_pred = pred[-1] if len(pred.shape) == 1 else pred.mean()
                    valid_predictions.append(ensemble_pred)
                    valid_weights.append(weights.get(model_name, 0))

            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = None

        # Calculate ensemble prediction
        if valid_predictions:
            valid_predictions = np.array(valid_predictions)
            valid_weights = np.array(valid_weights)

            # Normalize weights for valid predictions only
            if valid_weights.sum() > 0:
                valid_weights = valid_weights / valid_weights.sum()
                ensemble_prediction = np.average(valid_predictions, weights=valid_weights)
            else:
                ensemble_prediction = np.mean(valid_predictions)

            # Calculate prediction confidence (inverse of prediction variance)
            if len(valid_predictions) > 1:
                pred_std = np.std(valid_predictions)
                confidence = 1.0 / (1.0 + pred_std)
            else:
                confidence = 0.8  # Default confidence for single model

        else:
            ensemble_prediction = None
            confidence = 0.0

        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'confidence': confidence,
            'models_used': models,
            'weights_used': weights
        }

    def predict_with_uncertainty(self, data: Union[pd.DataFrame, np.ndarray],
                                model_name: str, n_samples: int = 100) -> Dict[str, Any]:
        """
        Prediction with uncertainty estimation using Monte Carlo dropout

        Args:
            data: Input data
            model_name: Model to use
            n_samples: Number of Monte Carlo samples

        Returns:
            Prediction statistics with uncertainty
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Only works with PyTorch models that have dropout
        if not hasattr(model, 'forward'):
            # For non-neural models, use ensemble for uncertainty
            return self.predict_ensemble(data, [model_name])

        model.train()  # Enable dropout
        predictions = []

        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.predict_single(model_name, data)
                if len(pred) > 0:
                    predictions.append(pred[-1] if len(pred.shape) == 1 else pred.mean())

        model.eval()  # Restore eval mode

        if predictions:
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence_interval = np.percentile(predictions, [2.5, 97.5])

            return {
                'prediction': mean_pred,
                'uncertainty': std_pred,
                'confidence_interval': confidence_interval,
                'samples': predictions
            }
        else:
            return {
                'prediction': None,
                'uncertainty': float('inf'),
                'confidence_interval': [None, None],
                'samples': []
            }

    def batch_predict(self, data_batch: List[Union[pd.DataFrame, np.ndarray]],
                     model_name: str) -> List[np.ndarray]:
        """
        Batch prediction for multiple data points

        Args:
            data_batch: List of data points
            model_name: Model to use

        Returns:
            List of predictions
        """
        predictions = []

        for data in data_batch:
            try:
                pred = self.predict_single(model_name, data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Batch prediction failed for item: {e}")
                predictions.append(np.array([]))

        return predictions

    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models

        Args:
            model_name: Model name

        Returns:
            Feature importance dictionary
        """
        if model_name not in self.models:
            return None

        model = self.models[model_name]

        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = getattr(self.data_processor, 'feature_names', None)
            if feature_names:
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(),
                                               key=lambda x: x[1], reverse=True))
                return sorted_importance
            else:
                return dict(enumerate(model.feature_importances_))

        return None

    def explain_prediction(self, data: Union[pd.DataFrame, np.ndarray],
                          model_name: str) -> Dict[str, Any]:
        """
        Explain individual prediction (simplified SHAP-like explanation)

        Args:
            data: Input data point
            model_name: Model to explain

        Returns:
            Explanation dictionary
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not loaded"}

        try:
            # Get base prediction
            prediction = self.predict_single(model_name, data)

            # Get feature importance if available
            feature_importance = self.get_feature_importance(model_name)

            # For tree-based models, we can get more detailed explanations
            explanation = {
                'prediction': prediction,
                'model_type': type(self.models[model_name]).__name__,
                'feature_importance': feature_importance
            }

            return explanation

        except Exception as e:
            return {"error": f"Explanation failed: {e}"}

    def model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of loaded models"""
        summary = {}

        for model_name in self.models.keys():
            try:
                model_info = self.registry.get_model_info(model_name)
                summary[model_name] = {
                    'version': model_info.get('version', 'unknown'),
                    'model_type': model_info.get('model_type', 'unknown'),
                    'metrics': model_info.get('metrics', {}),
                    'created_at': model_info.get('created_at', 'unknown')
                }
            except Exception as e:
                summary[model_name] = {'error': str(e)}

        return summary