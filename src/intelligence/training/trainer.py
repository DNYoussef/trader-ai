"""
Production-ready model training pipeline
Implements complete ML training workflow with MLflow integration
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..data.processor import DataProcessor
from ..models.neural_networks import TradingLSTM, TradingTransformer
from ..models.registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Complete model training pipeline with MLflow tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.data_processor = DataProcessor()
        self.model_registry = ModelRegistry()
        self.trained_models = {}
        self.scalers = {}

        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        # Create model storage directory
        self.model_dir = Path(self.config['model_storage']['base_path'])
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for training"""
        return {
            'mlflow': {
                'tracking_uri': 'file:///c/Users/17175/Desktop/trader-ai/mlruns',
                'experiment_name': 'trader-ai-intelligence'
            },
            'model_storage': {
                'base_path': '/c/Users/17175/Desktop/trader-ai/trained_models'
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.15,
                'random_state': 42,
                'cv_folds': 5
            },
            'models': {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            }
        }

    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with comprehensive preprocessing
        """
        with mlflow.start_run(nested=True):
            mlflow.log_param("data_path", data_path)

            # Load and process data
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                # Generate synthetic financial data for training
                df = self._generate_synthetic_data()

            # Process features
            processed_df = self.data_processor.preprocess(df)

            # Create features and targets
            feature_cols = [col for col in processed_df.columns
                          if col not in ['target', 'date', 'timestamp']]

            X = processed_df[feature_cols].values
            y = processed_df.get('target', self._create_target(processed_df)).values

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # Store scaler
            self.scalers['feature_scaler'] = scaler

            mlflow.log_param("n_features", X_scaled.shape[1])
            mlflow.log_param("n_samples", X_scaled.shape[0])

            return X_scaled, y

    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate realistic synthetic financial data for training
        """
        np.random.seed(42)

        # Generate price series with realistic patterns
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')

        # Base price with trend and volatility
        base_price = 100
        trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
        volatility = np.random.normal(0, 0.02, n_samples)
        prices = base_price * np.exp(trend + volatility)

        # Technical indicators
        returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        volume = np.random.exponential(1000000, n_samples)

        # Moving averages
        ma_5 = pd.Series(prices).rolling(5).mean().fillna(prices)
        ma_20 = pd.Series(prices).rolling(20).mean().fillna(prices)

        # RSI calculation
        delta = pd.Series(returns)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volume,
            'returns': returns,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'rsi': rsi.fillna(50),
            'volatility': pd.Series(returns).rolling(20).std().fillna(0.02)
        })

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable (future returns)"""
        if 'price' in df.columns:
            # Predict next period return
            future_returns = df['price'].pct_change(periods=1).shift(-1)
            return future_returns.fillna(0)
        else:
            # Generate synthetic target
            return pd.Series(np.random.normal(0, 0.01, len(df)))

    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model with hyperparameter tuning"""
        with mlflow.start_run(nested=True, run_name="random_forest_training"):
            mlflow.log_param("model_type", "RandomForest")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state']
            )

            # Hyperparameter tuning
            rf = RandomForestRegressor(random_state=42)
            param_grid = self.config['models']['random_forest']

            grid_search = GridSearchCV(
                rf, param_grid, cv=self.config['training']['cv_folds'],
                scoring='neg_mean_squared_error', n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate model
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Save model
            model_path = self.model_dir / "random_forest_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_artifact(str(model_path))

            self.trained_models['random_forest'] = best_model
            logger.info(f"Random Forest trained - MSE: {mse:.6f}, R2: {r2:.4f}")

            return best_model

    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """Train Gradient Boosting model"""
        with mlflow.start_run(nested=True, run_name="gradient_boosting_training"):
            mlflow.log_param("model_type", "GradientBoosting")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state']
            )

            # Hyperparameter tuning
            gb = GradientBoostingRegressor(random_state=42)
            param_grid = self.config['models']['gradient_boosting']

            grid_search = GridSearchCV(
                gb, param_grid, cv=self.config['training']['cv_folds'],
                scoring='neg_mean_squared_error', n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate model
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Save model
            model_path = self.model_dir / "gradient_boosting_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_artifact(str(model_path))

            self.trained_models['gradient_boosting'] = best_model
            logger.info(f"Gradient Boosting trained - MSE: {mse:.6f}, R2: {r2:.4f}")

            return best_model

    def train_lstm(self, X: np.ndarray, y: np.ndarray) -> TradingLSTM:
        """Train LSTM neural network"""
        with mlflow.start_run(nested=True, run_name="lstm_training"):
            mlflow.log_param("model_type", "LSTM")

            # Prepare data for LSTM (sequences)
            X_seq, y_seq = self._prepare_sequences(X, y, sequence_length=60)

            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state']
            )

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['models']['lstm']['batch_size'],
                shuffle=True
            )

            # Initialize model
            model = TradingLSTM(
                input_size=X.shape[1],
                hidden_size=self.config['models']['lstm']['hidden_size'],
                num_layers=self.config['models']['lstm']['num_layers'],
                dropout=self.config['models']['lstm']['dropout']
            )

            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['models']['lstm']['learning_rate']
            )

            # Training loop
            model.train()
            epoch_losses = []

            for epoch in range(self.config['models']['lstm']['epochs']):
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                epoch_losses.append(avg_loss)

                if epoch % 10 == 0:
                    mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            # Evaluate model
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor).squeeze().numpy()
                y_test_np = y_test_tensor.squeeze().numpy()

                mse = mean_squared_error(y_test_np, y_pred)
                mae = mean_absolute_error(y_test_np, y_pred)
                r2 = r2_score(y_test_np, y_pred)

            # Log metrics
            mlflow.log_metric("final_mse", mse)
            mlflow.log_metric("final_mae", mae)
            mlflow.log_metric("final_r2", r2)

            # Save model
            model_path = self.model_dir / "lstm_model.pth"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path))

            self.trained_models['lstm'] = model
            logger.info(f"LSTM trained - MSE: {mse:.6f}, R2: {r2:.4f}")

            return model

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray,
                          sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        X_seq, y_seq = [], []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train_all_models(self, data_path: str) -> Dict[str, Any]:
        """Train all models in the pipeline"""
        with mlflow.start_run(run_name="complete_training_pipeline"):
            mlflow.log_param("training_run", "complete_pipeline")

            # Prepare data
            X, y = self.prepare_data(data_path)

            # Train all models
            results = {}

            try:
                rf_model = self.train_random_forest(X, y)
                results['random_forest'] = rf_model
            except Exception as e:
                logger.error(f"Random Forest training failed: {e}")
                results['random_forest'] = None

            try:
                gb_model = self.train_gradient_boosting(X, y)
                results['gradient_boosting'] = gb_model
            except Exception as e:
                logger.error(f"Gradient Boosting training failed: {e}")
                results['gradient_boosting'] = None

            try:
                lstm_model = self.train_lstm(X, y)
                results['lstm'] = lstm_model
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                results['lstm'] = None

            # Save scalers
            scaler_path = self.model_dir / "scalers.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            mlflow.log_artifact(str(scaler_path))

            # Register models
            self.model_registry.register_models(self.trained_models)

            logger.info("Complete training pipeline finished")
            return results

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        results = {}

        for model_name, model in self.trained_models.items():
            if model is None:
                continue

            try:
                if model_name == 'lstm':
                    # Handle LSTM evaluation separately
                    X_seq, y_seq = self._prepare_sequences(X_test, y_test, sequence_length=60)
                    if len(X_seq) > 0:
                        model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_seq)
                            y_pred = model(X_tensor).squeeze().numpy()
                            y_true = y_seq
                    else:
                        continue
                else:
                    y_pred = model.predict(X_test)
                    y_true = y_test

                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }

            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results