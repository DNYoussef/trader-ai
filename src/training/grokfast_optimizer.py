"""
GrokFast Gradient Filtering for Anti-Memorization
Implements gradient filtering to amplify slow gradients (generalization) and suppress fast gradients (overfitting)
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Optional, Literal, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GrokFastOptimizer:
    """
    GrokFast gradient filtering optimizer wrapper
    Prevents memorization by amplifying generalization-inducing slow gradients
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 filter_type: Literal['ema', 'ma'] = 'ema',
                 alpha: float = 0.98,
                 lamb: float = 2.0,
                 window_size: int = 100,
                 warmup_steps: int = 50):
        """
        Initialize GrokFast optimizer wrapper

        Args:
            optimizer: Base PyTorch optimizer
            filter_type: 'ema' for exponential moving average, 'ma' for moving average
            alpha: EMA decay factor (for EMA filter)
            lamb: Amplification factor for slow gradients
            window_size: Window size for moving average (for MA filter)
            warmup_steps: Steps before applying filtering
        """
        self.optimizer = optimizer
        self.filter_type = filter_type
        self.alpha = alpha
        self.lamb = lamb
        self.window_size = window_size
        self.warmup_steps = warmup_steps

        # Gradient storage for filtering
        self.ema_grads: Optional[Dict[str, torch.Tensor]] = None
        self.ma_grads: Optional[Dict[str, deque]] = None
        self.step_count = 0

        logger.info(f"GrokFast optimizer initialized: {filter_type}, alpha={alpha}, lamb={lamb}")

    def gradfilter_ema(self, model: nn.Module) -> None:
        """
        Apply exponential moving average gradient filtering
        Amplifies slow-varying gradient components for better generalization
        """
        if self.ema_grads is None:
            # Initialize EMA gradients
            self.ema_grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.ema_grads[name] = param.grad.data.detach().clone()
            return

        # Apply EMA filtering and amplification
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Update EMA of gradients (slow component)
                self.ema_grads[name] = (self.alpha * self.ema_grads[name] +
                                      (1 - self.alpha) * param.grad.data.detach())

                # Amplify slow gradient component
                param.grad.data = param.grad.data + self.lamb * self.ema_grads[name]

    def gradfilter_ma(self, model: nn.Module) -> None:
        """
        Apply moving average gradient filtering
        Uses sliding window to identify and amplify slow gradient patterns
        """
        if self.ma_grads is None:
            # Initialize MA gradient storage
            self.ma_grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.ma_grads[name] = deque(maxlen=self.window_size)

        # Collect gradients in sliding window
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.ma_grads[name].append(param.grad.data.detach().clone())

        # Apply filtering after warmup
        if self.step_count >= self.warmup_steps:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and name in self.ma_grads:
                    if len(self.ma_grads[name]) >= self.window_size:
                        # Calculate moving average (slow component)
                        avg_grad = sum(self.ma_grads[name]) / len(self.ma_grads[name])

                        # Amplify slow gradient component
                        param.grad.data = param.grad.data + self.lamb * avg_grad

    def step(self, model: nn.Module, closure=None):
        """
        Perform optimization step with GrokFast gradient filtering

        Args:
            model: PyTorch model being optimized
            closure: Optional closure for optimizer
        """
        self.step_count += 1

        # Apply gradient filtering after warmup
        if self.step_count > self.warmup_steps:
            if self.filter_type == 'ema':
                self.gradfilter_ema(model)
            elif self.filter_type == 'ma':
                self.gradfilter_ma(model)
            else:
                raise ValueError(f"Unknown filter type: {self.filter_type}")

        # Perform optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        """Zero gradients in base optimizer"""
        return self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state including GrokFast components"""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'filter_type': self.filter_type,
            'alpha': self.alpha,
            'lamb': self.lamb,
            'window_size': self.window_size,
            'step_count': self.step_count,
            'ema_grads': self.ema_grads,
            'ma_grads': self.ma_grads
        }
        return state

    def load_state_dict(self, state_dict):
        """Load optimizer state including GrokFast components"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.filter_type = state_dict['filter_type']
        self.alpha = state_dict['alpha']
        self.lamb = state_dict['lamb']
        self.window_size = state_dict['window_size']
        self.step_count = state_dict['step_count']
        self.ema_grads = state_dict['ema_grads']
        self.ma_grads = state_dict['ma_grads']

class AntiMemorizationValidator:
    """
    Validates that models learn generalizable patterns rather than memorizing
    Uses noise injection and consistency testing
    """

    def __init__(self, noise_std: float = 0.02, noise_rate: float = 0.3):
        """
        Initialize anti-memorization validator

        Args:
            noise_std: Standard deviation of noise injection
            noise_rate: Fraction of samples to add noise to
        """
        self.noise_std = noise_std
        self.noise_rate = noise_rate

    def inject_feature_noise(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Inject controlled noise to market features
        Small changes that shouldn't affect optimal strategy choice
        """
        noisy_features = features.copy()

        # Define noise-tolerant features with appropriate scales
        noise_config = {
            'vix_level': 1.5,           # ±1.5 VIX points
            'spy_returns_5d': 0.005,    # ±0.5% returns
            'spy_returns_20d': 0.003,   # ±0.3% returns
            'volume_ratio': 0.1,        # ±10% volume
            'market_breadth': 0.05,     # ±5% breadth
            'correlation': 0.05,        # ±5% correlation
            'put_call_ratio': 0.1,      # ±10% put/call
            'gini_coefficient': 0.01,   # ±1% inequality
            'sector_dispersion': 0.002, # ±0.2% dispersion
            'signal_quality_score': 0.05 # ±5% quality
        }

        for feature, noise_scale in noise_config.items():
            if feature in noisy_features:
                noise = np.random.normal(0, noise_scale)
                noisy_features[feature] += noise

                # Apply reasonable bounds
                if feature == 'vix_level':
                    noisy_features[feature] = max(10, min(80, noisy_features[feature]))
                elif feature in ['market_breadth', 'correlation', 'signal_quality_score']:
                    noisy_features[feature] = max(0, min(1, noisy_features[feature]))
                elif feature == 'volume_ratio':
                    noisy_features[feature] = max(0.5, min(5.0, noisy_features[feature]))
                elif feature == 'put_call_ratio':
                    noisy_features[feature] = max(0.3, min(3.0, noisy_features[feature]))

        return noisy_features

    def validate_memorization_resistance(self,
                                       model_predictions: Dict[str, Any],
                                       clean_features: Dict[str, float],
                                       num_noise_tests: int = 10) -> Dict[str, Any]:
        """
        Test model's resistance to memorization using noise injection

        Args:
            model_predictions: Function that takes features and returns strategy predictions
            clean_features: Original market features
            num_noise_tests: Number of noise injection tests

        Returns:
            Memorization resistance analysis
        """
        # Get prediction on clean data
        clean_prediction = model_predictions(clean_features)

        # Test with noise injection
        noisy_predictions = []
        for _ in range(num_noise_tests):
            noisy_features = self.inject_feature_noise(clean_features)
            noisy_pred = model_predictions(noisy_features)
            noisy_predictions.append(noisy_pred)

        # Analyze consistency
        consistency_rate = sum(1 for pred in noisy_predictions
                             if pred == clean_prediction) / num_noise_tests

        # Calculate prediction stability
        if hasattr(clean_prediction, '__iter__') and isinstance(clean_prediction, (list, np.ndarray)):
            # For probability distributions - ensure numeric data
            try:
                numeric_predictions = [np.array(pred, dtype=float) for pred in noisy_predictions if isinstance(pred, (list, np.ndarray))]
                if len(numeric_predictions) > 0:
                    pred_variance = np.var(numeric_predictions, axis=0)
                    stability_score = 1.0 - np.mean(pred_variance)
                else:
                    stability_score = consistency_rate
            except (ValueError, TypeError):
                # Fallback for non-numeric predictions
                stability_score = consistency_rate
        else:
            # For single predictions (including strings)
            stability_score = consistency_rate

        return {
            'memorization_resistant': consistency_rate > 0.7,
            'consistency_rate': consistency_rate,
            'stability_score': stability_score,
            'clean_prediction': clean_prediction,
            'noisy_predictions': noisy_predictions,
            'noise_tests_conducted': num_noise_tests
        }

class StrategyLearningOrchestrator:
    """
    Orchestrates GrokFast-enhanced strategy learning with anti-memorization
    """

    def __init__(self,
                 model: nn.Module,
                 base_optimizer: torch.optim.Optimizer,
                 grokfast_config: Dict[str, Any] = None):
        """
        Initialize learning orchestrator

        Args:
            model: Strategy selection model
            base_optimizer: Base PyTorch optimizer
            grokfast_config: GrokFast configuration parameters
        """
        self.model = model

        # Default GrokFast configuration
        if grokfast_config is None:
            grokfast_config = {
                'filter_type': 'ema',
                'alpha': 0.98,
                'lamb': 2.0,
                'window_size': 100,
                'warmup_steps': 50
            }

        # Initialize GrokFast optimizer
        self.optimizer = GrokFastOptimizer(base_optimizer, **grokfast_config)

        # Initialize anti-memorization validator
        self.validator = AntiMemorizationValidator()

        # Training metrics
        self.training_metrics = {
            'generalization_score': [],
            'memorization_resistance': [],
            'gradient_amplification': [],
            'loss_trajectory': []
        }

    def training_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform single training step with GrokFast gradient filtering

        Args:
            batch_data: Training batch with market features and strategy labels

        Returns:
            Training step metrics
        """
        self.optimizer.zero_grad()

        # Forward pass
        features = batch_data['features']
        targets = batch_data['targets']

        predictions = self.model(features)
        loss = self._calculate_loss(predictions, targets)

        # Backward pass
        loss.backward()

        # GrokFast gradient filtering and optimization step
        self.optimizer.step(self.model)

        # Calculate metrics
        with torch.no_grad():
            generalization_score = self._estimate_generalization(predictions, targets)

        return {
            'loss': loss.item(),
            'generalization_score': generalization_score
        }

    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss for strategy selection"""
        # Use cross-entropy for strategy classification
        return nn.CrossEntropyLoss()(predictions, targets)

    def _estimate_generalization(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Estimate generalization quality during training"""
        with torch.no_grad():
            accuracy = (predictions.argmax(dim=1) == targets).float().mean()

            # Additional generalization metrics
            prediction_entropy = -torch.sum(torch.softmax(predictions, dim=1) *
                                          torch.log_softmax(predictions, dim=1), dim=1).mean()

            # Combine accuracy and entropy for generalization score
            generalization_score = (accuracy.item() + prediction_entropy.item() / 3.0) / 2.0

        return generalization_score

    def validate_epoch(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model with anti-memorization testing

        Args:
            validation_data: Validation dataset

        Returns:
            Comprehensive validation metrics
        """
        self.model.eval()

        total_loss = 0
        total_accuracy = 0
        memorization_tests = []

        with torch.no_grad():
            for batch in validation_data:
                # Standard validation
                features = batch['features']
                targets = batch['targets']
                predictions = self.model(features)
                loss = self._calculate_loss(predictions, targets)
                accuracy = (predictions.argmax(dim=1) == targets).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()

                # Anti-memorization testing on subset
                for i in range(min(5, len(features))):  # Test 5 samples per batch
                    sample_features = {k: v[i].item() if hasattr(v[i], 'item') else v[i]
                                     for k, v in features.items()}

                    def model_pred_func(feat_dict):
                        # Convert features back to tensor for model
                        feat_tensor = torch.tensor([list(feat_dict.values())],
                                                 dtype=torch.float32)
                        with torch.no_grad():
                            pred = self.model(feat_tensor)
                            return pred.argmax(dim=1).item()

                    memorization_test = self.validator.validate_memorization_resistance(
                        model_pred_func, sample_features, num_noise_tests=5
                    )
                    memorization_tests.append(memorization_test)

        # Aggregate memorization resistance
        avg_consistency = np.mean([test['consistency_rate'] for test in memorization_tests])
        memorization_resistant = avg_consistency > 0.7

        self.model.train()

        return {
            'validation_loss': total_loss / len(validation_data),
            'validation_accuracy': total_accuracy / len(validation_data),
            'memorization_resistance': memorization_resistant,
            'avg_noise_consistency': avg_consistency,
            'grokfast_active': self.optimizer.step_count > self.optimizer.warmup_steps
        }

if __name__ == "__main__":
    # Test GrokFast implementation
    print("=== Testing GrokFast Gradient Filtering ===")

    # Create simple test model
    test_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 8)  # 8 strategies
    )

    # Create base optimizer
    base_opt = torch.optim.Adam(test_model.parameters(), lr=0.001)

    # Wrap with GrokFast
    grokfast_opt = GrokFastOptimizer(base_opt, filter_type='ema', alpha=0.98, lamb=2.0)

    print(f"GrokFast optimizer created: {grokfast_opt.filter_type}")
    print(f"Parameters: alpha={grokfast_opt.alpha}, lamb={grokfast_opt.lamb}")

    # Test anti-memorization validator
    validator = AntiMemorizationValidator()

    test_features = {
        'vix_level': 22.5,
        'spy_returns_5d': -0.02,
        'market_breadth': 0.4,
        'volume_ratio': 1.5
    }

    noisy_features = validator.inject_feature_noise(test_features)
    print(f"\\nOriginal features: {test_features}")
    print(f"Noisy features: {noisy_features}")

    print("\\n=== GrokFast Implementation Ready ===")