"""
TimesFM Integration for Advanced Time-Series Forecasting
Google Research TimesFM (200M parameter foundation model) for volatility and price forecasting

Note: Requires Python 3.10-3.11 for full TimesFM support
For Python 3.12+, use fallback forecasting or install in separate environment
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import TimesFM (will fail gracefully if not available)
try:
    import timesfm
    TIMESFM_AVAILABLE = True
    logger.info("TimesFM successfully imported")
except ImportError as e:
    TIMESFM_AVAILABLE = False
    logger.warning(f"TimesFM not available: {e}. Using fallback forecasting.")


@dataclass
class ForecastResult:
    """Time-series forecast result with uncertainty"""
    horizon: int
    point_forecast: np.ndarray
    quantile_forecasts: Dict[float, np.ndarray]  # {quantile: forecast}
    confidence_interval_95: Tuple[np.ndarray, np.ndarray]  # (lower, upper)
    forecast_timestamps: List[datetime]
    model_used: str
    metadata: Dict[str, Any]


@dataclass
class VolatilityForecast:
    """VIX/Volatility-specific forecast"""
    vix_forecast: np.ndarray
    spike_probability: float  # P(VIX > 30 in forecast horizon)
    crisis_probability: float  # P(VIX > 40 in forecast horizon)
    regime_forecast: List[str]  # ['calm', 'normal', 'volatile', 'crisis']
    horizon_hours: int
    confidence: float


class TimesFMForecaster:
    """
    TimesFM-based forecasting wrapper for trader-ai system
    Provides multi-horizon volatility and price forecasting with quantile estimates
    """

    def __init__(self,
                 model_version: str = "2.5",
                 checkpoint_path: Optional[str] = None,
                 use_fallback: bool = True):
        """
        Initialize TimesFM forecaster

        Args:
            model_version: TimesFM version ("2.5" = 200M params)
            checkpoint_path: Optional custom checkpoint path
            use_fallback: Use fallback forecasting if TimesFM unavailable
        """
        self.model_version = model_version
        self.use_fallback = use_fallback
        self.model = None
        self.is_initialized = False

        # VIX thresholds for regime classification
        self.vix_thresholds = {
            'calm': (0, 15),
            'normal': (15, 25),
            'volatile': (25, 40),
            'crisis': (40, 100)
        }

        # Initialize model if available
        if TIMESFM_AVAILABLE:
            self._initialize_timesfm(checkpoint_path)
        elif use_fallback:
            logger.info("Initializing fallback forecasting (sklearn-based)")
            self._initialize_fallback()
        else:
            raise RuntimeError("TimesFM not available and fallback disabled")

    def _initialize_timesfm(self, checkpoint_path: Optional[str] = None):
        """Initialize TimesFM model"""
        try:
            # Load TimesFM 2.5 (200M parameters)
            self.model = timesfm.TimesFM_2p5_200M_torch()

            # Load checkpoint
            if checkpoint_path:
                self.model.load_checkpoint(checkpoint_path)
            else:
                self.model.load_checkpoint()  # Load default from HuggingFace

            # Compile with optimal config for financial data
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,  # Use 1024 historical points (can go up to 16k)
                    max_horizon=256,   # Forecast up to 256 steps ahead
                    normalize_inputs=True,  # Normalize for stability
                    use_continuous_quantile_head=True,  # Enable quantile forecasting
                    flip_invariance=False  # Directional forecasting matters
                )
            )

            self.is_initialized = True
            logger.info(f"TimesFM {self.model_version} initialized successfully (200M params)")

        except Exception as e:
            logger.error(f"Failed to initialize TimesFM: {e}")
            if self.use_fallback:
                self._initialize_fallback()
            else:
                raise

    def _initialize_fallback(self):
        """Initialize fallback forecasting using simpler models"""
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        self.fallback_models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        }
        self.is_initialized = True
        logger.info("Fallback forecasting initialized")

    def forecast_volatility(self,
                           vix_history: np.ndarray,
                           horizon_hours: int = 24,
                           quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> VolatilityForecast:
        """
        Forecast VIX volatility for specified horizon

        Args:
            vix_history: Historical VIX values (at least 100 points recommended)
            horizon_hours: Forecast horizon in hours (max 1000)
            quantiles: Quantiles to forecast

        Returns:
            VolatilityForecast with point estimates, quantiles, and probabilities
        """
        if not self.is_initialized:
            raise RuntimeError("Forecaster not initialized")

        # Prepare input
        if len(vix_history) < 50:
            logger.warning(f"VIX history too short ({len(vix_history)}), padding with mean")
            mean_vix = np.mean(vix_history)
            padding = np.full(50 - len(vix_history), mean_vix)
            vix_history = np.concatenate([padding, vix_history])

        # Forecast using TimesFM or fallback
        if TIMESFM_AVAILABLE and self.model is not None:
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon_hours,
                inputs=[vix_history],
                quantiles=quantiles
            )

            # Extract forecasts
            vix_forecast = point_forecast[0]
            quantile_dict = {q: quantile_forecast[i][0] for i, q in enumerate(quantiles)}

        else:
            # Fallback: Simple AR forecast
            vix_forecast, quantile_dict = self._fallback_volatility_forecast(
                vix_history, horizon_hours, quantiles
            )

        # Calculate spike probabilities
        spike_prob = np.mean(vix_forecast > 30)
        crisis_prob = np.mean(vix_forecast > 40)

        # Forecast regimes
        regime_forecast = [self._classify_vix_regime(v) for v in vix_forecast]

        # Calculate confidence based on quantile spread
        q95 = quantile_dict.get(0.95, vix_forecast)
        q05 = quantile_dict.get(0.05, vix_forecast)
        spread = np.mean(q95 - q05)
        confidence = 1.0 / (1.0 + spread / np.mean(vix_history))

        return VolatilityForecast(
            vix_forecast=vix_forecast,
            spike_probability=float(spike_prob),
            crisis_probability=float(crisis_prob),
            regime_forecast=regime_forecast,
            horizon_hours=horizon_hours,
            confidence=float(confidence)
        )

    def forecast_price(self,
                      price_history: np.ndarray,
                      horizon: int = 12,
                      return_quantiles: bool = True) -> ForecastResult:
        """
        Forecast price movements with uncertainty estimates

        Args:
            price_history: Historical price data
            horizon: Forecast horizon (number of steps)
            return_quantiles: Whether to return quantile forecasts

        Returns:
            ForecastResult with point and quantile forecasts
        """
        if not self.is_initialized:
            raise RuntimeError("Forecaster not initialized")

        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95] if return_quantiles else [0.5]

        if TIMESFM_AVAILABLE and self.model is not None:
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=[price_history],
                quantiles=quantiles
            )

            point = point_forecast[0]
            quantile_dict = {q: quantile_forecast[i][0] for i, q in enumerate(quantiles)}

        else:
            # Fallback forecast
            point, quantile_dict = self._fallback_price_forecast(
                price_history, horizon, quantiles
            )

        # Calculate 95% confidence interval
        ci_95 = (quantile_dict[0.05], quantile_dict[0.95]) if return_quantiles else (point, point)

        # Generate forecast timestamps (assuming hourly data)
        base_time = datetime.now()
        forecast_times = [base_time + timedelta(hours=i) for i in range(1, horizon + 1)]

        return ForecastResult(
            horizon=horizon,
            point_forecast=point,
            quantile_forecasts=quantile_dict,
            confidence_interval_95=ci_95,
            forecast_timestamps=forecast_times,
            model_used="TimesFM-2.5" if TIMESFM_AVAILABLE else "Fallback",
            metadata={'quantiles': quantiles}
        )

    def multi_horizon_forecast(self,
                               data: np.ndarray,
                               horizons: List[int] = [1, 6, 24, 168]) -> Dict[int, ForecastResult]:
        """
        Generate forecasts for multiple horizons

        Args:
            data: Historical data
            horizons: List of forecast horizons (in hours)

        Returns:
            Dictionary mapping horizon to ForecastResult
        """
        results = {}

        for horizon in horizons:
            try:
                result = self.forecast_price(data, horizon=horizon)
                results[horizon] = result
            except Exception as e:
                logger.error(f"Failed to forecast horizon {horizon}: {e}")

        return results

    def _classify_vix_regime(self, vix_value: float) -> str:
        """Classify VIX value into regime"""
        for regime, (low, high) in self.vix_thresholds.items():
            if low <= vix_value < high:
                return regime
        return 'crisis'  # Default for extreme values

    def _fallback_volatility_forecast(self,
                                      vix_history: np.ndarray,
                                      horizon: int,
                                      quantiles: List[float]) -> Tuple[np.ndarray, Dict]:
        """Fallback volatility forecasting using AR model"""
        from sklearn.linear_model import Ridge

        # Simple AR(5) model
        X, y = [], []
        for i in range(5, len(vix_history)):
            X.append(vix_history[i-5:i])
            y.append(vix_history[i])

        X = np.array(X)
        y = np.array(y)

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Forecast iteratively
        forecast = []
        current = vix_history[-5:].copy()

        for _ in range(horizon):
            pred = model.predict(current.reshape(1, -1))[0]
            forecast.append(pred)
            current = np.roll(current, -1)
            current[-1] = pred

        forecast = np.array(forecast)

        # Simple quantile estimation (add noise)
        std = np.std(y - model.predict(X))
        quantile_dict = {}
        for q in quantiles:
            z_score = np.percentile(np.random.randn(1000), q * 100)
            quantile_dict[q] = forecast + z_score * std

        return forecast, quantile_dict

    def _fallback_price_forecast(self,
                                 price_history: np.ndarray,
                                 horizon: int,
                                 quantiles: List[float]) -> Tuple[np.ndarray, Dict]:
        """Fallback price forecasting"""
        # Use last value + small random walk
        last_price = price_history[-1]
        returns_std = np.std(np.diff(price_history) / price_history[:-1])

        forecast = []
        for _ in range(horizon):
            drift = np.random.normal(0, returns_std)
            forecast.append(last_price * (1 + drift))

        forecast = np.array(forecast)

        # Quantile estimates
        quantile_dict = {}
        for q in quantiles:
            z = np.percentile(np.random.randn(1000), q * 100)
            quantile_dict[q] = forecast + z * returns_std * last_price

        return forecast, quantile_dict


if __name__ == "__main__":
    # Test TimesFM forecaster
    print("=== Testing TimesFM Forecaster ===")

    # Create synthetic VIX data
    vix_history = np.random.normal(20, 5, 200)  # 200 hours of VIX data
    vix_history = np.clip(vix_history, 10, 50)  # Clip to reasonable range

    forecaster = TimesFMForecaster(use_fallback=True)

    # Test volatility forecast
    print("\n1. Testing 24-hour VIX forecast...")
    vol_forecast = forecaster.forecast_volatility(vix_history, horizon_hours=24)
    print(f"   Spike probability (VIX>30): {vol_forecast.spike_probability:.2%}")
    print(f"   Crisis probability (VIX>40): {vol_forecast.crisis_probability:.2%}")
    print(f"   Confidence: {vol_forecast.confidence:.2f}")
    print(f"   Regime forecast: {vol_forecast.regime_forecast[:5]}...")

    # Test price forecast
    print("\n2. Testing price forecast...")
    price_history = np.random.normal(400, 10, 200)  # SPY-like prices
    price_forecast = forecaster.forecast_price(price_history, horizon=12)
    print(f"   Point forecast: {price_forecast.point_forecast[:5]}")
    print(f"   95% CI: [{price_forecast.confidence_interval_95[0][:3]}, {price_forecast.confidence_interval_95[1][:3]}]")
    print(f"   Model used: {price_forecast.model_used}")

    # Test multi-horizon
    print("\n3. Testing multi-horizon forecast...")
    multi_results = forecaster.multi_horizon_forecast(vix_history, horizons=[1, 6, 24])
    for horizon, result in multi_results.items():
        print(f"   {horizon}h forecast: {result.point_forecast[0]:.2f} VIX")

    print("\n=== TimesFM Forecaster Test Complete ===")