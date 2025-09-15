"""
Taleb Antifragility Engine - REAL IMPLEMENTATION
Implements actual Nassim Nicholas Taleb antifragility concepts:
- Barbell Strategy (80% safe, 20% convex)
- Extreme Value Theory for tail risk
- Convexity optimization
- Antifragile position management

NO THEATER - REAL MATHEMATICS
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from statistics import stdev, mean

# Set high precision for financial calculations
getcontext().prec = 10

logger = logging.getLogger(__name__)

@dataclass
class BarbellAllocation:
    """Barbell allocation following Taleb's 80/20 rule"""
    safe_allocation: float  # 80% safe assets (cash, treasuries)
    risky_allocation: float  # 20% convex opportunities
    safe_instruments: List[str]
    risky_instruments: List[str]
    rebalance_threshold: float = 0.05  # 5% drift threshold

@dataclass
class ConvexityMetrics:
    """Convexity assessment for position optimization"""
    symbol: str
    convexity_score: float  # Positive = convex, Negative = concave
    gamma: float  # Second derivative of price sensitivity
    vega: float  # Volatility sensitivity
    tail_risk_potential: float  # EVT-based tail risk score
    kelly_fraction: float  # Optimal position size

@dataclass
class TailRiskModel:
    """Extreme Value Theory model for tail events"""
    symbol: str
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall: float  # Conditional VaR
    tail_index: float  # Heavy tail parameter (xi in EVT)
    scale_parameter: float  # Scale parameter (sigma in EVT)

class AntifragilityEngine:
    """
    Nassim Taleb's Antifragility Engine

    Core Principles:
    1. Barbell Strategy: 80% safe, 20% convex
    2. Benefit from disorder/volatility
    3. Extreme Value Theory for tail modeling
    4. Position sizing via Kelly Criterion with convexity adjustment
    5. Antifragile rebalancing during volatility spikes
    """

    def __init__(self, portfolio_value: float, risk_tolerance: float = 0.02):
        """
        Initialize antifragility engine

        Args:
            portfolio_value: Total portfolio value
            risk_tolerance: Max acceptable loss per position (2% default)
        """
        self.portfolio_value = portfolio_value
        self.risk_tolerance = risk_tolerance
        self.volatility_lookback = 252  # 1 year of daily returns
        self.evt_threshold_percentile = 95  # Top 5% for extreme events

        # Barbell configuration - Taleb's actual allocation
        self.barbell_config = BarbellAllocation(
            safe_allocation=0.80,  # 80% safe
            risky_allocation=0.20,  # 20% convex
            safe_instruments=['CASH', 'SHY', 'TLT'],  # Cash, short/long treasuries
            risky_instruments=['QQQ', 'ARKK', 'TSLA'],  # High-growth/volatile assets
            rebalance_threshold=0.05
        )

        logger.info(f"Antifragility Engine initialized - Portfolio: ${portfolio_value:,.2f}")

    def calculate_barbell_allocation(self, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate REAL barbell allocation per Taleb methodology

        Returns:
            Dict with safe_amount and risky_amount in dollars
        """
        safe_amount = portfolio_value * self.barbell_config.safe_allocation
        risky_amount = portfolio_value * self.barbell_config.risky_allocation

        allocation = {
            'safe_amount': safe_amount,
            'risky_amount': risky_amount,
            'safe_percentage': self.barbell_config.safe_allocation * 100,
            'risky_percentage': self.barbell_config.risky_allocation * 100,
            'safe_instruments': self.barbell_config.safe_instruments,
            'risky_instruments': self.barbell_config.risky_instruments,
            'total_allocated': safe_amount + risky_amount,
            'allocation_date': datetime.now().isoformat()
        }

        logger.info(f"Barbell allocation calculated: Safe=${safe_amount:,.2f} Risky=${risky_amount:,.2f}")
        return allocation

    def assess_convexity(self, symbol: str, price_history: List[float],
                        position_size: float) -> ConvexityMetrics:
        """
        Assess convexity of a position using REAL mathematical analysis

        Convexity = d²P/dS² where P is position value, S is underlying price
        Positive convexity = benefits from volatility (antifragile)
        Negative convexity = hurt by volatility (fragile)

        Args:
            symbol: Asset symbol
            price_history: Historical prices for analysis
            position_size: Current position size in dollars

        Returns:
            ConvexityMetrics with mathematical assessment
        """
        if len(price_history) < 50:
            raise ValueError(f"Insufficient price history for {symbol}: need >= 50 points")

        prices = np.array(price_history)
        returns = np.diff(np.log(prices))  # Log returns

        # Calculate volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # Calculate convexity via second derivative approximation
        # Using finite differences: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        price_changes = np.diff(prices)
        second_derivatives = []

        for i in range(1, len(price_changes) - 1):
            h = price_changes[i]
            if h != 0:  # Avoid division by zero
                second_deriv = (price_changes[i+1] - 2*price_changes[i] + price_changes[i-1]) / (h**2)
                second_derivatives.append(second_deriv)

        # Average convexity (gamma)
        gamma = np.mean(second_derivatives) if second_derivatives else 0.0

        # Vega approximation (sensitivity to volatility changes)
        vol_periods = [returns[i:i+20] for i in range(0, len(returns)-20, 5)]  # Rolling 20-day periods
        vol_changes = [np.std(period) for period in vol_periods]

        # Calculate price changes corresponding to volatility periods
        if len(vol_changes) > 1:
            # Use price changes over same periods as volatility calculation
            price_changes_for_corr = []
            for i in range(0, len(returns)-20, 5):
                if i + 25 < len(prices):  # Ensure we have enough data
                    period_start_price = prices[i]
                    period_end_price = prices[i + 20]
                    price_change = (period_end_price - period_start_price) / period_start_price
                    price_changes_for_corr.append(price_change)

            # Match array lengths for correlation
            min_len = min(len(vol_changes), len(price_changes_for_corr))
            if min_len > 1:
                price_vol_corr = np.corrcoef(vol_changes[:min_len], price_changes_for_corr[:min_len])[0,1]
            else:
                price_vol_corr = 0.0
        else:
            price_vol_corr = 0.0

        vega = price_vol_corr * volatility if not np.isnan(price_vol_corr) else 0.0  # Simplified vega proxy

        # Tail risk assessment using EVT
        tail_risk_model = self._estimate_tail_risk(returns)
        tail_risk_potential = abs(tail_risk_model.expected_shortfall)

        # Kelly Criterion with convexity adjustment
        if volatility > 0:
            # Standard Kelly: f* = (bp - q) / b where b=odds, p=win_prob, q=lose_prob
            # For continuous case: f* = μ/σ² where μ=expected return, σ²=variance
            mean_return = np.mean(returns)
            kelly_base = mean_return / (volatility**2) if volatility > 0 else 0.0

            # Convexity adjustment: increase allocation for positive convexity
            convexity_multiplier = 1.0 + max(0, gamma * 0.1)  # Scale gamma impact
            kelly_fraction = kelly_base * convexity_multiplier

            # Cap at reasonable levels (max 25% of risky allocation)
            kelly_fraction = min(max(kelly_fraction, 0.01), 0.25)
        else:
            kelly_fraction = 0.01  # Minimal allocation

        # Convexity score: positive = antifragile, negative = fragile
        convexity_score = gamma + (vega * 0.5) - (tail_risk_potential * 0.3)

        metrics = ConvexityMetrics(
            symbol=symbol,
            convexity_score=convexity_score,
            gamma=gamma,
            vega=vega,
            tail_risk_potential=tail_risk_potential,
            kelly_fraction=kelly_fraction
        )

        logger.info(f"Convexity assessment for {symbol}: Score={convexity_score:.4f}, Kelly={kelly_fraction:.4f}")
        return metrics

    def model_tail_risk(self, symbol: str, returns: List[float],
                       confidence_level: float = 0.95) -> TailRiskModel:
        """
        Model tail risk using Extreme Value Theory (EVT)

        Implements Peaks Over Threshold (POT) method:
        1. Select threshold (95th percentile by default)
        2. Fit Generalized Pareto Distribution to exceedances
        3. Calculate VaR and Expected Shortfall

        Args:
            symbol: Asset symbol
            returns: Historical returns
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            TailRiskModel with EVT parameters and risk metrics
        """
        returns_array = np.array(returns)

        # Use negative returns for loss modeling (right tail of losses)
        losses = -returns_array

        # Determine threshold (95th percentile of losses)
        threshold = np.percentile(losses, self.evt_threshold_percentile)

        # Extract exceedances above threshold
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            logger.warning(f"Insufficient exceedances for {symbol}: {len(exceedances)}")
            # Return conservative estimates
            return TailRiskModel(
                symbol=symbol,
                var_95=np.percentile(losses, 95),
                var_99=np.percentile(losses, 99),
                expected_shortfall=np.mean(losses[losses > np.percentile(losses, 95)]),
                tail_index=0.5,  # Moderate fat tail
                scale_parameter=np.std(losses)
            )

        # Fit Generalized Pareto Distribution using Method of Moments
        # GPD: F(x) = 1 - (1 + ξx/σ)^(-1/ξ) for ξ ≠ 0
        mean_excess = np.mean(exceedances)
        var_excess = np.var(exceedances)

        # Method of moments estimators
        if var_excess > 0:
            xi = 0.5 * (mean_excess**2 / var_excess - 1)  # Tail index (shape parameter)
            sigma = 0.5 * mean_excess * (mean_excess**2 / var_excess + 1)  # Scale parameter
        else:
            xi = 0.1  # Small positive tail index
            sigma = mean_excess

        # Ensure stability (xi < 1 for finite mean)
        xi = min(xi, 0.9)
        sigma = max(sigma, 0.001)  # Prevent zero scale

        # Calculate VaR using EVT
        n = len(returns_array)
        n_exceedances = len(exceedances)

        # VaR formula: VaR_p = u + (σ/ξ) * [((n/Nu)(1-p))^(-ξ) - 1]
        # where u=threshold, Nu=number of exceedances, p=confidence level

        var_95 = self._calculate_evt_var(threshold, sigma, xi, n, n_exceedances, 0.95)
        var_99 = self._calculate_evt_var(threshold, sigma, xi, n, n_exceedances, 0.99)

        # Expected Shortfall (Conditional VaR)
        # ES_p = VaR_p / (1-ξ) + (σ - ξ*u) / (1-ξ)
        if abs(xi) < 0.0001:  # xi ≈ 0 (exponential case)
            expected_shortfall = var_95 + sigma
        else:
            expected_shortfall = var_95 / (1 - xi) + (sigma - xi * threshold) / (1 - xi)

        model = TailRiskModel(
            symbol=symbol,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            tail_index=xi,
            scale_parameter=sigma
        )

        logger.info(f"Tail risk model for {symbol}: VaR95={var_95:.4f}, VaR99={var_99:.4f}, ES={expected_shortfall:.4f}")
        return model

    def _calculate_evt_var(self, threshold: float, sigma: float, xi: float,
                          n: int, n_exceedances: int, confidence: float) -> float:
        """Calculate VaR using EVT formula"""
        if abs(xi) < 0.0001:  # Exponential case (xi ≈ 0)
            return threshold + sigma * np.log((n / n_exceedances) * (1 - confidence))
        else:
            return threshold + (sigma / xi) * (((n / n_exceedances) * (1 - confidence))**(-xi) - 1)

    def _estimate_tail_risk(self, returns: List[float]) -> TailRiskModel:
        """Quick tail risk estimation for convexity assessment"""
        return self.model_tail_risk("temp", returns)

    def rebalance_on_volatility(self, portfolio: dict, volatility_spike: float,
                              market_stress_indicators: Optional[dict] = None) -> dict:
        """
        ANTIFRAGILE rebalancing during volatility spikes

        Key Taleb principle: Increase exposure to convex positions during volatility
        This is counterintuitive but mathematically sound for antifragile assets

        Args:
            portfolio: Current portfolio with positions
            volatility_spike: Current volatility vs historical mean (e.g., 2.0 = 2x normal)
            market_stress_indicators: Additional stress metrics

        Returns:
            Rebalanced portfolio with antifragile adjustments
        """
        rebalanced_portfolio = portfolio.copy()

        # Antifragile rebalancing rules:
        # 1. If volatility spike > 1.5x: Increase convex positions
        # 2. If volatility spike > 2.0x: Reduce fragile positions
        # 3. Always maintain barbell structure

        volatility_threshold_1 = 1.5  # Moderate spike
        volatility_threshold_2 = 2.0  # Major spike

        adjustment_factor = 1.0

        if volatility_spike > volatility_threshold_2:
            # Major volatility spike - aggressive antifragile positioning
            adjustment_factor = 1.2  # Increase convex exposure by 20%
            logger.info(f"Major volatility spike detected: {volatility_spike:.2f}x - Aggressive antifragile rebalancing")

        elif volatility_spike > volatility_threshold_1:
            # Moderate spike - modest antifragile adjustment
            adjustment_factor = 1.1  # Increase convex exposure by 10%
            logger.info(f"Moderate volatility spike detected: {volatility_spike:.2f}x - Modest antifragile rebalancing")

        # Apply antifragile rebalancing
        for symbol in rebalanced_portfolio.get('positions', {}):
            position = rebalanced_portfolio['positions'][symbol]

            # Check if position is in risky (potentially convex) allocation
            if symbol in self.barbell_config.risky_instruments:
                # Assess current convexity
                if 'price_history' in position:
                    convexity_metrics = self.assess_convexity(
                        symbol,
                        position['price_history'],
                        position.get('value', 0)
                    )

                    # Increase allocation for convex assets during volatility
                    if convexity_metrics.convexity_score > 0:  # Positive convexity
                        old_size = position.get('size', 0)
                        new_size = old_size * adjustment_factor

                        # Respect position size limits (max 5% of portfolio per position)
                        max_position_size = self.portfolio_value * 0.05
                        new_value = new_size * position.get('price', 0)

                        if new_value <= max_position_size:
                            position['size'] = new_size
                            position['rebalance_reason'] = f"Antifragile adjustment: volatility={volatility_spike:.2f}x"
                            logger.info(f"Increased {symbol} position by {(adjustment_factor-1)*100:.1f}% due to positive convexity")
                        else:
                            logger.warning(f"Position size limit reached for {symbol}")

                    elif convexity_metrics.convexity_score < -0.1:  # Significantly negative convexity
                        # Reduce fragile positions during volatility
                        old_size = position.get('size', 0)
                        new_size = old_size * (2.0 - adjustment_factor)  # Inverse adjustment
                        position['size'] = new_size
                        position['rebalance_reason'] = f"Fragility reduction: volatility={volatility_spike:.2f}x"
                        logger.info(f"Reduced {symbol} position due to negative convexity during volatility")

        # Ensure barbell allocation is maintained
        rebalanced_portfolio = self._maintain_barbell_discipline(rebalanced_portfolio)

        # Add rebalancing metadata
        rebalanced_portfolio['rebalance_info'] = {
            'timestamp': datetime.now().isoformat(),
            'volatility_spike': volatility_spike,
            'adjustment_factor': adjustment_factor,
            'rebalance_type': 'antifragile_volatility_response',
            'barbell_safe_target': self.barbell_config.safe_allocation,
            'barbell_risky_target': self.barbell_config.risky_allocation
        }

        return rebalanced_portfolio

    def _maintain_barbell_discipline(self, portfolio: dict) -> dict:
        """Ensure portfolio maintains barbell allocation discipline"""
        total_value = sum(pos.get('value', 0) for pos in portfolio.get('positions', {}).values())
        safe_value = 0
        risky_value = 0

        # Calculate current allocation
        for symbol, position in portfolio.get('positions', {}).items():
            if symbol in self.barbell_config.safe_instruments:
                safe_value += position.get('value', 0)
            elif symbol in self.barbell_config.risky_instruments:
                risky_value += position.get('value', 0)

        # Check if rebalancing is needed
        if total_value > 0:
            safe_pct = safe_value / total_value
            risky_pct = risky_value / total_value

            safe_drift = abs(safe_pct - self.barbell_config.safe_allocation)
            risky_drift = abs(risky_pct - self.barbell_config.risky_allocation)

            if safe_drift > self.barbell_config.rebalance_threshold or risky_drift > self.barbell_config.rebalance_threshold:
                logger.info(f"Barbell rebalancing needed: Safe={safe_pct:.1%} (target {self.barbell_config.safe_allocation:.1%}), Risky={risky_pct:.1%} (target {self.barbell_config.risky_allocation:.1%})")

                # Apply proportional adjustment to maintain discipline
                target_safe_value = total_value * self.barbell_config.safe_allocation
                target_risky_value = total_value * self.barbell_config.risky_allocation

                portfolio['barbell_rebalance_needed'] = {
                    'current_safe_pct': safe_pct,
                    'current_risky_pct': risky_pct,
                    'target_safe_value': target_safe_value,
                    'target_risky_value': target_risky_value,
                    'drift_safe': safe_drift,
                    'drift_risky': risky_drift
                }

        return portfolio

    def calculate_antifragility_score(self, portfolio: dict,
                                    historical_performance: List[float]) -> dict:
        """
        Calculate overall portfolio antifragility score

        Antifragility = ability to benefit from volatility and stress
        Score components:
        1. Convexity of positions
        2. Tail risk protection
        3. Barbell structure adherence
        4. Historical volatility response

        Returns:
            Dictionary with antifragility metrics and score
        """
        if not historical_performance or len(historical_performance) < 30:
            logger.warning("Insufficient historical performance data for antifragility assessment")
            return {'antifragility_score': 0.0, 'confidence': 'low'}

        returns = np.array(historical_performance)

        # Component 1: Average convexity of portfolio
        total_convexity = 0.0
        total_weight = 0.0

        for symbol, position in portfolio.get('positions', {}).items():
            if 'price_history' in position and len(position['price_history']) >= 50:
                try:
                    convexity_metrics = self.assess_convexity(
                        symbol, position['price_history'], position.get('value', 0)
                    )
                    weight = position.get('value', 0) / self.portfolio_value
                    total_convexity += convexity_metrics.convexity_score * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Could not assess convexity for {symbol}: {e}")

        avg_convexity = total_convexity / total_weight if total_weight > 0 else 0.0

        # Component 2: Tail risk protection (lower tail risk = higher score)
        tail_risk_model = self._estimate_tail_risk(returns.tolist())
        tail_protection_score = max(0, 1.0 - (abs(tail_risk_model.expected_shortfall) / 0.10))  # Normalized to 10% loss

        # Component 3: Barbell adherence
        barbell_score = self._calculate_barbell_adherence(portfolio)

        # Component 4: Volatility response (does portfolio benefit from volatility?)
        volatility_response_score = self._calculate_volatility_response(returns)

        # Combined antifragility score (weighted average)
        weights = {
            'convexity': 0.35,
            'tail_protection': 0.25,
            'barbell_adherence': 0.25,
            'volatility_response': 0.15
        }

        antifragility_score = (
            avg_convexity * weights['convexity'] +
            tail_protection_score * weights['tail_protection'] +
            barbell_score * weights['barbell_adherence'] +
            volatility_response_score * weights['volatility_response']
        )

        # Normalize to 0-1 scale
        antifragility_score = max(-1.0, min(1.0, antifragility_score))

        # Determine confidence level
        data_quality = min(total_weight, 1.0)  # How much of portfolio was assessed
        confidence = 'high' if data_quality > 0.8 else 'medium' if data_quality > 0.5 else 'low'

        result = {
            'antifragility_score': antifragility_score,
            'confidence': confidence,
            'components': {
                'avg_convexity': avg_convexity,
                'tail_protection_score': tail_protection_score,
                'barbell_adherence_score': barbell_score,
                'volatility_response_score': volatility_response_score
            },
            'weights': weights,
            'data_quality': data_quality,
            'assessment_date': datetime.now().isoformat()
        }

        logger.info(f"Antifragility score calculated: {antifragility_score:.3f} (confidence: {confidence})")
        return result

    def _calculate_barbell_adherence(self, portfolio: dict) -> float:
        """Calculate how well portfolio adheres to barbell structure"""
        total_value = sum(pos.get('value', 0) for pos in portfolio.get('positions', {}).values())

        if total_value == 0:
            return 0.0

        safe_value = sum(
            pos.get('value', 0) for symbol, pos in portfolio.get('positions', {}).items()
            if symbol in self.barbell_config.safe_instruments
        )

        risky_value = sum(
            pos.get('value', 0) for symbol, pos in portfolio.get('positions', {}).items()
            if symbol in self.barbell_config.risky_instruments
        )

        safe_pct = safe_value / total_value
        risky_pct = risky_value / total_value

        # Score based on proximity to target allocation
        safe_drift = abs(safe_pct - self.barbell_config.safe_allocation)
        risky_drift = abs(risky_pct - self.barbell_config.risky_allocation)

        # Perfect adherence = 1.0, maximum drift (0.5) = 0.0
        adherence_score = max(0.0, 1.0 - (safe_drift + risky_drift))

        return adherence_score

    def _calculate_volatility_response(self, returns: np.ndarray) -> float:
        """
        Calculate how portfolio responds to volatility periods
        Antifragile portfolios should perform better during high volatility
        """
        if len(returns) < 60:  # Need at least 3 months of data
            return 0.0

        # Calculate rolling volatility (20-day windows)
        window_size = 20
        rolling_volatility = []
        rolling_returns = []

        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            volatility = np.std(window_returns) * np.sqrt(252)  # Annualized
            period_return = np.mean(window_returns)

            rolling_volatility.append(volatility)
            rolling_returns.append(period_return)

        if len(rolling_volatility) < 10:
            return 0.0

        # Calculate correlation between volatility and returns
        # Positive correlation = antifragile (benefits from volatility)
        # Negative correlation = fragile (hurt by volatility)
        volatility_return_corr = np.corrcoef(rolling_volatility, rolling_returns)[0,1]

        # Handle NaN correlation
        if np.isnan(volatility_return_corr):
            volatility_return_corr = 0.0

        # Normalize to 0-1 scale (correlation ranges from -1 to 1)
        volatility_response_score = (volatility_return_corr + 1.0) / 2.0

        return volatility_response_score

    def get_antifragile_recommendations(self, portfolio: dict,
                                      market_conditions: Optional[dict] = None) -> List[str]:
        """
        Generate actionable antifragility recommendations

        Args:
            portfolio: Current portfolio
            market_conditions: Current market metrics (volatility, correlations, etc.)

        Returns:
            List of specific recommendations to improve antifragility
        """
        recommendations = []

        # Assess current state
        antifragility_assessment = self.calculate_antifragility_score(
            portfolio,
            portfolio.get('historical_returns', [])
        )

        barbell_allocation = self.calculate_barbell_allocation(self.portfolio_value)

        # Recommendation 1: Barbell structure
        if antifragility_assessment['components']['barbell_adherence_score'] < 0.8:
            recommendations.append(
                f"REBALANCE TO BARBELL: Target {barbell_allocation['safe_percentage']:.0f}% safe assets, "
                f"{barbell_allocation['risky_percentage']:.0f}% convex opportunities. "
                f"Current adherence: {antifragility_assessment['components']['barbell_adherence_score']:.1%}"
            )

        # Recommendation 2: Convexity optimization
        if antifragility_assessment['components']['avg_convexity'] < 0:
            recommendations.append(
                "INCREASE CONVEXITY: Replace concave positions with convex alternatives. "
                "Look for assets that benefit from volatility (growth stocks, options, crypto)."
            )

        # Recommendation 3: Tail risk management
        if antifragility_assessment['components']['tail_protection_score'] < 0.6:
            recommendations.append(
                "IMPROVE TAIL PROTECTION: Increase safe asset allocation or add hedging instruments. "
                "Consider increasing cash/treasury allocation."
            )

        # Recommendation 4: Volatility response
        if antifragility_assessment['components']['volatility_response_score'] < 0.4:
            recommendations.append(
                "ENHANCE VOLATILITY RESPONSE: Portfolio is negatively affected by volatility. "
                "Add more convex positions or reduce fragile holdings."
            )

        # Market-specific recommendations
        if market_conditions:
            current_volatility = market_conditions.get('volatility_regime', 'normal')

            if current_volatility == 'high':
                recommendations.append(
                    "VOLATILITY OPPORTUNITY: Consider increasing convex position sizes while "
                    "maintaining barbell discipline. High volatility benefits antifragile portfolios."
                )
            elif current_volatility == 'low':
                recommendations.append(
                    "LOW VOLATILITY PERIOD: Maintain discipline and prepare for eventual volatility increase. "
                    "Ensure convex positions are properly sized for future opportunities."
                )

        # Position-specific recommendations
        for symbol, position in portfolio.get('positions', {}).items():
            if 'price_history' in position and len(position['price_history']) >= 50:
                try:
                    convexity_metrics = self.assess_convexity(
                        symbol, position['price_history'], position.get('value', 0)
                    )

                    if convexity_metrics.convexity_score < -0.2:  # Highly fragile
                        recommendations.append(
                            f"REDUCE/ELIMINATE {symbol}: Highly fragile position (convexity: {convexity_metrics.convexity_score:.3f}). "
                            f"Consider replacing with more antifragile alternative."
                        )
                    elif convexity_metrics.convexity_score > 0.2 and convexity_metrics.kelly_fraction > 0.05:  # Highly convex
                        current_weight = position.get('value', 0) / self.portfolio_value
                        if current_weight < convexity_metrics.kelly_fraction * 0.8:  # Under-allocated
                            recommendations.append(
                                f"INCREASE {symbol}: Highly convex position under-allocated. "
                                f"Current: {current_weight:.1%}, Optimal: {convexity_metrics.kelly_fraction:.1%}"
                            )
                except Exception as e:
                    logger.warning(f"Could not generate recommendation for {symbol}: {e}")

        # Overall score recommendation
        score = antifragility_assessment['antifragility_score']
        if score < 0:
            recommendations.insert(0,
                f"PORTFOLIO IS FRAGILE (Score: {score:.3f}). Priority: Implement barbell structure and reduce fragile positions."
            )
        elif score < 0.3:
            recommendations.insert(0,
                f"PORTFOLIO LACKS ANTIFRAGILITY (Score: {score:.3f}). Focus on adding convex positions and improving structure."
            )
        elif score > 0.7:
            recommendations.insert(0,
                f"PORTFOLIO IS ANTIFRAGILE (Score: {score:.3f}). Maintain discipline and capitalize on volatility opportunities."
            )

        if not recommendations:
            recommendations.append("Portfolio shows good antifragile characteristics. Continue monitoring and maintain discipline.")

        return recommendations[:10]  # Return top 10 recommendations

# Example usage and testing functions
def demo_antifragility_engine():
    """Demonstrate the antifragility engine with sample data"""

    # Initialize engine
    engine = AntifragilityEngine(portfolio_value=100000, risk_tolerance=0.02)

    print("=== TALEB ANTIFRAGILITY ENGINE DEMO ===\n")

    # 1. Calculate barbell allocation
    allocation = engine.calculate_barbell_allocation(100000)
    print("1. BARBELL ALLOCATION:")
    print(f"   Safe Allocation: ${allocation['safe_amount']:,.2f} ({allocation['safe_percentage']:.0f}%)")
    print(f"   Risky Allocation: ${allocation['risky_amount']:,.2f} ({allocation['risky_percentage']:.0f}%)")
    print(f"   Safe Instruments: {allocation['safe_instruments']}")
    print(f"   Risky Instruments: {allocation['risky_instruments']}\n")

    # 2. Sample price data for convexity assessment
    np.random.seed(42)  # For reproducible demo
    n_days = 252

    # Generate sample price series with different characteristics
    # Safe asset: low volatility, steady growth
    safe_returns = np.random.normal(0.0002, 0.005, n_days)  # 5% vol, 5% annual return
    safe_prices = [100]
    for r in safe_returns:
        safe_prices.append(safe_prices[-1] * (1 + r))

    # Risky asset: higher volatility, more convex response
    risky_returns = np.random.normal(0.0005, 0.02, n_days)  # 20% vol, 12% annual return
    # Add some non-linear effects (convexity)
    for i in range(len(risky_returns)):
        if abs(risky_returns[i]) > 0.02:  # Extreme moves
            risky_returns[i] *= 1.5  # Amplify extreme moves (convexity)

    risky_prices = [100]
    for r in risky_returns:
        risky_prices.append(risky_prices[-1] * (1 + r))

    # 3. Assess convexity
    print("2. CONVEXITY ASSESSMENT:")
    safe_convexity = engine.assess_convexity('SHY', safe_prices, 80000)
    print(f"   Safe Asset (SHY): Convexity Score = {safe_convexity.convexity_score:.4f}")
    print(f"                     Kelly Fraction = {safe_convexity.kelly_fraction:.4f}")

    risky_convexity = engine.assess_convexity('QQQ', risky_prices, 20000)
    print(f"   Risky Asset (QQQ): Convexity Score = {risky_convexity.convexity_score:.4f}")
    print(f"                      Kelly Fraction = {risky_convexity.kelly_fraction:.4f}\n")

    # 4. Tail risk modeling
    print("3. TAIL RISK MODELING:")
    tail_risk = engine.model_tail_risk('QQQ', risky_returns.tolist(), 0.95)
    print(f"   VaR (95%): {tail_risk.var_95:.4f}")
    print(f"   VaR (99%): {tail_risk.var_99:.4f}")
    print(f"   Expected Shortfall: {tail_risk.expected_shortfall:.4f}")
    print(f"   Tail Index (xi): {tail_risk.tail_index:.4f}\n")

    # 5. Sample portfolio for testing
    sample_portfolio = {
        'positions': {
            'SHY': {
                'size': 800,
                'price': safe_prices[-1],
                'value': 80000,
                'price_history': safe_prices
            },
            'QQQ': {
                'size': 100,
                'price': risky_prices[-1],
                'value': 20000,
                'price_history': risky_prices
            }
        },
        'historical_returns': (np.array(safe_prices[1:]) / np.array(safe_prices[:-1]) - 1).tolist()
    }

    # 6. Antifragile rebalancing simulation
    print("4. ANTIFRAGILE REBALANCING (Volatility Spike Simulation):")
    rebalanced = engine.rebalance_on_volatility(sample_portfolio, volatility_spike=2.5)
    print(f"   Original QQQ size: {sample_portfolio['positions']['QQQ']['size']}")
    print(f"   Rebalanced QQQ size: {rebalanced['positions']['QQQ']['size']}")
    print(f"   Rebalance reason: {rebalanced['positions']['QQQ'].get('rebalance_reason', 'N/A')}\n")

    # 7. Overall antifragility score
    print("5. ANTIFRAGILITY ASSESSMENT:")
    assessment = engine.calculate_antifragility_score(sample_portfolio, sample_portfolio['historical_returns'])
    print(f"   Overall Score: {assessment['antifragility_score']:.3f} ({assessment['confidence']} confidence)")
    print(f"   Components:")
    for component, score in assessment['components'].items():
        print(f"     {component}: {score:.3f}")

    # 8. Recommendations
    print("\n6. ANTIFRAGILE RECOMMENDATIONS:")
    recommendations = engine.get_antifragile_recommendations(sample_portfolio)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

    print("\n=== END DEMO ===")

if __name__ == "__main__":
    demo_antifragility_engine()