"""
Narrative Gap (NG) Calculation - Core Alpha Generation Mechanism

This module implements the core alpha generation formula from the Super-Gary vision:
measuring the gap between consensus forecasts and actual distribution estimates
to identify mispricing opportunities.

Mathematical Foundation:
- NG = abs(consensus_forecast - distribution_estimate) / market_price
- Higher NG indicates greater mispricing opportunity
- NG values range from 0-1 (normalized)
- Used as multiplier for position sizing (up to 2x)

Alpha Generation Logic:
- Large gaps suggest market inefficiency
- Gary's edge comes from superior distribution estimation
- NG amplifies Kelly sizing when opportunities are detected
"""

import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NarrativeGapComponents:
    """Components of the Narrative Gap calculation"""
    market_price: float
    consensus_forecast: float
    distribution_estimate: float
    raw_gap: float
    normalized_ng: float
    confidence_score: float


class NarrativeGap:
    """
    Narrative Gap Calculator - Core Alpha Generation Engine

    Calculates the gap between market consensus and Gary's distribution estimates
    to identify mispricing opportunities for alpha generation.
    """

    def __init__(
        self,
        max_ng: float = 1.0,          # Maximum NG value (100%)
        min_price_threshold: float = 1.0,  # Minimum price for calculation
        confidence_threshold: float = 0.3  # Minimum confidence required
    ):
        """
        Initialize Narrative Gap calculator

        Args:
            max_ng: Maximum allowed NG value (default 1.0)
            min_price_threshold: Minimum price for valid calculation
            confidence_threshold: Minimum confidence required
        """
        self.max_ng = max_ng
        self.min_price_threshold = min_price_threshold
        self.confidence_threshold = confidence_threshold

        logger.info(f"NarrativeGap initialized: max_ng={max_ng}, min_price={min_price_threshold}")

    def calculate_ng(
        self,
        market_price: float,
        consensus_forecast: float,
        distribution_estimate: float,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Narrative Gap for alpha generation

        Args:
            market_price: Current market price
            consensus_forecast: Market consensus price forecast
            distribution_estimate: Gary's distribution-based estimate
            confidence: Optional confidence in distribution estimate

        Returns:
            Normalized NG value [0, 1] for position sizing multiplication
        """
        try:
            # Input validation
            if market_price <= self.min_price_threshold:
                logger.warning(f"Market price {market_price} below threshold {self.min_price_threshold}")
                return 0.0

            if market_price <= 0 or np.isnan(market_price):
                logger.error(f"Invalid market price: {market_price}")
                return 0.0

            if np.isnan(consensus_forecast) or np.isnan(distribution_estimate):
                logger.error("Invalid forecast values (NaN detected)")
                return 0.0

            # Core NG calculation: absolute gap normalized by price
            raw_gap = abs(consensus_forecast - distribution_estimate)
            ng = raw_gap / market_price

            # Apply maximum constraint
            ng = min(ng, self.max_ng)

            # Apply confidence scaling if provided
            if confidence is not None:
                if confidence < self.confidence_threshold:
                    logger.debug(f"Low confidence {confidence:.2f}, scaling NG down")
                    ng *= max(0.1, confidence / self.confidence_threshold)
                else:
                    # High confidence can amplify NG slightly
                    ng *= min(1.2, 1.0 + (confidence - self.confidence_threshold))

            # Ensure non-negative result
            ng = max(0.0, ng)

            logger.debug(f"NG calculated: price={market_price:.2f}, "
                        f"consensus={consensus_forecast:.2f}, "
                        f"estimate={distribution_estimate:.2f}, "
                        f"NG={ng:.4f}")

            return ng

        except Exception as e:
            logger.error(f"Error calculating Narrative Gap: {e}")
            return 0.0

    def calculate_ng_detailed(
        self,
        market_price: float,
        consensus_forecast: float,
        distribution_estimate: float,
        confidence: Optional[float] = None
    ) -> NarrativeGapComponents:
        """
        Calculate Narrative Gap with detailed component breakdown

        Args:
            market_price: Current market price
            consensus_forecast: Market consensus price forecast
            distribution_estimate: Gary's distribution-based estimate
            confidence: Optional confidence in distribution estimate

        Returns:
            NarrativeGapComponents with full calculation details
        """
        try:
            # Calculate basic NG
            ng = self.calculate_ng(market_price, consensus_forecast, distribution_estimate, confidence)

            # Calculate components
            raw_gap = abs(consensus_forecast - distribution_estimate)
            confidence_score = confidence if confidence is not None else 0.5

            return NarrativeGapComponents(
                market_price=market_price,
                consensus_forecast=consensus_forecast,
                distribution_estimate=distribution_estimate,
                raw_gap=raw_gap,
                normalized_ng=ng,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error(f"Error calculating detailed NG: {e}")
            return NarrativeGapComponents(
                market_price=market_price,
                consensus_forecast=0.0,
                distribution_estimate=0.0,
                raw_gap=0.0,
                normalized_ng=0.0,
                confidence_score=0.0
            )

    def get_position_multiplier(self, ng: float) -> float:
        """
        Convert NG to position sizing multiplier

        Args:
            ng: Narrative Gap value [0, 1]

        Returns:
            Position multiplier [1.0, 2.0] where higher NG = larger positions
        """
        try:
            # Linear scaling from 1.0 to 2.0 based on NG
            # NG of 0 = 1.0x (normal sizing)
            # NG of 1 = 2.0x (double sizing)
            multiplier = 1.0 + ng

            # Ensure bounds
            multiplier = max(1.0, min(2.0, multiplier))

            logger.debug(f"NG {ng:.4f} -> position multiplier {multiplier:.2f}x")

            return multiplier

        except Exception as e:
            logger.error(f"Error calculating position multiplier: {e}")
            return 1.0  # Safe default

    def validate_inputs(
        self,
        market_price: float,
        consensus_forecast: float,
        distribution_estimate: float
    ) -> Tuple[bool, str]:
        """
        Validate inputs for NG calculation

        Args:
            market_price: Current market price
            consensus_forecast: Market consensus forecast
            distribution_estimate: Distribution estimate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check price validity
            if market_price <= 0:
                return False, f"Invalid market price: {market_price}"

            if market_price < self.min_price_threshold:
                return False, f"Price {market_price} below minimum threshold {self.min_price_threshold}"

            # Check for NaN values
            if np.isnan(market_price) or np.isnan(consensus_forecast) or np.isnan(distribution_estimate):
                return False, "NaN values detected in inputs"

            # Check for infinite values
            if np.isinf(market_price) or np.isinf(consensus_forecast) or np.isinf(distribution_estimate):
                return False, "Infinite values detected in inputs"

            # Basic reasonableness check (forecasts shouldn't be wildly different from price)
            price_ratio_consensus = abs(consensus_forecast / market_price) if market_price != 0 else float('inf')
            price_ratio_estimate = abs(distribution_estimate / market_price) if market_price != 0 else float('inf')

            if price_ratio_consensus > 10 or price_ratio_estimate > 10:
                return False, "Forecasts too far from market price (>10x difference)"

            return True, "Valid inputs"

        except Exception as e:
            return False, f"Validation error: {e}"


def create_narrative_gap_calculator(**kwargs) -> NarrativeGap:
    """
    Factory function to create NarrativeGap calculator with standard settings

    Args:
        **kwargs: Optional parameters for NarrativeGap initialization

    Returns:
        Configured NarrativeGap instance
    """
    return NarrativeGap(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Simple test
    ng_calc = NarrativeGap()

    # Test case: market thinks $100, consensus says $105, Gary estimates $110
    market_price = 100.0
    consensus = 105.0
    gary_estimate = 110.0

    ng = ng_calc.calculate_ng(market_price, consensus, gary_estimate)
    multiplier = ng_calc.get_position_multiplier(ng)

    print(f"Market Price: ${market_price}")
    print(f"Consensus Forecast: ${consensus}")
    print(f"Gary's Estimate: ${gary_estimate}")
    print(f"Narrative Gap: {ng:.4f}")
    print(f"Position Multiplier: {multiplier:.2f}x")

    # Detailed breakdown
    components = ng_calc.calculate_ng_detailed(market_price, consensus, gary_estimate, confidence=0.8)
    print("\nDetailed Components:")
    print(f"Raw Gap: ${components.raw_gap:.2f}")
    print(f"Normalized NG: {components.normalized_ng:.4f}")
    print(f"Confidence: {components.confidence_score:.2f}")