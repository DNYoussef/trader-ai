"""
Portfolio Context Features

Extends the 110 market features with portfolio-specific context:
- Milestone tracking state (current milestone, progress, days)
- Risk tier information (loss aversion, target allocation)
- Capital awareness (normalized capital, distance to goals)

This allows the AI to adjust decisions based on portfolio state,
not just market conditions.

Feature Dimensions:
- Market features: 110
- Milestone features: 6
- Risk tier features: 5
- Capital features: 4
- TOTAL: 125 features
"""

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import our portfolio modules
from src.portfolio.milestone_tracker import (
    MilestoneTracker,
    Milestone,
    MILESTONE_CONFIG,
)
from src.simulation.graduated_risk_simulator import (
    get_risk_tier,
    interpolate_risk_params,
    RiskTier,
    TIER_CONFIGS,
)

logger = logging.getLogger(__name__)


# Feature indices for the extended feature vector
MARKET_FEATURE_START = 0
MARKET_FEATURE_END = 110
MILESTONE_FEATURE_START = 110
MILESTONE_FEATURE_END = 116
RISK_TIER_FEATURE_START = 116
RISK_TIER_FEATURE_END = 121
CAPITAL_FEATURE_START = 121
CAPITAL_FEATURE_END = 125

TOTAL_FEATURES = 125


@dataclass
class PortfolioContextFeatures:
    """Container for portfolio context features."""

    # Milestone features (6)
    milestone_idx: float          # 0-3 normalized (M0=0, M1=0.33, M2=0.67, M3=1)
    milestone_progress: float     # 0-1 progress within current milestone
    days_at_milestone: float      # Normalized days at current milestone
    milestones_achieved: float    # 0-1 count of achieved milestones
    is_goal_achieved: float       # Binary: 1 if M3 reached
    milestone_momentum: float     # Rate of milestone progress

    # Risk tier features (5)
    risk_tier_idx: float          # 0-3 normalized tier
    loss_aversion: float          # 1.2-3.0 normalized
    target_spy_pct: float         # Target SPY allocation
    target_tlt_pct: float         # Target TLT allocation
    target_cash_pct: float        # Target cash allocation

    # Capital features (4)
    capital_normalized: float     # Capital / 500 (goal)
    capital_log: float            # Log-normalized capital
    distance_to_goal: float       # Normalized distance to $500
    drawdown_from_peak: float     # Current drawdown from peak

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            # Milestone features
            self.milestone_idx,
            self.milestone_progress,
            self.days_at_milestone,
            self.milestones_achieved,
            self.is_goal_achieved,
            self.milestone_momentum,
            # Risk tier features
            self.risk_tier_idx,
            self.loss_aversion,
            self.target_spy_pct,
            self.target_tlt_pct,
            self.target_cash_pct,
            # Capital features
            self.capital_normalized,
            self.capital_log,
            self.distance_to_goal,
            self.drawdown_from_peak,
        ], dtype=np.float32)


class PortfolioContextExtractor:
    """
    Extract portfolio context features for AI training.

    Usage:
        extractor = PortfolioContextExtractor()
        context = extractor.extract(capital=250.0, peak_capital=280.0)
        features = context.to_array()  # 15 features
    """

    def __init__(
        self,
        data_dir: str = "D:/Projects/trader-ai/data/milestones",
        goal_capital: float = 500.0,
        start_capital: float = 200.0,
    ):
        self.data_dir = Path(data_dir)
        self.goal_capital = goal_capital
        self.start_capital = start_capital

        # Milestone index mapping
        self.milestone_to_idx = {
            Milestone.M0_START: 0,
            Milestone.M1_TRACTION: 1,
            Milestone.M2_MOMENTUM: 2,
            Milestone.M3_GOAL: 3,
        }

        # Risk tier index mapping
        self.tier_to_idx = {
            RiskTier.AGGRESSIVE: 0,
            RiskTier.MODERATE: 1,
            RiskTier.CONSERVATIVE: 2,
            RiskTier.PRESERVATION: 3,
        }

        # For momentum calculation
        self._capital_history: List[float] = []
        self._max_history = 20

    def extract(
        self,
        capital: float,
        peak_capital: Optional[float] = None,
        days_at_milestone: int = 0,
        milestones_achieved: int = 0,
    ) -> PortfolioContextFeatures:
        """
        Extract portfolio context features.

        Args:
            capital: Current portfolio capital
            peak_capital: Peak capital (for drawdown), defaults to capital
            days_at_milestone: Days spent at current milestone
            milestones_achieved: Number of milestones achieved so far

        Returns:
            PortfolioContextFeatures dataclass
        """
        if peak_capital is None:
            peak_capital = capital

        # Update capital history for momentum
        self._capital_history.append(capital)
        if len(self._capital_history) > self._max_history:
            self._capital_history.pop(0)

        # === MILESTONE FEATURES ===
        milestone = self._get_milestone(capital)
        milestone_config = MILESTONE_CONFIG[milestone]

        milestone_idx = self.milestone_to_idx[milestone] / 3.0  # Normalize 0-1

        # Progress within milestone
        cap_min = milestone_config['capital_min']
        cap_max = milestone_config['capital_max']
        if cap_max == float('inf'):
            milestone_progress = 1.0  # Goal achieved
        else:
            milestone_progress = (capital - cap_min) / (cap_max - cap_min + 0.01)
            milestone_progress = np.clip(milestone_progress, 0, 1)

        # Days normalized (assume max ~60 days at a milestone)
        days_normalized = min(days_at_milestone / 60.0, 1.0)

        # Milestones achieved normalized
        milestones_norm = milestones_achieved / 3.0  # Max 3 transitions

        is_goal_achieved = 1.0 if milestone == Milestone.M3_GOAL else 0.0

        # Momentum: rate of capital change
        if len(self._capital_history) >= 5:
            recent_avg = np.mean(self._capital_history[-5:])
            older_avg = np.mean(self._capital_history[:-5]) if len(self._capital_history) > 5 else recent_avg
            momentum = (recent_avg - older_avg) / (older_avg + 1e-6)
            momentum = np.clip(momentum, -0.5, 0.5)  # Cap at +/-50%
        else:
            momentum = 0.0

        # === RISK TIER FEATURES ===
        risk_params = interpolate_risk_params(capital)
        risk_tier = get_risk_tier(capital)

        risk_tier_idx = self.tier_to_idx[risk_tier] / 3.0

        # Normalize loss aversion (1.2-3.0 range)
        loss_aversion_norm = (risk_params['loss_aversion'] - 1.2) / (3.0 - 1.2)
        loss_aversion_norm = np.clip(loss_aversion_norm, 0, 1)

        # === CAPITAL FEATURES ===
        capital_normalized = capital / self.goal_capital  # 0.4-1.0+ for $200-$500+

        # Log-normalized capital (smoother for large values)
        capital_log = np.log(capital / self.start_capital + 1) / np.log(3)  # ~1 at $500

        # Distance to goal normalized
        distance_to_goal = max(0, self.goal_capital - capital) / (self.goal_capital - self.start_capital)

        # Drawdown from peak
        drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        drawdown = np.clip(drawdown, 0, 1)

        return PortfolioContextFeatures(
            # Milestone
            milestone_idx=milestone_idx,
            milestone_progress=milestone_progress,
            days_at_milestone=days_normalized,
            milestones_achieved=milestones_norm,
            is_goal_achieved=is_goal_achieved,
            milestone_momentum=momentum,
            # Risk tier
            risk_tier_idx=risk_tier_idx,
            loss_aversion=loss_aversion_norm,
            target_spy_pct=risk_params['target_spy'],
            target_tlt_pct=risk_params['target_tlt'],
            target_cash_pct=risk_params['target_cash'],
            # Capital
            capital_normalized=capital_normalized,
            capital_log=capital_log,
            distance_to_goal=distance_to_goal,
            drawdown_from_peak=drawdown,
        )

    def _get_milestone(self, capital: float) -> Milestone:
        """Determine milestone based on capital."""
        for milestone, config in MILESTONE_CONFIG.items():
            if config['capital_min'] <= capital <= config['capital_max']:
                return milestone
        return Milestone.M3_GOAL if capital >= 500 else Milestone.M0_START

    def reset_history(self):
        """Reset capital history (call at start of new simulation)."""
        self._capital_history = []


class ExtendedFeatureExtractor:
    """
    Combine market features (110) with portfolio context (15) = 125 features.

    Usage:
        extractor = ExtendedFeatureExtractor()

        # Get extended features for current state
        market_features = compute_market_features(...)  # 110 features
        extended = extractor.extend_features(
            market_features,
            capital=250.0,
            peak_capital=280.0
        )  # 125 features
    """

    def __init__(self, data_dir: str = "D:/Projects/trader-ai/data/milestones"):
        self.context_extractor = PortfolioContextExtractor(data_dir=data_dir)

    def extend_features(
        self,
        market_features: np.ndarray,
        capital: float,
        peak_capital: Optional[float] = None,
        days_at_milestone: int = 0,
        milestones_achieved: int = 0,
    ) -> np.ndarray:
        """
        Extend market features with portfolio context.

        Args:
            market_features: Array of 110 market features
            capital: Current portfolio capital
            peak_capital: Peak capital for drawdown calculation
            days_at_milestone: Days at current milestone
            milestones_achieved: Count of achieved milestones

        Returns:
            Extended feature array of shape (125,)
        """
        assert len(market_features) == 110, f"Expected 110 market features, got {len(market_features)}"

        context = self.context_extractor.extract(
            capital=capital,
            peak_capital=peak_capital,
            days_at_milestone=days_at_milestone,
            milestones_achieved=milestones_achieved,
        )

        context_features = context.to_array()

        return np.concatenate([market_features, context_features])

    def extend_batch(
        self,
        market_features: np.ndarray,
        capitals: np.ndarray,
        peak_capitals: Optional[np.ndarray] = None,
        days_at_milestones: Optional[np.ndarray] = None,
        milestones_achieved: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extend a batch of market features.

        Args:
            market_features: (batch_size, 110) market features
            capitals: (batch_size,) current capitals
            peak_capitals: (batch_size,) peak capitals
            days_at_milestones: (batch_size,) days at milestone
            milestones_achieved: (batch_size,) milestone counts

        Returns:
            Extended features (batch_size, 125)
        """
        batch_size = len(market_features)

        if peak_capitals is None:
            peak_capitals = capitals
        if days_at_milestones is None:
            days_at_milestones = np.zeros(batch_size)
        if milestones_achieved is None:
            milestones_achieved = np.zeros(batch_size)

        extended = np.zeros((batch_size, TOTAL_FEATURES), dtype=np.float32)

        for i in range(batch_size):
            extended[i] = self.extend_features(
                market_features[i],
                capital=capitals[i],
                peak_capital=peak_capitals[i],
                days_at_milestone=int(days_at_milestones[i]),
                milestones_achieved=int(milestones_achieved[i]),
            )

        return extended

    def reset(self):
        """Reset the context extractor history."""
        self.context_extractor.reset_history()


def generate_synthetic_capital_trajectory(
    n_days: int = 252,
    start_capital: float = 200.0,
    daily_return_mean: float = 0.001,
    daily_return_std: float = 0.015,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic capital trajectory for training data augmentation.

    Returns:
        capitals: (n_days,) capital values
        peak_capitals: (n_days,) running peak values
    """
    if seed is not None:
        np.random.seed(seed)

    returns = np.random.normal(daily_return_mean, daily_return_std, n_days)

    capitals = np.zeros(n_days)
    capitals[0] = start_capital

    for i in range(1, n_days):
        capitals[i] = capitals[i-1] * (1 + returns[i])
        capitals[i] = max(capitals[i], 50)  # Floor at $50

    # Running peak
    peak_capitals = np.maximum.accumulate(capitals)

    return capitals, peak_capitals


def create_training_dataset_with_context(
    market_features: np.ndarray,
    time_indices: np.ndarray,
    capitals: np.ndarray,
    peak_capitals: np.ndarray,
) -> np.ndarray:
    """
    Create extended training dataset combining market features with portfolio context.

    This is the main function to call when preparing training data.
    """
    extractor = ExtendedFeatureExtractor()

    # Calculate days at milestone and milestones achieved
    n_samples = len(market_features)
    days_at_milestones = np.zeros(n_samples)
    milestones_achieved = np.zeros(n_samples)

    prev_milestone = None
    days_counter = 0
    milestone_count = 0

    for i in range(n_samples):
        capital = capitals[i]
        context = extractor.context_extractor.extract(capital)
        current_milestone = extractor.context_extractor._get_milestone(capital)

        if prev_milestone is not None and current_milestone != prev_milestone:
            # Milestone changed
            days_counter = 0
            if current_milestone.value > prev_milestone.value:  # Forward progress
                milestone_count += 1

        days_at_milestones[i] = days_counter
        milestones_achieved[i] = milestone_count

        prev_milestone = current_milestone
        days_counter += 1

    # Reset and extend all features
    extractor.reset()

    return extractor.extend_batch(
        market_features=market_features,
        capitals=capitals,
        peak_capitals=peak_capitals,
        days_at_milestones=days_at_milestones,
        milestones_achieved=milestones_achieved,
    )


def print_feature_summary():
    """Print summary of the extended feature set."""
    print()
    print("=" * 70)
    print("EXTENDED FEATURE SET (125 features)")
    print("=" * 70)
    print()
    print(f"Market Features:     {MARKET_FEATURE_START:3d} - {MARKET_FEATURE_END:3d}  (110 features)")
    print(f"Milestone Features:  {MILESTONE_FEATURE_START:3d} - {MILESTONE_FEATURE_END:3d}  (6 features)")
    print(f"  - milestone_idx:      Current milestone (0-1)")
    print(f"  - milestone_progress: Progress within milestone")
    print(f"  - days_at_milestone:  Days at current milestone")
    print(f"  - milestones_achieved: Count of achieved")
    print(f"  - is_goal_achieved:   Binary goal flag")
    print(f"  - milestone_momentum: Rate of progress")
    print()
    print(f"Risk Tier Features:  {RISK_TIER_FEATURE_START:3d} - {RISK_TIER_FEATURE_END:3d}  (5 features)")
    print(f"  - risk_tier_idx:      Current risk tier (0-1)")
    print(f"  - loss_aversion:      Loss aversion multiplier")
    print(f"  - target_spy_pct:     Target SPY allocation")
    print(f"  - target_tlt_pct:     Target TLT allocation")
    print(f"  - target_cash_pct:    Target cash allocation")
    print()
    print(f"Capital Features:    {CAPITAL_FEATURE_START:3d} - {CAPITAL_FEATURE_END:3d}  (4 features)")
    print(f"  - capital_normalized: Capital / goal")
    print(f"  - capital_log:        Log-normalized capital")
    print(f"  - distance_to_goal:   Distance to $500")
    print(f"  - drawdown_from_peak: Current drawdown")
    print()
    print(f"TOTAL: {TOTAL_FEATURES} features")
    print("=" * 70)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print_feature_summary()

    # Demo extraction
    print()
    print("DEMO: Portfolio Context Extraction")
    print("-" * 40)

    extractor = PortfolioContextExtractor()

    test_capitals = [200, 250, 300, 350, 400, 450, 500, 600]

    print(f"{'Capital':>10} {'Milestone':>8} {'Progress':>10} {'Tier':>12} {'Loss Av':>10} {'SPY%':>8}")
    print("-" * 70)

    for cap in test_capitals:
        ctx = extractor.extract(cap)
        milestone_names = ['M0', 'M1', 'M2', 'M3']
        tier_names = ['AGGR', 'MOD', 'CONS', 'PRES']

        m_idx = int(ctx.milestone_idx * 3)
        t_idx = int(ctx.risk_tier_idx * 3)

        print(f"${cap:>9} {milestone_names[m_idx]:>8} {ctx.milestone_progress:>10.1%} "
              f"{tier_names[t_idx]:>12} {ctx.loss_aversion:>10.2f} {ctx.target_spy_pct:>8.0%}")

    print()
    print("Extended Feature Demo:")
    print("-" * 40)

    # Create dummy market features
    market_features = np.random.randn(110).astype(np.float32)

    extended_extractor = ExtendedFeatureExtractor()
    extended = extended_extractor.extend_features(
        market_features,
        capital=275.0,
        peak_capital=290.0,
        days_at_milestone=15,
        milestones_achieved=0,
    )

    print(f"Market features shape: {market_features.shape}")
    print(f"Extended features shape: {extended.shape}")
    print(f"Context features (last 15): {extended[-15:]}")
