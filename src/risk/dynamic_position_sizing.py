"""
Dynamic Position Sizing System - Integration Layer

This module provides the integration layer between Kelly Criterion calculations,
Gary's DPI system, and the gate management framework for dynamic position sizing.

Key Features:
- Real-time position sizing with Kelly + DPI optimization
- Gate constraint compliance
- Overleverage prevention with hard limits
- Volatility-adjusted allocations
- Multi-asset portfolio optimization
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time

from .kelly_criterion import (
    KellyCriterionCalculator,
    PositionSizeRecommendation,
    KellyRegime
)
from ..strategies.dpi_calculator import DistributionalPressureIndex
from ..gates.gate_manager import GateManager, GateLevel

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation recommendation"""
    total_capital: float
    allocations: Dict[str, PositionSizeRecommendation] = field(default_factory=dict)
    total_allocated_pct: float = 0.0
    cash_reserve_pct: float = 0.0
    risk_budget_used: float = 0.0
    expected_sharpe: float = 0.0
    max_drawdown_estimate: float = 0.0
    gate_level: str = "G0"
    compliance_status: str = "COMPLIANT"
    calculation_time_ms: float = 0.0


@dataclass
class RiskBudget:
    """Risk budget allocation across positions"""
    total_risk_budget: float           # Total risk budget (e.g., 2% of capital)
    allocated_risk: Dict[str, float]   # Risk allocated per symbol
    remaining_risk: float              # Remaining risk budget
    max_position_risk: float           # Max risk per position
    diversification_factor: float      # Diversification adjustment


@dataclass
class DynamicSizingConfig:
    """Configuration for dynamic position sizing"""
    max_positions: int = 5             # Maximum number of positions
    min_position_size: float = 100.0   # Minimum position size in dollars
    max_position_risk: float = 0.02    # Max 2% risk per position
    total_risk_budget: float = 0.06    # Total 6% portfolio risk budget
    rebalance_threshold: float = 0.10  # 10% deviation triggers rebalance
    volatility_target: float = 0.15    # 15% target portfolio volatility
    kelly_scaling_factor: float = 0.5  # Conservative Kelly scaling (50%)


class DynamicPositionSizer:
    """
    Dynamic Position Sizing System

    Integrates Kelly Criterion, DPI analysis, and gate constraints for
    optimal position sizing with real-time risk management.
    """

    def __init__(
        self,
        kelly_calculator: KellyCriterionCalculator,
        dpi_calculator: DistributionalPressureIndex,
        gate_manager: GateManager,
        config: DynamicSizingConfig = None
    ):
        """
        Initialize dynamic position sizing system

        Args:
            kelly_calculator: Kelly Criterion calculator
            dpi_calculator: DPI calculator
            gate_manager: Gate management system
            config: Configuration parameters
        """
        self.kelly_calculator = kelly_calculator
        self.dpi_calculator = dpi_calculator
        self.gate_manager = gate_manager
        self.config = config or DynamicSizingConfig()

        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # Current portfolio state
        self.current_allocations: Dict[str, PositionSizeRecommendation] = {}
        self.last_rebalance = datetime.now()
        self.portfolio_metrics = {}

        logger.info("Dynamic Position Sizing System initialized")

    async def calculate_portfolio_allocation(
        self,
        symbols: List[str],
        current_prices: Dict[str, float],
        total_capital: float,
        current_positions: Dict[str, float] = None
    ) -> PortfolioAllocation:
        """
        Calculate optimal portfolio allocation using Kelly + DPI

        Args:
            symbols: List of symbols to analyze
            current_prices: Current prices for each symbol
            total_capital: Total available capital
            current_positions: Current position sizes (optional)

        Returns:
            PortfolioAllocation with complete allocation plan
        """
        start_time = time.time()
        logger.info(f"Calculating portfolio allocation for {len(symbols)} symbols")

        try:
            # 1. Get current gate configuration
            gate_config = self.gate_manager.get_current_config()
            available_capital = self._calculate_available_capital(
                total_capital, gate_config.cash_floor_pct
            )

            # 2. Filter symbols by gate allowances
            allowed_symbols = [s for s in symbols if s in gate_config.allowed_assets]
            if len(allowed_symbols) < len(symbols):
                logger.info(f"Gate {gate_config.level.value} filtered {len(symbols) - len(allowed_symbols)} symbols")

            # 3. Calculate individual Kelly positions in parallel
            individual_recommendations = await self._calculate_parallel_positions(
                allowed_symbols, current_prices, available_capital
            )

            # 4. Apply portfolio-level constraints and optimization
            optimized_allocation = self._optimize_portfolio_allocation(
                individual_recommendations, available_capital, total_capital
            )

            # 5. Validate complete portfolio against gate constraints
            portfolio_validation = self._validate_portfolio_allocation(
                optimized_allocation, total_capital
            )

            # 6. Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(optimized_allocation)

            # Create final allocation
            allocation = PortfolioAllocation(
                total_capital=total_capital,
                allocations=optimized_allocation,
                total_allocated_pct=sum(r.kelly_percentage for r in optimized_allocation.values()),
                cash_reserve_pct=1.0 - sum(r.kelly_percentage for r in optimized_allocation.values()),
                risk_budget_used=portfolio_metrics.get('risk_budget_used', 0.0),
                expected_sharpe=portfolio_metrics.get('expected_sharpe', 0.0),
                max_drawdown_estimate=portfolio_metrics.get('max_drawdown_estimate', 0.0),
                gate_level=gate_config.level.value,
                compliance_status="COMPLIANT" if portfolio_validation else "NON_COMPLIANT",
                calculation_time_ms=(time.time() - start_time) * 1000
            )

            logger.info(f"Portfolio allocation calculated in {allocation.calculation_time_ms:.1f}ms: "
                       f"{allocation.total_allocated_pct:.1%} allocated, "
                       f"{allocation.cash_reserve_pct:.1%} cash reserve")

            return allocation

        except Exception as e:
            logger.error(f"Error calculating portfolio allocation: {e}")
            return self._create_safe_allocation(total_capital, start_time)

    async def _calculate_parallel_positions(
        self,
        symbols: List[str],
        current_prices: Dict[str, float],
        available_capital: float
    ) -> Dict[str, PositionSizeRecommendation]:
        """Calculate Kelly positions for all symbols in parallel"""

        recommendations = {}

        # Create futures for parallel execution
        futures = []
        for symbol in symbols:
            if symbol in current_prices:
                future = self.thread_pool.submit(
                    self.kelly_calculator.calculate_kelly_position,
                    symbol,
                    current_prices[symbol],
                    available_capital
                )
                futures.append((symbol, future))

        # Collect results
        for symbol, future in futures:
            try:
                recommendation = future.result(timeout=5.0)  # 5 second timeout per symbol
                if recommendation.kelly_percentage > 0 and recommendation.gate_compliant:
                    recommendations[symbol] = recommendation
                else:
                    logger.debug(f"Skipped {symbol}: kelly={recommendation.kelly_percentage:.1%}, "
                               f"compliant={recommendation.gate_compliant}")
            except Exception as e:
                logger.error(f"Error calculating position for {symbol}: {e}")

        return recommendations

    def _optimize_portfolio_allocation(
        self,
        individual_recommendations: Dict[str, PositionSizeRecommendation],
        available_capital: float,
        total_capital: float
    ) -> Dict[str, PositionSizeRecommendation]:
        """
        Optimize portfolio allocation with constraints

        Args:
            individual_recommendations: Individual Kelly recommendations
            available_capital: Available capital after cash floor
            total_capital: Total portfolio capital

        Returns:
            Optimized position recommendations
        """
        try:
            if not individual_recommendations:
                return {}

            # 1. Calculate risk budget allocation
            risk_budget = self._calculate_risk_budget(
                individual_recommendations, total_capital
            )

            # 2. Apply portfolio-level constraints
            optimized_recommendations = {}
            total_risk_used = 0.0

            # Sort by confidence score (highest first)
            sorted_recommendations = sorted(
                individual_recommendations.items(),
                key=lambda x: x[1].confidence_score,
                reverse=True
            )

            for symbol, recommendation in sorted_recommendations:
                # Check if we can add this position
                if (len(optimized_recommendations) >= self.config.max_positions or
                    total_risk_used >= self.config.total_risk_budget):
                    break

                # Calculate position risk
                position_risk = self._calculate_position_risk(recommendation)

                if (position_risk <= self.config.max_position_risk and
                    total_risk_used + position_risk <= self.config.total_risk_budget):

                    # Apply Kelly scaling factor for conservative approach
                    scaled_kelly = recommendation.kelly_percentage * self.config.kelly_scaling_factor
                    scaled_dollar_amount = scaled_kelly * total_capital
                    scaled_shares = int(scaled_dollar_amount / (recommendation.dollar_amount / recommendation.share_quantity)) if recommendation.share_quantity > 0 else 0

                    # Create scaled recommendation
                    scaled_recommendation = PositionSizeRecommendation(
                        symbol=symbol,
                        kelly_percentage=scaled_kelly,
                        dollar_amount=scaled_dollar_amount,
                        share_quantity=scaled_shares,
                        confidence_score=recommendation.confidence_score,
                        risk_metrics=recommendation.risk_metrics,
                        gate_compliant=recommendation.gate_compliant,
                        execution_time_ms=recommendation.execution_time_ms
                    )

                    optimized_recommendations[symbol] = scaled_recommendation
                    total_risk_used += position_risk

            # 3. Normalize allocations if they exceed available capital
            total_allocated = sum(r.dollar_amount for r in optimized_recommendations.values())
            if total_allocated > available_capital:
                scale_factor = available_capital / total_allocated
                for symbol, recommendation in optimized_recommendations.items():
                    recommendation.kelly_percentage *= scale_factor
                    recommendation.dollar_amount *= scale_factor
                    recommendation.share_quantity = int(recommendation.share_quantity * scale_factor)

            return optimized_recommendations

        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return {}

    def _calculate_risk_budget(
        self,
        recommendations: Dict[str, PositionSizeRecommendation],
        total_capital: float
    ) -> RiskBudget:
        """Calculate risk budget allocation"""
        try:
            allocated_risk = {}
            total_risk = 0.0

            for symbol, recommendation in recommendations.items():
                # Estimate position risk based on volatility and size
                position_risk = self._calculate_position_risk(recommendation)
                allocated_risk[symbol] = position_risk
                total_risk += position_risk

            remaining_risk = max(0.0, self.config.total_risk_budget - total_risk)

            # Calculate diversification factor
            n_positions = len(recommendations)
            diversification_factor = np.sqrt(1.0 / max(1, n_positions)) if n_positions > 0 else 1.0

            return RiskBudget(
                total_risk_budget=self.config.total_risk_budget,
                allocated_risk=allocated_risk,
                remaining_risk=remaining_risk,
                max_position_risk=self.config.max_position_risk,
                diversification_factor=diversification_factor
            )

        except Exception as e:
            logger.error(f"Error calculating risk budget: {e}")
            return RiskBudget(
                total_risk_budget=self.config.total_risk_budget,
                allocated_risk={},
                remaining_risk=self.config.total_risk_budget,
                max_position_risk=self.config.max_position_risk,
                diversification_factor=1.0
            )

    def _calculate_position_risk(self, recommendation: PositionSizeRecommendation) -> float:
        """Calculate risk contribution of a position"""
        try:
            # Risk = Position Size % × Expected Volatility
            position_size_pct = recommendation.kelly_percentage
            volatility_estimate = 1.0 / max(recommendation.risk_metrics.volatility_adjustment, 0.1)

            # Normalize volatility (assume target is 15%)
            normalized_volatility = volatility_estimate / 0.15

            position_risk = position_size_pct * normalized_volatility

            return min(position_risk, self.config.max_position_risk)

        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return self.config.max_position_risk

    def _validate_portfolio_allocation(
        self,
        allocation: Dict[str, PositionSizeRecommendation],
        total_capital: float
    ) -> bool:
        """Validate complete portfolio against gate constraints"""
        try:
            # Check total allocation doesn't exceed limits
            total_allocated_pct = sum(r.kelly_percentage for r in allocation.values())
            gate_config = self.gate_manager.get_current_config()

            # Must maintain cash floor
            if total_allocated_pct > (1.0 - gate_config.cash_floor_pct):
                return False

            # Check individual position limits
            for symbol, recommendation in allocation.items():
                if not recommendation.gate_compliant:
                    return False

                # Check position size limit
                if recommendation.kelly_percentage > gate_config.max_position_pct:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating portfolio allocation: {e}")
            return False

    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, PositionSizeRecommendation]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        try:
            if not allocation:
                return {}

            # Aggregate metrics
            total_allocated = sum(r.kelly_percentage for r in allocation.values())
            weighted_sharpe = sum(
                r.kelly_percentage * r.risk_metrics.sharpe_expectation
                for r in allocation.values()
            ) / max(total_allocated, 0.01)

            weighted_drawdown = sum(
                r.kelly_percentage * r.risk_metrics.max_drawdown_risk
                for r in allocation.values()
            ) / max(total_allocated, 0.01)

            risk_budget_used = sum(
                self._calculate_position_risk(r) for r in allocation.values()
            )

            return {
                'expected_sharpe': weighted_sharpe,
                'max_drawdown_estimate': weighted_drawdown,
                'risk_budget_used': risk_budget_used,
                'diversification_ratio': len(allocation) / max(1, len(allocation)),
                'total_allocated_pct': total_allocated
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def _calculate_available_capital(self, total_capital: float, cash_floor_pct: float) -> float:
        """Calculate available capital after cash floor requirement"""
        return total_capital * (1.0 - cash_floor_pct)

    def _create_safe_allocation(self, total_capital: float, start_time: float) -> PortfolioAllocation:
        """Create safe default allocation on error"""
        return PortfolioAllocation(
            total_capital=total_capital,
            allocations={},
            total_allocated_pct=0.0,
            cash_reserve_pct=1.0,
            risk_budget_used=0.0,
            expected_sharpe=0.0,
            max_drawdown_estimate=0.0,
            gate_level=self.gate_manager.current_gate.value,
            compliance_status="SAFE_DEFAULT",
            calculation_time_ms=(time.time() - start_time) * 1000
        )

    def should_rebalance(
        self,
        current_allocation: PortfolioAllocation,
        target_allocation: PortfolioAllocation
    ) -> bool:
        """
        Determine if portfolio should be rebalanced

        Args:
            current_allocation: Current portfolio allocation
            target_allocation: Target portfolio allocation

        Returns:
            True if rebalance is recommended
        """
        try:
            # Check time-based rebalancing
            days_since_rebalance = (datetime.now() - self.last_rebalance).days
            if days_since_rebalance >= 7:  # Weekly rebalancing
                return True

            # Check deviation-based rebalancing
            max_deviation = 0.0

            current_positions = set(current_allocation.allocations.keys())
            target_positions = set(target_allocation.allocations.keys())

            all_positions = current_positions.union(target_positions)

            for symbol in all_positions:
                current_pct = current_allocation.allocations.get(symbol,
                    PositionSizeRecommendation("", 0, 0, 0, 0, None, False, 0)
                ).kelly_percentage

                target_pct = target_allocation.allocations.get(symbol,
                    PositionSizeRecommendation("", 0, 0, 0, 0, None, False, 0)
                ).kelly_percentage

                deviation = abs(current_pct - target_pct)
                max_deviation = max(max_deviation, deviation)

            return max_deviation >= self.config.rebalance_threshold

        except Exception as e:
            logger.error(f"Error checking rebalance condition: {e}")
            return False

    def get_rebalancing_trades(
        self,
        current_allocation: PortfolioAllocation,
        target_allocation: PortfolioAllocation,
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, Union[str, int, float]]]:
        """
        Generate trades needed to rebalance portfolio

        Args:
            current_allocation: Current portfolio allocation
            target_allocation: Target portfolio allocation
            current_prices: Current market prices

        Returns:
            Dictionary of trades needed for rebalancing
        """
        try:
            trades = {}

            current_positions = set(current_allocation.allocations.keys())
            target_positions = set(target_allocation.allocations.keys())
            all_positions = current_positions.union(target_positions)

            for symbol in all_positions:
                current_shares = current_allocation.allocations.get(symbol,
                    PositionSizeRecommendation("", 0, 0, 0, 0, None, False, 0)
                ).share_quantity

                target_shares = target_allocation.allocations.get(symbol,
                    PositionSizeRecommendation("", 0, 0, 0, 0, None, False, 0)
                ).share_quantity

                share_difference = target_shares - current_shares

                if abs(share_difference) > 0:
                    trades[symbol] = {
                        'symbol': symbol,
                        'side': 'BUY' if share_difference > 0 else 'SELL',
                        'quantity': abs(share_difference),
                        'current_price': current_prices.get(symbol, 0),
                        'trade_value': abs(share_difference) * current_prices.get(symbol, 0),
                        'reason': 'REBALANCE'
                    }

            logger.info(f"Generated {len(trades)} rebalancing trades")
            return trades

        except Exception as e:
            logger.error(f"Error generating rebalancing trades: {e}")
            return {}

    def update_portfolio_state(self, allocation: PortfolioAllocation):
        """Update internal portfolio state after allocation"""
        self.current_allocations = allocation.allocations
        self.last_rebalance = datetime.now()

        # Update portfolio metrics
        self.portfolio_metrics = {
            'total_allocated_pct': allocation.total_allocated_pct,
            'risk_budget_used': allocation.risk_budget_used,
            'expected_sharpe': allocation.expected_sharpe,
            'gate_level': allocation.gate_level
        }

    def get_allocation_summary(self) -> Dict:
        """Get current allocation summary"""
        return {
            'timestamp': datetime.now(),
            'positions': {
                symbol: {
                    'kelly_percentage': rec.kelly_percentage,
                    'dollar_amount': rec.dollar_amount,
                    'share_quantity': rec.share_quantity,
                    'confidence_score': rec.confidence_score
                }
                for symbol, rec in self.current_allocations.items()
            },
            'portfolio_metrics': self.portfolio_metrics,
            'last_rebalance': self.last_rebalance,
            'config': {
                'max_positions': self.config.max_positions,
                'total_risk_budget': self.config.total_risk_budget,
                'kelly_scaling_factor': self.config.kelly_scaling_factor
            }
        }