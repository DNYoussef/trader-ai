"""
Alpha Generation Integration System

This module integrates the three core alpha generation components:
1. Narrative Gap (NG) Engine
2. Shadow Book System
3. Policy Twin

The integration system coordinates signal generation, position sizing,
risk management, and ethical considerations to create a comprehensive
alpha generation framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import our alpha generation components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrative.narrative_gap import NarrativeGapEngine, NGSignal
from ..learning.shadow_book import ShadowBookEngine, Trade, TradeType, ActionType
from ..learning.policy_twin import PolicyTwin, EthicalTrade, AlphaType

logger = logging.getLogger(__name__)

@dataclass
class AlphaSignal:
    """Integrated alpha signal combining all components"""
    symbol: str
    timestamp: datetime

    # Narrative Gap components
    ng_score: float
    narrative_gap: float
    catalyst_proximity: float

    # Position sizing components
    recommended_size: float
    max_size: float
    confidence_adjusted_size: float

    # Ethical considerations
    ethical_score: float
    alpha_type: AlphaType
    social_impact: float

    # Risk components
    risk_adjusted_score: float
    var_impact: float
    correlation_impact: float

    # Final signal
    final_score: float
    action: str  # buy/sell/hold
    urgency: float  # 0-1 scale

    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioState:
    """Current portfolio state for integration decisions"""
    total_capital: float
    available_capital: float
    current_positions: Dict[str, float]
    risk_utilization: float
    var_usage: float
    sector_exposures: Dict[str, float] = field(default_factory=dict)

class AlphaIntegrationEngine:
    """Main engine for integrating all alpha generation components"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Initialize component engines
        self.ng_engine = NarrativeGapEngine()
        self.shadow_book = ShadowBookEngine()
        self.policy_twin = PolicyTwin()

        # Position sizing parameters
        self.max_position_size = self.config.get('max_position_size', 0.05)  # 5% of portfolio
        self.ng_position_multiplier = self.config.get('ng_position_multiplier', 2.0)
        self.base_position_size = self.config.get('base_position_size', 0.01)  # 1% base

        # Risk parameters
        self.max_var_usage = self.config.get('max_var_usage', 0.8)  # 80% of VaR budget
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)

        # Ethical parameters
        self.min_ethical_score = self.config.get('min_ethical_score', -0.3)
        self.ethical_size_adjustment = self.config.get('ethical_size_adjustment', True)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the integration engine"""
        return {
            'max_position_size': 0.05,
            'ng_position_multiplier': 2.0,
            'base_position_size': 0.01,
            'max_var_usage': 0.8,
            'correlation_threshold': 0.7,
            'min_ethical_score': -0.3,
            'ethical_size_adjustment': True,
            'signal_weights': {
                'narrative_gap': 0.4,
                'momentum': 0.2,
                'mean_reversion': 0.15,
                'ethical_adjustment': 0.15,
                'risk_adjustment': 0.1
            },
            'time_decay_hours': 24,
            'min_confidence': 0.3
        }

    async def generate_integrated_signal(self, symbol: str, current_price: float,
                                       portfolio_state: PortfolioState) -> AlphaSignal:
        """Generate integrated alpha signal for a symbol"""
        try:
            # Generate base narrative gap signal
            ng_signal = await self.ng_engine.calculate_narrative_gap(symbol, current_price)

            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(ng_signal, portfolio_state)

            # Evaluate ethical considerations
            trade_data = {
                'symbol': symbol,
                'quantity': position_sizing['recommended_size'] / current_price,
                'price': current_price,
                'strategy_id': 'integrated_alpha',
                'trade_id': f"ALPHA_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }

            ethical_trade = self.policy_twin.analyze_trade_ethics(trade_data)

            # Calculate risk adjustments
            risk_adjustments = self._calculate_risk_adjustments(ng_signal, portfolio_state, symbol)

            # Integrate all components into final signal
            integrated_signal = self._integrate_components(
                ng_signal, position_sizing, ethical_trade, risk_adjustments, portfolio_state
            )

            # Record in shadow book for learning
            if integrated_signal.final_score > 0.5:  # Only record significant signals
                await self._record_shadow_trades(integrated_signal, trade_data)

            logger.info(f"Generated integrated signal for {symbol}: score={integrated_signal.final_score:.3f}")

            return integrated_signal

        except Exception as e:
            logger.error(f"Error generating integrated signal for {symbol}: {e}")
            return self._create_neutral_signal(symbol, current_price)

    def _calculate_position_sizing(self, ng_signal: NGSignal, portfolio_state: PortfolioState) -> Dict[str, float]:
        """Calculate position sizing based on NG signal and portfolio state"""
        try:
            # Base size calculation
            base_size = self.base_position_size * portfolio_state.total_capital

            # NG score adjustment
            ng_multiplier = 1.0 + (ng_signal.ng_score * self.ng_position_multiplier)
            ng_adjusted_size = base_size * ng_multiplier

            # Confidence adjustment
            confidence_multiplier = ng_signal.confidence
            confidence_adjusted_size = ng_adjusted_size * confidence_multiplier

            # Portfolio constraint adjustment
            max_allowed = self.max_position_size * portfolio_state.total_capital
            available_capital = portfolio_state.available_capital

            # Risk-based sizing
            risk_multiplier = 1.0 - portfolio_state.risk_utilization
            risk_adjusted_size = confidence_adjusted_size * risk_multiplier

            # Final recommended size
            recommended_size = min(risk_adjusted_size, max_allowed, available_capital * 0.8)

            return {
                'base_size': base_size,
                'ng_adjusted_size': ng_adjusted_size,
                'confidence_adjusted_size': confidence_adjusted_size,
                'risk_adjusted_size': risk_adjusted_size,
                'recommended_size': max(0, recommended_size),
                'max_size': max_allowed,
                'size_utilization': recommended_size / max_allowed if max_allowed > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {
                'base_size': 0,
                'ng_adjusted_size': 0,
                'confidence_adjusted_size': 0,
                'risk_adjusted_size': 0,
                'recommended_size': 0,
                'max_size': 0,
                'size_utilization': 0
            }

    def _calculate_risk_adjustments(self, ng_signal: NGSignal, portfolio_state: PortfolioState,
                                  symbol: str) -> Dict[str, float]:
        """Calculate risk-based adjustments to the signal"""
        try:
            # VaR impact calculation
            var_impact = self._estimate_var_impact(symbol, portfolio_state)

            # Correlation impact
            correlation_impact = self._estimate_correlation_impact(symbol, portfolio_state)

            # Concentration risk
            current_weight = portfolio_state.current_positions.get(symbol, 0) / portfolio_state.total_capital
            concentration_penalty = max(0, (current_weight - self.max_position_size) * 2)

            # Sector exposure risk
            sector_risk = self._estimate_sector_risk(symbol, portfolio_state)

            # Overall risk score
            risk_score = 1.0 - (var_impact + correlation_impact + concentration_penalty + sector_risk) / 4
            risk_score = max(0.1, risk_score)  # Minimum 10% risk score

            return {
                'var_impact': var_impact,
                'correlation_impact': correlation_impact,
                'concentration_penalty': concentration_penalty,
                'sector_risk': sector_risk,
                'overall_risk_score': risk_score
            }

        except Exception as e:
            logger.error(f"Error calculating risk adjustments: {e}")
            return {
                'var_impact': 0.5,
                'correlation_impact': 0.5,
                'concentration_penalty': 0.0,
                'sector_risk': 0.2,
                'overall_risk_score': 0.5
            }

    def _integrate_components(self, ng_signal: NGSignal, position_sizing: Dict[str, float],
                            ethical_trade: EthicalTrade, risk_adjustments: Dict[str, float],
                            portfolio_state: PortfolioState) -> AlphaSignal:
        """Integrate all components into final alpha signal"""
        try:
            # Extract component scores
            ng_score = ng_signal.ng_score
            ethical_score = ethical_trade.social_impact_score
            risk_score = risk_adjustments['overall_risk_score']

            # Apply weighting scheme
            weights = self.config['signal_weights']

            # Base signal combination
            base_signal = (
                ng_score * weights['narrative_gap'] +
                risk_score * weights['risk_adjustment']
            )

            # Ethical adjustment
            if self.config['ethical_size_adjustment']:
                if ethical_score < self.min_ethical_score:
                    ethical_adjustment = 0.5  # Significant penalty
                else:
                    ethical_adjustment = 1.0 + (ethical_score * weights['ethical_adjustment'])
            else:
                ethical_adjustment = 1.0

            # Time decay adjustment
            time_since_signal = (datetime.now() - ng_signal.timestamp).total_seconds() / 3600
            time_decay = max(0.1, 1.0 - (time_since_signal / self.config['time_decay_hours']))

            # Calculate final score
            final_score = base_signal * ethical_adjustment * time_decay
            final_score = np.clip(final_score, 0.0, 1.0)

            # Determine action
            if final_score > 0.7 and ng_signal.confidence > self.config['min_confidence']:
                action = "buy" if ng_signal.gap_magnitude > 0 else "sell"
                urgency = min(1.0, final_score * ng_signal.catalyst_proximity)
            elif final_score < 0.3:
                action = "sell" if portfolio_state.current_positions.get(ng_signal.symbol, 0) > 0 else "hold"
                urgency = 0.3
            else:
                action = "hold"
                urgency = 0.5

            # Create integrated signal
            integrated_signal = AlphaSignal(
                symbol=ng_signal.symbol,
                timestamp=datetime.now(),
                ng_score=ng_score,
                narrative_gap=ng_signal.gap_magnitude,
                catalyst_proximity=ng_signal.catalyst_proximity,
                recommended_size=position_sizing['recommended_size'],
                max_size=position_sizing['max_size'],
                confidence_adjusted_size=position_sizing['confidence_adjusted_size'],
                ethical_score=ethical_score,
                alpha_type=ethical_trade.alpha_type,
                social_impact=ethical_trade.social_impact_score,
                risk_adjusted_score=risk_score,
                var_impact=risk_adjustments['var_impact'],
                correlation_impact=risk_adjustments['correlation_impact'],
                final_score=final_score,
                action=action,
                urgency=urgency,
                metadata={
                    'ng_signal_data': ng_signal.supporting_data,
                    'position_sizing_breakdown': position_sizing,
                    'risk_breakdown': risk_adjustments,
                    'ethical_considerations': ethical_trade.ethical_considerations,
                    'time_decay_factor': time_decay,
                    'ethical_adjustment': ethical_adjustment
                }
            )

            return integrated_signal

        except Exception as e:
            logger.error(f"Error integrating components: {e}")
            return self._create_neutral_signal(ng_signal.symbol, 0.0)

    async def _record_shadow_trades(self, signal: AlphaSignal, trade_data: Dict[str, Any]):
        """Record shadow trades for learning and comparison"""
        try:
            # Create actual trade record
            actual_trade = Trade(
                trade_id=trade_data['trade_id'],
                symbol=signal.symbol,
                trade_type=TradeType.ACTUAL,
                action_type=ActionType.ENTRY,
                quantity=signal.recommended_size / trade_data['price'],
                price=trade_data['price'],
                timestamp=signal.timestamp,
                strategy_id='integrated_alpha',
                confidence=signal.final_score,
                metadata={
                    'ng_score': signal.ng_score,
                    'ethical_score': signal.ethical_score,
                    'urgency': signal.urgency
                }
            )

            # Record in shadow book
            self.shadow_book.record_actual_trade(actual_trade)

            # Create alternative sizing scenarios
            alternative_sizes = [0.5, 1.5, 2.0]  # 50%, 150%, 200% of recommended

            for multiplier in alternative_sizes:
                shadow_trade = Trade(
                    trade_id=f"{trade_data['trade_id']}_shadow_{multiplier}x",
                    symbol=signal.symbol,
                    trade_type=TradeType.SHADOW,
                    action_type=ActionType.ENTRY,
                    quantity=(signal.recommended_size * multiplier) / trade_data['price'],
                    price=trade_data['price'],
                    timestamp=signal.timestamp,
                    strategy_id='integrated_alpha_shadow',
                    confidence=signal.final_score,
                    metadata={'size_multiplier': multiplier}
                )

                self.shadow_book.execute_shadow_trade(shadow_trade)

        except Exception as e:
            logger.error(f"Error recording shadow trades: {e}")

    def generate_portfolio_signals(self, symbols: List[str], current_prices: Dict[str, float],
                                 portfolio_state: PortfolioState) -> List[AlphaSignal]:
        """Generate signals for multiple symbols in portfolio context"""
        try:
            # Run signal generation in parallel
            async def generate_all_signals():
                tasks = [
                    self.generate_integrated_signal(symbol, current_prices[symbol], portfolio_state)
                    for symbol in symbols if symbol in current_prices
                ]
                return await asyncio.gather(*tasks)

            # Execute with timeout
            signals = asyncio.run(asyncio.wait_for(generate_all_signals(), timeout=30.0))

            # Sort by final score
            signals.sort(key=lambda x: x.final_score, reverse=True)

            # Apply portfolio-level constraints
            adjusted_signals = self._apply_portfolio_constraints(signals, portfolio_state)

            return adjusted_signals

        except Exception as e:
            logger.error(f"Error generating portfolio signals: {e}")
            return []

    def _apply_portfolio_constraints(self, signals: List[AlphaSignal],
                                   portfolio_state: PortfolioState) -> List[AlphaSignal]:
        """Apply portfolio-level constraints to signals"""
        try:
            adjusted_signals = []
            remaining_capital = portfolio_state.available_capital
            var_budget_used = 0.0

            for signal in signals:
                # Check capital constraints
                if signal.recommended_size > remaining_capital * 0.9:
                    # Reduce size to fit available capital
                    original_size = signal.recommended_size
                    signal.recommended_size = remaining_capital * 0.8

                    # Adjust final score proportionally
                    size_reduction_factor = signal.recommended_size / original_size
                    signal.final_score *= size_reduction_factor

                    signal.metadata['capital_constrained'] = True
                    signal.metadata['original_size'] = original_size

                # Check VaR budget
                estimated_var_usage = signal.var_impact * signal.recommended_size / portfolio_state.total_capital
                if var_budget_used + estimated_var_usage > self.max_var_usage:
                    # Reduce size to fit VaR budget
                    available_var_budget = self.max_var_usage - var_budget_used
                    if available_var_budget > 0:
                        var_size_limit = (available_var_budget / signal.var_impact) * portfolio_state.total_capital
                        if signal.recommended_size > var_size_limit:
                            signal.recommended_size = var_size_limit
                            signal.final_score *= 0.8  # Penalty for VaR constraint
                            signal.metadata['var_constrained'] = True
                    else:
                        signal.recommended_size = 0
                        signal.final_score = 0
                        signal.action = "hold"
                        signal.metadata['var_budget_exhausted'] = True

                # Update tracking variables
                remaining_capital -= signal.recommended_size
                var_budget_used += estimated_var_usage

                adjusted_signals.append(signal)

            return adjusted_signals

        except Exception as e:
            logger.error(f"Error applying portfolio constraints: {e}")
            return signals

    def _estimate_var_impact(self, symbol: str, portfolio_state: PortfolioState) -> float:
        """Estimate VaR impact of adding position in symbol"""
        # Simplified VaR calculation - would use actual risk model in production
        base_var_impact = 0.15  # 15% base impact

        # Adjust based on current position
        current_weight = portfolio_state.current_positions.get(symbol, 0) / portfolio_state.total_capital
        concentration_multiplier = 1.0 + current_weight * 2  # Higher impact for concentrated positions

        return min(0.8, base_var_impact * concentration_multiplier)

    def _estimate_correlation_impact(self, symbol: str, portfolio_state: PortfolioState) -> float:
        """Estimate correlation impact on portfolio"""
        # Simplified correlation calculation
        # In production, would use actual correlation matrix

        # Check sector concentration
        symbol_sector = self._get_symbol_sector(symbol)
        sector_exposure = portfolio_state.sector_exposures.get(symbol_sector, 0)

        if sector_exposure > 0.3:  # 30% sector concentration threshold
            return 0.6  # High correlation impact
        elif sector_exposure > 0.2:
            return 0.4  # Medium correlation impact
        else:
            return 0.2  # Low correlation impact

    def _estimate_sector_risk(self, symbol: str, portfolio_state: PortfolioState) -> float:
        """Estimate sector-specific risk"""
        symbol_sector = self._get_symbol_sector(symbol)
        sector_exposure = portfolio_state.sector_exposures.get(symbol_sector, 0)

        # Risk increases with sector concentration
        return min(0.5, sector_exposure * 1.5)

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol - simplified mapping"""
        # In production, would use actual sector classification
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS']

        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financials'
        else:
            return 'Other'

    def _create_neutral_signal(self, symbol: str, current_price: float) -> AlphaSignal:
        """Create neutral signal when calculation fails"""
        return AlphaSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            ng_score=0.0,
            narrative_gap=0.0,
            catalyst_proximity=0.0,
            recommended_size=0.0,
            max_size=0.0,
            confidence_adjusted_size=0.0,
            ethical_score=0.0,
            alpha_type=AlphaType.NEUTRAL,
            social_impact=0.0,
            risk_adjusted_score=0.5,
            var_impact=0.3,
            correlation_impact=0.3,
            final_score=0.0,
            action="hold",
            urgency=0.0,
            metadata={'error': 'Failed to generate signal'}
        )

# Example usage and testing
async def test_alpha_integration():
    """Test the integrated alpha generation system"""
    # Initialize integration engine
    integration_engine = AlphaIntegrationEngine()

    # Sample portfolio state
    portfolio_state = PortfolioState(
        total_capital=10_000_000,  # $10M portfolio
        available_capital=2_000_000,  # $2M available
        current_positions={'AAPL': 500_000, 'MSFT': 300_000},
        risk_utilization=0.6,  # 60% of risk budget used
        var_usage=0.5,  # 50% of VaR budget used
        sector_exposures={'Technology': 0.25, 'Financials': 0.15}
    )

    # Generate signals for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    current_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}

    signals = integration_engine.generate_portfolio_signals(symbols, current_prices, portfolio_state)

    print("Alpha Integration Test Results:")
    print(f"Generated {len(signals)} signals")

    for signal in signals:
        print(f"\n{signal.symbol}:")
        print(f"  Final Score: {signal.final_score:.3f}")
        print(f"  Action: {signal.action}")
        print(f"  Recommended Size: ${signal.recommended_size:,.0f}")
        print(f"  NG Score: {signal.ng_score:.3f}")
        print(f"  Ethical Score: {signal.ethical_score:.3f}")
        print(f"  Risk Score: {signal.risk_adjusted_score:.3f}")
        print(f"  Urgency: {signal.urgency:.3f}")

    return signals

if __name__ == "__main__":
    # Run test
    asyncio.run(test_alpha_integration())