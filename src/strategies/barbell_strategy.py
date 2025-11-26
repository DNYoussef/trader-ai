"""
Barbell Strategy Manager - Gary×Taleb Trading System

Implements the barbell approach:
- 80% Conservative: Index funds, bonds, stable assets (safety)
- 20% Aggressive: High-conviction contrarian bets (Gary moments)

This strategy provides antifragility through conservative base protection
while capturing massive upside when consensus is wrong about inequality effects.
"""

import logging
from typing import Dict, List, Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BarbellAllocation:
    """Represents current barbell allocation state"""
    conservative_allocation: Decimal  # Target: 80%
    aggressive_allocation: Decimal     # Target: 20%
    conservative_positions: Dict[str, Decimal]
    aggressive_positions: Dict[str, Decimal]
    total_value: Decimal
    rebalance_needed: bool
    last_rebalance: datetime


@dataclass
class ContrarianOpportunity:
    """Represents a contrarian trading opportunity (Gary moment)"""
    symbol: str
    thesis: str
    consensus_view: str
    contrarian_view: str
    inequality_correlation: float  # -1 to 1
    conviction_score: float        # 0 to 1
    expected_payoff: float         # Risk/reward ratio
    timeframe_days: int
    entry_price: Decimal
    target_price: Decimal
    stop_loss: Decimal
    created_at: datetime


class BarbellStrategy:
    """
    Barbell Strategy implementation combining safety with explosive upside.

    Core Philosophy:
    - Most of capital (80%) stays safe to survive any crisis
    - Small portion (20%) bets on massive mispricings from inequality blindness
    - When everyone's wrong about inequality effects, we make 10x+ returns
    """

    def __init__(self,
                 portfolio_manager,
                 inequality_hunter=None,
                 conservative_ratio: Decimal = Decimal("0.80"),
                 aggressive_ratio: Decimal = Decimal("0.20")):
        """
        Initialize Barbell Strategy.

        Args:
            portfolio_manager: Portfolio management system
            inequality_hunter: AI system for finding contrarian opportunities
            conservative_ratio: Portion for conservative allocation (default 80%)
            aggressive_ratio: Portion for aggressive bets (default 20%)
        """
        self.portfolio = portfolio_manager
        self.inequality_hunter = inequality_hunter
        self.conservative_ratio = conservative_ratio
        self.aggressive_ratio = aggressive_ratio

        # Conservative instruments (stable, liquid, minimal downside)
        self.conservative_instruments = {
            'SPY': Decimal("0.30"),   # S&P 500 index
            'TLT': Decimal("0.25"),   # Long-term treasuries
            'GLD': Decimal("0.15"),   # Gold
            'SGOV': Decimal("0.20"),  # Short-term treasuries
            'VTEB': Decimal("0.10"),  # Tax-exempt bonds
        }

        # Aggressive allocation tracking
        self.contrarian_opportunities: List[ContrarianOpportunity] = []
        self.active_contrarian_bets: Dict[str, ContrarianOpportunity] = {}

        # Performance tracking
        self.conservative_returns: List[float] = []
        self.aggressive_returns: List[float] = []
        self.total_returns: List[float] = []

        # Rebalancing parameters
        self.rebalance_threshold = Decimal("0.05")  # 5% deviation triggers rebalance
        self.min_rebalance_interval = timedelta(days=30)
        self.last_rebalance = datetime.now()

        # Risk parameters for aggressive bets
        self.max_single_bet_size = Decimal("0.05")  # Max 5% of total capital per bet
        self.max_correlation = 0.7  # Avoid highly correlated contrarian bets
        self.min_conviction_score = 0.7  # Only take high-conviction bets
        self.min_expected_payoff = 3.0  # Minimum 3:1 reward/risk ratio

        logger.info(f"Initialized Barbell Strategy: {conservative_ratio:.0%} conservative, "
                   f"{aggressive_ratio:.0%} aggressive")

    def analyze_current_allocation(self) -> BarbellAllocation:
        """
        Analyze current portfolio allocation vs target barbell.

        Returns:
            BarbellAllocation with current state
        """
        positions = self.portfolio.get_positions()
        total_value = self.portfolio.get_total_value()

        conservative_value = Decimal("0")
        aggressive_value = Decimal("0")
        conservative_positions = {}
        aggressive_positions = {}

        for symbol, position in positions.items():
            if symbol in self.conservative_instruments:
                conservative_value += position.market_value
                conservative_positions[symbol] = position.market_value
            elif symbol in self.active_contrarian_bets:
                aggressive_value += position.market_value
                aggressive_positions[symbol] = position.market_value

        conservative_pct = (conservative_value / total_value) if total_value > 0 else Decimal("0")
        aggressive_pct = (aggressive_value / total_value) if total_value > 0 else Decimal("0")

        # Check if rebalancing needed
        conservative_drift = abs(conservative_pct - self.conservative_ratio)
        aggressive_drift = abs(aggressive_pct - self.aggressive_ratio)
        time_since_rebalance = datetime.now() - self.last_rebalance

        rebalance_needed = (
            (conservative_drift > self.rebalance_threshold or
             aggressive_drift > self.rebalance_threshold) and
            time_since_rebalance > self.min_rebalance_interval
        )

        return BarbellAllocation(
            conservative_allocation=conservative_pct,
            aggressive_allocation=aggressive_pct,
            conservative_positions=conservative_positions,
            aggressive_positions=aggressive_positions,
            total_value=total_value,
            rebalance_needed=rebalance_needed,
            last_rebalance=self.last_rebalance
        )

    def find_contrarian_opportunities(self) -> List[ContrarianOpportunity]:
        """
        Find contrarian opportunities where consensus is wrong about inequality effects.

        This is where we look for "Gary moments" - times when everyone's missing
        the inequality angle that will drive massive repricing.

        Returns:
            List of high-conviction contrarian opportunities
        """
        opportunities = []

        if self.inequality_hunter:
            # Use AI to find mispricings
            raw_opportunities = self.inequality_hunter.find_consensus_blindspots()

            for opp in raw_opportunities:
                # Filter for high-conviction, high-payoff opportunities
                if (opp.conviction_score >= self.min_conviction_score and
                    opp.expected_payoff >= self.min_expected_payoff):

                    # Check correlation with existing bets
                    if not self._is_highly_correlated(opp):
                        opportunities.append(opp)
                        logger.info(f"Found contrarian opportunity: {opp.symbol} - {opp.thesis}")

        else:
            # Manual contrarian themes based on Gary's framework
            themes = self._get_manual_contrarian_themes()
            opportunities = self._evaluate_themes(themes)

        return opportunities

    def _get_manual_contrarian_themes(self) -> List[Dict[str, Any]]:
        """
        Manual contrarian themes based on Gary's inequality framework.

        These are the kinds of trades Gary would make:
        - Betting on continued wealth concentration
        - Betting against middle-class recovery
        - Betting on asset inflation from money printing
        """
        return [
            {
                'theme': 'wealth_concentration',
                'symbols': ['LVMUY', 'RL', 'RACE'],  # Luxury goods
                'thesis': 'Rich getting richer drives luxury consumption',
                'consensus': 'Recession will hurt luxury spending',
            },
            {
                'theme': 'housing_unaffordability',
                'symbols': ['REZ', 'XLRE', 'VNQ'],  # Real estate
                'thesis': 'Asset inflation continues despite rate hikes',
                'consensus': 'High rates will crash housing',
            },
            {
                'theme': 'consumer_weakness',
                'symbols': ['XRT', 'RTH', 'PMR'],  # Retail (short candidates)
                'thesis': 'Middle class has no money left to spend',
                'consensus': 'Consumer is resilient',
            },
            {
                'theme': 'financial_assets',
                'symbols': ['GLD', 'BTC-USD', 'GBTC'],  # Inflation hedges
                'thesis': 'Money printing drives asset prices despite economy',
                'consensus': 'Fed will successfully control inflation',
            }
        ]

    def _evaluate_themes(self, themes: List[Dict[str, Any]]) -> List[ContrarianOpportunity]:
        """
        Evaluate manual themes for trading opportunities.
        """
        opportunities = []

        for theme in themes:
            for symbol in theme['symbols']:
                # Get current market data
                try:
                    current_price = self.portfolio.market_data.get_current_price(symbol)

                    # Simple technical setup (real system would be more sophisticated)
                    opportunity = ContrarianOpportunity(
                        symbol=symbol,
                        thesis=theme['thesis'],
                        consensus_view=theme['consensus'],
                        contrarian_view=theme['thesis'],
                        inequality_correlation=0.8,  # High correlation with inequality
                        conviction_score=0.75,  # Manual themes get moderate conviction
                        expected_payoff=4.0,  # Target 4:1 payoff
                        timeframe_days=90,
                        entry_price=current_price,
                        target_price=current_price * Decimal("1.20"),  # 20% target
                        stop_loss=current_price * Decimal("0.95"),  # 5% stop
                        created_at=datetime.now()
                    )
                    opportunities.append(opportunity)

                except Exception as e:
                    logger.warning(f"Could not evaluate {symbol}: {e}")

        return opportunities

    def _is_highly_correlated(self, opportunity: ContrarianOpportunity) -> bool:
        """
        Check if opportunity is highly correlated with existing bets.

        We want diversification even within our contrarian bets.
        """
        if not self.active_contrarian_bets:
            return False

        # Simple correlation check (real system would use returns correlation)
        for symbol, existing in self.active_contrarian_bets.items():
            if opportunity.thesis == existing.thesis:
                return True  # Same thesis = high correlation

        return False

    def calculate_position_sizes(self,
                                opportunities: List[ContrarianOpportunity]) -> Dict[str, Decimal]:
        """
        Calculate position sizes for contrarian opportunities.

        Uses Kelly Criterion modified for barbell approach.
        """
        allocation = self.analyze_current_allocation()
        aggressive_capital = allocation.total_value * self.aggressive_ratio

        position_sizes = {}
        remaining_capital = aggressive_capital

        # Sort by conviction * expected payoff (best opportunities first)
        opportunities.sort(
            key=lambda x: x.conviction_score * x.expected_payoff,
            reverse=True
        )

        for opp in opportunities:
            # Kelly Criterion for position sizing
            win_prob = opp.conviction_score
            loss_prob = 1 - win_prob
            win_amount = opp.expected_payoff - 1  # Payoff minus principal

            # Kelly formula: f = (p*b - q) / b
            # where p = win prob, q = loss prob, b = win/loss ratio
            kelly_fraction = (win_prob * win_amount - loss_prob) / win_amount

            # Apply safety factors
            kelly_fraction = max(0, kelly_fraction)  # Never negative
            kelly_fraction *= 0.25  # Kelly/4 for safety (Gary would probably use Kelly/2)

            # Calculate position size
            position_value = min(
                aggressive_capital * Decimal(str(kelly_fraction)),
                allocation.total_value * self.max_single_bet_size,
                remaining_capital
            )

            if position_value > 0:
                shares = position_value / opp.entry_price
                position_sizes[opp.symbol] = shares.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                remaining_capital -= position_value

                logger.info(f"Sized {opp.symbol}: {shares:.0f} shares "
                          f"(${position_value:.2f}, Kelly={kelly_fraction:.2%})")

        return position_sizes

    def rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalance portfolio to target barbell allocation.

        Returns:
            Rebalancing trades and results
        """
        allocation = self.analyze_current_allocation()

        if not allocation.rebalance_needed:
            return {'status': 'no_rebalance_needed', 'allocation': allocation}

        logger.info("Rebalancing portfolio to target barbell allocation")

        trades = []
        total_value = allocation.total_value

        # Calculate target values
        target_conservative = total_value * self.conservative_ratio
        total_value * self.aggressive_ratio

        # Rebalance conservative portfolio
        for symbol, target_weight in self.conservative_instruments.items():
            target_value = target_conservative * target_weight
            current_value = allocation.conservative_positions.get(symbol, Decimal("0"))

            diff_value = target_value - current_value

            if abs(diff_value) > Decimal("10"):  # Minimum trade size
                current_price = self.portfolio.market_data.get_current_price(symbol)
                shares = (diff_value / current_price).quantize(Decimal("1"), rounding=ROUND_HALF_UP)

                if shares != 0:
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy' if shares > 0 else 'sell',
                        'shares': abs(shares),
                        'reason': 'barbell_rebalance_conservative'
                    })

        # Update contrarian bets if needed
        opportunities = self.find_contrarian_opportunities()
        if opportunities:
            position_sizes = self.calculate_position_sizes(opportunities)

            for symbol, shares in position_sizes.items():
                opp = next(o for o in opportunities if o.symbol == symbol)
                self.active_contrarian_bets[symbol] = opp

                trades.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'reason': f'contrarian_bet: {opp.thesis[:50]}'
                })

        # Execute trades
        results = []
        for trade in trades:
            try:
                result = self.portfolio.execute_trade(
                    symbol=trade['symbol'],
                    action=trade['action'],
                    shares=trade['shares'],
                    reason=trade['reason']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Trade failed: {trade} - {e}")

        self.last_rebalance = datetime.now()

        return {
            'status': 'rebalanced',
            'trades': trades,
            'results': results,
            'new_allocation': self.analyze_current_allocation()
        }

    def monitor_contrarian_bets(self) -> List[Dict[str, Any]]:
        """
        Monitor active contrarian bets for exit signals.

        Returns:
            List of exit signals and actions
        """
        exit_signals = []

        for symbol, opportunity in self.active_contrarian_bets.items():
            current_price = self.portfolio.market_data.get_current_price(symbol)

            # Check exit conditions
            if current_price >= opportunity.target_price:
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'take_profit',
                    'reason': f'Target reached: {current_price:.2f} >= {opportunity.target_price:.2f}',
                    'return': float((current_price - opportunity.entry_price) / opportunity.entry_price)
                })

            elif current_price <= opportunity.stop_loss:
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'stop_loss',
                    'reason': f'Stop hit: {current_price:.2f} <= {opportunity.stop_loss:.2f}',
                    'return': float((current_price - opportunity.entry_price) / opportunity.entry_price)
                })

            elif (datetime.now() - opportunity.created_at).days > opportunity.timeframe_days:
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'timeout',
                    'reason': f'Timeframe exceeded: {opportunity.timeframe_days} days',
                    'return': float((current_price - opportunity.entry_price) / opportunity.entry_price)
                })

        # Execute exits
        for signal in exit_signals:
            position = self.portfolio.get_position(signal['symbol'])
            if position:
                self.portfolio.execute_trade(
                    symbol=signal['symbol'],
                    action='sell',
                    shares=position.quantity,
                    reason=signal['reason']
                )

                # Track performance
                self.aggressive_returns.append(signal['return'])

                # Remove from active bets
                del self.active_contrarian_bets[signal['symbol']]

                logger.info(f"Exited contrarian bet: {signal}")

        return exit_signals

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for the barbell strategy.

        Returns:
            Dictionary of performance and risk metrics
        """
        allocation = self.analyze_current_allocation()

        # Calculate returns
        conservative_return = np.mean(self.conservative_returns) if self.conservative_returns else 0
        aggressive_return = np.mean(self.aggressive_returns) if self.aggressive_returns else 0
        total_return = np.mean(self.total_returns) if self.total_returns else 0

        # Calculate risk metrics
        conservative_vol = np.std(self.conservative_returns) if len(self.conservative_returns) > 1 else 0
        aggressive_vol = np.std(self.aggressive_returns) if len(self.aggressive_returns) > 1 else 0

        # Sharpe ratios (assuming 3% risk-free rate)
        risk_free = 0.03
        conservative_sharpe = ((conservative_return - risk_free) / conservative_vol
                              if conservative_vol > 0 else 0)
        aggressive_sharpe = ((aggressive_return - risk_free) / aggressive_vol
                           if aggressive_vol > 0 else 0)

        return {
            'allocation': {
                'conservative_pct': float(allocation.conservative_allocation),
                'aggressive_pct': float(allocation.aggressive_allocation),
                'rebalance_needed': allocation.rebalance_needed
            },
            'returns': {
                'conservative': conservative_return,
                'aggressive': aggressive_return,
                'total': total_return
            },
            'risk': {
                'conservative_volatility': conservative_vol,
                'aggressive_volatility': aggressive_vol,
                'conservative_sharpe': conservative_sharpe,
                'aggressive_sharpe': aggressive_sharpe
            },
            'contrarian_bets': {
                'active': len(self.active_contrarian_bets),
                'opportunities': len(self.find_contrarian_opportunities()),
                'avg_conviction': np.mean([b.conviction_score for b in self.active_contrarian_bets.values()])
                                if self.active_contrarian_bets else 0,
                'avg_payoff': np.mean([b.expected_payoff for b in self.active_contrarian_bets.values()])
                            if self.active_contrarian_bets else 0
            }
        }

    def explain_strategy(self) -> str:
        """
        Explain the barbell strategy in Gary's terms.

        Returns:
            Human-readable explanation of current strategy state
        """
        metrics = self.get_strategy_metrics()
        allocation = self.analyze_current_allocation()

        explanation = f"""
BARBELL STRATEGY STATUS - The Gary Way
=====================================

PHILOSOPHY: "Don't tell me your opinion, show me your position"

CURRENT ALLOCATION:
- Conservative (Safety Net): {metrics['allocation']['conservative_pct']:.1%}
  Target: {float(self.conservative_ratio):.0%}
  Purpose: Survive anything, even if we're wrong

- Aggressive (Gary Bets): {metrics['allocation']['aggressive_pct']:.1%}
  Target: {float(self.aggressive_ratio):.0%}
  Purpose: Massive wins when consensus is wrong about inequality

CONTRARIAN BETS (The Gary Moments):
- Active Positions: {metrics['contrarian_bets']['active']}
- Available Opportunities: {metrics['contrarian_bets']['opportunities']}
- Average Conviction: {metrics['contrarian_bets']['avg_conviction']:.1%}
- Average Expected Payoff: {metrics['contrarian_bets']['avg_payoff']:.1f}x

PERFORMANCE:
- Conservative Return: {metrics['returns']['conservative']:.1%}
- Aggressive Return: {metrics['returns']['aggressive']:.1%}
- Total Return: {metrics['returns']['total']:.1%}

RISK METRICS:
- Conservative Sharpe: {metrics['risk']['conservative_sharpe']:.2f}
- Aggressive Sharpe: {metrics['risk']['aggressive_sharpe']:.2f}

ACTIVE CONTRARIAN THEMES:
"""

        for symbol, opp in self.active_contrarian_bets.items():
            explanation += f"\n{symbol}: {opp.thesis}"
            explanation += f"\n  Consensus Wrong About: {opp.consensus_view}"
            explanation += f"\n  We Believe: {opp.contrarian_view}"
            explanation += f"\n  Conviction: {opp.conviction_score:.0%}, Expected Payoff: {opp.expected_payoff:.1f}x\n"

        if allocation.rebalance_needed:
            explanation += "\n!!! REBALANCE NEEDED - Allocation has drifted from targets"

        explanation += """

REMEMBER:
"Growing inequality is destroying the economy. Everything that happens
can be predicted by understanding what will happen with growing inequality."
- Gary Stevenson

The 80% keeps us alive. The 20% makes us rich when we're right about
what everyone else is wrong about.
"""

        return explanation