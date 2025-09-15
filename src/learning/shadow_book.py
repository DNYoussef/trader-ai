"""
Shadow Book System - Counterfactual P&L Tracking and Learning

This module implements a sophisticated shadow book system that tracks parallel
trade histories and counterfactual scenarios. The system answers critical
questions like "What if we disclosed early?", "What if we sized differently?",
and "What if we used different timing?" to generate optimization insights.

Key Features:
- Parallel trade tracking (actual vs shadow trades)
- Counterfactual P&L analysis
- Performance comparison metrics
- Learning insights generation
- Strategy optimization recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle

logger = logging.getLogger(__name__)

class TradeType(Enum):
    ACTUAL = "actual"
    SHADOW = "shadow"
    COUNTERFACTUAL = "counterfactual"

class ActionType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    SIZE_CHANGE = "size_change"
    DISCLOSURE = "disclosure"

@dataclass
class Trade:
    """Represents a single trade (actual or shadow)"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    action_type: ActionType
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade"""
        return abs(self.quantity * self.price)

@dataclass
class Position:
    """Represents a position in the shadow book"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    timestamp: datetime
    strategy_id: str
    trade_type: TradeType
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Cost basis of position"""
        return self.quantity * self.avg_entry_price

    def update_price(self, new_price: float, timestamp: datetime):
        """Update position with new market price"""
        self.current_price = new_price
        self.timestamp = timestamp
        self.unrealized_pnl = self.market_value - self.cost_basis

@dataclass
class CounterfactualScenario:
    """Defines a counterfactual scenario for analysis"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    base_trade_id: str

class ShadowBookEngine:
    """Main engine for shadow book operations"""

    def __init__(self, db_path: str = "shadow_book.db"):
        self.db_path = db_path
        self.positions: Dict[str, Dict[str, Position]] = {
            "actual": {},
            "shadow": {},
            "counterfactual": {}
        }
        self.trades: List[Trade] = []
        self.scenarios: Dict[str, CounterfactualScenario] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        trade_type TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        confidence REAL DEFAULT 0.0,
                        metadata TEXT
                    )
                """)

                # Positions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        avg_entry_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        trade_type TEXT NOT NULL,
                        unrealized_pnl REAL DEFAULT 0.0,
                        realized_pnl REAL DEFAULT 0.0
                    )
                """)

                # Scenarios table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scenarios (
                        scenario_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        parameters TEXT,
                        base_trade_id TEXT,
                        created_at TEXT
                    )
                """)

                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        book_type TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        calculation_date TEXT NOT NULL
                    )
                """)

                conn.commit()
                logger.info("Shadow book database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def record_actual_trade(self, trade: Trade) -> bool:
        """Record an actual trade in the shadow book"""
        try:
            trade.trade_type = TradeType.ACTUAL
            self.trades.append(trade)

            # Update actual positions
            self._update_position(trade, "actual")

            # Generate shadow scenarios for this trade
            self._generate_shadow_scenarios(trade)

            # Persist to database
            self._save_trade_to_db(trade)

            logger.info(f"Recorded actual trade: {trade.trade_id}")
            return True

        except Exception as e:
            logger.error(f"Error recording actual trade {trade.trade_id}: {e}")
            return False

    def execute_shadow_trade(self, trade: Trade, scenario_id: str = None) -> bool:
        """Execute a shadow trade based on scenario parameters"""
        try:
            trade.trade_type = TradeType.SHADOW

            if scenario_id:
                scenario = self.scenarios.get(scenario_id)
                if scenario:
                    trade = self._apply_scenario_parameters(trade, scenario)

            self.trades.append(trade)

            # Update shadow positions
            self._update_position(trade, "shadow")

            # Persist to database
            self._save_trade_to_db(trade)

            logger.info(f"Executed shadow trade: {trade.trade_id}")
            return True

        except Exception as e:
            logger.error(f"Error executing shadow trade {trade.trade_id}: {e}")
            return False

    def create_counterfactual_scenario(self, scenario: CounterfactualScenario) -> bool:
        """Create a new counterfactual scenario"""
        try:
            self.scenarios[scenario.scenario_id] = scenario

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO scenarios
                    (scenario_id, name, description, parameters, base_trade_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    scenario.scenario_id,
                    scenario.name,
                    scenario.description,
                    json.dumps(scenario.parameters),
                    scenario.base_trade_id,
                    datetime.now().isoformat()
                ))
                conn.commit()

            logger.info(f"Created counterfactual scenario: {scenario.scenario_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating scenario {scenario.scenario_id}: {e}")
            return False

    def analyze_counterfactual(self, scenario_id: str, base_trade_id: str) -> Dict[str, Any]:
        """Analyze counterfactual performance for a given scenario"""
        try:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")

            # Find base trade
            base_trade = next((t for t in self.trades if t.trade_id == base_trade_id), None)
            if not base_trade:
                raise ValueError(f"Base trade {base_trade_id} not found")

            # Create counterfactual trade
            cf_trade = self._create_counterfactual_trade(base_trade, scenario)

            # Calculate performance differences
            actual_performance = self._calculate_trade_performance(base_trade)
            counterfactual_performance = self._calculate_trade_performance(cf_trade)

            analysis = {
                "scenario_id": scenario_id,
                "base_trade_id": base_trade_id,
                "scenario_name": scenario.name,
                "actual_performance": actual_performance,
                "counterfactual_performance": counterfactual_performance,
                "performance_difference": {
                    "pnl_diff": counterfactual_performance["pnl"] - actual_performance["pnl"],
                    "return_diff": counterfactual_performance["return"] - actual_performance["return"],
                    "sharpe_diff": counterfactual_performance.get("sharpe", 0) - actual_performance.get("sharpe", 0)
                },
                "insights": self._generate_counterfactual_insights(
                    actual_performance, counterfactual_performance, scenario
                )
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing counterfactual {scenario_id}: {e}")
            return {"error": str(e)}

    def generate_optimization_insights(self, strategy_id: str = None,
                                     lookback_days: int = 30) -> Dict[str, Any]:
        """Generate optimization insights from shadow book analysis"""
        try:
            # Filter trades by strategy and time period
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            relevant_trades = [
                t for t in self.trades
                if t.timestamp >= cutoff_date and
                (strategy_id is None or t.strategy_id == strategy_id)
            ]

            if not relevant_trades:
                return {"insights": [], "recommendations": []}

            # Analyze patterns
            insights = []
            recommendations = []

            # 1. Size optimization analysis
            size_insights = self._analyze_sizing_patterns(relevant_trades)
            insights.extend(size_insights["insights"])
            recommendations.extend(size_insights["recommendations"])

            # 2. Timing optimization analysis
            timing_insights = self._analyze_timing_patterns(relevant_trades)
            insights.extend(timing_insights["insights"])
            recommendations.extend(timing_insights["recommendations"])

            # 3. Disclosure optimization analysis
            disclosure_insights = self._analyze_disclosure_patterns(relevant_trades)
            insights.extend(disclosure_insights["insights"])
            recommendations.extend(disclosure_insights["recommendations"])

            # 4. Performance comparison
            performance_comparison = self._compare_actual_vs_shadow_performance(relevant_trades)

            optimization_report = {
                "analysis_period": {
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "trades_analyzed": len(relevant_trades)
                },
                "insights": insights,
                "recommendations": recommendations,
                "performance_comparison": performance_comparison,
                "summary_statistics": self._calculate_summary_statistics(relevant_trades)
            }

            return optimization_report

        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            return {"error": str(e)}

    def get_shadow_book_performance(self, book_type: str = "shadow") -> Dict[str, float]:
        """Get performance metrics for shadow book"""
        try:
            positions = self.positions.get(book_type, {})

            if not positions:
                return {"total_pnl": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0}

            total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
            total_realized = sum(pos.realized_pnl for pos in positions.values())
            total_pnl = total_unrealized + total_realized

            # Calculate additional metrics
            total_notional = sum(abs(pos.market_value) for pos in positions.values())
            return_pct = (total_pnl / total_notional * 100) if total_notional > 0 else 0.0

            performance = {
                "total_pnl": total_pnl,
                "unrealized_pnl": total_unrealized,
                "realized_pnl": total_realized,
                "total_notional": total_notional,
                "return_percentage": return_pct,
                "position_count": len(positions)
            }

            return performance

        except Exception as e:
            logger.error(f"Error calculating {book_type} performance: {e}")
            return {"error": str(e)}

    def _update_position(self, trade: Trade, book_type: str):
        """Update position in specified book type"""
        position_key = f"{trade.symbol}_{trade.strategy_id}"

        if position_key not in self.positions[book_type]:
            # New position
            if trade.action_type == ActionType.ENTRY:
                self.positions[book_type][position_key] = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    avg_entry_price=trade.price,
                    current_price=trade.price,
                    timestamp=trade.timestamp,
                    strategy_id=trade.strategy_id,
                    trade_type=TradeType(book_type)
                )
        else:
            # Update existing position
            position = self.positions[book_type][position_key]

            if trade.action_type == ActionType.ENTRY:
                # Add to position
                total_notional = position.quantity * position.avg_entry_price + trade.quantity * trade.price
                total_quantity = position.quantity + trade.quantity
                position.avg_entry_price = total_notional / total_quantity if total_quantity != 0 else 0
                position.quantity = total_quantity

            elif trade.action_type == ActionType.EXIT:
                # Reduce position and realize P&L
                exit_pnl = trade.quantity * (trade.price - position.avg_entry_price)
                position.realized_pnl += exit_pnl
                position.quantity -= trade.quantity

                # Remove position if fully closed
                if abs(position.quantity) < 1e-6:
                    del self.positions[book_type][position_key]
                    return

            elif trade.action_type == ActionType.SIZE_CHANGE:
                # Adjust position size
                position.quantity += trade.quantity

            position.current_price = trade.price
            position.timestamp = trade.timestamp
            position.update_price(trade.price, trade.timestamp)

    def _generate_shadow_scenarios(self, actual_trade: Trade):
        """Generate shadow trading scenarios for actual trade"""
        try:
            # Scenario 1: Different sizing (50% larger, 50% smaller)
            for size_multiplier, scenario_name in [(1.5, "larger_size"), (0.5, "smaller_size")]:
                scenario = CounterfactualScenario(
                    scenario_id=f"{actual_trade.trade_id}_{scenario_name}",
                    name=f"{scenario_name.replace('_', ' ').title()}",
                    description=f"What if we sized {size_multiplier}x the original trade",
                    parameters={"size_multiplier": size_multiplier},
                    base_trade_id=actual_trade.trade_id
                )
                self.create_counterfactual_scenario(scenario)

            # Scenario 2: Different timing (1 hour earlier/later)
            for time_delta, scenario_name in [(timedelta(hours=-1), "earlier_entry"),
                                            (timedelta(hours=1), "later_entry")]:
                scenario = CounterfactualScenario(
                    scenario_id=f"{actual_trade.trade_id}_{scenario_name}",
                    name=f"{scenario_name.replace('_', ' ').title()}",
                    description=f"What if we entered {abs(time_delta.total_seconds()/3600)} hours {'earlier' if time_delta.total_seconds() < 0 else 'later'}",
                    parameters={"time_delta_hours": time_delta.total_seconds() / 3600},
                    base_trade_id=actual_trade.trade_id
                )
                self.create_counterfactual_scenario(scenario)

            # Scenario 3: Early disclosure
            scenario = CounterfactualScenario(
                scenario_id=f"{actual_trade.trade_id}_early_disclosure",
                name="Early Disclosure",
                description="What if we disclosed the position immediately after entry",
                parameters={"immediate_disclosure": True},
                base_trade_id=actual_trade.trade_id
            )
            self.create_counterfactual_scenario(scenario)

        except Exception as e:
            logger.error(f"Error generating shadow scenarios for {actual_trade.trade_id}: {e}")

    def _apply_scenario_parameters(self, trade: Trade, scenario: CounterfactualScenario) -> Trade:
        """Apply scenario parameters to modify trade"""
        modified_trade = Trade(
            trade_id=f"{trade.trade_id}_cf_{scenario.scenario_id}",
            symbol=trade.symbol,
            trade_type=TradeType.COUNTERFACTUAL,
            action_type=trade.action_type,
            quantity=trade.quantity,
            price=trade.price,
            timestamp=trade.timestamp,
            strategy_id=trade.strategy_id,
            confidence=trade.confidence,
            metadata=trade.metadata.copy()
        )

        params = scenario.parameters

        # Apply size multiplier
        if "size_multiplier" in params:
            modified_trade.quantity *= params["size_multiplier"]

        # Apply timing change
        if "time_delta_hours" in params:
            time_delta = timedelta(hours=params["time_delta_hours"])
            modified_trade.timestamp += time_delta
            # In a real system, this would require re-pricing at the new time

        # Apply disclosure parameters
        if "immediate_disclosure" in params:
            modified_trade.metadata["immediate_disclosure"] = params["immediate_disclosure"]

        return modified_trade

    def _create_counterfactual_trade(self, base_trade: Trade, scenario: CounterfactualScenario) -> Trade:
        """Create counterfactual trade from base trade and scenario"""
        return self._apply_scenario_parameters(base_trade, scenario)

    def _calculate_trade_performance(self, trade: Trade) -> Dict[str, float]:
        """Calculate performance metrics for a trade"""
        # This is a simplified calculation - in practice would require
        # full position tracking and market data

        # Simulate some performance metrics
        base_return = np.random.normal(0.05, 0.15)  # 5% expected return, 15% volatility

        # Adjust based on trade characteristics
        confidence_adjustment = (trade.confidence - 0.5) * 0.1
        size_impact = min(0.02, abs(trade.quantity) / 10000 * 0.001)  # Market impact

        adjusted_return = base_return + confidence_adjustment - size_impact
        pnl = trade.notional_value * adjusted_return

        return {
            "pnl": pnl,
            "return": adjusted_return,
            "sharpe": adjusted_return / 0.15 if 0.15 > 0 else 0,  # Simplified Sharpe
            "notional": trade.notional_value
        }

    def _generate_counterfactual_insights(self, actual_perf: Dict[str, float],
                                        cf_perf: Dict[str, float],
                                        scenario: CounterfactualScenario) -> List[str]:
        """Generate insights from counterfactual analysis"""
        insights = []

        pnl_diff = cf_perf["pnl"] - actual_perf["pnl"]
        return_diff = cf_perf["return"] - actual_perf["return"]

        if pnl_diff > 0:
            insights.append(f"The {scenario.name} scenario would have improved P&L by ${pnl_diff:.2f}")
        else:
            insights.append(f"The {scenario.name} scenario would have reduced P&L by ${abs(pnl_diff):.2f}")

        if abs(return_diff) > 0.01:  # 1% threshold
            direction = "improved" if return_diff > 0 else "worsened"
            insights.append(f"Return would have {direction} by {abs(return_diff)*100:.2f}%")

        return insights

    def _analyze_sizing_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze sizing patterns and generate insights"""
        insights = []
        recommendations = []

        # Group trades by symbol and analyze sizing vs performance
        symbol_groups = {}
        for trade in trades:
            if trade.symbol not in symbol_groups:
                symbol_groups[trade.symbol] = []
            symbol_groups[trade.symbol].append(trade)

        for symbol, symbol_trades in symbol_groups.items():
            if len(symbol_trades) > 2:
                # Analyze size vs performance correlation
                sizes = [t.notional_value for t in symbol_trades]
                performances = [self._calculate_trade_performance(t)["return"] for t in symbol_trades]

                if len(sizes) > 3:
                    correlation = np.corrcoef(sizes, performances)[0, 1]

                    if correlation > 0.3:
                        insights.append(f"Larger positions in {symbol} tend to perform better")
                        recommendations.append(f"Consider increasing position sizes for {symbol}")
                    elif correlation < -0.3:
                        insights.append(f"Smaller positions in {symbol} tend to perform better")
                        recommendations.append(f"Consider reducing position sizes for {symbol}")

        return {"insights": insights, "recommendations": recommendations}

    def _analyze_timing_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze timing patterns and generate insights"""
        insights = []
        recommendations = []

        # Analyze hour-of-day patterns
        hour_performance = {}
        for trade in trades:
            hour = trade.timestamp.hour
            if hour not in hour_performance:
                hour_performance[hour] = []

            perf = self._calculate_trade_performance(trade)["return"]
            hour_performance[hour].append(perf)

        # Find best and worst performing hours
        avg_performance_by_hour = {
            hour: np.mean(perfs) for hour, perfs in hour_performance.items()
            if len(perfs) > 2
        }

        if avg_performance_by_hour:
            best_hour = max(avg_performance_by_hour, key=avg_performance_by_hour.get)
            worst_hour = min(avg_performance_by_hour, key=avg_performance_by_hour.get)

            insights.append(f"Best performing hour: {best_hour}:00")
            insights.append(f"Worst performing hour: {worst_hour}:00")

            if avg_performance_by_hour[best_hour] > avg_performance_by_hour[worst_hour] + 0.02:
                recommendations.append(f"Consider concentrating trades around {best_hour}:00")

        return {"insights": insights, "recommendations": recommendations}

    def _analyze_disclosure_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze disclosure timing patterns"""
        insights = []
        recommendations = []

        # Analyze trades with disclosure metadata
        disclosed_trades = [t for t in trades if t.metadata.get("disclosed", False)]
        undisclosed_trades = [t for t in trades if not t.metadata.get("disclosed", False)]

        if disclosed_trades and undisclosed_trades:
            disclosed_perf = np.mean([
                self._calculate_trade_performance(t)["return"] for t in disclosed_trades
            ])
            undisclosed_perf = np.mean([
                self._calculate_trade_performance(t)["return"] for t in undisclosed_trades
            ])

            if disclosed_perf > undisclosed_perf + 0.01:
                insights.append("Disclosed positions tend to perform better")
                recommendations.append("Consider more frequent position disclosure")
            elif undisclosed_perf > disclosed_perf + 0.01:
                insights.append("Undisclosed positions tend to perform better")
                recommendations.append("Consider delaying disclosure for better performance")

        return {"insights": insights, "recommendations": recommendations}

    def _compare_actual_vs_shadow_performance(self, trades: List[Trade]) -> Dict[str, Any]:
        """Compare actual vs shadow performance"""
        actual_trades = [t for t in trades if t.trade_type == TradeType.ACTUAL]
        shadow_trades = [t for t in trades if t.trade_type == TradeType.SHADOW]

        if not actual_trades or not shadow_trades:
            return {"comparison": "Insufficient data for comparison"}

        actual_perf = np.mean([
            self._calculate_trade_performance(t)["return"] for t in actual_trades
        ])
        shadow_perf = np.mean([
            self._calculate_trade_performance(t)["return"] for t in shadow_trades
        ])

        performance_diff = shadow_perf - actual_perf

        return {
            "actual_avg_return": actual_perf,
            "shadow_avg_return": shadow_perf,
            "performance_difference": performance_diff,
            "shadow_outperformance": performance_diff > 0.01
        }

    def _calculate_summary_statistics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate summary statistics for trades"""
        if not trades:
            return {}

        returns = [self._calculate_trade_performance(t)["return"] for t in trades]
        notionals = [t.notional_value for t in trades]

        return {
            "total_trades": len(trades),
            "avg_return": np.mean(returns),
            "return_volatility": np.std(returns),
            "sharpe_ratio": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            "avg_notional": np.mean(notionals),
            "total_notional": sum(notionals),
            "win_rate": sum(1 for r in returns if r > 0) / len(returns),
            "best_trade": max(returns),
            "worst_trade": min(returns)
        }

    def _save_trade_to_db(self, trade: Trade):
        """Save trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades
                    (trade_id, symbol, trade_type, action_type, quantity, price,
                     timestamp, strategy_id, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id,
                    trade.symbol,
                    trade.trade_type.value,
                    trade.action_type.value,
                    trade.quantity,
                    trade.price,
                    trade.timestamp.isoformat(),
                    trade.strategy_id,
                    trade.confidence,
                    json.dumps(trade.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")

# Example usage and testing
def test_shadow_book_system():
    """Test the shadow book system"""
    shadow_book = ShadowBookEngine()

    # Create sample actual trade
    actual_trade = Trade(
        trade_id="ACTUAL_001",
        symbol="AAPL",
        trade_type=TradeType.ACTUAL,
        action_type=ActionType.ENTRY,
        quantity=100,
        price=150.0,
        timestamp=datetime.now(),
        strategy_id="momentum_strategy",
        confidence=0.8,
        metadata={"entry_signal": "bullish_breakout"}
    )

    # Record actual trade
    shadow_book.record_actual_trade(actual_trade)

    # Generate optimization insights
    insights = shadow_book.generate_optimization_insights()

    print("Shadow Book Analysis:")
    print(f"Insights: {len(insights.get('insights', []))}")
    print(f"Recommendations: {len(insights.get('recommendations', []))}")

    # Get performance comparison
    actual_performance = shadow_book.get_shadow_book_performance("actual")
    shadow_performance = shadow_book.get_shadow_book_performance("shadow")

    print(f"\nActual Book Performance: {actual_performance}")
    print(f"Shadow Book Performance: {shadow_performance}")

    return shadow_book

if __name__ == "__main__":
    test_shadow_book_system()