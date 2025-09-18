"""
Launch Black Swan Trading System
Integrates the black swan hunting AI with the existing trading engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing trading system components
from src.trading_engine import TradingEngine
from src.brokers.alpaca_adapter import AlpacaAdapter
from src.gates.gate_manager import GateManager
from src.portfolio.portfolio_manager import PortfolioManager
from src.market.market_data_provider import MarketDataProvider

# Import black swan hunting components
from src.data.historical_data_manager import HistoricalDataManager
from src.data.black_swan_labeler import BlackSwanLabeler
from src.strategies.black_swan_strategies import (
    BlackSwanStrategyToolbox,
    MarketState,
    StrategySignal
)
from src.strategies.convex_reward_function import (
    ConvexRewardFunction,
    TradeOutcome
)
from src.intelligence.local_llm_orchestrator import (
    LocalLLMOrchestrator,
    MarketContext
)

class BlackSwanTradingSystem:
    """
    Main integration class for Black Swan Trading System
    Combines Taleb's barbell strategy with AI-selected trading algorithms
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Black Swan Trading System

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get('mode', 'paper')

        # Initialize existing trading components
        logger.info("Initializing core trading components...")
        self.trading_engine = TradingEngine(config)
        self.gate_manager = GateManager()
        self.portfolio_manager = self.trading_engine.portfolio_manager
        self.market_data_provider = self.trading_engine.market_data_provider

        # Initialize black swan components
        logger.info("Initializing black swan hunting components...")
        self.historical_manager = HistoricalDataManager()
        self.black_swan_labeler = BlackSwanLabeler()
        self.strategy_toolbox = BlackSwanStrategyToolbox(
            portfolio_manager=self.portfolio_manager,
            market_data_provider=self.market_data_provider
        )
        self.convex_reward = ConvexRewardFunction()
        self.llm_orchestrator = LocalLLMOrchestrator()

        # Barbell strategy allocation (80% safe, 20% aggressive)
        self.barbell_allocation = {
            'safe': 0.80,  # 80% in boring investments
            'aggressive': 0.20  # 20% for black swan hunting
        }

        # Track active strategies
        self.active_strategies = {}
        self.performance_history = []

        logger.info(f"Black Swan Trading System initialized in {self.mode} mode")

    async def start(self):
        """Start the black swan trading system"""
        logger.info("Starting Black Swan Trading System...")

        # Initialize trading engine
        await self.trading_engine.initialize()

        # Check LLM availability
        llm_available = self.llm_orchestrator.check_ollama_status()
        if llm_available:
            logger.info("✅ Local LLM (Ollama/Mistral) is available")
        else:
            logger.warning("⚠️ Local LLM not available, using fallback strategy selection")

        # Load historical data for context
        self._load_historical_context()

        # Start main trading loop
        await self._trading_loop()

    def _load_historical_context(self):
        """Load historical market data and black swan events"""
        logger.info("Loading historical context...")

        # Get recent market data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')

        self.recent_data = self.historical_manager.get_training_data(
            start_date=start_date,
            end_date=end_date,
            symbols=['SPY', 'QQQ', 'VIX']
        )

        # Get black swan events
        self.black_swan_events = self.historical_manager.get_black_swan_events()
        logger.info(f"Loaded {len(self.black_swan_events)} historical black swan events")

    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("Entering main trading loop...")

        while True:
            try:
                # Get current market state
                market_state = await self._get_market_state()

                # Determine capital allocation
                available_capital = self._calculate_available_capital()

                # Get AI strategy recommendation
                strategy_signal = await self._get_ai_strategy_recommendation(
                    market_state,
                    available_capital
                )

                if strategy_signal:
                    # Execute the strategy
                    await self._execute_strategy(strategy_signal)

                # Monitor existing positions
                await self._monitor_positions()

                # Update performance metrics
                self._update_performance_metrics()

                # Sleep based on market conditions
                sleep_time = self._calculate_sleep_time(market_state)
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

    async def _get_market_state(self) -> MarketState:
        """Get current market state"""

        # Get real-time market data
        spy_data = await self.market_data_provider.get_latest_data('SPY')
        vix_data = await self.market_data_provider.get_latest_data('VIX')

        # Calculate metrics
        spy_returns_5d = self._calculate_returns(spy_data, 5)
        spy_returns_20d = self._calculate_returns(spy_data, 20)

        # Get market internals
        market_breadth = await self._calculate_market_breadth()
        put_call_ratio = await self._get_put_call_ratio()
        correlation = await self._calculate_correlation()
        volume_ratio = await self._calculate_volume_ratio()

        # Determine regime
        vix_level = vix_data.get('close', 20.0) if vix_data else 20.0
        regime = self._determine_regime(vix_level, spy_returns_20d)

        return MarketState(
            timestamp=datetime.now(),
            vix_level=vix_level,
            vix_percentile=self._calculate_vix_percentile(vix_level),
            spy_returns_5d=spy_returns_5d,
            spy_returns_20d=spy_returns_20d,
            put_call_ratio=put_call_ratio,
            market_breadth=market_breadth,
            correlation=correlation,
            volume_ratio=volume_ratio,
            regime=regime
        )

    def _calculate_available_capital(self) -> float:
        """Calculate capital available for black swan hunting"""

        total_capital = self.portfolio_manager.get_total_value()
        current_gate = self.gate_manager.current_gate

        # Apply barbell allocation
        aggressive_capital = total_capital * self.barbell_allocation['aggressive']

        # Apply gate constraints
        max_position_size = current_gate.max_position_size * total_capital

        return min(aggressive_capital, max_position_size)

    async def _get_ai_strategy_recommendation(self,
                                             market_state: MarketState,
                                             available_capital: float) -> Optional[StrategySignal]:
        """Get AI-recommended strategy"""

        # Create market context for LLM
        market_context = MarketContext(
            timestamp=market_state.timestamp,
            vix_level=market_state.vix_level,
            vix_percentile=market_state.vix_percentile,
            spy_returns_5d=market_state.spy_returns_5d,
            spy_returns_20d=market_state.spy_returns_20d,
            put_call_ratio=market_state.put_call_ratio,
            market_breadth=market_state.market_breadth,
            correlation=market_state.correlation,
            volume_ratio=market_state.volume_ratio,
            recent_events=self._get_recent_events(),
            sector_performance=await self._get_sector_performance(),
            black_swan_indicators=self._calculate_black_swan_indicators(market_state)
        )

        # Get LLM recommendation
        recommendation = self.llm_orchestrator.get_strategy_recommendation(
            market_context,
            available_capital
        )

        if not recommendation:
            return None

        # Convert to strategy signal
        strategy = self.strategy_toolbox.strategies.get(recommendation.strategy_name)
        if not strategy:
            logger.warning(f"Strategy {recommendation.strategy_name} not found")
            return None

        # Generate signal with LLM parameters
        signal = strategy.generate_signal(market_state, self.recent_data)

        if signal:
            # Override with LLM recommendations
            signal.confidence = recommendation.confidence
            signal.position_size *= recommendation.position_size
            signal.reasoning = recommendation.reasoning

        return signal

    async def _execute_strategy(self, signal: StrategySignal):
        """Execute a trading strategy"""

        logger.info(f"Executing strategy: {signal.strategy_name}")
        logger.info(f"Action: {signal.action} {signal.symbol}")
        logger.info(f"Confidence: {signal.confidence:.1%}")
        logger.info(f"Expected convexity: {signal.convexity_ratio:.1f}x")

        try:
            # Calculate position size
            position_size = signal.position_size

            # Execute trade through trading engine
            if signal.action == 'BUY':
                order = await self.trading_engine.execute_trade(
                    symbol=signal.symbol,
                    quantity=position_size,
                    side='buy',
                    strategy_name=signal.strategy_name
                )
            elif signal.action == 'SELL':
                order = await self.trading_engine.execute_trade(
                    symbol=signal.symbol,
                    quantity=position_size,
                    side='sell',
                    strategy_name=signal.strategy_name
                )
            else:
                logger.warning(f"Unknown action: {signal.action}")
                return

            # Track active strategy
            if order:
                self.active_strategies[order['id']] = {
                    'strategy_name': signal.strategy_name,
                    'entry_time': datetime.now(),
                    'signal': signal,
                    'order': order
                }

                logger.info(f"Strategy executed successfully: {order['id']}")

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")

    async def _monitor_positions(self):
        """Monitor and manage existing positions"""

        positions = self.portfolio_manager.get_positions()

        for position_id, position in positions.items():
            if position_id in self.active_strategies:
                strategy_info = self.active_strategies[position_id]

                # Check exit conditions
                should_exit = await self._check_exit_conditions(
                    position,
                    strategy_info
                )

                if should_exit:
                    await self._close_position(position_id, position, strategy_info)

    async def _check_exit_conditions(self,
                                    position: Dict[str, Any],
                                    strategy_info: Dict[str, Any]) -> bool:
        """Check if position should be closed"""

        # Get current price
        current_data = await self.market_data_provider.get_latest_data(
            position['symbol']
        )
        current_price = current_data.get('close', position['entry_price'])

        # Calculate returns
        returns = (current_price - position['entry_price']) / position['entry_price']

        # Check stop loss
        if returns <= -0.05:  # 5% stop loss
            logger.info(f"Stop loss triggered for {position['symbol']}")
            return True

        # Check take profit
        if returns >= 0.20:  # 20% take profit
            logger.info(f"Take profit triggered for {position['symbol']}")
            return True

        # Check strategy-specific exit conditions
        signal = strategy_info['signal']
        if hasattr(signal, 'exit_conditions'):
            # Would need more sophisticated exit condition checking
            pass

        return False

    async def _close_position(self,
                            position_id: str,
                            position: Dict[str, Any],
                            strategy_info: Dict[str, Any]):
        """Close a position and record outcome"""

        logger.info(f"Closing position: {position_id}")

        # Execute close order
        close_order = await self.trading_engine.execute_trade(
            symbol=position['symbol'],
            quantity=position['quantity'],
            side='sell' if position['side'] == 'buy' else 'buy',
            strategy_name=strategy_info['strategy_name']
        )

        if close_order:
            # Calculate outcome
            exit_price = close_order.get('price', position['entry_price'])
            returns = (exit_price - position['entry_price']) / position['entry_price']

            # Create trade outcome
            outcome = TradeOutcome(
                strategy_name=strategy_info['strategy_name'],
                entry_date=strategy_info['entry_time'],
                exit_date=datetime.now(),
                symbol=position['symbol'],
                returns=returns,
                max_drawdown=self._calculate_max_drawdown(position_id),
                holding_period_days=(datetime.now() - strategy_info['entry_time']).days,
                volatility_during_trade=self._calculate_trade_volatility(position_id),
                is_black_swan_period=self._is_black_swan_period(),
                black_swan_captured=returns > 0.5,  # 50%+ gain
                convexity_achieved=max(returns / 0.05, 1.0)  # Actual vs expected
            )

            # Calculate reward
            reward_metrics = self.convex_reward.calculate_reward(outcome)

            logger.info(f"Position closed - Returns: {returns:.2%}, "
                       f"Reward: {reward_metrics.final_reward:.2f}")

            # Store outcome
            self.performance_history.append({
                'outcome': outcome,
                'reward': reward_metrics
            })

            # Remove from active strategies
            del self.active_strategies[position_id]

    def _update_performance_metrics(self):
        """Update and log performance metrics"""

        if not self.performance_history:
            return

        # Calculate aggregate metrics
        total_trades = len(self.performance_history)
        winning_trades = sum(1 for h in self.performance_history
                           if h['outcome'].returns > 0)
        black_swan_captures = sum(1 for h in self.performance_history
                                if h['outcome'].black_swan_captured)

        avg_return = sum(h['outcome'].returns for h in self.performance_history) / total_trades
        avg_reward = sum(h['reward'].final_reward for h in self.performance_history) / total_trades

        logger.info(f"Performance Update - Trades: {total_trades}, "
                   f"Win Rate: {winning_trades/total_trades:.1%}, "
                   f"Black Swans: {black_swan_captures}, "
                   f"Avg Return: {avg_return:.2%}, "
                   f"Avg Reward: {avg_reward:.2f}")

    # Helper methods
    def _calculate_returns(self, data: Dict[str, Any], days: int) -> float:
        """Calculate returns over specified days"""
        # Simplified calculation
        return 0.0

    async def _calculate_market_breadth(self) -> float:
        """Calculate market breadth (advancing vs declining)"""
        return 0.5  # Placeholder

    async def _get_put_call_ratio(self) -> float:
        """Get put/call ratio"""
        return 1.0  # Placeholder

    async def _calculate_correlation(self) -> float:
        """Calculate average correlation"""
        return 0.5  # Placeholder

    async def _calculate_volume_ratio(self) -> float:
        """Calculate volume ratio"""
        return 1.0  # Placeholder

    def _determine_regime(self, vix: float, returns: float) -> str:
        """Determine market regime"""
        if vix > 30:
            return 'crisis'
        elif vix > 20:
            return 'volatile'
        else:
            return 'normal'

    def _calculate_vix_percentile(self, vix: float) -> float:
        """Calculate VIX percentile"""
        # Would need historical VIX data
        return 0.5

    def _get_recent_events(self) -> List[str]:
        """Get recent market events"""
        return ["Market opened normally", "No major news"]

    async def _get_sector_performance(self) -> Dict[str, float]:
        """Get sector performance"""
        return {
            'XLF': 0.01,
            'XLK': 0.02,
            'XLE': -0.01
        }

    def _calculate_black_swan_indicators(self, market_state: MarketState) -> Dict[str, float]:
        """Calculate black swan probability indicators"""
        return self.llm_orchestrator.analyze_black_swan_probability(self.recent_data)

    def _calculate_max_drawdown(self, position_id: str) -> float:
        """Calculate maximum drawdown for position"""
        return -0.05  # Placeholder

    def _calculate_trade_volatility(self, position_id: str) -> float:
        """Calculate volatility during trade"""
        return 0.15  # Placeholder

    def _is_black_swan_period(self) -> bool:
        """Check if currently in black swan period"""
        # Would check against known events and current volatility
        return False

    def _calculate_sleep_time(self, market_state: MarketState) -> int:
        """Calculate sleep time based on market conditions"""
        if market_state.regime == 'crisis':
            return 60  # Check every minute during crisis
        elif market_state.regime == 'volatile':
            return 300  # Check every 5 minutes
        else:
            return 900  # Check every 15 minutes

async def main():
    """Main entry point"""

    # Load configuration
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "mode": "paper",
            "broker": "alpaca",
            "initial_capital": 200,
            "siphon_enabled": True,
            "audit_enabled": True
        }

    # Create and start system
    system = BlackSwanTradingSystem(config)

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Black Swan Trading System...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not used
    asyncio.run(main())