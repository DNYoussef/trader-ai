"""
Real-time Alpha Generation Pipeline

This module implements a real-time signal generation pipeline that orchestrates
all alpha generation components to provide live trading signals. The pipeline
handles data ingestion, signal generation, risk management, and execution
coordination in real-time.

Key Features:
- Real-time market data ingestion
- Continuous signal generation
- Risk monitoring and position management
- Execution coordination
- Performance tracking
- Alert system integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, AsyncIterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod

# Import alpha generation components
from alpha_integration import AlphaIntegrationEngine, AlphaSignal, PortfolioState

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_trade_size: float = 0.0

@dataclass
class SignalEvent:
    """Real-time signal event"""
    signal: AlphaSignal
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionOrder:
    """Order for execution"""
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: str = "market"  # market/limit
    limit_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC/IOC/FOK
    signal_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class DataFeed(ABC):
    """Abstract base class for market data feeds"""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data feed"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def get_data_stream(self) -> AsyncIterable[MarketData]:
        """Get continuous data stream"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data feed"""
        pass

class SimulatedDataFeed(DataFeed):
    """Simulated data feed for testing"""

    def __init__(self, symbols: List[str], update_frequency: float = 1.0):
        self.symbols = symbols
        self.update_frequency = update_frequency
        self.connected = False
        self.subscribed_symbols = set()
        self.last_prices = {symbol: np.random.uniform(50, 500) for symbol in symbols}

    async def connect(self) -> bool:
        """Connect to simulated feed"""
        self.connected = True
        logger.info("Connected to simulated data feed")
        return True

    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols"""
        if not self.connected:
            return False

        self.subscribed_symbols.update(symbols)
        logger.info(f"Subscribed to symbols: {symbols}")
        return True

    async def get_data_stream(self) -> AsyncIterable[MarketData]:
        """Generate simulated market data stream"""
        while self.connected:
            try:
                for symbol in self.subscribed_symbols:
                    # Simulate price movement
                    last_price = self.last_prices[symbol]
                    price_change = np.random.normal(0, 0.001)  # 0.1% std price change
                    new_price = last_price * (1 + price_change)
                    self.last_prices[symbol] = new_price

                    # Generate market data
                    spread = new_price * 0.001  # 10 bps spread
                    bid = new_price - spread / 2
                    ask = new_price + spread / 2

                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=new_price,
                        volume=np.random.uniform(1000, 10000),
                        bid=bid,
                        ask=ask,
                        bid_size=np.random.uniform(100, 1000),
                        ask_size=np.random.uniform(100, 1000),
                        last_trade_size=np.random.uniform(10, 500)
                    )

                    yield market_data

                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                logger.error(f"Error in data stream: {e}")
                break

    async def disconnect(self):
        """Disconnect from feed"""
        self.connected = False
        logger.info("Disconnected from simulated data feed")

class ExecutionHandler(ABC):
    """Abstract base class for order execution"""

    @abstractmethod
    async def submit_order(self, order: ExecutionOrder) -> str:
        """Submit order for execution"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass

class SimulatedExecutionHandler(ExecutionHandler):
    """Simulated execution handler for testing"""

    def __init__(self):
        self.orders = {}
        self.order_counter = 0

    async def submit_order(self, order: ExecutionOrder) -> str:
        """Submit simulated order"""
        try:
            self.order_counter += 1
            order_id = f"ORDER_{self.order_counter:06d}"

            # Simulate execution with some delay and slippage
            await asyncio.sleep(0.1)  # Execution delay

            execution_price = order.limit_price if order.order_type == "limit" else None
            if execution_price is None:
                # Market order - apply slippage
                slippage = np.random.normal(0, 0.0005)  # 5 bps std slippage
                execution_price = order.limit_price * (1 + slippage) if order.limit_price else 100.0

            self.orders[order_id] = {
                'order': order,
                'status': 'filled',
                'fill_price': execution_price,
                'fill_quantity': order.quantity,
                'fill_time': datetime.now()
            }

            logger.info(f"Order {order_id} executed: {order.side} {order.quantity} {order.symbol} @ {execution_price:.2f}")
            return order_id

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return ""

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            return True
        return False

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        return self.orders.get(order_id, {})

class RiskMonitor:
    """Real-time risk monitoring system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_limits = config.get('position_limits', {})
        self.var_limit = config.get('var_limit', 1000000)  # $1M VaR limit
        self.concentration_limit = config.get('concentration_limit', 0.1)  # 10% max concentration

        self.current_positions = {}
        self.current_var = 0.0
        self.risk_violations = []

    def update_positions(self, positions: Dict[str, float]):
        """Update current positions"""
        self.current_positions = positions.copy()
        self._calculate_risk_metrics()

    def check_pre_trade_risk(self, order: ExecutionOrder, current_price: float) -> Tuple[bool, List[str]]:
        """Check if trade violates risk limits"""
        violations = []

        try:
            # Calculate position after trade
            symbol = order.symbol
            current_position = self.current_positions.get(symbol, 0)

            if order.side == 'buy':
                new_position = current_position + order.quantity
            else:
                new_position = current_position - order.quantity

            # Check position limits
            position_limit = self.position_limits.get(symbol, float('inf'))
            if abs(new_position * current_price) > position_limit:
                violations.append(f"Position limit exceeded for {symbol}")

            # Check concentration limit
            total_portfolio_value = sum(abs(pos * 100) for pos in self.current_positions.values())  # Simplified
            position_weight = abs(new_position * current_price) / max(total_portfolio_value, 1)

            if position_weight > self.concentration_limit:
                violations.append(f"Concentration limit exceeded for {symbol}")

            # Check VaR limit (simplified)
            estimated_var_increase = abs(order.quantity * current_price) * 0.02  # 2% VaR estimate
            if self.current_var + estimated_var_increase > self.var_limit:
                violations.append("VaR limit would be exceeded")

            return len(violations) == 0, violations

        except Exception as e:
            logger.error(f"Error checking pre-trade risk: {e}")
            return False, ["Risk check failed"]

    def _calculate_risk_metrics(self):
        """Calculate current risk metrics"""
        try:
            # Simplified VaR calculation
            position_values = [abs(pos * 100) for pos in self.current_positions.values()]  # Simplified pricing
            self.current_var = sum(position_values) * 0.02  # 2% of total exposure

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")

class RealTimePipeline:
    """Main real-time alpha generation pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', [])
        self.running = False

        # Initialize components
        self.alpha_engine = AlphaIntegrationEngine(config.get('alpha_config', {}))
        self.data_feed = SimulatedDataFeed(self.symbols, config.get('data_frequency', 1.0))
        self.execution_handler = SimulatedExecutionHandler()
        self.risk_monitor = RiskMonitor(config.get('risk_config', {}))

        # Pipeline queues
        self.market_data_queue = asyncio.Queue(maxsize=1000)
        self.signal_queue = asyncio.Queue(maxsize=100)
        self.execution_queue = asyncio.Queue(maxsize=50)

        # Current state
        self.current_prices = {}
        self.current_positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_state = PortfolioState(
            total_capital=config.get('initial_capital', 10_000_000),
            available_capital=config.get('initial_capital', 10_000_000),
            current_positions={},
            risk_utilization=0.0,
            var_usage=0.0
        )

        # Performance tracking
        self.signal_history = []
        self.execution_history = []
        self.performance_metrics = {}

        # Event callbacks
        self.signal_callbacks: List[Callable[[SignalEvent], None]] = []
        self.execution_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    async def start(self):
        """Start the real-time pipeline"""
        try:
            logger.info("Starting real-time alpha generation pipeline")
            self.running = True

            # Connect to data feed
            await self.data_feed.connect()
            await self.data_feed.subscribe(self.symbols)

            # Start pipeline components
            tasks = [
                asyncio.create_task(self._market_data_processor()),
                asyncio.create_task(self._signal_generator()),
                asyncio.create_task(self._execution_processor()),
                asyncio.create_task(self._performance_monitor())
            ]

            # Start data ingestion
            asyncio.create_task(self._ingest_market_data())

            logger.info("Real-time pipeline started successfully")

            # Wait for all tasks
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            await self.stop()

    async def stop(self):
        """Stop the real-time pipeline"""
        try:
            logger.info("Stopping real-time pipeline")
            self.running = False

            # Disconnect from data feed
            await self.data_feed.disconnect()

            logger.info("Real-time pipeline stopped")

        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")

    async def _ingest_market_data(self):
        """Ingest market data from feed"""
        try:
            async for market_data in self.data_feed.get_data_stream():
                if not self.running:
                    break

                await self.market_data_queue.put(market_data)

        except Exception as e:
            logger.error(f"Error ingesting market data: {e}")

    async def _market_data_processor(self):
        """Process incoming market data"""
        try:
            while self.running:
                try:
                    # Get market data with timeout
                    market_data = await asyncio.wait_for(
                        self.market_data_queue.get(), timeout=1.0
                    )

                    # Update current prices
                    self.current_prices[market_data.symbol] = market_data.price

                    # Update portfolio state
                    self._update_portfolio_state()

                    # Mark task as done
                    self.market_data_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")

        except Exception as e:
            logger.error(f"Error in market data processor: {e}")

    async def _signal_generator(self):
        """Generate alpha signals based on current market data"""
        try:
            last_signal_time = {}
            signal_frequency = self.config.get('signal_frequency', 10.0)  # seconds

            while self.running:
                try:
                    current_time = datetime.now()

                    # Generate signals for symbols that need updates
                    for symbol in self.symbols:
                        if symbol not in self.current_prices:
                            continue

                        last_time = last_signal_time.get(symbol, datetime.min)
                        if (current_time - last_time).total_seconds() >= signal_frequency:

                            # Generate signal
                            signal = await self.alpha_engine.generate_integrated_signal(
                                symbol, self.current_prices[symbol], self.portfolio_state
                            )

                            # Create signal event
                            if signal.final_score > 0.3:  # Only queue significant signals
                                priority = 1 if signal.urgency > 0.7 else 2 if signal.urgency > 0.4 else 3

                                signal_event = SignalEvent(
                                    signal=signal,
                                    timestamp=current_time,
                                    priority=priority,
                                    metadata={'source': 'real_time_pipeline'}
                                )

                                await self.signal_queue.put(signal_event)
                                self.signal_history.append(signal_event)

                                # Notify callbacks
                                for callback in self.signal_callbacks:
                                    try:
                                        callback(signal_event)
                                    except Exception as e:
                                        logger.error(f"Error in signal callback: {e}")

                            last_signal_time[symbol] = current_time

                    await asyncio.sleep(1.0)  # Check frequency

                except Exception as e:
                    logger.error(f"Error generating signals: {e}")
                    await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"Error in signal generator: {e}")

    async def _execution_processor(self):
        """Process signals for execution"""
        try:
            while self.running:
                try:
                    # Get signal event with timeout
                    signal_event = await asyncio.wait_for(
                        self.signal_queue.get(), timeout=1.0
                    )

                    signal = signal_event.signal

                    # Check if we should execute
                    if signal.action in ['buy', 'sell'] and signal.final_score > 0.5:

                        # Calculate order parameters
                        current_price = self.current_prices[signal.symbol]
                        order_quantity = signal.recommended_size / current_price

                        if signal.action == 'sell':
                            # Check available position to sell
                            available_position = self.current_positions.get(signal.symbol, 0)
                            order_quantity = min(order_quantity, available_position)

                        if order_quantity > 0:
                            # Create execution order
                            order = ExecutionOrder(
                                symbol=signal.symbol,
                                side=signal.action,
                                quantity=order_quantity,
                                order_type="market",
                                signal_id=f"{signal.symbol}_{signal.timestamp.strftime('%H%M%S')}"
                            )

                            # Risk check
                            risk_ok, violations = self.risk_monitor.check_pre_trade_risk(
                                order, current_price
                            )

                            if risk_ok:
                                # Submit order
                                order_id = await self.execution_handler.submit_order(order)

                                if order_id:
                                    execution_record = {
                                        'order_id': order_id,
                                        'signal_event': signal_event,
                                        'order': order,
                                        'timestamp': datetime.now()
                                    }

                                    self.execution_history.append(execution_record)

                                    # Update positions (simplified - assumes immediate fill)
                                    if order.side == 'buy':
                                        self.current_positions[order.symbol] += order.quantity
                                    else:
                                        self.current_positions[order.symbol] -= order.quantity

                                    self.risk_monitor.update_positions(self.current_positions)

                                    # Notify callbacks
                                    for callback in self.execution_callbacks:
                                        try:
                                            callback(order_id, execution_record)
                                        except Exception as e:
                                            logger.error(f"Error in execution callback: {e}")

                                    logger.info(f"Executed order {order_id}: {order.side} {order.quantity:.2f} {order.symbol}")

                            else:
                                logger.warning(f"Risk check failed for {signal.symbol}: {violations}")

                    # Mark task as done
                    self.signal_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing execution: {e}")

        except Exception as e:
            logger.error(f"Error in execution processor: {e}")

    async def _performance_monitor(self):
        """Monitor and track performance metrics"""
        try:
            while self.running:
                try:
                    # Calculate current portfolio metrics
                    portfolio_value = self.portfolio_state.available_capital
                    for symbol, position in self.current_positions.items():
                        if symbol in self.current_prices:
                            portfolio_value += position * self.current_prices[symbol]

                    # Calculate returns (simplified)
                    initial_capital = self.config.get('initial_capital', 10_000_000)
                    total_return = (portfolio_value / initial_capital) - 1

                    # Update performance metrics
                    self.performance_metrics.update({
                        'timestamp': datetime.now(),
                        'portfolio_value': portfolio_value,
                        'total_return': total_return,
                        'total_signals': len(self.signal_history),
                        'total_executions': len(self.execution_history),
                        'current_positions': len([p for p in self.current_positions.values() if p != 0]),
                        'risk_utilization': self.risk_monitor.current_var / self.risk_monitor.var_limit
                    })

                    # Log performance periodically
                    if datetime.now().minute % 5 == 0:  # Every 5 minutes
                        logger.info(f"Portfolio Value: ${portfolio_value:,.0f}, "
                                  f"Return: {total_return:.2%}, "
                                  f"Signals: {len(self.signal_history)}, "
                                  f"Executions: {len(self.execution_history)}")

                    await asyncio.sleep(60.0)  # Update every minute

                except Exception as e:
                    logger.error(f"Error monitoring performance: {e}")
                    await asyncio.sleep(60.0)

        except Exception as e:
            logger.error(f"Error in performance monitor: {e}")

    def _update_portfolio_state(self):
        """Update portfolio state with current market data"""
        try:
            # Calculate position values
            position_values = {}
            total_value = self.portfolio_state.available_capital

            for symbol, position in self.current_positions.items():
                if symbol in self.current_prices and position != 0:
                    value = position * self.current_prices[symbol]
                    position_values[symbol] = value
                    total_value += value

            # Update portfolio state
            self.portfolio_state.current_positions = position_values
            self.portfolio_state.total_capital = total_value

            # Calculate risk utilization (simplified)
            total_exposure = sum(abs(v) for v in position_values.values())
            self.portfolio_state.risk_utilization = min(1.0, total_exposure / total_value)

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    def add_signal_callback(self, callback: Callable[[SignalEvent], None]):
        """Add callback for signal events"""
        self.signal_callbacks.append(callback)

    def add_execution_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for execution events"""
        self.execution_callbacks.append(callback)

    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_signal_summary(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent signals"""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_signals = [s for s in self.signal_history if s.timestamp >= cutoff_time]

        if not recent_signals:
            return {'total_signals': 0}

        signal_scores = [s.signal.final_score for s in recent_signals]
        ng_scores = [s.signal.ng_score for s in recent_signals]
        ethical_scores = [s.signal.ethical_score for s in recent_signals]

        return {
            'total_signals': len(recent_signals),
            'avg_signal_score': np.mean(signal_scores),
            'avg_ng_score': np.mean(ng_scores),
            'avg_ethical_score': np.mean(ethical_scores),
            'high_priority_signals': len([s for s in recent_signals if s.priority == 1]),
            'actions_buy': len([s for s in recent_signals if s.signal.action == 'buy']),
            'actions_sell': len([s for s in recent_signals if s.signal.action == 'sell']),
            'actions_hold': len([s for s in recent_signals if s.signal.action == 'hold'])
        }

# Example usage and testing
async def test_realtime_pipeline():
    """Test the real-time pipeline"""
    config = {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'initial_capital': 10_000_000,
        'data_frequency': 0.5,  # 0.5 second updates
        'signal_frequency': 5.0,  # 5 second signal generation
        'alpha_config': {
            'max_position_size': 0.05,
            'ng_position_multiplier': 2.0
        },
        'risk_config': {
            'var_limit': 1_000_000,
            'concentration_limit': 0.1,
            'position_limits': {
                'AAPL': 500_000,
                'MSFT': 500_000,
                'GOOGL': 500_000
            }
        }
    }

    pipeline = RealTimePipeline(config)

    # Add callbacks for monitoring
    def signal_callback(signal_event: SignalEvent):
        print(f"Signal: {signal_event.signal.symbol} - {signal_event.signal.action} "
              f"(Score: {signal_event.signal.final_score:.3f})")

    def execution_callback(order_id: str, execution_record: Dict[str, Any]):
        order = execution_record['order']
        print(f"Execution: {order_id} - {order.side} {order.quantity:.2f} {order.symbol}")

    pipeline.add_signal_callback(signal_callback)
    pipeline.add_execution_callback(execution_callback)

    try:
        # Start pipeline and run for a test period
        asyncio.create_task(pipeline.start())

        # Let it run for 30 seconds
        await asyncio.sleep(30)

        # Get performance summary
        performance = pipeline.get_current_performance()
        signal_summary = pipeline.get_signal_summary()

        print("\n=== REAL-TIME PIPELINE TEST RESULTS ===")
        print(f"Portfolio Value: ${performance.get('portfolio_value', 0):,.0f}")
        print(f"Total Return: {performance.get('total_return', 0):.2%}")
        print(f"Total Signals: {signal_summary.get('total_signals', 0)}")
        print(f"Average Signal Score: {signal_summary.get('avg_signal_score', 0):.3f}")
        print(f"Total Executions: {performance.get('total_executions', 0)}")
        print(f"Active Positions: {performance.get('current_positions', 0)}")

        # Stop pipeline
        await pipeline.stop()

    except Exception as e:
        logger.error(f"Error in pipeline test: {e}")
        await pipeline.stop()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_realtime_pipeline())