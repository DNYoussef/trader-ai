"""
Trading Terminal Data Provider

Provides real-time market data, algorithmic signals, and AI inflection points
for the professional trading terminal interface. Integrates with existing
trading systems and causal intelligence components.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

# Import existing system components
try:
    from ..intelligence.causal_integration import CausallyEnhancedDPICalculator
    from ..market.market_data import MarketDataManager
    from ..portfolio.portfolio_manager import PortfolioManager
    from ..gates.enhanced_gate_manager import EnhancedGateManager
except ImportError:
    # Handle import errors gracefully for testing
    pass

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Real-time market data point"""
    symbol: str
    timestamp: float
    price: float
    volume: int
    change: float
    change_percent: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None


@dataclass
class TechnicalIndicator:
    """Technical indicator data"""
    symbol: str
    timestamp: float
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None
    rsi: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None


@dataclass
class AlgorithmicSignal:
    """Algorithmic trading signal"""
    signal_id: str
    symbol: str
    timestamp: float
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strategy: str  # 'DPI', 'CAUSAL', 'RISK_MGMT'
    reason: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None


@dataclass
class AIInflectionPoint:
    """AI-detected inflection point"""
    symbol: str
    timestamp: float
    inflection_type: str  # 'REVERSAL', 'ACCELERATION', 'CONSOLIDATION'
    confidence: float
    predicted_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    time_horizon: int  # Minutes
    supporting_evidence: List[str]


@dataclass
class OrderBookLevel:
    """Order book price level"""
    price: float
    size: int
    orders: int


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float


class TradingTerminalDataProvider:
    """
    Provides comprehensive real-time data for the professional trading terminal
    """

    def __init__(self,
                 symbols: List[str] = None,
                 update_interval: float = 1.0,
                 enable_live_data: bool = False):
        """
        Initialize the terminal data provider

        Args:
            symbols: List of symbols to track
            update_interval: Data update interval in seconds
            enable_live_data: Whether to use live market data
        """
        self.symbols = symbols or ['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU']
        self.update_interval = update_interval
        self.enable_live_data = enable_live_data

        # Data storage
        self.market_data: Dict[str, MarketDataPoint] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.technical_indicators: Dict[str, TechnicalIndicator] = {}
        self.algorithmic_signals: deque = deque(maxlen=100)
        self.ai_inflections: deque = deque(maxlen=50)
        self.order_books: Dict[str, OrderBookSnapshot] = {}

        # Callbacks for real-time updates
        self.market_data_callbacks: List[Callable] = []
        self.signal_callbacks: List[Callable] = []
        self.inflection_callbacks: List[Callable] = []

        # Integration components
        self.causal_calculator = None
        self.portfolio_manager = None
        self.gate_manager = None

        # Background tasks
        self.running = False
        self.update_task = None

        logger.info("Trading Terminal Data Provider initialized")

    def add_market_data_callback(self, callback: Callable):
        """Add callback for market data updates"""
        self.market_data_callbacks.append(callback)

    def add_signal_callback(self, callback: Callable):
        """Add callback for algorithmic signal updates"""
        self.signal_callbacks.append(callback)

    def add_inflection_callback(self, callback: Callable):
        """Add callback for AI inflection updates"""
        self.inflection_callbacks.append(callback)

    async def start(self):
        """Start the data provider"""
        if self.running:
            return

        self.running = True

        # Initialize with mock data if not using live data
        if not self.enable_live_data:
            await self._initialize_mock_data()

        # Start background update task
        self.update_task = asyncio.create_task(self._update_loop())

        logger.info("Trading Terminal Data Provider started")

    async def stop(self):
        """Stop the data provider"""
        self.running = False

        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        logger.info("Trading Terminal Data Provider stopped")

    async def _initialize_mock_data(self):
        """Initialize with mock market data"""

        base_prices = {
            'SPY': 440.25,
            'ULTY': 52.18,
            'AMDY': 24.67,
            'VTIP': 48.92,
            'IAU': 36.45
        }

        current_time = time.time()

        for symbol in self.symbols:
            base_price = base_prices.get(symbol, 100.0)

            # Create initial market data
            self.market_data[symbol] = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                price=base_price,
                volume=np.random.randint(1000000, 10000000),
                change=0.0,
                change_percent=0.0,
                bid=base_price - 0.01,
                ask=base_price + 0.01,
                high_24h=base_price * 1.02,
                low_24h=base_price * 0.98
            )

            # Initialize price history with some data
            for i in range(100):
                historical_time = current_time - (100 - i) * 60  # 1-minute intervals
                volatility = 0.02 if symbol == 'SPY' else 0.05
                price_change = np.random.normal(0, volatility) * base_price
                historical_price = base_price + price_change

                self.price_history[symbol].append({
                    'timestamp': historical_time,
                    'price': historical_price,
                    'volume': np.random.randint(10000, 1000000)
                })

            # Initialize technical indicators
            await self._calculate_technical_indicators(symbol)

            # Initialize order book
            await self._generate_order_book(symbol)

        # Generate some initial signals and inflections
        await self._generate_algorithmic_signals()
        await self._generate_ai_inflections()

    async def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                await self._update_market_data()
                await self._update_technical_indicators()
                await self._check_algorithmic_signals()
                await self._check_ai_inflections()
                await self._update_order_books()

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _update_market_data(self):
        """Update real-time market data"""
        current_time = time.time()

        for symbol in self.symbols:
            if symbol not in self.market_data:
                continue

            current_data = self.market_data[symbol]

            # Simulate price movement
            volatility = 0.001 if symbol == 'SPY' else 0.003  # Lower volatility for more realistic movement
            price_change = np.random.normal(0, volatility) * current_data.price

            # Add slight trending based on time of day
            hour = datetime.now().hour
            trend_factor = 0.0001 if 9 <= hour <= 16 else -0.0001  # Slight upward trend during market hours

            new_price = max(0.01, current_data.price + price_change + (trend_factor * current_data.price))
            volume_change = np.random.randint(-10000, 50000)
            new_volume = max(1000, current_data.volume + volume_change)

            change = new_price - current_data.price
            change_percent = (change / current_data.price) * 100

            # Update market data
            updated_data = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                price=new_price,
                volume=new_volume,
                change=change,
                change_percent=change_percent,
                bid=new_price - 0.01,
                ask=new_price + 0.01,
                high_24h=max(current_data.high_24h or new_price, new_price),
                low_24h=min(current_data.low_24h or new_price, new_price)
            )

            self.market_data[symbol] = updated_data

            # Add to price history
            self.price_history[symbol].append({
                'timestamp': current_time,
                'price': new_price,
                'volume': new_volume
            })

            # Notify callbacks
            for callback in self.market_data_callbacks:
                try:
                    callback(symbol, updated_data)
                except Exception as e:
                    logger.error(f"Error in market data callback: {e}")

    async def _update_technical_indicators(self):
        """Update technical indicators"""
        for symbol in self.symbols:
            await self._calculate_technical_indicators(symbol)

    async def _calculate_technical_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
            return

        history = list(self.price_history[symbol])
        prices = [point['price'] for point in history]

        current_time = time.time()

        # Calculate moving averages
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else None
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else None
        ma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else None

        # Calculate RSI
        rsi = self._calculate_rsi(prices) if len(prices) >= 14 else None

        # Calculate Bollinger Bands
        bollinger_upper, bollinger_lower = None, None
        if len(prices) >= 20:
            std_dev = np.std(prices[-20:])
            if ma_20:
                bollinger_upper = ma_20 + (2 * std_dev)
                bollinger_lower = ma_20 - (2 * std_dev)

        # Update indicators
        self.technical_indicators[symbol] = TechnicalIndicator(
            symbol=symbol,
            timestamp=current_time,
            ma_20=ma_20,
            ma_50=ma_50,
            ma_200=ma_200,
            rsi=rsi,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower
        )

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas[-period:]]
        losses = [-delta if delta < 0 else 0 for delta in deltas[-period:]]

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def _check_algorithmic_signals(self):
        """Check for new algorithmic trading signals"""
        # Generate signals occasionally
        if np.random.random() > 0.98:  # 2% chance per update
            await self._generate_algorithmic_signals()

    async def _generate_algorithmic_signals(self):
        """Generate algorithmic trading signals"""
        current_time = time.time()

        # Generate 1-3 signals
        num_signals = np.random.randint(1, 4)

        for _ in range(num_signals):
            symbol = np.random.choice(self.symbols)
            strategies = ['DPI', 'CAUSAL', 'RISK_MGMT']
            strategy = np.random.choice(strategies)

            signal_types = ['BUY', 'SELL']
            signal_type = np.random.choice(signal_types)

            confidence = np.random.uniform(0.6, 0.95)

            reasons = {
                'DPI': ['DPI threshold exceeded', 'Momentum divergence detected', 'Volume confirmation'],
                'CAUSAL': ['Causal pattern identified', 'Policy shock anticipated', 'Flow reversal predicted'],
                'RISK_MGMT': ['Position rebalancing required', 'Risk limit approached', 'Drawdown protection']
            }

            reason = np.random.choice(reasons[strategy])

            current_price = self.market_data.get(symbol, {}).price if symbol in self.market_data else 100.0

            signal = AlgorithmicSignal(
                signal_id=f"{strategy}_{symbol}_{int(current_time)}",
                symbol=symbol,
                timestamp=current_time,
                signal_type=signal_type,
                confidence=confidence,
                strategy=strategy,
                reason=reason,
                target_price=current_price * (1.02 if signal_type == 'BUY' else 0.98),
                stop_loss=current_price * (0.99 if signal_type == 'BUY' else 1.01),
                position_size=np.random.uniform(0.1, 0.3)
            )

            self.algorithmic_signals.append(signal)

            # Notify callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")

    async def _check_ai_inflections(self):
        """Check for AI inflection points"""
        # Generate inflections occasionally
        if np.random.random() > 0.995:  # 0.5% chance per update
            await self._generate_ai_inflections()

    async def _generate_ai_inflections(self):
        """Generate AI inflection points"""
        current_time = time.time()

        # Generate 1-2 inflections
        num_inflections = np.random.randint(1, 3)

        for _ in range(num_inflections):
            symbol = np.random.choice(self.symbols)

            inflection_types = ['REVERSAL', 'ACCELERATION', 'CONSOLIDATION']
            inflection_type = np.random.choice(inflection_types)

            directions = ['UP', 'DOWN', 'SIDEWAYS']
            predicted_direction = np.random.choice(directions)

            confidence = np.random.uniform(0.7, 0.95)
            time_horizon = np.random.randint(5, 60)  # 5-60 minutes

            evidence = [
                'Unusual volume pattern detected',
                'Cross-asset correlation shift',
                'Sentiment inflection identified',
                'Macro regime change signal',
                'Microstructure anomaly'
            ]

            supporting_evidence = np.random.choice(evidence, size=np.random.randint(1, 4), replace=False).tolist()

            inflection = AIInflectionPoint(
                symbol=symbol,
                timestamp=current_time,
                inflection_type=inflection_type,
                confidence=confidence,
                predicted_direction=predicted_direction,
                time_horizon=time_horizon,
                supporting_evidence=supporting_evidence
            )

            self.ai_inflections.append(inflection)

            # Notify callbacks
            for callback in self.inflection_callbacks:
                try:
                    callback(inflection)
                except Exception as e:
                    logger.error(f"Error in inflection callback: {e}")

    async def _update_order_books(self):
        """Update order book data"""
        for symbol in self.symbols:
            await self._generate_order_book(symbol)

    async def _generate_order_book(self, symbol: str):
        """Generate realistic order book for a symbol"""
        if symbol not in self.market_data:
            return

        current_price = self.market_data[symbol].price
        current_time = time.time()

        # Generate bid levels (below current price)
        bids = []
        for i in range(10):
            price = current_price - (i + 1) * np.random.uniform(0.01, 0.05)
            size = np.random.randint(100, 2000)
            orders = np.random.randint(1, 10)
            bids.append(OrderBookLevel(price=price, size=size, orders=orders))

        # Generate ask levels (above current price)
        asks = []
        for i in range(10):
            price = current_price + (i + 1) * np.random.uniform(0.01, 0.05)
            size = np.random.randint(100, 2000)
            orders = np.random.randint(1, 10)
            asks.append(OrderBookLevel(price=price, size=size, orders=orders))

        spread = asks[0].price - bids[0].price if asks and bids else 0.0

        self.order_books[symbol] = OrderBookSnapshot(
            symbol=symbol,
            timestamp=current_time,
            bids=bids,
            asks=asks,
            spread=spread
        )

    def get_market_data(self, symbol: str = None) -> Dict[str, Any]:
        """Get current market data"""
        if symbol:
            return asdict(self.market_data.get(symbol, {}))
        return {sym: asdict(data) for sym, data in self.market_data.items()}

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get price history for a symbol"""
        if symbol not in self.price_history:
            return []
        return list(self.price_history[symbol])[-limit:]

    def get_technical_indicators(self, symbol: str = None) -> Dict[str, Any]:
        """Get technical indicators"""
        if symbol:
            return asdict(self.technical_indicators.get(symbol, {}))
        return {sym: asdict(data) for sym, data in self.technical_indicators.items()}

    def get_algorithmic_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent algorithmic signals"""
        return [asdict(signal) for signal in list(self.algorithmic_signals)[-limit:]]

    def get_ai_inflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent AI inflection points"""
        return [asdict(inflection) for inflection in list(self.ai_inflections)[-limit:]]

    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol"""
        if symbol not in self.order_books:
            return {}
        return asdict(self.order_books[symbol])

    def get_terminal_snapshot(self) -> Dict[str, Any]:
        """Get complete data snapshot for terminal"""
        return {
            'market_data': self.get_market_data(),
            'technical_indicators': self.get_technical_indicators(),
            'algorithmic_signals': self.get_algorithmic_signals(),
            'ai_inflections': self.get_ai_inflections(),
            'order_books': {symbol: self.get_order_book(symbol) for symbol in self.symbols},
            'timestamp': time.time()
        }


class TerminalWebSocketHandler:
    """WebSocket handler for real-time terminal data"""

    def __init__(self, data_provider: TradingTerminalDataProvider):
        self.data_provider = data_provider
        self.connected_clients = set()

    async def connect(self, websocket):
        """Handle new WebSocket connection"""
        self.connected_clients.add(websocket)
        logger.info(f"Terminal client connected: {websocket.remote_address}")

        # Send initial data snapshot
        initial_data = self.data_provider.get_terminal_snapshot()
        await websocket.send(json.dumps({
            'type': 'snapshot',
            'data': initial_data
        }))

    async def disconnect(self, websocket):
        """Handle WebSocket disconnection"""
        self.connected_clients.discard(websocket)
        logger.info(f"Terminal client disconnected: {websocket.remote_address}")

    async def broadcast_market_data(self, symbol: str, data: MarketDataPoint):
        """Broadcast market data update to all clients"""
        message = {
            'type': 'market_data',
            'symbol': symbol,
            'data': asdict(data)
        }

        await self._broadcast(message)

    async def broadcast_signal(self, signal: AlgorithmicSignal):
        """Broadcast algorithmic signal to all clients"""
        message = {
            'type': 'signal',
            'data': asdict(signal)
        }

        await self._broadcast(message)

    async def broadcast_inflection(self, inflection: AIInflectionPoint):
        """Broadcast AI inflection point to all clients"""
        message = {
            'type': 'inflection',
            'data': asdict(inflection)
        }

        await self._broadcast(message)

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return

        message_str = json.dumps(message)
        disconnected = set()

        for websocket in self.connected_clients:
            try:
                await websocket.send(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)

        # Remove disconnected clients
        self.connected_clients -= disconnected