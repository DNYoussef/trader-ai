# API Reference Documentation

## Gary×Taleb Foundation Phase - Core API Reference

This document provides comprehensive API documentation for the Foundation phase components, including method signatures, parameters, return values, and usage examples.

---

## BrokerInterface

**Location:** `src/brokers/broker_interface.py`

### Abstract Base Class for Broker Implementations

The `BrokerInterface` defines the contract for all broker implementations, ensuring consistent behavior across different brokers.

#### Core Data Structures

##### Order
```python
@dataclass
class Order:
    id: Optional[str] = None                    # Broker-assigned order ID
    client_order_id: Optional[str] = None       # Client-assigned order ID
    symbol: str = ""                            # Trading symbol (e.g., "ULTY")
    qty: Decimal = Decimal("0")                 # Quantity (supports fractional)
    notional: Optional[Decimal] = None          # Dollar amount for market orders
    side: str = ""                              # "buy" or "sell"
    order_type: OrderType = OrderType.MARKET    # Order type enum
    time_in_force: TimeInForce = TimeInForce.DAY # Execution timing
    limit_price: Optional[Decimal] = None       # Limit price (if applicable)
    stop_price: Optional[Decimal] = None        # Stop price (if applicable)
    extended_hours: bool = False                # After-hours trading
    status: Optional[OrderStatus] = None        # Current order status
    filled_qty: Decimal = Decimal("0")          # Quantity filled
    # ... additional fields for tracking and metadata
```

##### Position
```python
@dataclass
class Position:
    asset_id: str                               # Unique asset identifier
    symbol: str                                 # Trading symbol
    exchange: str                               # Exchange (e.g., "NASDAQ")
    asset_class: str                            # "us_equity", "option", etc.
    avg_entry_price: Decimal                    # Average entry price
    qty: Decimal                                # Current quantity
    side: str                                   # "long" or "short"
    market_value: Optional[Decimal] = None      # Current market value
    unrealized_pl: Optional[Decimal] = None     # Unrealized P&L
    # ... additional P&L and pricing fields
```

#### Abstract Methods

##### Connection Management

```python
async def connect() -> bool:
    """
    Establish connection to broker.

    Returns:
        bool: True if connection successful, False otherwise

    Raises:
        ConnectionError: If connection fails
        AuthenticationError: If credentials invalid
    """
```

```python
async def disconnect() -> None:
    """
    Gracefully disconnect from broker.

    Performs cleanup and closes all connections.
    """
```

##### Account Information

```python
async def get_account_value() -> Decimal:
    """
    Get total account value (cash + positions).

    Returns:
        Decimal: Total portfolio value in USD

    Raises:
        ConnectionError: If not connected to broker
        BrokerError: If API call fails
    """
```

```python
async def get_cash_balance() -> Decimal:
    """
    Get available cash balance.

    Returns:
        Decimal: Available cash in USD
    """
```

```python
async def get_buying_power() -> Decimal:
    """
    Get total buying power (including margin if applicable).

    Returns:
        Decimal: Available buying power in USD
    """
```

##### Position Management

```python
async def get_positions() -> List[Position]:
    """
    Get all current positions.

    Returns:
        List[Position]: List of current positions
    """
```

```python
async def get_position(symbol: str) -> Optional[Position]:
    """
    Get position for specific symbol.

    Args:
        symbol: Trading symbol (e.g., "ULTY")

    Returns:
        Optional[Position]: Position if exists, None otherwise
    """
```

##### Order Management

```python
async def submit_order(order: Order) -> Order:
    """
    Submit trading order to broker.

    Args:
        order: Order object with trade details

    Returns:
        Order: Updated order with broker response data

    Raises:
        InsufficientFundsError: If insufficient funds
        InvalidOrderError: If order parameters invalid
        MarketClosedError: If market closed and order doesn't allow extended hours
        RateLimitError: If API rate limit exceeded
    """
```

```python
async def get_order(order_id: str) -> Optional[Order]:
    """
    Retrieve order by ID.

    Args:
        order_id: Broker-assigned order ID

    Returns:
        Optional[Order]: Order if found, None otherwise
    """
```

```python
async def cancel_order(order_id: str) -> bool:
    """
    Cancel specific order.

    Args:
        order_id: Broker-assigned order ID

    Returns:
        bool: True if cancellation successful
    """
```

```python
async def cancel_all_orders() -> int:
    """
    Cancel all open orders.

    Returns:
        int: Number of orders successfully canceled
    """
```

##### Market Data

```python
async def get_market_price(symbol: str) -> Optional[Decimal]:
    """
    Get current market price for symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Optional[Decimal]: Current price if available
    """
```

```python
async def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get current bid/ask quote.

    Args:
        symbol: Trading symbol

    Returns:
        Optional[Dict]: Quote data including:
            - bid: float
            - ask: float
            - midpoint: float
            - bid_size: int
            - ask_size: int
            - timestamp: datetime
    """
```

---

## AlpacaAdapter

**Location:** `src/brokers/alpaca_adapter.py`

### Concrete Implementation of BrokerInterface for Alpaca Trading

#### Initialization

```python
def __init__(self, config: Dict[str, Any]):
    """
    Initialize Alpaca adapter.

    Args:
        config: Configuration dictionary:
            - api_key: Alpaca API key
            - secret_key: Alpaca secret key
            - paper_trading: bool (default: True)
            - base_url: Optional custom base URL
    """
```

#### Mock Mode Support

When Alpaca library is unavailable, automatically falls back to mock mode:

```python
class MockAlpacaClient:
    """
    Mock client for development without Alpaca library.
    Provides realistic responses for testing and development.
    """

    async def get_account(self):
        """Returns mock account with $100k portfolio, $50k cash"""

    async def submit_order(self, order_data):
        """Returns mock filled order with realistic execution"""
```

#### Fractional Share Precision

- **Precision:** 6 decimal places (Alpaca limit)
- **Rounding:** ROUND_HALF_UP for consistent behavior
- **Validation:** Automatic precision adjustment in `_validate_order()`

#### Error Handling

Comprehensive error mapping from Alpaca API errors to standardized exceptions:

```python
# HTTP 401/403 -> AuthenticationError
# HTTP 429 -> RateLimitError
# "insufficient" in message -> InsufficientFundsError
# "invalid" in message -> InvalidOrderError
```

---

## GateManager

**Location:** `src/gates/gate_manager.py`

### Capital-Based Trading Gate System

#### Initialization

```python
def __init__(self, data_dir: str = "./data/gates"):
    """
    Initialize gate manager.

    Args:
        data_dir: Directory for persistent state storage
    """
```

#### Core Data Structures

##### GateConfig
```python
@dataclass
class GateConfig:
    level: GateLevel                           # G0, G1, G2, G3
    capital_min: float                         # Minimum capital for gate
    capital_max: float                         # Maximum capital for gate
    allowed_assets: Set[str]                   # Permitted trading symbols
    cash_floor_pct: float                      # Required cash percentage
    options_enabled: bool                      # Options trading allowed
    max_theta_pct: Optional[float] = None      # Theta exposure limit
    max_position_pct: float = 0.20             # Max single position size
    max_concentration_pct: float = 0.30        # Max sector concentration
```

##### TradeValidationResult
```python
@dataclass
class TradeValidationResult:
    is_valid: bool                             # Overall validation result
    violations: List[Dict[str, Any]]           # Constraint violations
    warnings: List[Dict[str, Any]]             # Warning conditions

    def add_violation(self, violation_type: ViolationType,
                     message: str, details: Dict = None):
        """Add constraint violation to result"""

    def add_warning(self, message: str, details: Dict = None):
        """Add warning condition to result"""
```

#### Gate Configurations

##### G0 Gate ($200-499)
```python
GateConfig(
    level=GateLevel.G0,
    capital_min=200.0,
    capital_max=499.99,
    allowed_assets={'ULTY', 'AMDY'},           # Only 2 ETFs
    cash_floor_pct=0.50,                       # 50% cash required
    options_enabled=False,                     # No options
    max_position_pct=0.25,                     # Conservative sizing
    max_concentration_pct=0.40
)
```

##### G1 Gate ($500-999)
```python
GateConfig(
    level=GateLevel.G1,
    capital_min=500.0,
    capital_max=999.99,
    allowed_assets={'ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP'},  # Adds gold/TIPS
    cash_floor_pct=0.60,                       # Higher cash floor
    options_enabled=False,
    max_position_pct=0.22,
    max_concentration_pct=0.35
)
```

#### Key Methods

```python
def update_capital(self, new_capital: float) -> bool:
    """
    Update current capital and check for gate transitions.

    Args:
        new_capital: New portfolio value

    Returns:
        bool: True if gate changed, False otherwise
    """
```

```python
def validate_trade(self, trade_details: Dict[str, Any],
                  current_portfolio: Dict[str, Any]) -> TradeValidationResult:
    """
    Validate trade against current gate constraints.

    Args:
        trade_details: Trade information:
            - symbol: str
            - side: "BUY" or "SELL"
            - quantity: float
            - price: float
            - trade_type: "STOCK" or "OPTION"
        current_portfolio: Portfolio state:
            - cash: float
            - positions: Dict[str, Dict]
            - total_value: float

    Returns:
        TradeValidationResult: Validation outcome
    """
```

```python
def check_graduation(self, portfolio_metrics: Dict[str, Any]) -> str:
    """
    Evaluate gate graduation/downgrade eligibility.

    Args:
        portfolio_metrics: Performance data:
            - sharpe_ratio_30d: float
            - max_drawdown_30d: float
            - avg_cash_utilization_30d: float

    Returns:
        str: "GRADUATE", "HOLD", or "DOWNGRADE"
    """
```

#### Graduation Criteria

Each gate has specific graduation requirements:

```python
# G0 -> G1 Requirements:
{
    'min_compliant_days': 14,
    'max_violations_30d': 2,
    'min_performance_score': 0.6,
    'min_capital': 500.0
}

# G1 -> G2 Requirements:
{
    'min_compliant_days': 21,
    'max_violations_30d': 1,
    'min_performance_score': 0.7,
    'min_capital': 1000.0
}
```

#### Performance Scoring

Composite performance score (0-1) calculation:

```python
def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
    """
    Components:
    - Sharpe ratio (0-0.4): Normalized around 2.0 Sharpe
    - Drawdown (0-0.3): Less than 5% = full points
    - Cash utilization (0-0.2): Optimal = slightly above cash floor
    - Compliance (0-0.1): Zero violations = full points

    Returns:
        float: Composite score 0.0-1.0
    """
```

---

## WeeklyCycle

**Location:** `src/cycles/weekly_cycle.py`

### Automated Weekly Buy/Siphon Trading Cycles

#### Initialization

```python
def __init__(self,
             portfolio_manager: PortfolioManager,
             trade_executor: TradeExecutor,
             market_data: MarketDataProvider,
             holiday_calendar: Optional[MarketHolidayCalendar] = None):
    """
    Initialize weekly cycle system.

    Args:
        portfolio_manager: Portfolio state management
        trade_executor: Order execution interface
        market_data: Market data provider
        holiday_calendar: Market holiday calendar (optional)
    """
```

#### Timing Configuration

```python
# Eastern Time timezone
ET = pytz.timezone('US/Eastern')

# Schedule times (in ET)
BUY_TIME = time(16, 10)      # 4:10 PM ET Friday
SIPHON_TIME = time(18, 0)    # 6:00 PM ET Friday
```

#### Gate-Specific Allocations

```python
GATE_ALLOCATIONS = {
    'G0': GateAllocation(ulty_pct=70.0, amdy_pct=30.0),
    'G1': GateAllocation(ulty_pct=50.0, amdy_pct=20.0,
                        iau_pct=15.0, vtip_pct=15.0)
}
```

#### Core Methods

```python
def should_execute_buy() -> bool:
    """
    Determine if buy phase should execute.

    Conditions:
    - Current time is Friday 4:10pm ET or later
    - Market is open (not holiday)
    - Haven't executed buy phase this week

    Returns:
        bool: True if should execute buy phase
    """
```

```python
def should_execute_siphon() -> bool:
    """
    Determine if siphon phase should execute.

    Conditions:
    - Current time is Friday 6:00pm ET or later
    - Market is open (not holiday)
    - Haven't executed siphon phase this week
    - Buy phase executed this week

    Returns:
        bool: True if should execute siphon phase
    """
```

```python
def execute_buy_phase(self, gate: str, available_cash: float) -> Dict:
    """
    Execute buy phase for specified gate.

    Args:
        gate: Gate identifier ('G0', 'G1', etc.)
        available_cash: Cash available for purchases

    Returns:
        Dict: Execution results:
            - gate: str
            - phase: "buy"
            - timestamp: datetime
            - total_cash: float
            - trades: List[Dict]
            - success: bool
            - errors: List[str]
    """
```

```python
def execute_siphon_phase(self, gate: str) -> Dict:
    """
    Execute siphon phase (50/50 rebalancing).

    Args:
        gate: Gate identifier

    Returns:
        Dict: Execution results:
            - gate: str
            - phase: "siphon"
            - operations: List[Dict]
            - success: bool
            - errors: List[str]
    """
```

#### Holiday Handling

```python
def handle_market_holiday(self, holiday_date: datetime.date) -> Dict:
    """
    Handle market holiday by deferring execution.

    Args:
        holiday_date: Date of market holiday

    Returns:
        Dict: Holiday handling information:
            - holiday_date: date
            - next_trading_day: date
            - action: "defer_execution"
            - message: str
    """
```

#### Performance Tracking

```python
@dataclass
class WeeklyDelta:
    """Weekly performance metrics"""
    week_start: datetime
    week_end: datetime
    nav_start: float
    nav_end: float
    deposits: float
    withdrawals: float
    delta: float          # Performance excluding cash flows
    delta_pct: float      # Percentage performance
```

```python
def calculate_weekly_delta(self, week_start: datetime) -> WeeklyDelta:
    """
    Calculate weekly performance delta.

    Args:
        week_start: Start of week to analyze

    Returns:
        WeeklyDelta: Weekly performance metrics
    """
```

---

## Error Handling & Exceptions

### Exception Hierarchy

```python
BrokerError(Exception)                    # Base broker exception
├── ConnectionError(BrokerError)          # Network/connection issues
├── AuthenticationError(BrokerError)      # Invalid credentials
├── InsufficientFundsError(BrokerError)   # Insufficient account funds
├── InvalidOrderError(BrokerError)        # Invalid order parameters
├── MarketClosedError(BrokerError)        # Market closed
└── RateLimitError(BrokerError)          # API rate limit exceeded
```

### Error Context

All exceptions include detailed context:

```python
class BrokerError(Exception):
    def __init__(self, message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code      # Broker-specific error code
        self.details = details or {}      # Additional error context
```

---

## Usage Examples

### Basic Trading Flow

```python
# Initialize components
broker = AlpacaAdapter(config)
gate_manager = GateManager()
weekly_cycle = WeeklyCycle(portfolio_manager, trade_executor, market_data)

# Connect to broker
await broker.connect()

# Check if buy phase should execute
if weekly_cycle.should_execute_buy():
    # Get available cash
    cash = await broker.get_cash_balance()

    # Execute buy phase
    result = weekly_cycle.execute_buy_phase('G0', float(cash))

    # Log results
    logger.info(f"Buy phase result: {result['success']}")
```

### Trade Validation

```python
# Prepare trade details
trade = {
    'symbol': 'ULTY',
    'side': 'BUY',
    'quantity': 10.0,
    'price': 5.57,
    'trade_type': 'STOCK'
}

# Get current portfolio
portfolio = {
    'cash': 150.0,
    'positions': {},
    'total_value': 200.0
}

# Validate trade
validation = gate_manager.validate_trade(trade, portfolio)

if validation.is_valid:
    # Proceed with trade
    order = Order(symbol='ULTY', qty=Decimal('10.0'), side='buy')
    result = await broker.submit_order(order)
else:
    # Handle violations
    for violation in validation.violations:
        logger.error(f"Violation: {violation['message']}")
```

---

**API Documentation Status:** ✅ **COMPLETE**
**Coverage:** All Foundation phase APIs documented
**Examples:** Comprehensive usage patterns included
**Error Handling:** Full exception documentation provided