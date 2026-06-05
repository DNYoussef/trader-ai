# Gate Validation Flow Diagram

## Trade Execution Flow - BUY ORDER

```
buy_market_order(symbol, dollar_amount, gate)
    |
    v
1. Validate order parameters
    - Symbol validation
    - Amount limits
    - Side validation
    - Gate designation
    - Broker connection
    - Market hours
    |
    v
2. Get current market price
    |
    v
3. GATE VALIDATION (NEW)  <--- SAFETY GATE ADDED HERE
    |
    +-- Build trade_details:
    |   - symbol
    |   - side: 'BUY'
    |   - quantity (calculated from dollar_amount)
    |   - price
    |   - trade_type: 'STOCK'
    |
    +-- Get current portfolio summary
    |
    +-- Call gate_manager.validate_trade()
    |   |
    |   +-- Check asset allowlist
    |   +-- Check cash floor requirements
    |   +-- Check options permissions
    |   +-- Check theta limits
    |   +-- Check position size limits
    |   +-- Check concentration limits
    |
    +-- If INVALID:
    |   - Log warning with violations
    |   - Raise ValueError
    |   - TRADE BLOCKED
    |
    +-- If VALID:
        - Continue to next step
    |
    v
4. Check buying power
    |
    v
5. Validate position size
    |
    v
6. Create order object
    |
    v
7. Submit to broker
    |
    v
8. Track order
    |
    v
9. Record transaction
    |
    v
DONE
```

## Trade Execution Flow - SELL ORDER

```
sell_market_order(symbol, dollar_amount, gate)
    |
    v
1. Validate order parameters
    |
    v
2. Get current position
    |
    v
3. Get current market price
    |
    v
4. Calculate sell quantity
    |
    v
5. GATE VALIDATION (NEW)  <--- SAFETY GATE ADDED HERE
    |
    +-- Build trade_details:
    |   - symbol
    |   - side: 'SELL'
    |   - quantity (calculated sell_quantity)
    |   - price
    |   - trade_type: 'STOCK'
    |
    +-- Get current portfolio summary
    |
    +-- Call gate_manager.validate_trade()
    |   |
    |   +-- Check asset allowlist
    |   +-- Check cash floor requirements
    |   +-- Check options permissions
    |   +-- Check theta limits
    |   +-- Check position size limits
    |   +-- Check concentration limits
    |
    +-- If INVALID:
    |   - Log warning with violations
    |   - Raise ValueError
    |   - TRADE BLOCKED
    |
    +-- If VALID:
        - Continue to next step
    |
    v
6. Create order object
    |
    v
7. Submit to broker
    |
    v
8. Track order
    |
    v
9. Record transaction
    |
    v
DONE
```

## Gate Validation Decision Tree

```
validate_trade(trade_details, current_portfolio)
    |
    v
Is symbol in allowed_assets for current gate?
    |
    +-- NO --> VIOLATION: asset_not_allowed --> BLOCK TRADE
    |
    +-- YES
        |
        v
    Is trade_type OPTION?
        |
        +-- YES --> Are options enabled for current gate?
        |           |
        |           +-- NO --> VIOLATION: options_not_allowed --> BLOCK TRADE
        |           |
        |           +-- YES --> Check theta limits
        |                       |
        |                       +-- EXCEEDED --> VIOLATION: theta_limit_exceeded --> BLOCK TRADE
        |                       |
        |                       +-- OK
        |
        +-- NO (STOCK trade)
            |
            v
        Is side BUY?
            |
            +-- YES --> Will post-trade cash violate cash floor?
            |           |
            |           +-- YES --> VIOLATION: cash_floor_violation --> BLOCK TRADE
            |           |
            |           +-- NO --> Will new position size exceed limit?
            |                       |
            |                       +-- YES --> VIOLATION: position_size_exceeded --> BLOCK TRADE
            |                       |
            |                       +-- NO --> ALL CHECKS PASSED --> ALLOW TRADE
            |
            +-- NO (SELL)
                |
                v
            ALL CHECKS PASSED --> ALLOW TRADE
```

## Gate Level Constraints

### G0: $200-499
- Allowed Assets: ULTY, AMDY
- Cash Floor: 50%
- Options: Disabled
- Max Position: 25%
- Max Concentration: 40%

### G1: $500-999
- Allowed Assets: ULTY, AMDY, IAU, GLDM, VTIP
- Cash Floor: 60%
- Options: Disabled
- Max Position: 22%
- Max Concentration: 35%

### G2: $1k-2.5k
- Allowed Assets: G1 + VTI, VTV, VUG, VEA, VWO, SCHD, DGRO, NOBL, VYM
- Cash Floor: 65%
- Options: Disabled
- Max Position: 20%
- Max Concentration: 30%

### G3: $2.5k-5k
- Allowed Assets: G2 + SPY, QQQ, IWM, DIA
- Cash Floor: 70%
- Options: Enabled (long only)
- Max Theta: 0.5%
- Max Position: 20%
- Max Concentration: 30%

## Safety Guarantees

### Before Gate Validation:
1. Order parameters validated
2. Market price obtained
3. (BUY only) Position checked

### After Gate Validation:
1. All gate constraints verified
2. Portfolio state validated
3. Violation logging complete

### If Validation Fails:
- ValueError raised with detailed violation list
- Warning logged to system logs
- Trade execution halted
- No broker submission
- Order never created

### If Validation Passes:
- Continue to broker submission
- Order created
- Order submitted
- Transaction recorded

## Example Violations

### Asset Not Allowed
```
Gate: G0
Attempted: SPY
Violation: Asset SPY not allowed in G0
Allowed: ['ULTY', 'AMDY']
Result: TRADE BLOCKED
```

### Cash Floor Violation
```
Gate: G1
Current Cash: $300
Total Value: $600
Cash Floor Required: 60% ($360)
Attempted Buy: $50
Post-Trade Cash: $250
Violation: Would violate 60% cash floor
Result: TRADE BLOCKED
```

### Position Size Violation
```
Gate: G2
Portfolio Value: $1500
Current Position: $250
Attempted Buy: $100
New Position: $350
Position Percent: 23.3%
Max Allowed: 20%
Violation: Position would exceed 20% limit
Result: TRADE BLOCKED
```

## Integration Points

1. **TradeExecutor Initialization**
   - Add gate_manager parameter
   - Store reference to gate manager

2. **Buy Order Flow**
   - Validation after price check
   - Before buying power check

3. **Sell Order Flow**
   - Validation after quantity calculation
   - Before order creation

4. **Portfolio Manager**
   - Provides portfolio summary
   - Called during validation

5. **Gate Manager**
   - Performs validation
   - Returns validation result
   - Records violations

## Monitoring Recommendations

### Log Monitoring:
- Track validation failures by type
- Monitor violation patterns
- Alert on repeated violations

### Metrics to Track:
- Validation success rate
- Violations by gate level
- Violations by asset
- Violations by violation type
- Time to validate

### Alerts to Configure:
- High violation rate
- Repeated violations for same asset
- Cash floor violations (potential liquidity issue)
- Position size violations (potential concentration risk)
