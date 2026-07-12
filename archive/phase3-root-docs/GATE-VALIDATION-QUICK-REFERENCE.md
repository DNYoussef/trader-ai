# Gate Validation Quick Reference

## TL;DR
Gate validation now runs BEFORE all trade submissions. Trades that violate gate constraints are automatically blocked with detailed error messages.

## Setup (One-Time)

```python
from src.gates.gate_manager import GateManager
from src.trading.trade_executor import TradeExecutor

# Initialize gate manager
gate_manager = GateManager(data_dir="./data/gates")
gate_manager.update_capital(current_capital)

# Pass to trade executor
executor = TradeExecutor(
    broker_adapter=broker,
    portfolio_manager=portfolio,
    market_data_provider=market_data,
    gate_manager=gate_manager  # ADD THIS
)
```

## What Gets Validated

- Asset allowlist for current gate
- Cash floor requirements
- Options permissions
- Theta exposure limits (options)
- Position size limits
- Concentration limits

## What Happens on Violation

```python
# Attempt invalid trade
result = await executor.buy_market_order("SPY", Decimal("10.00"), "G0")

# Raises ValueError with details:
# "Trade blocked by gate validation: [{'type': 'asset_not_allowed', ...}]"

# Also logs warning:
# "Trade validation failed: [{'type': 'asset_not_allowed', ...}]"
```

## Gate Constraints Quick Table

| Gate | Capital Range | Assets | Cash Floor | Options | Max Position |
|------|--------------|--------|------------|---------|--------------|
| G0   | $200-499     | 2      | 50%        | No      | 25%          |
| G1   | $500-999     | 5      | 60%        | No      | 22%          |
| G2   | $1k-2.5k     | 13     | 65%        | No      | 20%          |
| G3   | $2.5k-5k     | 17     | 70%        | Yes     | 20%          |

## Validation Flow

```
BUY:  Parameters -> Price -> GATE CHECK -> Buying Power -> Position Size -> Submit
SELL: Parameters -> Position -> Price -> Quantity -> GATE CHECK -> Submit
```

## Common Violations

### G0 trying to trade non-ULTY/AMDY
```
ValueError: Trade blocked by gate validation:
[{'type': 'asset_not_allowed', 'message': 'Asset SPY not allowed in G0'}]
```

### Violating cash floor
```
ValueError: Trade blocked by gate validation:
[{'type': 'cash_floor_violation', 'message': 'Trade would violate 60% cash floor'}]
```

### Position too large
```
ValueError: Trade blocked by gate validation:
[{'type': 'position_size_exceeded', 'message': 'Trade would exceed 20% position size limit'}]
```

## Testing Your Integration

```python
# Test 1: Valid trade (should succeed)
result = await executor.buy_market_order("ULTY", Decimal("10.00"), "G0")
assert result.status != "error"

# Test 2: Invalid asset (should fail)
try:
    result = await executor.buy_market_order("SPY", Decimal("10.00"), "G0")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "gate validation" in str(e).lower()

# Test 3: Cash floor violation (should fail)
# ... reduce cash to near floor, then try large buy
```

## Backward Compatibility

Old code without gate_manager still works:
```python
# This still works (no validation)
executor = TradeExecutor(broker, portfolio, market_data)
```

## Files Modified

- `src/trading/trade_executor.py` - Added validation logic

## Files Referenced

- `src/gates/gate_manager.py` - Validation implementation

## Documentation

- `GATE-VALIDATION-IMPLEMENTATION.md` - Full implementation details
- `GATE-VALIDATION-FLOW.md` - Flow diagrams and decision trees
- `GATE-VALIDATION-QUICK-REFERENCE.md` - This file

## Need Help?

1. Check logs for violation details
2. Review gate constraints in gate_manager.py
3. Verify current gate level matches capital
4. Check portfolio summary matches expectations

## Common Issues

### Issue: Validation not running
**Solution:** Verify gate_manager parameter was passed to TradeExecutor

### Issue: All trades blocked
**Solution:** Check current gate level with `gate_manager.current_gate`

### Issue: Unexpected violations
**Solution:** Call `gate_manager.get_status_report()` to see current constraints

## Production Checklist

- [ ] gate_manager passed to TradeExecutor
- [ ] Gate capital updated on portfolio changes
- [ ] Violation logging configured
- [ ] Alerts set up for high violation rates
- [ ] Tests cover all violation types
- [ ] Documentation updated
- [ ] Team trained on gate constraints
