# Daily Loss Limit - Quick Reference

## What It Does
Automatically stops all trading if portfolio loses more than 2% in a single day.

## Key Features

### Automatic Protection
- Triggers at -2.00% daily loss
- Blocks ALL trades until next trading day
- No manual intervention needed
- Resets automatically at market open

### Monitoring
Check daily P&L in logs:
```
Daily P&L: -1.25% (Limit: -2.00%)  <- Safe
Daily P&L: -2.50% (Limit: -2.00%)  <- BLOCKED
```

### When Triggered
You'll see:
```
DAILY LOSS LIMIT TRIGGERED: -2.10%
TRADING BLOCKED: Daily loss limit triggered
```

## How to Adjust Limit

Edit `src/portfolio/portfolio_manager.py`:

```python
# Change from -2% to -3%
self.daily_loss_limit_pct = Decimal("-0.03")
```

## Testing

Run unit tests:
```bash
cd D:/Projects/trader-ai
python -m pytest tests/test_daily_loss_limit.py -v
```

Run simulation:
```bash
python tests/test_daily_loss_limit.py
```

## Implementation Details

### Files Modified
1. `src/portfolio/portfolio_manager.py` - Added daily loss tracking
2. `src/trading_engine.py` - Added protection check

### New Methods
- `PortfolioManager.check_daily_loss()` - Main safety check
- `PortfolioManager._should_reset_daily()` - Reset logic

### State Variables
- `daily_start_value` - Portfolio value at day start
- `daily_reset_time` - Last reset timestamp
- `daily_loss_limit_pct` - Threshold (-0.02 = -2%)
- `daily_loss_triggered` - Protection active flag

## Examples

### Scenario 1: Normal Day
```
10:00 AM - Start: $200.00
11:00 AM - Value: $198.00 (-1.0%) - TRADING
12:00 PM - Value: $197.00 (-1.5%) - TRADING
```

### Scenario 2: Protection Triggered
```
10:00 AM - Start: $200.00
11:00 AM - Value: $198.00 (-1.0%) - TRADING
12:00 PM - Value: $195.00 (-2.5%) - BLOCKED
 2:00 PM - Value: $197.00 (-1.5%) - STILL BLOCKED
```

### Scenario 3: Next Day Reset
```
Day 1, 2 PM - Value: $195.00 (triggered, blocked)
Day 2, 9 AM - Value: $195.00 (reset, new baseline)
Day 2, 10 AM - Value: $194.00 (-0.5% today) - TRADING
```

## Monitoring Integration

Daily loss status appears in:
1. **Console Logs** - Every trading cycle
2. **Audit Trail** - `.claude/.artifacts/audit_log.jsonl`
3. **Critical Alerts** - When limit triggered

## Safety Guarantees

- Cannot be bypassed within same trading day
- Persists across system restarts (uses timestamp comparison)
- Independent of kill switch (additional layer)
- Protects against cascading losses

## Related Safety Features

This complements:
- Kill switch (manual emergency stop)
- Position size limits
- Broker-level protections
- Risk management rules

## Support

See full documentation: `DAILY_LOSS_LIMIT_IMPLEMENTATION.md`

Run tests: `python -m pytest tests/test_daily_loss_limit.py -v`
