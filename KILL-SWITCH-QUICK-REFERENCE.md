# Kill Switch Quick Reference

## What It Does
Automatically blocks ALL trading when safety system enters CRITICAL state.

## Activation Triggers
- CRITICAL safety state detected by safety_integration system
- Manual activation via `activate_kill_switch()` method

## What Happens When Activated

1. `self.kill_switch_activated` set to `True`
2. All pending orders cancelled immediately
3. Audit log entry created
4. All future trading blocked:
   - Automated rebalancing: BLOCKED
   - Manual trades: REJECTED with exception
   - Main loop: EXITS

## Code Locations

### Trading Cycle Protection
**File**: `src/trading_engine.py`
**Line**: 331-334
**Method**: `_execute_trading_cycle()`

### CRITICAL State Handler
**File**: `src/trading_engine.py`
**Line**: 528-543
**Method**: `_check_system_health()`

### Manual Trade Protection
**File**: `src/trading_engine.py`
**Line**: 577-579
**Method**: `execute_manual_trade()`

## Log Messages to Watch For

```
CRITICAL STATE - ACTIVATING KILL SWITCH!
All pending orders cancelled
Trading cycle blocked: Kill switch active
Trading BLOCKED: Kill switch active (CRITICAL safety state)
```

## Audit Log Events

```json
{
  "event": "kill_switch_activated",
  "reason": "CRITICAL safety state",
  "timestamp": "ISO-8601 timestamp"
}
```

## Recovery Steps

1. Check logs for CRITICAL state cause
2. Fix underlying issue
3. Verify safety systems operational
4. Restart trading engine
5. Monitor closely

## Testing

```python
# Simulate kill switch
engine.kill_switch_activated = True

# Should block trading
result = await engine._execute_trading_cycle()  # Returns early

# Should raise exception
try:
    await engine.execute_manual_trade("SPY", 100, "buy")
except Exception as e:
    assert "Kill switch active" in str(e)
```

## Safety Guarantee

**NO TRADING CAN OCCUR WHEN KILL SWITCH IS ACTIVE**

The system is designed to FAIL SAFE - it blocks trading by default when in doubt.
