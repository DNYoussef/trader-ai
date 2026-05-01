# Trader-AI Architecture

Last verified locally: 2026-05-01.

## Active Runtime

The active equity runtime is:

```text
main.py -> src/trading_engine.py
```

`TradingEngine` loads `config/config.json`, initializes Alpaca, portfolio management, market data, `GateManager`, safety/circuit systems, and the trade executor. The main loop checks every `MAIN_LOOP_INTERVAL_SECONDS = 300` seconds and runs a trading cycle when `rebalance_frequency_minutes` has elapsed. The current config sets that value to `5`.

`src/cycles/weekly_cycle.py` exists, but it is not the scheduler used by `TradingEngine.start()`.

## Equity Trading Flow

```text
config -> TradingEngine.initialize()
       -> AlpacaAdapter
       -> MarketDataProvider
       -> PortfolioManager
       -> GateManager
       -> TradingSafetyIntegration / circuit manager
       -> TradeExecutor
       -> rebalancing cycle
       -> audit JSONL
```

The audit path is `.claude/.artifacts/audit_log.jsonl`. It is append-only JSONL in the current code. Do not treat it as a complete WORM/hash-chain audit system.

## Gates

Runtime gate implementation is `G0` through `G12` in `src/gates/gate_manager.py`:

- `G0`: $200-499, restricted starting universe
- `G1`: $500-999
- `G2`: $1k-2.5k
- `G3`: $2.5k-5k, long-options related constraints
- `G4`-`G12`: $5k through $10M+, conservative ETF/proxy guardrails with progressively tighter position, concentration, and theta limits

Higher-gate configs intentionally use listed ETFs/proxies only. Futures, FX derivatives, swaps, and direct-treasury workflows remain target capabilities until they have venue-specific validators and execution/reconciliation coverage.

## Broker Boundary

`AlpacaAdapter` requires:

- `alpaca-py`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

The current adapter does not provide a mock-broker fallback. Some market-data paths have fallback behavior, but trading initialization fails without Alpaca credentials.

## Safety Boundary

Trade execution is intended to fail closed:

- Safety integration must initialize.
- Circuit manager must exist.
- Kill switch and daily loss checks can block cycles.
- Manual trades check kill switch and circuit-breaker status before submission.

Dashboard `/api/trading/execute` is not a wired live order path and currently returns `501`.

## Mieza Prediction-Market Integration

The prediction-market path is separate from the equity loop. Trader-AI begins at signed Mieza signal envelopes, not at Mieza Quant internals.

```text
signed mieza.signal.v1 envelope
  -> MiezaSignalBridge
  -> MiezaSQLiteStore
  -> MultiAgentPredictionMarketCommittee
  -> MiezaMOOAllocationService
  -> PredictionMarketRiskGate
  -> DryRunPredictionMarketExecutor by default
```

Live prediction-market execution requires explicit opt-in and venue configuration. See `docs/MIEZA-PREDICTION-MARKET-INTEGRATION.md`.

## Testing Posture

Current verified commands and results live in `docs/TESTING_GUIDE.md`. The latest audit verified:

- unit marker slice
- e2e marker slice
- focused Mieza/prediction-market/MOO/Kelly/circuit-breaker bundle
- Bandit high/medium findings at zero for `src/`

## Remaining Architecture Risks

- Equity live mode relies on the interactive `main.py --mode live` prompt; `validate_trading_mode()` is a separate helper and is not the startup authority.
- Prediction-market live execution has fail-closed submit/read-back reconciliation in code, but still needs monitored venue operations, credential procedures, partial-fill handling, and production runbooks before production use.
- Some older docs and dashboard copy still describe roadmap features as if fully shipped. Treat this document and the top-level README as the current source of truth.
