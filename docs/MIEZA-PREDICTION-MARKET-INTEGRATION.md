# Mieza Prediction-Market Integration

Last verified: 2026-05-01

## Boundary

Trader-AI does not scan Kalshi or Polymarket directly through the Mieza Quant codebase. The integration boundary is a signed Mieza signal envelope.

`src/integration/mieza_quant_bridge.py` accepts `mieza.signal.v1` JSON envelopes, verifies them with `MIEZA_SIGNAL_SIGNING_KEY`, rejects stale or future-dated envelopes, rejects replayed nonces, enforces a strict schema, and only allows `polymarket` and `kalshi` platforms.

## Current Flow

1. `MiezaSignalBridge` validates a signed envelope.
2. `MiezaSQLiteStore` records the envelope and nonce durably.
3. `MultiAgentPredictionMarketCommittee` creates a signed, auditable `PortfolioDecision`.
4. `MiezaMOOAllocationService` scores the decision and can reduce size.
5. `PredictionMarketRiskGate` remains the execution authority.
6. `DryRunPredictionMarketExecutor` records a dry-run submission by default.
7. `LivePredictionMarketExecutor` is available only when explicit live gates and venue clients are configured.

MOO sizing is advisory. It can shrink or reject sizing, but it does not grant execution permission by itself.

## Durable State

The default SQLite database is `data/mieza_signals.db`. It stores signal envelopes, nonces, committee decisions, execution intents/submissions, and outcome records.

Outcome reflection only records executions that reached `live_reconciled`; dry-run submissions are intentionally not treated as settled P&L.

## Live Gates

Live prediction-market execution is off by default. It requires:

- `TRADER_AI_ENABLE_LIVE_PREDICTION_MARKETS=true`
- Durable SQLite store, not process-local state
- Venue client configuration for the selected platforms
- Kalshi credentials via `KALSHI_API_KEY_ID` and either `KALSHI_PRIVATE_KEY_PEM` or `KALSHI_PRIVATE_KEY_PATH`
- Platform instrument resolution before order submission
- Durable idempotency reservation before submit
- Venue order read-back reconciliation before a live result is accepted

Venue-backed outcome resolution exists for mocked Kalshi and Polymarket response shapes and fails closed on schema drift. Live order submission also has submit/read-back reconciliation in code. Production use still requires monitored venue operations, credential management, partial-fill handling, and runbooks.

## Verified Commands

```powershell
python -m pytest tests\test_prediction_market_moo.py tests\test_prediction_market_committee.py tests\test_prediction_market_risk.py tests\test_mieza_signal_ingestion.py tests\test_mieza_quant_bridge.py -q
```

This slice is also covered by the broader focused bundle with Kelly and circuit-breaker tests.
