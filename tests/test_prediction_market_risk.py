import pytest
from decimal import Decimal

from src.risk.prediction_market_risk import (
    PredictionMarketExposure,
    PredictionMarketRiskConfig,
    PredictionMarketRiskGate,
)
from src.trading.prediction_market_executor import (
    DryRunPredictionMarketExecutor,
    LivePredictionMarketExecutor,
    PredictionMarketExecutionError,
    PredictionMarketInstrument,
    PredictionMarketOrder,
    PredictionMarketVenueSubmission,
    reconcile_prediction_market_order,
)


def _event(**overrides):
    event = {
        "source": "mieza-quant",
        "asset_class": "prediction_market",
        "platform": "polymarket",
        "market_id": "market-1",
        "question": "Will it resolve yes?",
        "signal_type": "equilibrium",
        "side": "yes",
        "direction": 1,
        "edge": 0.08,
        "confidence": 0.72,
        "recommended_size": 3,
        "market_price": 0.42,
        "estimated_fair_price": 0.50,
        "generated_at": "2026-04-30T12:00:00Z",
        "nonce": "risk-test-nonce",
    }
    event.update(overrides)
    return event


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_approves_clean_event():
    decision = PredictionMarketRiskGate().validate_event(_event())

    assert decision.approved
    assert decision.violations == []
    assert decision.contracts == 3
    assert decision.notional == pytest.approx(1.26)


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_rejects_bad_probability_bounds():
    decision = PredictionMarketRiskGate().validate_event(
        _event(market_price=0.0, estimated_fair_price=0.08)
    )

    assert not decision.approved
    assert "market_price outside executable probability bounds" in decision.violations


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_rejects_edge_mismatch():
    decision = PredictionMarketRiskGate().validate_event(
        _event(edge=0.08, market_price=0.42, estimated_fair_price=0.43)
    )

    assert not decision.approved
    assert "estimated fair price does not clear executable edge floor" in decision.violations
    assert "edge disagrees with fair-price delta" in decision.violations


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_rejects_existing_exposure_breach():
    gate = PredictionMarketRiskGate(
        PredictionMarketRiskConfig(max_market_exposure=2.0)
    )
    exposure = PredictionMarketExposure(market_exposure={"market-1": 1.0})

    decision = gate.validate_event(_event(recommended_size=3, market_price=0.42), exposure=exposure)

    assert not decision.approved
    assert "market exposure limit exceeded" in decision.violations


@pytest.mark.unit
@pytest.mark.security
def test_dry_run_prediction_market_executor_is_deterministic():
    event = _event()
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision)
    executor = DryRunPredictionMarketExecutor()

    first = executor.execute(order)
    second = executor.execute(order)

    assert first.order_id == second.order_id
    assert first.idempotency_key == second.idempotency_key
    assert first.status == "dry_run_accepted"
    assert first.dry_run is True


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_accumulates_batch_exposure():
    gate = PredictionMarketRiskGate(
        PredictionMarketRiskConfig(max_platform_exposure=2.0, max_unresolved_exposure=10.0)
    )

    decisions = gate.validate_batch(
        [
            _event(market_id="market-1", nonce="batch-exposure"),
            _event(market_id="market-2", nonce="batch-exposure"),
        ]
    )

    assert decisions[0].approved
    assert not decisions[1].approved
    assert "platform exposure limit exceeded" in decisions[1].violations


@pytest.mark.unit
@pytest.mark.security
def test_prediction_market_risk_gate_rejects_duplicate_execution_key_in_batch():
    decisions = PredictionMarketRiskGate().validate_batch(
        [
            _event(signal_type="equilibrium", edge=0.08),
            _event(signal_type="momentum", edge=0.09, estimated_fair_price=0.51),
        ]
    )

    assert not decisions[0].approved
    assert not decisions[1].approved
    assert "duplicate execution idempotency key in batch" in decisions[0].violations
    assert "duplicate execution idempotency key in batch" in decisions[1].violations


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_fails_closed_until_enabled():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _FakeVenueClient()
    journal = _MemoryExecutionJournal()

    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=False,
        journal=journal,
    )

    with pytest.raises(PredictionMarketExecutionError, match="disabled"):
        executor.execute(order)
    assert client.submits == 0
    assert journal.keys == set()


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_requires_instrument_resolution():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _NoResolveVenueClient()
    journal = _MemoryExecutionJournal()

    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=True,
        journal=journal,
    )

    with pytest.raises(PredictionMarketExecutionError, match="no instrument resolver"):
        executor.execute(order)
    assert client.submits == 0
    assert journal.keys == set()


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_rejects_closed_resolved_instrument_before_submit():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _FakeVenueClient(status="closed")
    journal = _MemoryExecutionJournal()

    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=True,
        journal=journal,
    )

    with pytest.raises(PredictionMarketExecutionError, match="not open"):
        executor.execute(order)
    assert client.submits == 0
    assert journal.keys == set()


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_rejects_tick_mismatch_before_submit():
    event = _event(platform="kalshi", market_price=0.425, estimated_fair_price=0.505, edge=0.08)
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _FakeVenueClient(tick_size="0.01")
    journal = _MemoryExecutionJournal()

    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=True,
        journal=journal,
    )

    with pytest.raises(PredictionMarketExecutionError, match="tick size"):
        executor.execute(order)
    assert client.submits == 0
    assert journal.keys == set()


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_reserves_submits_and_reconciles():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _FakeVenueClient()
    journal = _MemoryExecutionJournal()
    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=True,
        journal=journal,
    )

    result = executor.execute(order)

    assert result.status == "live_reconciled"
    assert result.venue_market_id == "venue-market-1"
    assert result.venue_order_id == "venue-order-1"
    assert result.reconciliation_status == "matched"
    assert client.submits == 1
    assert order.idempotency_key in journal.keys

    with pytest.raises(PredictionMarketExecutionError, match="duplicate"):
        executor.execute(order)
    assert client.submits == 1


@pytest.mark.unit
@pytest.mark.security
def test_live_prediction_market_executor_reports_reconciliation_mismatch():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)
    client = _FakeVenueClient(market_id="wrong-market")
    executor = LivePredictionMarketExecutor(
        {"kalshi": client},
        enable_live=True,
        journal=_MemoryExecutionJournal(),
    )

    result = executor.execute(order)

    assert result.status == "live_reconcile_failed"
    assert result.reconciliation_status == "mismatch"
    assert "market id mismatch" in result.error


@pytest.mark.unit
@pytest.mark.security
def test_reconcile_prediction_market_order_requires_client_order_id_when_configured():
    event = _event(platform="kalshi")
    decision = PredictionMarketRiskGate().validate_event(event)
    order = PredictionMarketOrder.from_alpha_event(event, decision, dry_run=False)

    reconciliation = reconcile_prediction_market_order(
        order,
        "venue-order-1",
        {
            "order": {
                "order_id": "venue-order-1",
                "ticker": order.market_id,
                "side": "yes",
                "action": "buy",
                "status": "resting",
            }
        },
        client_order_id_required=True,
    )

    assert not reconciliation.matched
    assert "venue order snapshot missing client order id" in reconciliation.violations


class _MemoryExecutionJournal:
    def __init__(self):
        self.keys = set()

    def reserve_execution_order(self, order):
        if order.idempotency_key in self.keys:
            return False
        self.keys.add(order.idempotency_key)
        return True


class _FakeVenueClient:
    platform = "kalshi"
    supports_client_order_id = True

    def __init__(self, market_id=None, status="open", tick_size="0.01"):
        self.market_id = market_id
        self.status = status
        self.tick_size = tick_size
        self.submits = 0

    def resolve_instrument(self, order):
        return PredictionMarketInstrument(
            platform=self.platform,
            source_market_id=order.market_id,
            venue_market_id="venue-market-1",
            status=self.status,
            tick_size=Decimal(str(self.tick_size)),
        )

    def submit_order(self, order):
        self.submits += 1
        self._last_order = order
        return PredictionMarketVenueSubmission(
            venue_order_id="venue-order-1",
            venue_status="submitted",
            raw_response={"order_id": "venue-order-1"},
        )

    def get_order(self, venue_order_id):
        return {
            "order": {
                "order_id": venue_order_id,
                "client_order_id": self._last_order.idempotency_key,
                "ticker": self.market_id or self._last_order.venue_market_id,
                "side": "yes",
                "action": "buy",
                "status": "resting",
            }
        }

    def reconcile_order(self, order, venue_order_id, order_snapshot):
        self._last_order = order
        return reconcile_prediction_market_order(
            order,
            venue_order_id,
            order_snapshot,
            client_order_id_required=True,
        )


class _NoResolveVenueClient:
    platform = "kalshi"
    supports_client_order_id = True

    def __init__(self):
        self.submits = 0

    def submit_order(self, order):
        self.submits += 1
        return PredictionMarketVenueSubmission(
            venue_order_id="venue-order-1",
            venue_status="submitted",
            raw_response={"order_id": "venue-order-1"},
        )

    def get_order(self, venue_order_id):
        return {}

    def reconcile_order(self, order, venue_order_id, order_snapshot):
        raise AssertionError("should not reconcile without instrument resolution")
