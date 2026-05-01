import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.integration.mieza_quant_bridge import MiezaSignalBridge, sign_mieza_envelope
from src.integration.mieza_signal_ingestion import MiezaSignalIngestionService
from src.integration.mieza_signal_store import MiezaSQLiteStore
from src.risk.prediction_market_risk import PredictionMarketRiskConfig, PredictionMarketRiskGate
from src.trading.prediction_market_executor import PredictionMarketOrder


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mieza_signal_envelope.json"
SIGNING_KEY = "unit-test-mieza-signing-key"
BASE_TIME = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


def _fixture_payload(nonce: str = "ingest-nonce") -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["nonce"] = nonce
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)
    return payload


def _service(db_path: Path, risk_gate: PredictionMarketRiskGate | None = None) -> MiezaSignalIngestionService:
    store = MiezaSQLiteStore(db_path)
    bridge = MiezaSignalBridge(
        signing_key=SIGNING_KEY,
        nonce_store=store,
        max_age_seconds=600,
    )
    return MiezaSignalIngestionService(bridge=bridge, store=store, risk_gate=risk_gate)


@pytest.mark.integration
@pytest.mark.security
def test_valid_envelope_persists_nonce_events_and_dry_run_execution(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("valid-ingest")

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "accepted"
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.execution_results[0]["status"] == "dry_run_accepted"
    assert result.execution_results[0]["dry_run"] is True
    assert service.store.has_nonce("valid-ingest")
    [alpha_event] = service.store.list_alpha_events()
    [committee_decision] = service.store.list_committee_decisions()
    [moo_decision] = service.store.list_moo_decisions()
    [execution_result] = service.store.list_execution_results()
    [audit] = service.store.list_audits()
    assert committee_decision["nonce"] == alpha_event["nonce"]
    assert committee_decision["event_key"] == alpha_event["event_key"]
    assert moo_decision["event_key"] == alpha_event["event_key"]
    assert moo_decision["committee_decision_id"] == committee_decision["decision_id"]
    assert execution_result["idempotency_key"] == result.execution_results[0]["idempotency_key"]
    assert audit["status"] == "accepted"
    assert len(result.committee_decisions[0].analyst_reports) == 3
    assert result.moo_decisions[0].selected_candidate["contracts"] == 3


@pytest.mark.integration
@pytest.mark.security
def test_nonce_replay_is_rejected_across_store_restarts(tmp_path):
    db_path = tmp_path / "mieza.db"
    payload = _fixture_payload("persistent-replay")

    first = _service(db_path)
    assert first.ingest_payload(payload, now=BASE_TIME).status == "accepted"
    first.store.close()

    second = _service(db_path)
    replay = second.ingest_payload(payload, now=BASE_TIME)

    assert replay.status == "rejected"
    assert "duplicate envelope nonce" in replay.errors[0]
    assert len(second.store.list_execution_results()) == 1


@pytest.mark.integration
@pytest.mark.security
def test_tampered_envelope_records_rejected_audit(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("tampered-ingest")
    payload["signals"][0]["confidence"] = 0.99

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "rejected"
    assert "invalid Mieza signal envelope signature" in result.errors[0]
    audits = service.store.list_audits()
    assert audits[-1]["status"] == "rejected"


@pytest.mark.integration
@pytest.mark.security
def test_prediction_market_risk_rejection_prevents_execution(tmp_path):
    risk_gate = PredictionMarketRiskGate(
        PredictionMarketRiskConfig(enabled_platforms={"kalshi"})
    )
    service = _service(tmp_path / "mieza.db", risk_gate=risk_gate)
    payload = _fixture_payload("disabled-platform")

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert "platform disabled or unsupported: polymarket" in result.errors
    assert len(service.store.list_committee_decisions()) == 1
    assert service.store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_committee_approval_cannot_override_deterministic_risk_rejection(tmp_path):
    risk_gate = PredictionMarketRiskGate(
        PredictionMarketRiskConfig(min_confidence=0.99)
    )
    service = _service(tmp_path / "mieza.db", risk_gate=risk_gate)
    payload = _fixture_payload("committee-approve-risk-reject")

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert result.committee_decisions[0].approved
    assert result.moo_decisions[0].approved
    assert "confidence below minimum" in result.errors[0]
    assert service.store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_committee_rejection_prevents_risk_and_execution(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("committee-reject")
    payload["signals"][0]["question"] = ""
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "committee_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert "committee requires a market question" in result.errors
    assert result.execution_results == []
    assert len(service.store.list_committee_decisions()) == 1
    assert service.store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_committee_exception_fails_closed_without_execution(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    bridge = MiezaSignalBridge(
        signing_key=SIGNING_KEY,
        nonce_store=store,
        max_age_seconds=600,
    )
    service = MiezaSignalIngestionService(
        bridge=bridge,
        store=store,
        committee=_ExplodingCommittee(),
    )
    payload = _fixture_payload("committee-exception")

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "committee_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert "committee review failed" in result.errors[0]
    assert store.list_committee_decisions() == []
    assert store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_conflicting_yes_no_batch_is_rejected(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("conflict-batch")
    no_signal = copy.deepcopy(payload["signals"][0])
    no_signal["side"] = "no"
    no_signal["market_price"] = 0.45
    no_signal["estimated_fair_price"] = 0.53
    no_signal["edge"] = 0.08
    payload["signals"].append(no_signal)
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert result.execution_results == []
    assert "conflicting yes/no signals for the same market" in result.errors


@pytest.mark.integration
@pytest.mark.security
def test_validate_only_mode_does_not_create_execution_records(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("validate-only")

    result = service.ingest_payload(payload, now=BASE_TIME, dry_run_execute=False)

    assert result.status == "validated"
    assert result.accepted_count == 1
    assert result.execution_results == []
    assert len(service.store.list_alpha_events()) == 1
    assert len(service.store.list_committee_decisions()) == 1
    assert len(service.store.list_moo_decisions()) == 1
    assert service.store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_moo_sizing_reduces_execution_contracts_before_risk_gate(tmp_path):
    risk_gate = PredictionMarketRiskGate(
        PredictionMarketRiskConfig(max_dollars_per_signal=1.0)
    )
    service = _service(tmp_path / "mieza.db", risk_gate=risk_gate)
    payload = _fixture_payload("moo-size-reduction")
    payload["signals"][0]["recommended_size"] = 10
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "accepted"
    assert result.moo_decisions[0].selected_candidate["contracts"] == 2
    assert result.risk_decisions[0].contracts == 2
    assert result.execution_results[0]["contracts"] == 2
    [moo_row] = service.store.list_moo_decisions()
    assert moo_row["approved"] == 1


@pytest.mark.integration
@pytest.mark.security
def test_moo_exception_fails_closed_without_risk_or_execution(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    bridge = MiezaSignalBridge(
        signing_key=SIGNING_KEY,
        nonce_store=store,
        max_age_seconds=600,
    )
    service = MiezaSignalIngestionService(
        bridge=bridge,
        store=store,
        moo_allocator=_ExplodingMOOAllocator(),
    )
    payload = _fixture_payload("moo-exception")

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert "MOO allocation failed" in result.errors[0]
    assert len(store.list_committee_decisions()) == 1
    assert store.list_moo_decisions() == []
    assert store.list_execution_results() == []


@pytest.mark.integration
@pytest.mark.security
def test_duplicate_same_side_execution_key_batch_is_rejected(tmp_path):
    service = _service(tmp_path / "mieza.db")
    payload = _fixture_payload("duplicate-execution-key")
    duplicate_signal = copy.deepcopy(payload["signals"][0])
    duplicate_signal["signal_type"] = "momentum"
    duplicate_signal["edge"] = 0.09
    duplicate_signal["estimated_fair_price"] = 0.51
    payload["signals"].append(duplicate_signal)
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)

    result = service.ingest_payload(payload, now=BASE_TIME)

    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert result.execution_results == []
    assert "duplicate execution idempotency key in batch" in result.errors


@pytest.mark.integration
@pytest.mark.security
def test_execution_intent_reservation_is_durable_and_upsertable(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    event = {
        "nonce": "intent-reserve",
        "platform": "kalshi",
        "market_id": "market-1",
        "side": "yes",
    }
    order = PredictionMarketOrder(
        platform="kalshi",
        market_id="market-1",
        side="yes",
        contracts=2,
        limit_price=0.42,
        notional=0.84,
        idempotency_key="intent-key",
        dry_run=False,
    )

    assert store.reserve_execution_order(order)
    assert not store.reserve_execution_order(order)
    assert store.list_execution_results()[0]["status"] == "intent_reserved"

    store.record_execution_results(
        [
            {
                "order_id": "venue-order-1",
                "idempotency_key": "intent-key",
                "platform": event["platform"],
                "market_id": event["market_id"],
                "side": event["side"],
                "contracts": 2,
                "limit_price": 0.42,
                "notional": 0.84,
                "status": "live_reconciled",
                "dry_run": False,
                "venue_order_id": "venue-order-1",
                "venue_status": "resting",
                "reconciliation_status": "matched",
                "error": None,
                "raw_response": {"order": {"id": "venue-order-1"}},
                "created_at": "2026-04-30T12:00:00Z",
            }
        ]
    )

    [row] = store.list_execution_results()
    assert row["status"] == "live_reconciled"
    assert row["venue_order_id"] == "venue-order-1"
    assert row["reconciliation_status"] == "matched"


class _ExplodingCommittee:
    def review_event(self, event):
        raise RuntimeError("synthetic committee failure")


class _ExplodingMOOAllocator:
    def score_event(self, event, committee_decision, exposure=None):
        raise RuntimeError("synthetic MOO failure")
