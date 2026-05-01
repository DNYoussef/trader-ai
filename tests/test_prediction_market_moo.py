import sqlite3

import pytest
from pydantic import ValidationError

from src.intelligence.prediction_markets.committee_graph import (
    MultiAgentPredictionMarketCommittee,
)
from src.intelligence.prediction_markets.moo_allocation import MiezaMOOAllocationService
from src.integration.mieza_signal_store import MiezaSQLiteStore
from src.optimization.moo_decision import MOOScoredDecision
from src.risk.prediction_market_risk import PredictionMarketExposure, PredictionMarketRiskConfig


def _event(**overrides):
    event = {
        "source": "mieza-quant",
        "asset_class": "prediction_market",
        "platform": "kalshi",
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
        "nonce": "moo-test-nonce",
    }
    event.update(overrides)
    return event


@pytest.mark.unit
@pytest.mark.security
def test_moo_decision_has_stable_identity_and_tamper_detection():
    event = _event()
    committee_decision = MultiAgentPredictionMarketCommittee().review_event(event)
    allocator = MiezaMOOAllocationService()

    first = allocator.score_event(event, committee_decision)
    second = allocator.score_event(event, committee_decision)

    assert first.decision_id == second.decision_id
    assert first.approved
    assert first.selected_candidate["contracts"] == 3

    tampered = first.model_dump(mode="json")
    tampered["selected_candidate"]["contracts"] = 99
    with pytest.raises(ValidationError, match="payload_hash"):
        MOOScoredDecision(**tampered)

    timestamp_tampered = first.model_dump(mode="json")
    timestamp_tampered["created_at"] = "2026-05-01T00:00:00Z"
    with pytest.raises(ValidationError, match="payload_hash"):
        MOOScoredDecision(**timestamp_tampered)


@pytest.mark.unit
@pytest.mark.security
def test_moo_size_never_exceeds_signed_or_risk_caps():
    event = _event(recommended_size=99)
    committee_decision = MultiAgentPredictionMarketCommittee().review_event(event)
    allocator = MiezaMOOAllocationService(
        PredictionMarketRiskConfig(max_contracts_per_signal=10)
    )

    decision = allocator.score_event(event, committee_decision)

    assert decision.selected_candidate["contracts"] <= 10
    assert decision.selected_candidate["contracts"] <= event["recommended_size"]
    assert decision.constraints["cannot_increase_signed_size"] is True


@pytest.mark.unit
@pytest.mark.security
def test_moo_zero_edge_selects_zero_contracts():
    base_event = _event()
    committee_decision = MultiAgentPredictionMarketCommittee().review_event(base_event)
    flat_event = _event(edge=0.0, estimated_fair_price=0.42)

    decision = MiezaMOOAllocationService().score_event(flat_event, committee_decision)

    assert not decision.approved
    assert decision.selected_candidate["contracts"] == 0
    assert "zero contracts" in decision.final_reason


@pytest.mark.unit
@pytest.mark.security
def test_moo_exposure_pressure_reduces_candidate_size():
    event = _event(recommended_size=10)
    committee_decision = MultiAgentPredictionMarketCommittee().review_event(event)
    exposure = PredictionMarketExposure(
        market_exposure={"market-1": 199.2},
        platform_exposure={"kalshi": 0.0},
        unresolved_exposure=0.0,
    )

    decision = MiezaMOOAllocationService().score_event(
        event,
        committee_decision,
        exposure=exposure,
    )

    assert decision.selected_candidate["contracts"] <= 1
    assert decision.constraints["market_cap_contracts"] == 1


@pytest.mark.integration
@pytest.mark.security
def test_duplicate_moo_decision_is_rejected_by_store(tmp_path):
    event = _event()
    committee_decision = MultiAgentPredictionMarketCommittee().review_event(event)
    decision = MiezaMOOAllocationService().score_event(event, committee_decision)
    store = MiezaSQLiteStore(tmp_path / "mieza.db")

    store.record_moo_decision(decision)

    with pytest.raises(sqlite3.IntegrityError):
        store.record_moo_decision(decision)
