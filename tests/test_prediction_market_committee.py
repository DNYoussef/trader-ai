from datetime import datetime, timezone
import json
import sqlite3

import pytest
from pydantic import ValidationError

from src.intelligence.prediction_markets.committee_graph import (
    DeterministicPredictionMarketCommittee,
    FunctionCommitteeAgent,
    MultiAgentPredictionMarketCommittee,
)
from src.intelligence.prediction_markets.committee_schemas import (
    AnalystReport,
    CommitteeRating,
    PortfolioDecision,
    committee_payload_hash,
    sign_committee_decision,
)
from src.intelligence.prediction_markets.outcome_reflection import (
    CompositePredictionMarketResolutionProvider,
    FilePredictionMarketResolutionProvider,
    KalshiMarketResolutionProvider,
    PolymarketGammaResolutionProvider,
    PredictionMarketOutcomeReflectionJob,
    PredictionMarketOutcomeReflector,
    PredictionMarketResolutionError,
    PredictionMarketResolution,
    StaticPredictionMarketResolutionProvider,
    compute_prediction_market_pnl,
    load_prediction_market_resolutions,
)
from src.integration.mieza_outcome_reflection_cli import main as reflection_cli_main
from src.integration.mieza_signal_store import MiezaSQLiteStore


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
        "nonce": "committee-test-nonce",
    }
    event.update(overrides)
    return event


@pytest.mark.unit
@pytest.mark.security
def test_deterministic_committee_outputs_stable_auditable_decision():
    committee = DeterministicPredictionMarketCommittee(signing_key="committee-key")

    decision = committee.review_event(_event())

    assert decision.approved
    assert decision.rating == CommitteeRating.APPROVE
    assert decision.payload_hash == committee_payload_hash(decision.model_dump(mode="json"))
    assert decision.signature == sign_committee_decision(
        decision.model_copy(update={"signature": None}),
        "committee-key",
    )
    assert decision.trader_proposal.action.value == "buy_yes"
    assert "prediction_market_risk_gate" in decision.required_gates


@pytest.mark.unit
@pytest.mark.security
def test_multi_agent_committee_outputs_replaceable_analyst_reports():
    committee = MultiAgentPredictionMarketCommittee(signing_key="committee-key")

    decision = committee.review_event(_event())

    roles = {report.role for report in decision.analyst_reports}
    assert decision.approved
    assert roles == {"game_theory_edge", "liquidity_proxy", "resolution_rules"}
    assert decision.payload_hash == committee_payload_hash(decision.model_dump(mode="json"))
    assert decision.signature == sign_committee_decision(
        decision.model_copy(update={"signature": None}),
        "committee-key",
    )
    assert decision.debate_state.vote_summary["game_theory_edge"] == "approve"


@pytest.mark.unit
@pytest.mark.security
def test_multi_agent_committee_treats_blocking_agent_risk_as_hard_rejection():
    committee = MultiAgentPredictionMarketCommittee(
        agents=[
            FunctionCommitteeAgent(
                "liquidity_proxy",
                lambda event: AnalystReport(
                    role="liquidity_proxy",
                    platform=event["platform"],
                    market_id=event["market_id"],
                    thesis="Liquidity is insufficient for this signal.",
                    evidence=["spread too wide"],
                    risks=["BLOCK: liquidity too thin"],
                    confidence=0.4,
                    data_sources=["test-agent"],
                ),
            )
        ]
    )

    decision = committee.review_event(_event())

    assert not decision.approved
    assert decision.rating == CommitteeRating.REJECT
    assert decision.trader_proposal.action.value == "reject"
    assert "liquidity too thin" in decision.risk_review.violations
    assert decision.debate_state.vote_summary["liquidity_proxy"] == "reject"


@pytest.mark.unit
@pytest.mark.security
def test_multi_agent_committee_rejects_duplicate_agent_roles():
    agents = [
        FunctionCommitteeAgent("dup", _report_with_role("dup")),
        FunctionCommitteeAgent("dup", _report_with_role("dup")),
    ]

    with pytest.raises(ValueError, match="roles must be unique"):
        MultiAgentPredictionMarketCommittee(agents=agents).review_event(_event())


@pytest.mark.unit
@pytest.mark.security
def test_multi_agent_committee_propagates_agent_failures_for_fail_closed_ingestion():
    def explode(event):
        raise RuntimeError("synthetic analyst failure")

    committee = MultiAgentPredictionMarketCommittee(
        agents=[FunctionCommitteeAgent("broken_agent", explode)]
    )

    with pytest.raises(RuntimeError, match="synthetic analyst failure"):
        committee.review_event(_event())


@pytest.mark.unit
@pytest.mark.security
def test_portfolio_decision_rejects_tampered_hash_or_decision_id():
    decision = DeterministicPredictionMarketCommittee().review_event(_event())

    tampered_payload = decision.model_dump(mode="json")
    tampered_payload["final_reason"] = "changed after signing"
    with pytest.raises(ValidationError, match="payload_hash"):
        PortfolioDecision(**tampered_payload)

    tampered_id_payload = decision.model_dump(mode="json")
    tampered_id_payload["decision_id"] = "0" * 64
    with pytest.raises(ValidationError, match="decision_id"):
        PortfolioDecision(**tampered_id_payload)


@pytest.mark.unit
@pytest.mark.security
def test_deterministic_committee_rejects_incomplete_research_event():
    decision = DeterministicPredictionMarketCommittee().review_event(
        _event(question="", confidence=0.0)
    )

    assert not decision.approved
    assert decision.rating == CommitteeRating.REJECT
    assert "committee requires a market question" in decision.risk_review.violations
    assert "committee requires positive confidence" in decision.risk_review.violations


@pytest.mark.integration
@pytest.mark.security
def test_committee_decision_and_outcome_are_durable(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    decision = DeterministicPredictionMarketCommittee().review_event(_event())
    outcome = PredictionMarketOutcomeReflector().build_outcome(
        decision,
        resolved_outcome="yes",
        entry_price=0.42,
        exit_or_resolution_value=1.0,
        contracts=3,
        venue_order_id="venue-order-1",
        fees=0.01,
        slippage=0.02,
        resolved_at=datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )

    store.record_committee_decision(decision)
    store.record_market_outcome(outcome)

    [decision_row] = store.list_committee_decisions()
    [outcome_row] = store.list_market_outcomes()
    assert decision_row["decision_id"] == decision.decision_id
    assert decision_row["approved"] == 1
    assert outcome_row["decision_id"] == decision.decision_id
    assert outcome_row["pnl"] == pytest.approx((1.0 - 0.42) * 3 - 0.03)
    assert outcome_row["contracts"] == 3
    assert outcome_row["thesis_accuracy"] == "correct"


@pytest.mark.integration
@pytest.mark.security
def test_duplicate_committee_decision_is_rejected_by_store(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    decision = DeterministicPredictionMarketCommittee().review_event(_event())

    store.record_committee_decision(decision)

    with pytest.raises(sqlite3.IntegrityError):
        store.record_committee_decision(decision)


@pytest.mark.integration
@pytest.mark.security
def test_outcome_reflection_job_records_live_resolution_once(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    event = _event(nonce="reflection-live")
    decision = MultiAgentPredictionMarketCommittee().review_event(event)
    store.record_committee_decision(decision)
    store.record_execution_results([_live_execution_record(event)])
    provider = StaticPredictionMarketResolutionProvider(
        {
            "market-1": PredictionMarketResolution(
                platform="kalshi",
                market_id="market-1",
                venue_order_id="venue-order-1",
                resolved_outcome="yes",
                resolved_at="2026-05-06T12:00:00Z",
                fees=0.02,
                slippage=0.01,
                source="unit-test-resolution-feed",
                metadata={"settlement_price": 1.0},
            )
        }
    )
    job = PredictionMarketOutcomeReflectionJob(store, provider)

    first = job.run()
    second = job.run()

    assert first.errors == []
    assert first.scanned_executions == 1
    assert first.recorded_outcomes == 1
    assert second.recorded_outcomes == 0
    [outcome_row] = store.list_market_outcomes()
    assert outcome_row["decision_id"] == decision.decision_id
    assert outcome_row["venue_order_id"] == "venue-order-1"
    assert outcome_row["contracts"] == 3
    assert outcome_row["pnl"] == pytest.approx((1.0 - 0.42) * 3 - 0.03)
    assert len(store.list_market_outcomes()) == 1


@pytest.mark.integration
@pytest.mark.security
def test_outcome_reflection_job_skips_dry_run_executions(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    event = _event(nonce="reflection-dry-run")
    decision = MultiAgentPredictionMarketCommittee().review_event(event)
    dry_run_record = _live_execution_record(event)
    dry_run_record["status"] = "dry_run_accepted"
    dry_run_record["dry_run"] = True
    store.record_committee_decision(decision)
    store.record_execution_results([dry_run_record])
    provider = StaticPredictionMarketResolutionProvider(
        {
            "market-1": PredictionMarketResolution(
                platform="kalshi",
                market_id="market-1",
                resolved_outcome="yes",
                resolved_at="2026-05-06T12:00:00Z",
            )
        }
    )

    result = PredictionMarketOutcomeReflectionJob(store, provider).run()

    assert result.errors == []
    assert result.scanned_executions == 1
    assert result.skipped_count == 1
    assert result.recorded_outcomes == 0
    assert store.list_market_outcomes() == []


@pytest.mark.integration
@pytest.mark.security
def test_outcome_reflection_job_contains_resolution_provider_failure(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    event = _event(nonce="reflection-provider-failure")
    decision = MultiAgentPredictionMarketCommittee().review_event(event)
    store.record_committee_decision(decision)
    store.record_execution_results([_live_execution_record(event)])

    result = PredictionMarketOutcomeReflectionJob(
        store,
        _ExplodingResolutionProvider(),
    ).run()

    assert result.recorded_outcomes == 0
    assert result.skipped_count == 1
    assert "resolution lookup failed" in result.errors[0]
    assert store.list_market_outcomes() == []


@pytest.mark.unit
@pytest.mark.security
def test_kalshi_resolution_provider_parses_settled_market_response():
    session = _FakeSession(
        {
            "market": {
                "ticker": "market-1",
                "status": "settled",
                "result": "yes",
                "settlement_value_dollars": "1.0000",
                "settlement_ts": "2026-05-06T12:00:00Z",
            }
        }
    )
    provider = KalshiMarketResolutionProvider(
        base_url="https://kalshi.test/trade-api/v2",
        session=session,
    )

    resolution = provider.resolve(
        platform="kalshi",
        market_id="market-1",
        venue_order_id="venue-order-1",
    )

    assert resolution is not None
    assert resolution.resolved_outcome == "yes"
    assert resolution.exit_or_resolution_value is None
    assert resolution.source == "kalshi-market-api"
    assert session.urls == ["https://kalshi.test/trade-api/v2/markets/market-1"]


@pytest.mark.unit
@pytest.mark.security
def test_kalshi_resolution_provider_skips_unsettled_market_response():
    provider = KalshiMarketResolutionProvider(
        session=_FakeSession(
            {
                "market": {
                    "ticker": "market-1",
                    "status": "active",
                    "result": "yes",
                    "updated_time": "2026-05-06T12:00:00Z",
                }
            }
        )
    )

    assert provider.resolve(platform="kalshi", market_id="market-1") is None


@pytest.mark.unit
@pytest.mark.security
def test_kalshi_resolution_provider_fails_closed_on_schema_drift():
    provider = KalshiMarketResolutionProvider(session=_FakeSession({"not_market": {}}))

    with pytest.raises(PredictionMarketResolutionError, match="missing market object"):
        provider.resolve(platform="kalshi", market_id="market-1")


@pytest.mark.unit
@pytest.mark.security
def test_polymarket_gamma_resolution_provider_parses_closed_market_response():
    session = _FakeSession(
        {
            "id": "123",
            "slug": "market-1",
            "closed": True,
            "closedTime": "2026-05-06T12:00:00Z",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1","0"]',
            "umaResolutionStatus": "resolved",
            "resolvedBy": "uma",
        }
    )
    provider = PolymarketGammaResolutionProvider(
        base_url="https://gamma.test",
        session=session,
    )

    resolution = provider.resolve(platform="polymarket", market_id="market-1")

    assert resolution is not None
    assert resolution.resolved_outcome == "yes"
    assert resolution.exit_or_resolution_value is None
    assert resolution.source == "polymarket-gamma-api"
    assert session.urls == ["https://gamma.test/markets/slug/market-1"]


@pytest.mark.unit
@pytest.mark.security
def test_polymarket_gamma_resolution_provider_skips_open_market_response():
    provider = PolymarketGammaResolutionProvider(
        session=_FakeSession(
            {
                "id": "123",
                "slug": "market-1",
                "closed": False,
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.6","0.4"]',
            }
        )
    )

    assert provider.resolve(platform="polymarket", market_id="market-1") is None


@pytest.mark.unit
@pytest.mark.security
def test_polymarket_gamma_resolution_provider_fails_closed_on_ambiguous_prices():
    provider = PolymarketGammaResolutionProvider(
        session=_FakeSession(
            {
                "id": "123",
                "slug": "market-1",
                "closed": True,
                "closedTime": "2026-05-06T12:00:00Z",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.5","0.5"]',
            }
        )
    )

    with pytest.raises(PredictionMarketResolutionError, match="unambiguous"):
        provider.resolve(platform="polymarket", market_id="market-1")


@pytest.mark.integration
@pytest.mark.security
def test_venue_resolution_job_uses_side_aware_no_payout(tmp_path):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    event = _event(nonce="venue-no-payout", platform="kalshi", side="no")
    decision = MultiAgentPredictionMarketCommittee().review_event(event)
    store.record_committee_decision(decision)
    store.record_execution_results([_live_execution_record(event)])
    provider = KalshiMarketResolutionProvider(
        session=_FakeSession(
            {
                "market": {
                    "ticker": "market-1",
                    "status": "settled",
                    "result": "no",
                    "settlement_value_dollars": "0.0000",
                    "settlement_ts": "2026-05-06T12:00:00Z",
                }
            }
        )
    )

    result = PredictionMarketOutcomeReflectionJob(store, provider).run()

    assert result.errors == []
    assert result.recorded_outcomes == 1
    [outcome_row] = store.list_market_outcomes()
    assert outcome_row["resolved_outcome"] == "no"
    assert outcome_row["pnl"] == pytest.approx((1.0 - 0.42) * 3)


@pytest.mark.unit
@pytest.mark.security
def test_composite_resolution_provider_tries_providers_in_order():
    resolution = PredictionMarketResolution(
        platform="kalshi",
        market_id="market-1",
        resolved_outcome="yes",
        resolved_at="2026-05-06T12:00:00Z",
    )
    provider = CompositePredictionMarketResolutionProvider(
        [_EmptyResolutionProvider(), StaticPredictionMarketResolutionProvider([resolution])]
    )

    assert provider.resolve(platform="kalshi", market_id="market-1") == resolution


@pytest.mark.unit
@pytest.mark.security
def test_resolution_file_provider_loads_list_and_wrapped_shapes(tmp_path):
    list_path = tmp_path / "resolutions-list.json"
    wrapped_path = tmp_path / "resolutions-wrapped.json"
    payload = {
        "platform": "kalshi",
        "market_id": "market-1",
        "venue_order_id": "venue-order-1",
        "resolved_outcome": "yes",
        "resolved_at": "2026-05-06T12:00:00Z",
        "source": "unit-test-resolution-feed",
    }
    list_path.write_text(json.dumps([payload]), encoding="utf-8")
    wrapped_path.write_text(json.dumps({"resolutions": [payload]}), encoding="utf-8")

    assert load_prediction_market_resolutions(list_path)[0].market_id == "market-1"
    provider = FilePredictionMarketResolutionProvider(wrapped_path)

    resolution = provider.resolve(
        platform="kalshi",
        market_id="market-1",
        venue_order_id="venue-order-1",
    )
    assert resolution is not None
    assert resolution.resolved_outcome == "yes"


@pytest.mark.integration
@pytest.mark.security
def test_outcome_reflection_cli_records_and_replays(tmp_path, capsys):
    db_path = tmp_path / "mieza.db"
    resolution_path = tmp_path / "resolutions.json"
    store = MiezaSQLiteStore(db_path)
    event = _event(nonce="reflection-cli")
    decision = MultiAgentPredictionMarketCommittee().review_event(event)
    store.record_committee_decision(decision)
    store.record_execution_results([_live_execution_record(event)])
    store.close()
    resolution_path.write_text(
        json.dumps(
            {
                "resolutions": [
                    {
                        "platform": "kalshi",
                        "market_id": "market-1",
                        "venue_order_id": "venue-order-1",
                        "resolved_outcome": "yes",
                        "resolved_at": "2026-05-06T12:00:00Z",
                        "fees": 0.02,
                        "slippage": 0.01,
                        "source": "unit-test-resolution-feed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    first_code = reflection_cli_main(
        ["--db", str(db_path), "--resolutions", str(resolution_path)]
    )
    first_output = json.loads(capsys.readouterr().out)
    second_code = reflection_cli_main(
        ["--db", str(db_path), "--resolutions", str(resolution_path)]
    )
    second_output = json.loads(capsys.readouterr().out)

    assert first_code == 0
    assert first_output["status"] == "completed"
    assert first_output["recorded_outcomes"] == 1
    assert len(first_output["outcome_ids"]) == 1
    assert second_code == 0
    assert second_output["recorded_outcomes"] == 0
    assert second_output["skipped_count"] == 1


@pytest.mark.unit
@pytest.mark.security
def test_outcome_reflection_cli_rejects_malformed_resolution_file(tmp_path, capsys):
    bad_path = tmp_path / "bad-resolutions.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    code = reflection_cli_main(
        ["--db", str(tmp_path / "mieza.db"), "--resolutions", str(bad_path)]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 1
    assert output["status"] == "rejected"
    assert output["errors"]


@pytest.mark.unit
@pytest.mark.security
def test_outcome_ids_are_stable_across_replays():
    decision = MultiAgentPredictionMarketCommittee().review_event(_event())
    reflector = PredictionMarketOutcomeReflector()
    kwargs = {
        "resolved_outcome": "yes",
        "entry_price": 0.42,
        "exit_or_resolution_value": 1.0,
        "contracts": 3,
        "venue_order_id": "venue-order-1",
        "fees": 0.02,
        "slippage": 0.01,
        "resolved_at": "2026-05-06T12:00:00Z",
        "resolution_source": "unit-test-resolution-feed",
    }

    first = reflector.build_outcome(decision, **kwargs)
    second = reflector.build_outcome(decision, **kwargs)

    assert first.outcome_id == second.outcome_id


@pytest.mark.unit
def test_prediction_market_pnl_uses_resolution_value_minus_entry_costs():
    assert compute_prediction_market_pnl(
        entry_price=0.4,
        exit_or_resolution_value=1.0,
        contracts=5,
        fees=0.05,
        slippage=0.10,
    ) == pytest.approx(2.85)


def _report_with_role(role: str):
    def build(event):
        return AnalystReport(
            role=role,
            platform=event["platform"],
            market_id=event["market_id"],
            thesis="Synthetic report.",
            evidence=["unit test"],
            risks=[],
            confidence=0.7,
            data_sources=["test-agent"],
        )

    return build


def _live_execution_record(event):
    idempotency_key = _prediction_market_idempotency_key(event)
    return {
        "order_id": "venue-order-1",
        "idempotency_key": idempotency_key,
        "platform": event["platform"],
        "market_id": event["market_id"],
        "side": event["side"],
        "contracts": int(event["recommended_size"]),
        "limit_price": float(event["market_price"]),
        "notional": int(event["recommended_size"]) * float(event["market_price"]),
        "status": "live_reconciled",
        "dry_run": False,
        "venue_order_id": "venue-order-1",
        "venue_status": "filled",
        "reconciliation_status": "matched",
        "error": None,
        "raw_response": {"order": {"id": "venue-order-1"}},
        "created_at": "2026-04-30T12:00:00Z",
    }


def _prediction_market_idempotency_key(event):
    import hashlib

    raw = "|".join(
        [
            str(event["nonce"]),
            str(event["platform"]),
            str(event["market_id"]),
            str(event["side"]),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _ExplodingResolutionProvider:
    def resolve(self, *, platform, market_id, venue_order_id=None):
        raise RuntimeError("synthetic resolution outage")


class _EmptyResolutionProvider:
    def resolve(self, *, platform, market_id, venue_order_id=None):
        return None


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return _FakeResponse(self.payload, self.status_code)
