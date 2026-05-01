"""End-to-end Mieza signal ingestion into Trader AI execution workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .mieza_quant_bridge import MiezaBridgeError, MiezaSignalBridge
from .mieza_signal_store import MiezaSQLiteStore
from ..intelligence.prediction_markets.committee_graph import (
    MultiAgentPredictionMarketCommittee,
    PredictionMarketCommittee,
)
from ..intelligence.prediction_markets.committee_schemas import PortfolioDecision
from ..intelligence.prediction_markets.moo_allocation import (
    MiezaMOOAllocationService,
    PredictionMarketMOOAllocator,
    event_with_moo_size,
)
from ..optimization.moo_decision import MOOScoredDecision
from ..risk.prediction_market_risk import (
    PredictionMarketExposure,
    PredictionMarketRiskDecision,
    PredictionMarketRiskGate,
)
from ..trading.prediction_market_executor import (
    DryRunPredictionMarketExecutor,
    PredictionMarketExecutionError,
    PredictionMarketOrder,
)


EXECUTION_MODES = {"validate_only", "dry_run", "live"}


@dataclass(frozen=True)
class MiezaIngestionResult:
    status: str
    accepted_count: int = 0
    rejected_count: int = 0
    alpha_events: List[Dict[str, Any]] = field(default_factory=list)
    committee_decisions: List[PortfolioDecision] = field(default_factory=list)
    moo_decisions: List[MOOScoredDecision] = field(default_factory=list)
    risk_decisions: List[PredictionMarketRiskDecision] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MiezaSignalIngestionService:
    """Validate, persist, risk-gate, and execute Mieza signal envelopes."""

    def __init__(
        self,
        bridge: MiezaSignalBridge,
        store: MiezaSQLiteStore,
        risk_gate: Optional[PredictionMarketRiskGate] = None,
        committee: Optional[PredictionMarketCommittee] = None,
        moo_allocator: Optional[PredictionMarketMOOAllocator] = None,
        executor: Optional[Any] = None,
    ):
        self.bridge = bridge
        self.store = store
        self.risk_gate = risk_gate or PredictionMarketRiskGate()
        self.committee = committee or MultiAgentPredictionMarketCommittee()
        self.moo_allocator = moo_allocator or MiezaMOOAllocationService(self.risk_gate.config)
        self.executor = executor or DryRunPredictionMarketExecutor()

    def ingest_file(
        self,
        path: str | Path,
        now: Optional[datetime] = None,
        dry_run_execute: Optional[bool] = None,
        execution_mode: Optional[str] = None,
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> MiezaIngestionResult:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.ingest_payload(
            payload,
            envelope_ref=str(path),
            now=now,
            dry_run_execute=dry_run_execute,
            execution_mode=execution_mode,
            exposure=exposure,
        )

    def ingest_payload(
        self,
        payload: Mapping[str, Any],
        envelope_ref: Optional[str] = None,
        now: Optional[datetime] = None,
        dry_run_execute: Optional[bool] = None,
        execution_mode: Optional[str] = None,
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> MiezaIngestionResult:
        mode = self._resolve_execution_mode(execution_mode, dry_run_execute)
        try:
            envelope = self.bridge.validate_envelope(payload, now=now)
        except MiezaBridgeError as exc:
            message = str(exc)
            self.store.record_audit("rejected", message, envelope_ref=envelope_ref)
            return MiezaIngestionResult(status="rejected", rejected_count=1, errors=[message])

        alpha_events = envelope.to_alpha_events()

        alpha_event_rows = self.store.record_alpha_events(alpha_events)
        committee_decisions: List[Optional[PortfolioDecision]] = []
        moo_decisions: List[Optional[MOOScoredDecision]] = []
        decisions: List[PredictionMarketRiskDecision] = []
        execution_results: List[Dict[str, Any]] = []
        errors: List[str] = []

        if len(alpha_event_rows) != len(alpha_events):
            message = "duplicate alpha event already persisted"
            self.store.record_audit(
                "duplicate_rejected",
                message,
                envelope_ref=envelope_ref,
                payload={"nonce": envelope.nonce, "signals": len(envelope.signals)},
            )
            return MiezaIngestionResult(
                status="duplicate_rejected",
                accepted_count=0,
                rejected_count=len(alpha_events),
                alpha_events=alpha_events,
                committee_decisions=[],
                moo_decisions=[],
                risk_decisions=decisions,
                errors=[message],
            )

        approved_events: List[Mapping[str, Any]] = []
        approved_event_indexes: List[int] = []
        for index, event in enumerate(alpha_events):
            try:
                committee_decision = self.committee.review_event(event)
            except Exception as exc:
                committee_decision = None
                errors.append(f"committee review failed: {exc}")

            if committee_decision is None:
                committee_decisions.append(None)
                moo_decisions.append(None)
                decisions.append(_committee_failure_decision(event, "committee review failed"))
                continue

            committee_decisions.append(committee_decision)
            self.store.record_committee_decision(committee_decision)
            if not committee_decision.approved:
                moo_decisions.append(None)
                message = f"committee rejected {committee_decision_ref(committee_decision)}"
                errors.extend(committee_decision.risk_review.violations or [committee_decision.final_reason])
                decisions.append(_committee_failure_decision(event, message))
                continue

            try:
                moo_decision = self.moo_allocator.score_event(
                    event,
                    committee_decision,
                    exposure=exposure,
                )
            except Exception as exc:
                moo_decisions.append(None)
                message = f"MOO allocation failed: {exc}"
                errors.append(message)
                decisions.append(_moo_failure_decision(event, message))
                continue

            moo_decisions.append(moo_decision)
            self.store.record_moo_decision(moo_decision)
            if not moo_decision.approved:
                errors.append(moo_decision.final_reason)
                decisions.append(_moo_failure_decision(event, moo_decision.final_reason))
                continue

            approved_events.append(event_with_moo_size(event, moo_decision))
            approved_event_indexes.append(index)
            decisions.append(_committee_placeholder_decision(event))

        approved_risk_decisions = self.risk_gate.validate_batch(approved_events, exposure=exposure)
        for event_index, risk_decision in zip(approved_event_indexes, approved_risk_decisions):
            decisions[event_index] = risk_decision

        for index, event in enumerate(alpha_events):
            committee_decision = committee_decisions[index]
            decision = decisions[index]
            if committee_decision is None or not committee_decision.approved:
                continue
            if not decision.approved:
                errors.extend(decision.violations)
                continue

            if mode in {"dry_run", "live"}:
                order = PredictionMarketOrder.from_alpha_event(
                    event,
                    decision,
                    dry_run=(mode == "dry_run"),
                )
                try:
                    result = self.executor.execute(order).to_record()
                except PredictionMarketExecutionError as exc:
                    errors.append(str(exc))
                    continue
                except Exception as exc:
                    errors.append(f"prediction-market execution failed: {exc}")
                    continue

                if mode == "live" and result.get("status") != "live_reconciled":
                    errors.append(
                        str(result.get("error") or "live prediction-market reconciliation failed")
                    )
                execution_results.append(result)

            elif mode == "validate_only":
                continue

        if execution_results:
            self.store.record_execution_results(execution_results)

        risk_approved_count = sum(1 for decision in decisions if decision.approved)
        accepted_count = self._accepted_count(mode, risk_approved_count, execution_results)
        rejected_count = len(decisions) - accepted_count

        if mode == "validate_only" and not errors and not rejected_count:
            status = "validated"
            reason = "Mieza envelope validated without execution"
        elif errors and accepted_count:
            status = "partially_accepted"
            reason = "some Mieza alpha events failed prediction-market execution or risk gates"
        elif errors and risk_approved_count:
            status = "execution_failed"
            reason = "approved Mieza alpha events failed prediction-market execution"
        elif errors and all(not decision.approved for decision in decisions):
            status = "committee_rejected" if _only_committee_rejections(decisions) else "risk_rejected"
            reason = "all Mieza alpha events failed committee or risk gates"
        elif rejected_count and accepted_count:
            status = "partially_accepted"
            reason = "some Mieza alpha events failed prediction-market risk gates"
        elif rejected_count:
            status = "risk_rejected"
            reason = "all Mieza alpha events failed prediction-market risk gates"
        else:
            status = "accepted"
            reason = "Mieza envelope accepted"

        self.store.record_audit(
            status,
            reason if not errors else "; ".join(sorted(set(errors))),
            envelope_ref=envelope_ref,
            payload={"nonce": envelope.nonce, "signals": len(envelope.signals)},
        )

        return MiezaIngestionResult(
            status=status,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            alpha_events=alpha_events,
            committee_decisions=[decision for decision in committee_decisions if decision is not None],
            moo_decisions=[decision for decision in moo_decisions if decision is not None],
            risk_decisions=decisions,
            execution_results=execution_results,
            errors=errors,
        )

    @staticmethod
    def _resolve_execution_mode(
        execution_mode: Optional[str],
        dry_run_execute: Optional[bool],
    ) -> str:
        if execution_mode is None:
            execution_mode = "dry_run" if dry_run_execute is not False else "validate_only"
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {sorted(EXECUTION_MODES)}")
        return execution_mode

    @staticmethod
    def _accepted_count(
        mode: str,
        risk_approved_count: int,
        execution_results: List[Dict[str, Any]],
    ) -> int:
        if mode == "validate_only":
            return risk_approved_count
        if mode == "live":
            return sum(1 for result in execution_results if result.get("status") == "live_reconciled")
        return sum(1 for result in execution_results if result.get("status") == "dry_run_accepted")


def committee_decision_ref(decision: PortfolioDecision) -> str:
    return f"{decision.platform}:{decision.market_id}:{decision.side}:{decision.decision_id}"


def _committee_failure_decision(
    event: Mapping[str, Any],
    message: str,
) -> PredictionMarketRiskDecision:
    contracts = _as_int(event.get("recommended_size", 0))
    price = _as_float(event.get("market_price", 0.0))
    return PredictionMarketRiskDecision(
        approved=False,
        violations=[message],
        contracts=contracts,
        notional=contracts * price,
        platform=str(event.get("platform", "")),
        market_id=str(event.get("market_id", "")),
        side=str(event.get("side", "")),
    )


def _committee_placeholder_decision(event: Mapping[str, Any]) -> PredictionMarketRiskDecision:
    contracts = _as_int(event.get("recommended_size", 0))
    price = _as_float(event.get("market_price", 0.0))
    return PredictionMarketRiskDecision(
        approved=True,
        violations=[],
        contracts=contracts,
        notional=contracts * price,
        platform=str(event.get("platform", "")),
        market_id=str(event.get("market_id", "")),
        side=str(event.get("side", "")),
    )


def _moo_failure_decision(
    event: Mapping[str, Any],
    message: str,
) -> PredictionMarketRiskDecision:
    contracts = _as_int(event.get("recommended_size", 0))
    price = _as_float(event.get("market_price", 0.0))
    return PredictionMarketRiskDecision(
        approved=False,
        violations=[message],
        contracts=contracts,
        notional=contracts * price,
        platform=str(event.get("platform", "")),
        market_id=str(event.get("market_id", "")),
        side=str(event.get("side", "")),
    )


def _only_committee_rejections(decisions: List[PredictionMarketRiskDecision]) -> bool:
    return bool(decisions) and all(
        any("committee" in violation for violation in decision.violations)
        for decision in decisions
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
