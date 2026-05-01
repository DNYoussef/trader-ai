"""Prediction-market research committee interface.

The committee creates typed, auditable research decisions. It cannot submit
orders and it cannot override the hard risk gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

from .committee_schemas import (
    AnalystReport,
    CommitteeRating,
    DebateState,
    PortfolioDecision,
    RiskReview,
    TraderAction,
    TraderProposal,
    committee_decision_id,
    committee_payload_hash,
    sign_committee_decision,
    utc_now_text,
)


class PredictionMarketCommittee(Protocol):
    """Committee implementations review an alpha event and return a decision."""

    def review_event(self, event: Mapping[str, Any]) -> PortfolioDecision:
        """Return an auditable committee decision for a single alpha event."""


class CommitteeAgent(Protocol):
    """Research agents produce evidence, not orders."""

    role: str

    def analyze(self, event: Mapping[str, Any]) -> AnalystReport:
        """Return one typed analyst report for the alpha event."""


@dataclass(frozen=True)
class FunctionCommitteeAgent:
    """Adapter for deterministic functions and future model-backed agents."""

    role: str
    analyzer: Callable[[Mapping[str, Any]], AnalystReport | Mapping[str, Any]]

    def analyze(self, event: Mapping[str, Any]) -> AnalystReport:
        raw_report = self.analyzer(event)
        report = raw_report if isinstance(raw_report, AnalystReport) else AnalystReport(**raw_report)
        if report.role != self.role:
            raise ValueError(
                f"committee agent role mismatch: expected {self.role}, got {report.role}"
            )
        return report


class MultiAgentPredictionMarketCommittee:
    """Deterministic multi-agent committee behind the production interface.

    The default agents are deliberately local and deterministic. They are a
    scaffold for LLM/LangGraph agents, but the boundary is already strict:
    agents return evidence, this class creates the signed decision, and the
    executor still only sees events after hard gates approve them.
    """

    def __init__(
        self,
        signing_key: Optional[str] = None,
        agents: Optional[Iterable[CommitteeAgent]] = None,
    ):
        self.signing_key = signing_key
        self.agents = list(agents) if agents is not None else default_committee_agents()
        if not self.agents:
            raise ValueError("prediction-market committee requires at least one agent")

    def review_event(self, event: Mapping[str, Any]) -> PortfolioDecision:
        reports = [agent.analyze(event) for agent in self.agents]
        _validate_unique_roles(reports)
        return build_committee_decision(event, reports, signing_key=self.signing_key)


class DeterministicPredictionMarketCommittee(MultiAgentPredictionMarketCommittee):
    """Compatibility baseline with a single deterministic analyst."""

    def __init__(self, signing_key: Optional[str] = None):
        super().__init__(
            signing_key=signing_key,
            agents=[FunctionCommitteeAgent("game_theory_edge", game_theory_edge_agent)],
        )


def default_committee_agents() -> list[CommitteeAgent]:
    return [
        FunctionCommitteeAgent("game_theory_edge", game_theory_edge_agent),
        FunctionCommitteeAgent("liquidity_proxy", liquidity_proxy_agent),
        FunctionCommitteeAgent("resolution_rules", resolution_rules_agent),
    ]


def build_committee_decision(
    event: Mapping[str, Any],
    analyst_reports: Iterable[AnalystReport],
    signing_key: Optional[str] = None,
) -> PortfolioDecision:
    reports = list(analyst_reports)
    if not reports:
        raise ValueError("committee decision requires at least one analyst report")

    platform = str(event.get("platform", ""))
    market_id = str(event.get("market_id", ""))
    side = str(event.get("side", ""))
    market_price = _as_float(event.get("market_price"))
    violations = _dedupe_preserving_order(
        _structural_violations(event) + _blocking_report_violations(reports)
    )

    approved = not violations
    event_key = alpha_event_key(event)
    action = TraderAction.BUY_YES if side == "yes" else TraderAction.BUY_NO
    if not approved:
        action = TraderAction.REJECT

    payload = {
        "schema_version": "prediction_market_committee.v1",
        "decision_id": "pending",
        "nonce": str(event.get("nonce", "")),
        "event_key": event_key,
        "platform": platform,
        "market_id": market_id,
        "side": side,
        "approved": approved,
        "rating": CommitteeRating.APPROVE.value if approved else CommitteeRating.REJECT.value,
        "final_reason": _final_reason(event, approved, violations, len(reports)),
        "required_gates": [
            "signed_envelope",
            "committee_approval",
            "prediction_market_risk_gate",
            "execution_journal",
            "venue_reconciliation",
        ],
        "analyst_reports": [report.model_dump(mode="json") for report in reports],
        "debate_state": _debate_state(reports, approved).model_dump(mode="json"),
        "trader_proposal": TraderProposal(
            action=action,
            platform=platform,
            market_id=market_id,
            side=side if approved else None,
            max_price=market_price if approved else None,
            size_hint=_as_int(event.get("recommended_size")) if approved else None,
            thesis=(
                "Committee forwards the event to hard risk gates only."
                if approved
                else "Committee blocks the event."
            ),
        ).model_dump(mode="json"),
        "risk_review": RiskReview(
            approved=approved,
            violations=violations,
            liquidity_notes="Committee proxy cannot replace deterministic venue checks.",
            exposure_notes="PredictionMarketRiskGate remains authoritative.",
            kill_switch_notes="Committee cannot bypass kill switches or execution gates.",
        ).model_dump(mode="json"),
        "created_at": utc_now_text(),
        "payload_hash": "pending",
        "signature": None,
    }
    payload["payload_hash"] = committee_payload_hash(payload)
    payload["decision_id"] = committee_decision_id(payload)
    decision = PortfolioDecision(**payload)

    if signing_key:
        signed_payload = decision.model_dump(mode="json")
        signed_payload["signature"] = sign_committee_decision(decision, signing_key)
        decision = PortfolioDecision(**signed_payload)
    return decision


def game_theory_edge_agent(event: Mapping[str, Any]) -> AnalystReport:
    platform = str(event.get("platform", ""))
    market_id = str(event.get("market_id", ""))
    confidence = _as_float(event.get("confidence"))
    edge = _as_float(event.get("edge"))
    market_price = _as_float(event.get("market_price"))
    fair_price = _as_float(event.get("estimated_fair_price"))
    risks = [
        "committee baseline does not inspect live order book depth",
        "committee baseline does not verify venue resolution rules",
    ]
    if confidence <= 0:
        risks.append("BLOCK: committee requires positive confidence")
    if edge <= 0:
        risks.append("BLOCK: committee requires positive edge")
    return AnalystReport(
        role="game_theory_edge",
        platform=platform,
        market_id=market_id,
        thesis=(
            f"Mieza estimates fair value {fair_price:.4f} against "
            f"market price {market_price:.4f}."
        ),
        evidence=[
            f"edge={edge:.4f}",
            f"confidence={confidence:.4f}",
            f"signal_type={event.get('signal_type', '')}",
        ],
        risks=risks,
        confidence=_clamp_probability(confidence),
        data_sources=["mieza-quant-signed-envelope"],
    )


def liquidity_proxy_agent(event: Mapping[str, Any]) -> AnalystReport:
    platform = str(event.get("platform", ""))
    market_id = str(event.get("market_id", ""))
    confidence = _as_float(event.get("confidence"))
    market_price = _as_float(event.get("market_price"))
    recommended_size = _as_int(event.get("recommended_size"))
    risks = [
        "live order book depth is not available in the signed envelope",
        "executor must re-check venue status before live submission",
    ]
    if not 0 < market_price < 1:
        risks.append("BLOCK: committee requires executable market_price")
    return AnalystReport(
        role="liquidity_proxy",
        platform=platform,
        market_id=market_id,
        thesis="Signed signal contains an executable probability proxy for risk review.",
        evidence=[
            f"market_price={market_price:.4f}",
            f"recommended_size={recommended_size}",
            f"platform={platform}",
        ],
        risks=risks,
        confidence=_clamp_probability(confidence),
        data_sources=["mieza-quant-signed-envelope"],
    )


def resolution_rules_agent(event: Mapping[str, Any]) -> AnalystReport:
    platform = str(event.get("platform", ""))
    market_id = str(event.get("market_id", ""))
    question = str(event.get("question", ""))
    confidence = _as_float(event.get("confidence"))
    fair_price = _as_float(event.get("estimated_fair_price"))
    risks = [
        "venue-specific resolution text is not independently fetched by committee",
        "outcome reflection must compare thesis against final resolution metadata",
    ]
    if not question:
        risks.append("BLOCK: committee requires a market question")
    if not 0 <= fair_price <= 1:
        risks.append("BLOCK: committee requires estimated_fair_price between 0 and 1")
    return AnalystReport(
        role="resolution_rules",
        platform=platform,
        market_id=market_id,
        thesis=question or "Committee cannot evaluate a market without a question.",
        evidence=[
            f"question_present={bool(question)}",
            f"estimated_fair_price={fair_price:.4f}",
        ],
        risks=risks,
        confidence=_clamp_probability(confidence),
        data_sources=["mieza-quant-signed-envelope"],
    )


def _final_reason(
    event: Mapping[str, Any],
    approved: bool,
    violations: list[str],
    report_count: int,
) -> str:
    if not approved:
        return "Committee rejected event: " + "; ".join(violations)
    return (
        f"Committee approved {event.get('side')} review from {report_count} analyst reports "
        f"for {event.get('platform')} "
        f"{event.get('market_id')} subject to deterministic risk and execution gates."
    )


def _debate_state(reports: list[AnalystReport], approved: bool) -> DebateState:
    blocking = _blocking_report_violations(reports)
    non_blocking_risks = [
        risk
        for report in reports
        for risk in report.risks
        if not risk.startswith("BLOCK:")
    ]
    return DebateState(
        bull_case=_bull_case(reports),
        bear_case=_bear_case(non_blocking_risks, blocking),
        unresolved_questions=_dedupe_preserving_order(
            non_blocking_risks
            or [
                "Has venue instrument resolution confirmed the live market?",
                "Does the latest order book still support the submitted limit price?",
            ]
        ),
        vote_summary={
            report.role: _agent_vote(report, approved)
            for report in reports
        },
    )


def _bull_case(reports: list[AnalystReport]) -> str:
    edge_report = next((report for report in reports if report.role == "game_theory_edge"), reports[0])
    return edge_report.thesis


def _bear_case(non_blocking_risks: list[str], blocking: list[str]) -> str:
    if blocking:
        return "Hard committee blockers: " + "; ".join(blocking)
    if non_blocking_risks:
        return "Residual risks require deterministic venue and execution gates."
    return "No residual committee risks were reported before deterministic gates."


def _agent_vote(report: AnalystReport, approved: bool) -> str:
    if any(risk.startswith("BLOCK:") for risk in report.risks):
        return "reject"
    if not approved:
        return "hold"
    if report.confidence >= 0.5:
        return "approve"
    return "hold"


def _structural_violations(event: Mapping[str, Any]) -> list[str]:
    confidence = _as_float(event.get("confidence"))
    edge = _as_float(event.get("edge"))
    market_price = _as_float(event.get("market_price"))
    fair_price = _as_float(event.get("estimated_fair_price"))
    question = str(event.get("question", ""))
    violations = []

    if not question:
        violations.append("committee requires a market question")
    if confidence <= 0:
        violations.append("committee requires positive confidence")
    if edge <= 0:
        violations.append("committee requires positive edge")
    if not 0 < market_price < 1:
        violations.append("committee requires executable market_price")
    if not 0 <= fair_price <= 1:
        violations.append("committee requires estimated_fair_price between 0 and 1")
    return violations


def _blocking_report_violations(reports: Iterable[AnalystReport]) -> list[str]:
    violations = []
    for report in reports:
        for risk in report.risks:
            if risk.startswith("BLOCK:"):
                violation = risk.removeprefix("BLOCK:").strip()
                if violation:
                    violations.append(violation)
    return violations


def _validate_unique_roles(reports: list[AnalystReport]) -> None:
    roles = [report.role for report in reports]
    duplicates = sorted({role for role in roles if roles.count(role) > 1})
    if duplicates:
        raise ValueError(f"committee agent roles must be unique: {', '.join(duplicates)}")


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


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


def alpha_event_key(event: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(event["nonce"]),
            str(event["platform"]),
            str(event["market_id"]),
            str(event["side"]),
            str(event["signal_type"]),
        ]
    )
