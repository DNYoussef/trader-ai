"""Prediction-market MOO sizing recommendations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Protocol

from .committee_graph import alpha_event_key
from .committee_schemas import PortfolioDecision
from ...optimization.moo_decision import (
    MOOScoredDecision,
    build_moo_decision,
    moo_inputs_hash,
    utc_now_text,
)
from ...risk.prediction_market_risk import (
    PredictionMarketExposure,
    PredictionMarketRiskConfig,
)


class PredictionMarketMOOAllocator(Protocol):
    """MOO allocators score sizing, never execution permission."""

    def score_event(
        self,
        event: Mapping[str, Any],
        committee_decision: PortfolioDecision,
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> MOOScoredDecision:
        """Return one audited sizing decision."""


class MiezaMOOAllocationService:
    """Deterministic prediction-market sizing scorer.

    This is the first, local MOO slice. It turns sizing ambiguity into a numeric
    recommendation while keeping the risk gate authoritative.
    """

    decision_type = "prediction_market_sizing"
    optimizer_source = "deterministic_local_v1"

    def __init__(self, risk_config: Optional[PredictionMarketRiskConfig] = None):
        self.risk_config = risk_config or PredictionMarketRiskConfig()

    def score_event(
        self,
        event: Mapping[str, Any],
        committee_decision: PortfolioDecision,
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> MOOScoredDecision:
        exposure = exposure or PredictionMarketExposure()
        platform = str(event.get("platform", ""))
        market_id = str(event.get("market_id", ""))
        side = str(event.get("side", ""))
        price = _as_float(event.get("market_price"))
        fair_price = _as_float(event.get("estimated_fair_price"))
        signed_size = max(0, _as_int(event.get("recommended_size")))
        confidence = _clamp(_as_float(event.get("confidence")), 0.0, 1.0)
        edge = max(0.0, min(_as_float(event.get("edge")), fair_price - price))
        notional_cap_contracts = _safe_floor(self.risk_config.max_dollars_per_signal, price)
        market_cap_contracts = _safe_floor(
            self.risk_config.max_market_exposure - exposure.market_exposure.get(market_id, 0.0),
            price,
        )
        platform_cap_contracts = _safe_floor(
            self.risk_config.max_platform_exposure - exposure.platform_exposure.get(platform, 0.0),
            price,
        )
        unresolved_cap_contracts = _safe_floor(
            self.risk_config.max_unresolved_exposure - exposure.unresolved_exposure,
            price,
        )
        candidate_max = max(
            0,
            min(
                signed_size,
                self.risk_config.max_contracts_per_signal,
                notional_cap_contracts,
                market_cap_contracts,
                platform_cap_contracts,
                unresolved_cap_contracts,
            ),
        )

        candidates = [
            self._candidate(
                contracts=contracts,
                price=price,
                signed_size=signed_size,
                edge=edge,
                confidence=confidence,
            )
            for contracts in range(candidate_max + 1)
        ]
        selected = max(candidates, key=lambda candidate: (candidate["score"], candidate["contracts"]))
        approved = int(selected["contracts"]) > 0
        final_reason = (
            f"MOO selected {selected['contracts']} contracts from signed max {signed_size}"
            if approved
            else "MOO selected zero contracts; expected edge or risk budget is insufficient"
        )
        inputs = {
            "event_key": alpha_event_key(event),
            "committee_decision_id": committee_decision.decision_id,
            "platform": platform,
            "market_id": market_id,
            "side": side,
            "signed_size": signed_size,
            "market_price": price,
            "estimated_fair_price": fair_price,
            "edge": edge,
            "confidence": confidence,
            "exposure": {
                "market": exposure.market_exposure.get(market_id, 0.0),
                "platform": exposure.platform_exposure.get(platform, 0.0),
                "unresolved": exposure.unresolved_exposure,
            },
            "risk_config": {
                "max_contracts_per_signal": self.risk_config.max_contracts_per_signal,
                "max_dollars_per_signal": self.risk_config.max_dollars_per_signal,
                "max_market_exposure": self.risk_config.max_market_exposure,
                "max_platform_exposure": self.risk_config.max_platform_exposure,
                "max_unresolved_exposure": self.risk_config.max_unresolved_exposure,
            },
        }
        return build_moo_decision(
            {
                "schema_version": "moo_scored_decision.v1",
                "decision_id": "pending",
                "decision_type": self.decision_type,
                "nonce": str(event.get("nonce", "")),
                "event_key": alpha_event_key(event),
                "committee_decision_id": committee_decision.decision_id,
                "platform": platform,
                "market_id": market_id,
                "side": side,
                "approved": approved,
                "optimizer_source": self.optimizer_source,
                "fallback_reason": None,
                "inputs_hash": moo_inputs_hash(inputs),
                "objectives": {
                    "neg_expected_value": -float(selected["expected_value"]),
                    "exposure": float(selected["notional"]),
                    "concentration": float(selected["concentration_penalty"]),
                    "price_fragility": float(selected["fragility_penalty"]),
                    "venue_risk": float(selected["venue_risk_penalty"]),
                },
                "constraints": {
                    "signed_max_contracts": signed_size,
                    "candidate_max_contracts": candidate_max,
                    "risk_max_contracts_per_signal": self.risk_config.max_contracts_per_signal,
                    "notional_cap_contracts": notional_cap_contracts,
                    "market_cap_contracts": market_cap_contracts,
                    "platform_cap_contracts": platform_cap_contracts,
                    "unresolved_cap_contracts": unresolved_cap_contracts,
                    "cannot_increase_signed_size": True,
                },
                "candidate_set": candidates,
                "selected_candidate": selected,
                "scores": {
                    "selected_score": float(selected["score"]),
                    "edge_quality": edge,
                    "confidence": confidence,
                    "recommended_contracts": float(selected["contracts"]),
                    "recommended_notional": float(selected["notional"]),
                },
                "final_reason": final_reason,
                "created_at": utc_now_text(),
                "payload_hash": "pending",
            }
        )

    def _candidate(
        self,
        *,
        contracts: int,
        price: float,
        signed_size: int,
        edge: float,
        confidence: float,
    ) -> Dict[str, Any]:
        notional = contracts * price
        expected_value = contracts * edge * confidence
        size_ratio = contracts / signed_size if signed_size > 0 else 0.0
        notional_ratio = (
            notional / self.risk_config.max_dollars_per_signal
            if self.risk_config.max_dollars_per_signal > 0
            else 1.0
        )
        concentration_penalty = size_ratio**2
        fragility_penalty = _clamp(price - edge, 0.0, 1.0) * size_ratio
        venue_risk_penalty = notional_ratio
        score = (
            expected_value
            - 0.02 * concentration_penalty
            - 0.01 * fragility_penalty
            - 0.01 * venue_risk_penalty
        )
        if contracts == 0:
            score = 0.0
        return {
            "contracts": contracts,
            "notional": round(notional, 8),
            "expected_value": round(expected_value, 8),
            "concentration_penalty": round(concentration_penalty, 8),
            "fragility_penalty": round(fragility_penalty, 8),
            "venue_risk_penalty": round(venue_risk_penalty, 8),
            "score": round(score, 8),
        }


def event_with_moo_size(
    event: Mapping[str, Any],
    decision: MOOScoredDecision,
) -> Dict[str, Any]:
    adjusted = dict(event)
    adjusted["recommended_size"] = int(decision.selected_candidate.get("contracts", 0))
    adjusted["moo_decision_id"] = decision.decision_id
    return adjusted


def _safe_floor(dollars: float, price: float) -> int:
    if dollars <= 0 or price <= 0:
        return 0
    return max(0, int(math.floor(dollars / price)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
