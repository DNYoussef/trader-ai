"""Prediction-market-specific risk gates for Mieza alpha events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


@dataclass(frozen=True)
class PredictionMarketRiskConfig:
    enabled_platforms: Set[str] = field(default_factory=lambda: {"polymarket", "kalshi"})
    min_edge: float = 0.02
    min_confidence: float = 0.55
    min_price: float = 0.01
    max_price: float = 0.99
    max_contracts_per_signal: int = 10
    max_dollars_per_signal: float = 100.0
    max_market_exposure: float = 200.0
    max_platform_exposure: float = 500.0
    max_unresolved_exposure: float = 1000.0
    edge_tolerance: float = 0.02


@dataclass(frozen=True)
class PredictionMarketExposure:
    market_exposure: Dict[str, float] = field(default_factory=dict)
    platform_exposure: Dict[str, float] = field(default_factory=dict)
    unresolved_exposure: float = 0.0


@dataclass(frozen=True)
class PredictionMarketRiskDecision:
    approved: bool
    violations: List[str]
    contracts: int
    notional: float
    platform: str
    market_id: str
    side: str


class PredictionMarketRiskGate:
    """Survival-first risk gate for prediction-market alpha events."""

    def __init__(self, config: Optional[PredictionMarketRiskConfig] = None):
        self.config = config or PredictionMarketRiskConfig()

    def validate_batch(
        self,
        events: Iterable[Mapping[str, Any]],
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> List[PredictionMarketRiskDecision]:
        events_list = list(events)
        exposure = exposure or PredictionMarketExposure()
        conflicts = self._conflicting_markets(events_list)
        duplicate_execution_keys = self._duplicate_execution_keys(events_list)
        batch_market_exposure = dict(exposure.market_exposure)
        batch_platform_exposure = dict(exposure.platform_exposure)
        batch_unresolved_exposure = exposure.unresolved_exposure
        decisions: List[PredictionMarketRiskDecision] = []

        for event in events_list:
            batch_exposure = PredictionMarketExposure(
                market_exposure=batch_market_exposure,
                platform_exposure=batch_platform_exposure,
                unresolved_exposure=batch_unresolved_exposure,
            )
            decision = self.validate_event(event, exposure=batch_exposure)
            if (event.get("platform"), event.get("market_id")) in conflicts:
                violations = list(decision.violations)
                violations.append("conflicting yes/no signals for the same market")
                decision = PredictionMarketRiskDecision(
                    approved=False,
                    violations=violations,
                    contracts=decision.contracts,
                    notional=decision.notional,
                    platform=decision.platform,
                    market_id=decision.market_id,
                    side=decision.side,
                )
            if self._execution_key(event) in duplicate_execution_keys:
                violations = list(decision.violations)
                violations.append("duplicate execution idempotency key in batch")
                decision = PredictionMarketRiskDecision(
                    approved=False,
                    violations=violations,
                    contracts=decision.contracts,
                    notional=decision.notional,
                    platform=decision.platform,
                    market_id=decision.market_id,
                    side=decision.side,
                )
            decisions.append(decision)
            if decision.approved:
                batch_market_exposure[decision.market_id] = (
                    batch_market_exposure.get(decision.market_id, 0.0) + decision.notional
                )
                batch_platform_exposure[decision.platform] = (
                    batch_platform_exposure.get(decision.platform, 0.0) + decision.notional
                )
                batch_unresolved_exposure += decision.notional

        return decisions

    def validate_event(
        self,
        event: Mapping[str, Any],
        exposure: Optional[PredictionMarketExposure] = None,
    ) -> PredictionMarketRiskDecision:
        exposure = exposure or PredictionMarketExposure()
        violations: List[str] = []

        platform = str(event.get("platform", ""))
        market_id = str(event.get("market_id", ""))
        side = str(event.get("side", ""))
        contracts = _as_int(event.get("recommended_size", 0))
        market_price = _as_float(event.get("market_price", 0.0))
        fair_price = _as_float(event.get("estimated_fair_price", 0.0))
        edge = _as_float(event.get("edge", 0.0))
        confidence = _as_float(event.get("confidence", 0.0))
        notional = contracts * market_price

        if event.get("asset_class") != "prediction_market":
            violations.append("asset_class must be prediction_market")
        if platform not in self.config.enabled_platforms:
            violations.append(f"platform disabled or unsupported: {platform}")
        if side not in {"yes", "no"}:
            violations.append("side must be yes or no")
        if not market_id:
            violations.append("market_id is required")
        if edge < self.config.min_edge:
            violations.append(f"edge below minimum: {edge:.4f} < {self.config.min_edge:.4f}")
        if confidence < self.config.min_confidence:
            violations.append(
                f"confidence below minimum: {confidence:.4f} < {self.config.min_confidence:.4f}"
            )
        if contracts <= 0:
            violations.append("recommended_size must be positive")
        if contracts > self.config.max_contracts_per_signal:
            violations.append("recommended_size exceeds per-signal contract limit")
        if not self.config.min_price <= market_price <= self.config.max_price:
            violations.append("market_price outside executable probability bounds")
        if not 0.0 <= fair_price <= 1.0:
            violations.append("estimated_fair_price must be between 0 and 1")

        implied_edge = fair_price - market_price
        if implied_edge < self.config.min_edge:
            violations.append("estimated fair price does not clear executable edge floor")
        if abs(implied_edge - edge) > self.config.edge_tolerance:
            violations.append("edge disagrees with fair-price delta")

        if notional > self.config.max_dollars_per_signal:
            violations.append("notional exceeds per-signal dollar limit")

        market_after = exposure.market_exposure.get(market_id, 0.0) + notional
        platform_after = exposure.platform_exposure.get(platform, 0.0) + notional
        unresolved_after = exposure.unresolved_exposure + notional

        if market_after > self.config.max_market_exposure:
            violations.append("market exposure limit exceeded")
        if platform_after > self.config.max_platform_exposure:
            violations.append("platform exposure limit exceeded")
        if unresolved_after > self.config.max_unresolved_exposure:
            violations.append("unresolved exposure limit exceeded")

        return PredictionMarketRiskDecision(
            approved=not violations,
            violations=violations,
            contracts=contracts,
            notional=notional,
            platform=platform,
            market_id=market_id,
            side=side,
        )

    @staticmethod
    def _conflicting_markets(events: List[Mapping[str, Any]]) -> Set[tuple[str, str]]:
        sides_by_market: Dict[tuple[str, str], Set[str]] = {}
        for event in events:
            key = (str(event.get("platform", "")), str(event.get("market_id", "")))
            sides_by_market.setdefault(key, set()).add(str(event.get("side", "")))
        return {key for key, sides in sides_by_market.items() if {"yes", "no"}.issubset(sides)}

    @classmethod
    def _duplicate_execution_keys(cls, events: List[Mapping[str, Any]]) -> Set[tuple[str, str, str, str]]:
        seen: Set[tuple[str, str, str, str]] = set()
        duplicates: Set[tuple[str, str, str, str]] = set()
        for event in events:
            key = cls._execution_key(event)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        return duplicates

    @staticmethod
    def _execution_key(event: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(event.get("nonce", "")),
            str(event.get("platform", "")),
            str(event.get("market_id", "")),
            str(event.get("side", "")),
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
