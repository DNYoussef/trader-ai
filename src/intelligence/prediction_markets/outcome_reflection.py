"""Outcome and reflection records for prediction-market committee decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .committee_schemas import PortfolioDecision, canonical_json, utc_now_text

try:  # Keep import-time behavior clean in minimal installs.
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class MarketOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_id: str = Field(min_length=16)
    decision_id: str = Field(min_length=16)
    venue_order_id: Optional[str] = None
    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    resolved_outcome: str = Field(min_length=1)
    entry_price: float = Field(ge=0.0, le=1.0)
    exit_or_resolution_value: float = Field(ge=0.0, le=1.0)
    contracts: int = Field(ge=0)
    fees: float = Field(default=0.0, ge=0.0)
    slippage: float = Field(default=0.0, ge=0.0)
    pnl: float
    thesis_accuracy: str = Field(min_length=1)
    reflection: Dict[str, Any] = Field(default_factory=dict)
    resolved_at: str
    created_at: str

    @field_validator("resolved_outcome")
    @classmethod
    def _validate_resolved_outcome(cls, value: str) -> str:
        if value not in {"yes", "no", "void", "unknown"}:
            raise ValueError("resolved_outcome must be yes, no, void, or unknown")
        return value

    @field_validator("thesis_accuracy")
    @classmethod
    def _validate_thesis_accuracy(cls, value: str) -> str:
        if value not in {"correct", "incorrect", "void", "unknown"}:
            raise ValueError("thesis_accuracy must be correct, incorrect, void, or unknown")
        return value


class PredictionMarketResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    resolved_outcome: str = Field(min_length=1)
    resolved_at: str
    venue_order_id: Optional[str] = None
    exit_or_resolution_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fees: float = Field(default=0.0, ge=0.0)
    slippage: float = Field(default=0.0, ge=0.0)
    source: str = Field(default="unknown", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("resolved_outcome")
    @classmethod
    def _validate_resolved_outcome(cls, value: str) -> str:
        if value not in {"yes", "no", "void", "unknown"}:
            raise ValueError("resolved_outcome must be yes, no, void, or unknown")
        return value


class PredictionMarketResolutionProvider(Protocol):
    def resolve(
        self,
        *,
        platform: str,
        market_id: str,
        venue_order_id: Optional[str] = None,
    ) -> Optional[PredictionMarketResolution]:
        """Return resolution metadata when a market has resolved."""


class ReflectionStore(Protocol):
    def list_committee_decisions(self) -> List[Dict[str, Any]]:
        """Return persisted committee decision rows."""

    def list_execution_results(self) -> List[Dict[str, Any]]:
        """Return persisted execution rows."""

    def list_market_outcomes(self) -> List[Dict[str, Any]]:
        """Return persisted market outcome rows."""

    def record_market_outcome(self, outcome: "MarketOutcome") -> int:
        """Persist one market outcome."""


class PredictionMarketResolutionError(RuntimeError):
    """Raised when venue resolution metadata is missing or unsafe to trust."""


@dataclass(frozen=True)
class ReflectionJobResult:
    scanned_executions: int = 0
    recorded_outcomes: int = 0
    skipped_count: int = 0
    outcomes: List[MarketOutcome] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PredictionMarketOutcomeReflector:
    """Builds durable outcome/reflection records after resolution."""

    def build_outcome(
        self,
        decision: PortfolioDecision,
        *,
        resolved_outcome: str,
        entry_price: float,
        exit_or_resolution_value: float,
        contracts: int,
        venue_order_id: Optional[str] = None,
        fees: float = 0.0,
        slippage: float = 0.0,
        resolved_at: Optional[str] = None,
        resolution_source: Optional[str] = None,
        resolution_metadata: Optional[Mapping[str, Any]] = None,
    ) -> MarketOutcome:
        pnl = compute_prediction_market_pnl(
            entry_price=entry_price,
            exit_or_resolution_value=exit_or_resolution_value,
            contracts=contracts,
            fees=fees,
            slippage=slippage,
        )
        thesis_accuracy = _thesis_accuracy(decision.side, resolved_outcome)
        reflection = {
            "summary": _reflection_summary(decision, thesis_accuracy, pnl),
            "decision_rating": decision.rating.value,
            "committee_reason": decision.final_reason,
        }
        if resolution_source:
            reflection["resolution_source"] = resolution_source
        if resolution_metadata:
            reflection["resolution_metadata"] = dict(resolution_metadata)
        payload = {
            "outcome_id": "pending",
            "decision_id": decision.decision_id,
            "venue_order_id": venue_order_id,
            "platform": decision.platform,
            "market_id": decision.market_id,
            "resolved_outcome": resolved_outcome,
            "entry_price": entry_price,
            "exit_or_resolution_value": exit_or_resolution_value,
            "contracts": contracts,
            "fees": fees,
            "slippage": slippage,
            "pnl": pnl,
            "thesis_accuracy": thesis_accuracy,
            "reflection": reflection,
            "resolved_at": resolved_at or utc_now_text(),
            "created_at": utc_now_text(),
        }
        payload["outcome_id"] = outcome_id(payload)
        return MarketOutcome(**payload)


class PredictionMarketOutcomeReflectionJob:
    """Replay-safe job that writes outcomes for reconciled live executions."""

    terminal_execution_status = "live_reconciled"

    def __init__(
        self,
        store: ReflectionStore,
        resolution_provider: PredictionMarketResolutionProvider,
        reflector: Optional[PredictionMarketOutcomeReflector] = None,
    ):
        self.store = store
        self.resolution_provider = resolution_provider
        self.reflector = reflector or PredictionMarketOutcomeReflector()

    def run(self) -> ReflectionJobResult:
        committee_by_execution_key, committee_errors = self._committee_by_execution_key()
        existing = {
            (str(row.get("decision_id")), str(row.get("venue_order_id") or ""))
            for row in self.store.list_market_outcomes()
        }
        scanned = 0
        skipped = 0
        outcomes: List[MarketOutcome] = []
        errors = list(committee_errors)

        for row in self.store.list_execution_results():
            scanned += 1
            if bool(row.get("dry_run")):
                skipped += 1
                continue
            if row.get("status") != self.terminal_execution_status:
                skipped += 1
                continue

            try:
                payload = _row_payload(row)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                skipped += 1
                errors.append(
                    f"invalid execution payload {row.get('idempotency_key')}: {exc}"
                )
                continue

            decision = committee_by_execution_key.get(str(row.get("idempotency_key")))
            if decision is None:
                skipped += 1
                errors.append(
                    "missing committee decision for execution "
                    f"{row.get('idempotency_key')}"
                )
                continue

            venue_order_id = str(row.get("venue_order_id") or "")
            outcome_key = (decision.decision_id, venue_order_id)
            if outcome_key in existing:
                skipped += 1
                continue

            try:
                resolution = self.resolution_provider.resolve(
                    platform=str(row.get("platform")),
                    market_id=str(row.get("market_id")),
                    venue_order_id=venue_order_id or None,
                )
            except Exception as exc:
                skipped += 1
                errors.append(
                    f"resolution lookup failed for execution {row.get('idempotency_key')}: {exc}"
                )
                continue
            if resolution is None:
                skipped += 1
                continue

            if not _resolution_matches_execution(row, resolution):
                skipped += 1
                errors.append(
                    "resolution metadata does not match execution "
                    f"{row.get('idempotency_key')}"
                )
                continue

            try:
                outcome = self.reflector.build_outcome(
                    decision,
                    resolved_outcome=resolution.resolved_outcome,
                    entry_price=float(payload["limit_price"]),
                    exit_or_resolution_value=_resolution_value_for_decision(
                        decision,
                        resolution,
                        entry_price=float(payload["limit_price"]),
                    ),
                    contracts=int(payload["contracts"]),
                    venue_order_id=venue_order_id or None,
                    fees=resolution.fees,
                    slippage=resolution.slippage,
                    resolved_at=resolution.resolved_at,
                    resolution_source=resolution.source,
                    resolution_metadata=resolution.metadata,
                )
            except (KeyError, TypeError, ValueError) as exc:
                skipped += 1
                errors.append(
                    f"failed to build outcome for execution {row.get('idempotency_key')}: {exc}"
                )
                continue

            self.store.record_market_outcome(outcome)
            existing.add(outcome_key)
            outcomes.append(outcome)

        return ReflectionJobResult(
            scanned_executions=scanned,
            recorded_outcomes=len(outcomes),
            skipped_count=skipped,
            outcomes=outcomes,
            errors=errors,
        )

    def _committee_by_execution_key(self) -> tuple[Dict[str, PortfolioDecision], List[str]]:
        decisions = {}
        errors = []
        for row in self.store.list_committee_decisions():
            try:
                payload = json.loads(str(row["payload_json"]))
                decision = PortfolioDecision(**payload)
                if not decision.approved:
                    continue
                decisions[_execution_key_for_decision(decision)] = decision
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid committee decision row {row.get('id')}: {exc}")
        return decisions, errors


class StaticPredictionMarketResolutionProvider:
    """In-memory resolution provider for tests and operator-maintained maps."""

    def __init__(self, resolutions: Mapping[str, PredictionMarketResolution] | List[PredictionMarketResolution]):
        values = resolutions.values() if isinstance(resolutions, Mapping) else resolutions
        self.resolutions = {
            _resolution_key(
                resolution.platform,
                resolution.market_id,
                resolution.venue_order_id,
            ): resolution
            for resolution in values
        }

    def resolve(
        self,
        *,
        platform: str,
        market_id: str,
        venue_order_id: Optional[str] = None,
    ) -> Optional[PredictionMarketResolution]:
        exact = self.resolutions.get(_resolution_key(platform, market_id, venue_order_id))
        if exact is not None:
            return exact
        return self.resolutions.get(_resolution_key(platform, market_id, None))


class FilePredictionMarketResolutionProvider(StaticPredictionMarketResolutionProvider):
    """Loads operator-supplied market resolution metadata from JSON."""

    def __init__(self, path: str | Path):
        super().__init__(load_prediction_market_resolutions(path))


class CompositePredictionMarketResolutionProvider:
    """Tries providers in order until one returns a resolution."""

    def __init__(self, providers: List[PredictionMarketResolutionProvider]):
        if not providers:
            raise ValueError("composite resolution provider requires at least one provider")
        self.providers = list(providers)

    def resolve(
        self,
        *,
        platform: str,
        market_id: str,
        venue_order_id: Optional[str] = None,
    ) -> Optional[PredictionMarketResolution]:
        for provider in self.providers:
            resolution = provider.resolve(
                platform=platform,
                market_id=market_id,
                venue_order_id=venue_order_id,
            )
            if resolution is not None:
                return resolution
        return None


class KalshiMarketResolutionProvider:
    """Read-only Kalshi market resolution provider.

    Kalshi exposes market lifecycle status, result, settlement timestamp, and
    settlement value on the public market endpoint.
    """

    platform = "kalshi"

    def __init__(
        self,
        *,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        session: Optional[Any] = None,
        timeout_seconds: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or _requests_session()
        self.timeout_seconds = timeout_seconds

    def resolve(
        self,
        *,
        platform: str,
        market_id: str,
        venue_order_id: Optional[str] = None,
    ) -> Optional[PredictionMarketResolution]:
        if platform.lower() != self.platform:
            return None

        body = _get_json(
            self.session,
            f"{self.base_url}/markets/{market_id}",
            timeout_seconds=self.timeout_seconds,
        )
        market = body.get("market")
        if not isinstance(market, Mapping):
            raise PredictionMarketResolutionError("Kalshi response missing market object")

        ticker = str(market.get("ticker") or "")
        if ticker and ticker != market_id:
            raise PredictionMarketResolutionError("Kalshi response market ticker mismatch")

        status = str(market.get("status") or "").lower()
        if status not in {"determined", "settled"}:
            return None

        resolved_outcome = _normalize_yes_no_void(market.get("result"))
        resolved_at = str(
            market.get("settlement_ts")
            or market.get("updated_time")
            or market.get("expiration_time")
            or ""
        )
        if not resolved_at:
            raise PredictionMarketResolutionError("Kalshi settled market missing settlement timestamp")

        _optional_float(market.get("settlement_value_dollars"))
        return PredictionMarketResolution(
            platform=self.platform,
            market_id=market_id,
            venue_order_id=venue_order_id,
            resolved_outcome=resolved_outcome,
            resolved_at=resolved_at,
            source="kalshi-market-api",
            metadata={
                "status": status,
                "result": market.get("result"),
                "ticker": ticker,
                "settlement_value_dollars": market.get("settlement_value_dollars"),
                "settlement_ts": market.get("settlement_ts"),
            },
        )


class PolymarketGammaResolutionProvider:
    """Read-only Polymarket Gamma resolution provider.

    Gamma exposes closed market metadata and 1:1 outcome/outcomePrices arrays.
    This provider records a resolution only when the closed market has exactly
    one outcome priced at 1 and all other outcomes are 0.
    """

    platform = "polymarket"

    def __init__(
        self,
        *,
        base_url: str = "https://gamma-api.polymarket.com",
        session: Optional[Any] = None,
        timeout_seconds: float = 10.0,
        lookup_by_slug: Optional[bool] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or _requests_session()
        self.timeout_seconds = timeout_seconds
        self.lookup_by_slug = lookup_by_slug

    def resolve(
        self,
        *,
        platform: str,
        market_id: str,
        venue_order_id: Optional[str] = None,
    ) -> Optional[PredictionMarketResolution]:
        if platform.lower() != self.platform:
            return None

        path = self._market_path(market_id)
        market = _get_json(
            self.session,
            f"{self.base_url}{path}",
            timeout_seconds=self.timeout_seconds,
        )
        return _polymarket_resolution_from_market(
            market,
            market_id=market_id,
            venue_order_id=venue_order_id,
        )

    def _market_path(self, market_id: str) -> str:
        if self.lookup_by_slug is True:
            return f"/markets/slug/{market_id}"
        if self.lookup_by_slug is False:
            return f"/markets/{market_id}"
        if market_id.isdigit():
            return f"/markets/{market_id}"
        return f"/markets/slug/{market_id}"


def load_prediction_market_resolutions(path: str | Path) -> List[PredictionMarketResolution]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_resolutions = payload.get("resolutions")
        if raw_resolutions is None:
            raw_resolutions = [payload]
    else:
        raw_resolutions = payload

    if not isinstance(raw_resolutions, list):
        raise ValueError("resolution file must contain a JSON object or list of objects")

    resolutions = []
    for index, item in enumerate(raw_resolutions):
        if not isinstance(item, Mapping):
            raise ValueError(f"resolution entry {index} must be a JSON object")
        resolutions.append(PredictionMarketResolution(**dict(item)))
    return resolutions


def compute_prediction_market_pnl(
    *,
    entry_price: float,
    exit_or_resolution_value: float,
    contracts: int,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> float:
    return (float(exit_or_resolution_value) - float(entry_price)) * int(contracts) - fees - slippage


def outcome_id(payload: Dict[str, object]) -> str:
    clean = dict(payload)
    clean.pop("outcome_id", None)
    clean.pop("created_at", None)
    seed = canonical_json(clean)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _thesis_accuracy(side: str, resolved_outcome: str) -> str:
    if resolved_outcome in {"void", "unknown"}:
        return resolved_outcome
    return "correct" if side == resolved_outcome else "incorrect"


def _reflection_summary(decision: PortfolioDecision, thesis_accuracy: str, pnl: float) -> str:
    return (
        f"Decision {decision.decision_id} resolved as {thesis_accuracy}; "
        f"PnL={pnl:.4f}. Use this only as research context, not as an execution override."
    )


def _execution_key_for_decision(decision: PortfolioDecision) -> str:
    raw = "|".join(
        [
            decision.nonce,
            decision.platform,
            decision.market_id,
            decision.side,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolution_key(
    platform: str,
    market_id: str,
    venue_order_id: Optional[str],
) -> str:
    return "|".join(
        [
            platform.lower(),
            market_id,
            venue_order_id or "",
        ]
    )


def _row_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload_json")
    if not payload:
        return {}
    decoded = json.loads(str(payload))
    if not isinstance(decoded, Mapping):
        return {}
    return dict(decoded)


def _resolution_matches_execution(
    row: Mapping[str, Any],
    resolution: PredictionMarketResolution,
) -> bool:
    return (
        resolution.platform.lower() == str(row.get("platform", "")).lower()
        and resolution.market_id == str(row.get("market_id", ""))
    )


def _resolution_value_for_decision(
    decision: PortfolioDecision,
    resolution: PredictionMarketResolution,
    *,
    entry_price: float,
) -> float:
    if resolution.exit_or_resolution_value is not None:
        return resolution.exit_or_resolution_value
    if resolution.resolved_outcome == "void":
        return entry_price
    if resolution.resolved_outcome == "unknown":
        return entry_price
    return 1.0 if decision.side == resolution.resolved_outcome else 0.0


def _polymarket_resolution_from_market(
    market: Mapping[str, Any],
    *,
    market_id: str,
    venue_order_id: Optional[str],
) -> Optional[PredictionMarketResolution]:
    closed = market.get("closed")
    if closed is not True:
        return None

    outcomes = _json_array_field(market, "outcomes")
    prices = [_float_exact(value) for value in _json_array_field(market, "outcomePrices")]
    if len(outcomes) != len(prices) or len(outcomes) < 2:
        raise PredictionMarketResolutionError(
            "Polymarket outcomes and outcomePrices must be parallel arrays"
        )

    winning_indexes = [index for index, price in enumerate(prices) if price == 1.0]
    losing_prices = [price for price in prices if price != 1.0]
    if len(winning_indexes) != 1 or any(price != 0.0 for price in losing_prices):
        raise PredictionMarketResolutionError(
            "Polymarket closed market does not have an unambiguous 1/0 outcome"
        )

    resolved_outcome = _normalize_binary_outcome_label(outcomes[winning_indexes[0]])
    resolved_at = str(market.get("closedTime") or market.get("updatedAt") or "")
    if not resolved_at:
        raise PredictionMarketResolutionError("Polymarket closed market missing resolution timestamp")

    response_market_id = str(market.get("id") or market.get("slug") or market_id)
    return PredictionMarketResolution(
        platform="polymarket",
        market_id=market_id,
        venue_order_id=venue_order_id,
        resolved_outcome=resolved_outcome,
        resolved_at=resolved_at,
        source="polymarket-gamma-api",
        metadata={
            "id": market.get("id"),
            "slug": market.get("slug"),
            "closed": market.get("closed"),
            "closedTime": market.get("closedTime"),
            "umaResolutionStatus": market.get("umaResolutionStatus"),
            "resolvedBy": market.get("resolvedBy"),
            "outcomes": outcomes,
            "outcomePrices": prices,
            "response_market_id": response_market_id,
        },
    )


def _requests_session() -> Any:
    if requests is None:
        raise PredictionMarketResolutionError("requests is required for venue resolution providers")
    return requests.Session()


def _get_json(session: Any, url: str, *, timeout_seconds: float) -> Dict[str, Any]:
    response = session.get(url, timeout=timeout_seconds)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    elif int(getattr(response, "status_code", 200)) >= 400:
        raise PredictionMarketResolutionError(
            f"resolution provider HTTP error: {getattr(response, 'status_code')}"
        )

    body = response.json()
    if not isinstance(body, Mapping):
        raise PredictionMarketResolutionError("resolution provider response must be a JSON object")
    return dict(body)


def _json_array_field(payload: Mapping[str, Any], field: str) -> List[Any]:
    value = payload.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PredictionMarketResolutionError(f"{field} must be a JSON array string") from exc
    if not isinstance(value, list):
        raise PredictionMarketResolutionError(f"{field} must be a JSON array")
    return value


def _normalize_yes_no_void(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"yes", "no", "void"}:
        return normalized
    raise PredictionMarketResolutionError(
        f"unsupported prediction-market resolved outcome: {value}"
    )


def _normalize_binary_outcome_label(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    raise PredictionMarketResolutionError(
        f"unsupported binary outcome label: {value}"
    )


def _float_exact(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionMarketResolutionError(f"invalid numeric outcome price: {value}") from exc


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionMarketResolutionError(f"invalid settlement value: {value}") from exc
