"""Prediction market execution primitives.

Live execution is intentionally fail-closed: a caller must opt in, provide a
venue client, reserve the idempotency key, and reconcile the submitted order
against the venue before the result is considered accepted.
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Mapping, Optional, Protocol

from ..risk.prediction_market_risk import PredictionMarketRiskDecision

try:  # requests is a project dependency, but keep import-time failures clean.
    import requests
except ImportError:  # pragma: no cover - exercised only in broken installs
    requests = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PredictionMarketOrder:
    platform: str
    market_id: str
    side: str
    contracts: int
    limit_price: float
    notional: float
    idempotency_key: str
    dry_run: bool = True
    venue_market_id: Optional[str] = None

    @classmethod
    def from_alpha_event(
        cls,
        event: Mapping[str, Any],
        risk_decision: PredictionMarketRiskDecision,
        dry_run: bool = True,
    ) -> "PredictionMarketOrder":
        idempotency_key = prediction_market_idempotency_key(event)
        return cls(
            platform=str(event["platform"]),
            market_id=str(event["market_id"]),
            side=str(event["side"]),
            contracts=risk_decision.contracts,
            limit_price=float(event["market_price"]),
            notional=risk_decision.notional,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )


@dataclass(frozen=True)
class PredictionMarketExecutionResult:
    order_id: str
    idempotency_key: str
    platform: str
    market_id: str
    side: str
    contracts: int
    limit_price: float
    notional: float
    status: str
    dry_run: bool
    created_at: str
    venue_market_id: Optional[str] = None
    venue_order_id: Optional[str] = None
    venue_status: Optional[str] = None
    reconciliation_status: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionMarketVenueSubmission:
    """Normalized venue response immediately after submit."""

    venue_order_id: str
    venue_status: str
    raw_response: Dict[str, Any]


@dataclass(frozen=True)
class PredictionMarketReconciliation:
    """Result of comparing local order intent with the venue's order view."""

    matched: bool
    status: str
    venue_status: str
    violations: tuple[str, ...]
    raw_response: Dict[str, Any]


@dataclass(frozen=True)
class PredictionMarketInstrument:
    """Resolved venue instrument required before live submission."""

    platform: str
    source_market_id: str
    venue_market_id: str
    status: str
    tick_size: Decimal = Decimal("0.01")
    min_price: Decimal = Decimal("0.01")
    max_price: Decimal = Decimal("0.99")
    raw_snapshot: Dict[str, Any] = field(default_factory=dict)


class PredictionMarketExecutionError(RuntimeError):
    """Raised when prediction-market execution cannot safely proceed."""


class PredictionMarketExecutionJournal(Protocol):
    """Durable journal required for live order submission."""

    def reserve_execution_order(self, order: PredictionMarketOrder) -> bool:
        """Reserve the order idempotency key before network submission."""


class PredictionMarketVenueClient(Protocol):
    """Venue adapter contract for live prediction-market execution."""

    platform: str
    supports_client_order_id: bool

    def submit_order(self, order: PredictionMarketOrder) -> PredictionMarketVenueSubmission:
        """Submit one order to the venue."""

    def get_order(self, venue_order_id: str) -> Mapping[str, Any]:
        """Return the venue's current view of an order."""

    def reconcile_order(
        self,
        order: PredictionMarketOrder,
        venue_order_id: str,
        order_snapshot: Mapping[str, Any],
    ) -> PredictionMarketReconciliation:
        """Compare local intent with venue state."""


class PredictionMarketInstrumentResolver(Protocol):
    """Resolves a Mieza market id to a concrete, tradable venue instrument."""

    def resolve_instrument(self, order: PredictionMarketOrder) -> PredictionMarketInstrument:
        """Return resolved instrument metadata or raise."""


class DryRunPredictionMarketExecutor:
    """Executor that records deterministic dry-run orders only."""

    def execute(self, order: PredictionMarketOrder) -> PredictionMarketExecutionResult:
        if not order.dry_run:
            raise ValueError("DryRunPredictionMarketExecutor refuses non-dry-run orders")

        order_id = "dryrun_" + hashlib.sha256(order.idempotency_key.encode("utf-8")).hexdigest()[:16]
        return PredictionMarketExecutionResult(
            order_id=order_id,
            idempotency_key=order.idempotency_key,
            platform=order.platform,
            market_id=order.market_id,
            side=order.side,
            contracts=order.contracts,
            limit_price=order.limit_price,
            notional=order.notional,
            status="dry_run_accepted",
            dry_run=True,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )


class LivePredictionMarketExecutor:
    """Fail-closed live executor with venue read-back reconciliation."""

    success_status = "live_reconciled"

    def __init__(
        self,
        venue_clients: Mapping[str, PredictionMarketVenueClient],
        *,
        enable_live: bool = False,
        journal: Optional[PredictionMarketExecutionJournal] = None,
        instrument_resolvers: Optional[Mapping[str, PredictionMarketInstrumentResolver]] = None,
        require_client_order_id: bool = True,
        require_instrument_resolution: bool = True,
    ):
        self.venue_clients = {platform.lower(): client for platform, client in venue_clients.items()}
        self.enable_live = enable_live
        self.journal = journal
        self.instrument_resolvers = {
            platform.lower(): resolver
            for platform, resolver in (instrument_resolvers or {}).items()
        }
        self.require_client_order_id = require_client_order_id
        self.require_instrument_resolution = require_instrument_resolution

    def execute(self, order: PredictionMarketOrder) -> PredictionMarketExecutionResult:
        if order.dry_run:
            raise PredictionMarketExecutionError("LivePredictionMarketExecutor refuses dry-run orders")
        if not self.enable_live:
            raise PredictionMarketExecutionError("live prediction-market execution is disabled")
        if self.journal is None:
            raise PredictionMarketExecutionError("live execution requires a durable execution journal")

        client = self.venue_clients.get(order.platform.lower())
        if client is None:
            raise PredictionMarketExecutionError(f"no live venue client configured for {order.platform}")
        if self.require_client_order_id and not client.supports_client_order_id:
            raise PredictionMarketExecutionError(
                f"{order.platform} client has no documented client-order-id idempotency"
            )
        order = self._resolve_order_instrument(order, client)
        if not self.journal.reserve_execution_order(order):
            raise PredictionMarketExecutionError(
                f"duplicate prediction-market idempotency key: {order.idempotency_key}"
            )

        submission = client.submit_order(order)
        try:
            snapshot = client.get_order(submission.venue_order_id)
            reconciliation = client.reconcile_order(order, submission.venue_order_id, snapshot)
        except Exception as exc:
            return _execution_result(
                order,
                order_id=submission.venue_order_id,
                status="live_reconcile_failed",
                dry_run=False,
                venue_order_id=submission.venue_order_id,
                venue_status=submission.venue_status,
                reconciliation_status="error",
                error=str(exc),
                raw_response=submission.raw_response,
            )

        if not reconciliation.matched:
            return _execution_result(
                order,
                order_id=submission.venue_order_id,
                status="live_reconcile_failed",
                dry_run=False,
                venue_order_id=submission.venue_order_id,
                venue_status=reconciliation.venue_status,
                reconciliation_status=reconciliation.status,
                error="; ".join(reconciliation.violations),
                raw_response=reconciliation.raw_response,
            )

        return _execution_result(
            order,
            order_id=submission.venue_order_id,
            status=self.success_status,
            dry_run=False,
            venue_order_id=submission.venue_order_id,
            venue_status=reconciliation.venue_status,
            reconciliation_status=reconciliation.status,
            raw_response=reconciliation.raw_response,
        )

    def _resolve_order_instrument(
        self,
        order: PredictionMarketOrder,
        client: PredictionMarketVenueClient,
    ) -> PredictionMarketOrder:
        resolver = self.instrument_resolvers.get(order.platform.lower())
        if resolver is None:
            maybe_resolver = getattr(client, "resolve_instrument", None)
            resolver = client if callable(maybe_resolver) else None  # type: ignore[assignment]

        if resolver is None:
            if self.require_instrument_resolution:
                raise PredictionMarketExecutionError(
                    f"no instrument resolver configured for {order.platform}"
                )
            return order

        instrument = resolver.resolve_instrument(order)
        _validate_resolved_instrument(order, instrument)
        return replace(order, venue_market_id=instrument.venue_market_id)


class StaticPredictionMarketInstrumentResolver:
    """In-memory resolver for tests and explicit operator-maintained maps."""

    def __init__(self, instruments: Mapping[str, PredictionMarketInstrument]):
        self.instruments = dict(instruments)

    def resolve_instrument(self, order: PredictionMarketOrder) -> PredictionMarketInstrument:
        instrument = self.instruments.get(order.market_id)
        if instrument is None:
            raise PredictionMarketExecutionError(f"unresolved prediction-market instrument: {order.market_id}")
        return instrument


class KalshiVenueClient:
    """Kalshi V2 event-market order client.

    Uses Kalshi's documented V2 event order endpoint. The signal side is
    interpreted as the outcome to buy: YES maps to a YES bid; NO maps to a YES
    ask at 1 - no_price, which is economically equivalent to buying NO.
    """

    platform = "kalshi"
    supports_client_order_id = True

    def __init__(
        self,
        *,
        api_key_id: str,
        private_key_pem: bytes | str,
        base_url: str = "https://demo-api.kalshi.co/trade-api/v2",
        session: Optional[Any] = None,
        timeout_seconds: float = 10.0,
    ):
        if not api_key_id:
            raise PredictionMarketExecutionError("Kalshi api_key_id is required")
        if not private_key_pem:
            raise PredictionMarketExecutionError("Kalshi private key is required")
        self.api_key_id = api_key_id
        self.base_url = base_url.rstrip("/")
        self.session = session or _requests_session()
        self.timeout_seconds = timeout_seconds
        self.private_key = _load_rsa_private_key(private_key_pem)

    def submit_order(self, order: PredictionMarketOrder) -> PredictionMarketVenueSubmission:
        payload = self._order_payload(order)
        response = _post_json(
            self.session,
            self.base_url + "/portfolio/events/orders",
            headers=self._auth_headers("POST", "/portfolio/events/orders"),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        body = _response_json(response)
        order_id = str(body.get("order_id") or body.get("id") or "")
        if not order_id:
            raise PredictionMarketExecutionError("Kalshi response missing order_id")
        return PredictionMarketVenueSubmission(
            venue_order_id=order_id,
            venue_status=str(body.get("status") or "submitted"),
            raw_response=dict(body),
        )

    def get_order(self, venue_order_id: str) -> Mapping[str, Any]:
        path = f"/portfolio/orders/{venue_order_id}"
        response = _get_json(
            self.session,
            self.base_url + path,
            headers=self._auth_headers("GET", path),
            timeout_seconds=self.timeout_seconds,
        )
        return _response_json(response)

    def resolve_instrument(self, order: PredictionMarketOrder) -> PredictionMarketInstrument:
        path = f"/markets/{order.market_id}"
        response = _get_json(
            self.session,
            self.base_url + path,
            headers=self._auth_headers("GET", path),
            timeout_seconds=self.timeout_seconds,
        )
        body = _response_json(response)
        market = body.get("market")
        if not isinstance(market, Mapping):
            raise PredictionMarketExecutionError("Kalshi market response missing market object")
        ticker = str(market.get("ticker") or order.market_id)
        return PredictionMarketInstrument(
            platform=self.platform,
            source_market_id=order.market_id,
            venue_market_id=ticker,
            status=str(market.get("status") or "unknown"),
            tick_size=_parse_tick_size(market.get("tick_size"), default=Decimal("0.01")),
            raw_snapshot=dict(market),
        )

    def reconcile_order(
        self,
        order: PredictionMarketOrder,
        venue_order_id: str,
        order_snapshot: Mapping[str, Any],
    ) -> PredictionMarketReconciliation:
        return reconcile_prediction_market_order(
            order,
            venue_order_id,
            order_snapshot,
            client_order_id_required=True,
        )

    def _order_payload(self, order: PredictionMarketOrder) -> Dict[str, Any]:
        if order.side == "yes":
            venue_side = "bid"
            venue_price = Decimal(str(order.limit_price))
        elif order.side == "no":
            venue_side = "ask"
            venue_price = Decimal("1") - Decimal(str(order.limit_price))
        else:
            raise PredictionMarketExecutionError(f"unsupported Kalshi side: {order.side}")

        return {
            "ticker": order.venue_market_id or order.market_id,
            "client_order_id": order.idempotency_key,
            "side": venue_side,
            "count": _fixed_decimal(order.contracts, places=2),
            "price": _fixed_decimal(venue_price, places=4),
            "time_in_force": "fill_or_kill",
            "self_trade_prevention_type": "taker_at_cross",
            "reduce_only": False,
        }

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = _rsa_pss_sha256_b64(self.private_key, timestamp, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }


class PolymarketUSVenueClient:
    """Polymarket US REST order client.

    The public docs expose order submission and read-back, but not a client
    order id/idempotency field. Submission is therefore disabled by default;
    callers must explicitly accept that venue risk before use.
    """

    platform = "polymarket"
    supports_client_order_id = False

    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        base_url: str = "https://api.polymarket.us",
        session: Optional[Any] = None,
        timeout_seconds: float = 10.0,
        allow_non_idempotent_submit: bool = False,
    ):
        if not key_id:
            raise PredictionMarketExecutionError("Polymarket US key_id is required")
        if not secret_key:
            raise PredictionMarketExecutionError("Polymarket US secret_key is required")
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self.session = session or _requests_session()
        self.timeout_seconds = timeout_seconds
        self.allow_non_idempotent_submit = allow_non_idempotent_submit
        self.private_key = _load_ed25519_private_key(secret_key)

    def submit_order(self, order: PredictionMarketOrder) -> PredictionMarketVenueSubmission:
        if not self.allow_non_idempotent_submit:
            raise PredictionMarketExecutionError(
                "Polymarket US order submission is disabled because the documented REST "
                "shape does not include client-order-id idempotency"
            )

        payload = self._order_payload(order)
        response = _post_json(
            self.session,
            self.base_url + "/v1/orders",
            headers=self._auth_headers("POST", "/v1/orders"),
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        body = _response_json(response)
        order_id = str(body.get("id") or "")
        if not order_id:
            raise PredictionMarketExecutionError("Polymarket US response missing id")
        return PredictionMarketVenueSubmission(
            venue_order_id=order_id,
            venue_status="submitted",
            raw_response=dict(body),
        )

    def get_order(self, venue_order_id: str) -> Mapping[str, Any]:
        path = f"/v1/order/{venue_order_id}"
        response = _get_json(
            self.session,
            self.base_url + path,
            headers=self._auth_headers("GET", path),
            timeout_seconds=self.timeout_seconds,
        )
        return _response_json(response)

    def resolve_instrument(self, order: PredictionMarketOrder) -> PredictionMarketInstrument:
        path = f"/v1/markets/{order.market_id}/book"
        response = _get_json(
            self.session,
            self.base_url + path,
            headers=self._auth_headers("GET", path),
            timeout_seconds=self.timeout_seconds,
        )
        body = _response_json(response)
        market_data = body.get("marketData")
        if not isinstance(market_data, Mapping):
            raise PredictionMarketExecutionError("Polymarket US market response missing marketData object")
        venue_market_id = str(market_data.get("marketSlug") or order.market_id)
        return PredictionMarketInstrument(
            platform=self.platform,
            source_market_id=order.market_id,
            venue_market_id=venue_market_id,
            status=str(market_data.get("state") or "unknown"),
            tick_size=_parse_tick_size(
                market_data.get("orderPriceMinTickSize")
                or market_data.get("minimumTickSize")
                or market_data.get("tickSize"),
                default=Decimal("0.01"),
            ),
            raw_snapshot=dict(market_data),
        )

    def reconcile_order(
        self,
        order: PredictionMarketOrder,
        venue_order_id: str,
        order_snapshot: Mapping[str, Any],
    ) -> PredictionMarketReconciliation:
        return reconcile_prediction_market_order(
            order,
            venue_order_id,
            order_snapshot,
            client_order_id_required=False,
        )

    def _order_payload(self, order: PredictionMarketOrder) -> Dict[str, Any]:
        return {
            "marketSlug": order.venue_market_id or order.market_id,
            "type": "ORDER_TYPE_LIMIT",
            "price": {
                "value": _fixed_decimal(order.limit_price, places=4),
                "currency": "USD",
            },
            "quantity": order.contracts,
            "tif": "TIME_IN_FORCE_FILL_OR_KILL",
            "outcomeSide": "OUTCOME_SIDE_YES" if order.side == "yes" else "OUTCOME_SIDE_NO",
            "action": "ORDER_ACTION_BUY",
            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
            "synchronousExecution": False,
        }

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method}{path}".encode("utf-8")
        signature = base64.b64encode(self.private_key.sign(message)).decode("utf-8")
        return {
            "X-PM-Access-Key": self.key_id,
            "X-PM-Timestamp": timestamp,
            "X-PM-Signature": signature,
            "Content-Type": "application/json",
        }


def prediction_market_idempotency_key(event: Mapping[str, Any]) -> str:
    raw = "|".join(
        [
            str(event["nonce"]),
            str(event["platform"]),
            str(event["market_id"]),
            str(event["side"]),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def reconcile_prediction_market_order(
    order: PredictionMarketOrder,
    venue_order_id: str,
    order_snapshot: Mapping[str, Any],
    *,
    client_order_id_required: bool,
) -> PredictionMarketReconciliation:
    normalized = _unwrap_order_snapshot(order_snapshot)
    violations = []

    snapshot_order_id = _first_present(normalized, "order_id", "id")
    if snapshot_order_id and str(snapshot_order_id) != venue_order_id:
        violations.append("venue order id mismatch")

    client_order_id = _first_present(normalized, "client_order_id", "clientOrderId", "clOrdId")
    if client_order_id_required:
        if not client_order_id:
            violations.append("venue order snapshot missing client order id")
        elif str(client_order_id) != order.idempotency_key:
            violations.append("client order id mismatch")

    expected_market_id = order.venue_market_id or order.market_id
    market_id = _first_present(normalized, "ticker", "marketSlug", "market_slug", "market")
    if market_id and str(market_id) != expected_market_id:
        violations.append("market id mismatch")

    if not _snapshot_side_matches(order, normalized):
        violations.append("side mismatch")

    venue_status = str(_first_present(normalized, "status", "state") or "unknown")
    status = "matched" if not violations else "mismatch"
    return PredictionMarketReconciliation(
        matched=not violations,
        status=status,
        venue_status=venue_status,
        violations=tuple(violations),
        raw_response=dict(order_snapshot),
    )


def _execution_result(
    order: PredictionMarketOrder,
    *,
    order_id: str,
    status: str,
    dry_run: bool,
    venue_order_id: Optional[str] = None,
    venue_status: Optional[str] = None,
    reconciliation_status: Optional[str] = None,
    error: Optional[str] = None,
    raw_response: Optional[Dict[str, Any]] = None,
) -> PredictionMarketExecutionResult:
    return PredictionMarketExecutionResult(
        order_id=order_id,
        idempotency_key=order.idempotency_key,
        platform=order.platform,
        market_id=order.market_id,
        side=order.side,
        contracts=order.contracts,
        limit_price=order.limit_price,
        notional=order.notional,
        status=status,
        dry_run=dry_run,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        venue_market_id=order.venue_market_id,
        venue_order_id=venue_order_id,
        venue_status=venue_status,
        reconciliation_status=reconciliation_status,
        error=error,
        raw_response=raw_response,
    )


def _requests_session() -> Any:
    if requests is None:
        raise PredictionMarketExecutionError("requests is required for live venue clients")
    return requests.Session()


def _post_json(
    session: Any,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    response = session.post(url, headers=dict(headers), json=dict(payload), timeout=timeout_seconds)
    _raise_for_status(response)
    return response


def _get_json(
    session: Any,
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Any:
    response = session.get(url, headers=dict(headers), timeout=timeout_seconds)
    _raise_for_status(response)
    return response


def _raise_for_status(response: Any) -> None:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
        return
    status_code = int(getattr(response, "status_code", 200))
    if status_code >= 400:
        raise PredictionMarketExecutionError(f"venue HTTP error: {status_code}")


def _response_json(response: Any) -> Dict[str, Any]:
    body = response.json()
    if not isinstance(body, Mapping):
        raise PredictionMarketExecutionError("venue response must be a JSON object")
    return dict(body)


def _load_rsa_private_key(private_key_pem: bytes | str) -> Any:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:  # pragma: no cover - dependency is installed in tests
        raise PredictionMarketExecutionError("cryptography is required for Kalshi signing") from exc

    raw = private_key_pem.encode("utf-8") if isinstance(private_key_pem, str) else private_key_pem
    return serialization.load_pem_private_key(raw, password=None, backend=default_backend())


def _load_ed25519_private_key(secret_key: str) -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - dependency is installed in tests
        raise PredictionMarketExecutionError("cryptography is required for Polymarket US signing") from exc

    raw = base64.b64decode(secret_key)
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])


def _rsa_pss_sha256_b64(private_key: Any, timestamp: str, method: str, path: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _fixed_decimal(value: Any, *, places: int) -> str:
    quantum = Decimal("1").scaleb(-places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{places}f}"


def _validate_resolved_instrument(
    order: PredictionMarketOrder,
    instrument: PredictionMarketInstrument,
) -> None:
    if instrument.platform.lower() != order.platform.lower():
        raise PredictionMarketExecutionError("resolved instrument platform mismatch")
    if instrument.source_market_id != order.market_id:
        raise PredictionMarketExecutionError("resolved instrument source market mismatch")
    if not instrument.venue_market_id:
        raise PredictionMarketExecutionError("resolved instrument missing venue market id")
    if not _status_is_tradable(instrument.status):
        raise PredictionMarketExecutionError(
            f"resolved instrument is not open for trading: {instrument.status}"
        )

    venue_price = _venue_yes_price(order)
    if not instrument.min_price <= venue_price <= instrument.max_price:
        raise PredictionMarketExecutionError("resolved venue price outside instrument bounds")
    if not _price_matches_tick(venue_price, instrument.tick_size):
        raise PredictionMarketExecutionError(
            f"resolved venue price {venue_price} violates tick size {instrument.tick_size}"
        )


def _venue_yes_price(order: PredictionMarketOrder) -> Decimal:
    price = Decimal(str(order.limit_price))
    if order.side == "no":
        return Decimal("1") - price
    return price


def _status_is_tradable(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized in {
        "open",
        "active",
        "trading",
        "market_state_open",
        "order_state_new",
    }


def _price_matches_tick(price: Decimal, tick_size: Decimal) -> bool:
    if tick_size <= 0:
        raise PredictionMarketExecutionError("resolved instrument tick size must be positive")
    ratio = price / tick_size
    return ratio == ratio.to_integral_value()


def _parse_tick_size(value: Any, *, default: Decimal) -> Decimal:
    if value is None:
        return default
    parsed = Decimal(str(value))
    if parsed <= 0:
        raise PredictionMarketExecutionError("instrument tick size must be positive")
    if parsed >= 1:
        return parsed / Decimal("100")
    return parsed


def _unwrap_order_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    current: Mapping[str, Any] = snapshot
    for key in ("order", "data"):
        nested = current.get(key)
        if isinstance(nested, Mapping):
            current = nested
    return dict(current)


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _snapshot_side_matches(order: PredictionMarketOrder, snapshot: Mapping[str, Any]) -> bool:
    side = str(_first_present(snapshot, "side", "outcomeSide", "outcome") or "").lower()
    action = str(_first_present(snapshot, "action") or "").lower()

    if not side and not action:
        return True

    buy_actions = {"", "buy", "order_action_buy"}
    sell_actions = {"sell", "order_action_sell"}
    yes_sides = {"yes", "bid", "buy", "order_side_buy", "outcome_side_yes"}
    no_sides = {"no", "ask", "outcome_side_no"}

    if action in sell_actions:
        return order.side == "no" and side in {"yes", "outcome_side_yes"}
    if action in buy_actions:
        if order.side == "yes":
            return side in yes_sides
        if order.side == "no":
            return side in no_sides
        return False

    if order.side == "yes":
        return side in yes_sides
    if order.side == "no":
        return side in no_sides
    return False
