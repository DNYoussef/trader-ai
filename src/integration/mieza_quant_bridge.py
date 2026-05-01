"""Signed JSON bridge for Mieza Quant signal ingestion.

The bridge intentionally stops at alpha-event ingestion. Mieza signals describe
prediction-market opportunities; they are not stock broker orders.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set


SCHEMA_VERSION = "mieza.signal.v1"
SOURCE = "mieza-quant"
SIGNING_KEY_ENV = "MIEZA_SIGNAL_SIGNING_KEY"

ENVELOPE_FIELDS = {
    "schema_version",
    "source",
    "generated_at",
    "nonce",
    "signals",
    "signature",
}
SIGNAL_FIELDS = {
    "market_id",
    "platform",
    "question",
    "signal_type",
    "side",
    "edge",
    "confidence",
    "recommended_size",
    "market_price",
    "estimated_fair_price",
}
PLATFORMS = {"polymarket", "kalshi"}
SIGNAL_TYPES = {"arbitrage", "equilibrium", "momentum"}
SIDES = {"yes", "no"}


class MiezaBridgeError(ValueError):
    """Base error for Mieza bridge failures."""


class MiezaSignalValidationError(MiezaBridgeError):
    """Raised when the signal envelope or signal fields are invalid."""


class MiezaSignatureError(MiezaBridgeError):
    """Raised when envelope signature verification fails."""


class MiezaReplayError(MiezaBridgeError):
    """Raised when an already consumed envelope nonce is seen again."""


class MiezaNonceStore(Protocol):
    """Persistent nonce store used to reject replay across process restarts."""

    def reserve_nonce(self, nonce: str, generated_at: datetime, source: str) -> bool:
        """Reserve nonce and return False when it already exists."""


@dataclass(frozen=True)
class MiezaSignal:
    """Validated Mieza prediction-market signal."""

    market_id: str
    platform: str
    question: str
    signal_type: str
    side: str
    edge: float
    confidence: float
    recommended_size: int
    market_price: float
    estimated_fair_price: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MiezaSignal":
        _validate_exact_fields(payload, SIGNAL_FIELDS, "signal")

        platform = _require_choice(payload, "platform", PLATFORMS)
        signal_type = _require_choice(payload, "signal_type", SIGNAL_TYPES)
        side = _require_choice(payload, "side", SIDES)
        edge = _require_number(payload, "edge")
        if edge < 0:
            raise MiezaSignalValidationError("signal.edge must be >= 0")

        return cls(
            market_id=_require_str(payload, "market_id"),
            platform=platform,
            question=_require_str(payload, "question"),
            signal_type=signal_type,
            side=side,
            edge=edge,
            confidence=_require_probability(payload, "confidence"),
            recommended_size=_require_positive_int(payload, "recommended_size"),
            market_price=_require_probability(payload, "market_price"),
            estimated_fair_price=_require_probability(payload, "estimated_fair_price"),
        )

    def to_alpha_event(self) -> Dict[str, Any]:
        """Convert to a neutral internal alpha event, not an execution order."""
        return {
            "source": SOURCE,
            "asset_class": "prediction_market",
            "platform": self.platform,
            "market_id": self.market_id,
            "question": self.question,
            "signal_type": self.signal_type,
            "side": self.side,
            "direction": 1 if self.side == "yes" else -1,
            "edge": self.edge,
            "confidence": self.confidence,
            "recommended_size": self.recommended_size,
            "market_price": self.market_price,
            "estimated_fair_price": self.estimated_fair_price,
        }


@dataclass(frozen=True)
class MiezaSignalEnvelope:
    """Validated signed signal batch from Mieza Quant."""

    schema_version: str
    source: str
    generated_at: datetime
    nonce: str
    signals: List[MiezaSignal]
    signature: str

    def to_alpha_events(self) -> List[Dict[str, Any]]:
        generated_at = self.generated_at.isoformat().replace("+00:00", "Z")
        events = []
        for signal in self.signals:
            event = signal.to_alpha_event()
            event["generated_at"] = generated_at
            event["nonce"] = self.nonce
            events.append(event)
        return events


class MiezaSignalBridge:
    """Verifier and parser for signed Mieza Quant signal envelopes."""

    def __init__(
        self,
        signing_key: Optional[str] = None,
        max_age_seconds: int = 300,
        max_future_skew_seconds: int = 30,
        seen_nonces: Optional[Set[str]] = None,
        nonce_store: Optional[MiezaNonceStore] = None,
    ):
        self.signing_key = signing_key if signing_key is not None else os.getenv(SIGNING_KEY_ENV)
        if not self.signing_key:
            raise MiezaSignalValidationError(f"{SIGNING_KEY_ENV} is required")

        self.max_age_seconds = max_age_seconds
        self.max_future_skew_seconds = max_future_skew_seconds
        self.seen_nonces = seen_nonces if seen_nonces is not None else set()
        self.nonce_store = nonce_store

    def load_file(self, path: str | Path, now: Optional[datetime] = None) -> MiezaSignalEnvelope:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.validate_envelope(payload, now=now)

    def validate_envelope(
        self,
        payload: Mapping[str, Any],
        now: Optional[datetime] = None,
    ) -> MiezaSignalEnvelope:
        if not isinstance(payload, Mapping):
            raise MiezaSignalValidationError("envelope must be a JSON object")

        _validate_exact_fields(payload, ENVELOPE_FIELDS, "envelope")
        self._verify_signature(payload)

        schema_version = _require_str(payload, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise MiezaSignalValidationError(f"unsupported schema_version: {schema_version}")

        source = _require_str(payload, "source")
        if source != SOURCE:
            raise MiezaSignalValidationError(f"unsupported source: {source}")

        generated_at = _require_datetime(payload, "generated_at")
        nonce = _require_str(payload, "nonce")
        if not nonce:
            raise MiezaSignalValidationError("envelope.nonce must not be empty")

        signals_payload = payload.get("signals")
        if not isinstance(signals_payload, list):
            raise MiezaSignalValidationError("envelope.signals must be a list")

        envelope = MiezaSignalEnvelope(
            schema_version=schema_version,
            source=source,
            generated_at=generated_at,
            nonce=nonce,
            signals=[MiezaSignal.from_mapping(signal) for signal in signals_payload],
            signature=_require_str(payload, "signature"),
        )
        self._validate_timestamp(envelope.generated_at, now)

        if self.nonce_store is not None:
            if not self.nonce_store.reserve_nonce(envelope.nonce, envelope.generated_at, envelope.source):
                raise MiezaReplayError(f"duplicate envelope nonce: {envelope.nonce}")
        else:
            if envelope.nonce in self.seen_nonces:
                raise MiezaReplayError(f"duplicate envelope nonce: {envelope.nonce}")
            self.seen_nonces.add(envelope.nonce)

        return envelope

    def _verify_signature(self, payload: Mapping[str, Any]) -> None:
        supplied = _require_str(payload, "signature")
        expected = sign_mieza_envelope(payload, self.signing_key)
        if not hmac.compare_digest(supplied, expected):
            raise MiezaSignatureError("invalid Mieza signal envelope signature")

    def _validate_timestamp(self, generated_at: datetime, now: Optional[datetime]) -> None:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age_seconds = (now_utc - generated_at).total_seconds()

        if age_seconds > self.max_age_seconds:
            raise MiezaSignalValidationError("stale Mieza signal envelope")
        if age_seconds < -self.max_future_skew_seconds:
            raise MiezaSignalValidationError("Mieza signal envelope timestamp is too far in the future")


def sign_mieza_envelope(envelope: Mapping[str, Any], signing_key: str) -> str:
    """Return the HMAC-SHA256 hex signature for an envelope payload."""
    if not signing_key:
        raise MiezaSignalValidationError("signing_key is required")

    payload = {key: value for key, value in envelope.items() if key != "signature"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(
        signing_key.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def alpha_events_from_envelope(
    payload: Mapping[str, Any],
    signing_key: Optional[str] = None,
    seen_nonces: Optional[Set[str]] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Validate an envelope and return neutral alpha events."""
    bridge = MiezaSignalBridge(signing_key=signing_key, seen_nonces=seen_nonces)
    return bridge.validate_envelope(payload, now=now).to_alpha_events()


def _validate_exact_fields(payload: Mapping[str, Any], expected_fields: Iterable[str], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise MiezaSignalValidationError(f"{label} must be a JSON object")

    expected = set(expected_fields)
    actual = set(payload.keys())
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise MiezaSignalValidationError(f"{label} missing required fields: {sorted(missing)}")
    if unexpected:
        raise MiezaSignalValidationError(f"{label} has unexpected fields: {sorted(unexpected)}")


def _require_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise MiezaSignalValidationError(f"{field} must be a string")
    return value


def _require_choice(payload: Mapping[str, Any], field: str, choices: Set[str]) -> str:
    value = _require_str(payload, field)
    if value not in choices:
        raise MiezaSignalValidationError(f"{field} must be one of {sorted(choices)}")
    return value


def _require_number(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MiezaSignalValidationError(f"{field} must be a number")
    if not math.isfinite(float(value)):
        raise MiezaSignalValidationError(f"{field} must be finite")
    return float(value)


def _require_probability(payload: Mapping[str, Any], field: str) -> float:
    value = _require_number(payload, field)
    if value < 0 or value > 1:
        raise MiezaSignalValidationError(f"{field} must be between 0 and 1")
    return value


def _require_positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MiezaSignalValidationError(f"{field} must be an integer")
    if value <= 0:
        raise MiezaSignalValidationError(f"{field} must be > 0")
    return value


def _require_datetime(payload: Mapping[str, Any], field: str) -> datetime:
    value = _require_str(payload, field)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise MiezaSignalValidationError(f"{field} must be an ISO-8601 datetime") from exc

    if parsed.tzinfo is None:
        raise MiezaSignalValidationError(f"{field} must include timezone")
    return parsed.astimezone(timezone.utc)
