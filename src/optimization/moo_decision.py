"""Durable scored-decision contract for multi-objective optimization."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "moo_scored_decision.v1"


class MOOScoredDecision(BaseModel):
    """Auditable MOO output.

    This is a recommendation record, not an execution permission. Hard gates
    remain authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    decision_id: str = Field(min_length=16)
    decision_type: str = Field(min_length=1)
    nonce: str = Field(min_length=1)
    event_key: str = Field(min_length=1)
    committee_decision_id: Optional[str] = None
    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    side: str = Field(min_length=1)
    approved: bool
    optimizer_source: str = Field(min_length=1)
    fallback_reason: Optional[str] = None
    inputs_hash: str = Field(min_length=64, max_length=64)
    objectives: Dict[str, float] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    candidate_set: List[Dict[str, Any]] = Field(default_factory=list)
    selected_candidate: Dict[str, Any]
    scores: Dict[str, float] = Field(default_factory=dict)
    final_reason: str = Field(min_length=1)
    created_at: str
    payload_hash: str = Field(min_length=64, max_length=64)

    @field_validator("side")
    @classmethod
    def _validate_side(cls, value: str) -> str:
        if value not in {"yes", "no"}:
            raise ValueError("side must be yes or no")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            raise ValueError("created_at must include timezone")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def _validate_integrity(self) -> "MOOScoredDecision":
        payload = self.model_dump(mode="json")
        expected_hash = moo_payload_hash(payload)
        if self.payload_hash != expected_hash:
            raise ValueError("payload_hash does not match MOO decision payload")

        expected_decision_id = moo_decision_id(payload)
        if self.decision_id != expected_decision_id:
            raise ValueError("decision_id does not match MOO decision payload")
        return self


def build_moo_decision(payload: Mapping[str, Any]) -> MOOScoredDecision:
    decision_payload = dict(payload)
    decision_payload["payload_hash"] = moo_payload_hash(decision_payload)
    decision_payload["decision_id"] = moo_decision_id(decision_payload)
    return MOOScoredDecision(**decision_payload)


def moo_inputs_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def moo_payload_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("decision_id", None)
    clean.pop("payload_hash", None)
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def moo_identity_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("decision_id", None)
    clean.pop("payload_hash", None)
    clean.pop("created_at", None)
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def moo_decision_id(payload: Mapping[str, Any]) -> str:
    seed = "|".join(
        [
            str(payload["schema_version"]),
            str(payload["decision_type"]),
            str(payload["nonce"]),
            str(payload["event_key"]),
            str(payload.get("committee_decision_id") or ""),
            str(payload["platform"]),
            str(payload["market_id"]),
            str(payload["side"]),
            moo_identity_hash(payload),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
