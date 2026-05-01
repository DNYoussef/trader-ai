"""Typed prediction-market committee decision contracts.

These schemas are deliberately execution-free. They describe research evidence,
debate, proposal, and approval state that can be audited before the deterministic
risk gate and executor are allowed to see an event.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "prediction_market_committee.v1"


class CommitteeRating(str, Enum):
    STRONG_APPROVE = "strong_approve"
    APPROVE = "approve"
    HOLD = "hold"
    REJECT = "reject"


class TraderAction(str, Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    HOLD = "hold"
    REJECT = "reject"


class AnalystReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    thesis: str = Field(min_length=1)
    evidence: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    data_sources: List[str] = Field(default_factory=list)


class DebateState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bull_case: str = Field(min_length=1)
    bear_case: str = Field(min_length=1)
    unresolved_questions: List[str] = Field(default_factory=list)
    vote_summary: Dict[str, str] = Field(default_factory=dict)


class TraderProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: TraderAction
    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    side: Optional[str] = None
    max_price: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    size_hint: Optional[int] = Field(default=None, ge=0)
    thesis: str = Field(min_length=1)

    @field_validator("side")
    @classmethod
    def _validate_side(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in {"yes", "no"}:
            raise ValueError("side must be yes or no")
        return value


class RiskReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    violations: List[str] = Field(default_factory=list)
    liquidity_notes: str = ""
    exposure_notes: str = ""
    kill_switch_notes: str = ""

    @field_validator("violations")
    @classmethod
    def _reject_requires_violation(cls, value: List[str], info: Any) -> List[str]:
        approved = info.data.get("approved")
        if approved is False and not value:
            raise ValueError("rejected risk reviews must include at least one violation")
        return value


class PortfolioDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    decision_id: str = Field(min_length=16)
    nonce: str = Field(min_length=1)
    event_key: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    market_id: str = Field(min_length=1)
    side: str = Field(min_length=1)
    approved: bool
    rating: CommitteeRating
    final_reason: str = Field(min_length=1)
    required_gates: List[str] = Field(default_factory=list)
    analyst_reports: List[AnalystReport] = Field(default_factory=list)
    debate_state: DebateState
    trader_proposal: TraderProposal
    risk_review: RiskReview
    created_at: str
    payload_hash: str = Field(min_length=64, max_length=64)
    signature: Optional[str] = Field(default=None, min_length=64, max_length=64)

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

    def canonical_payload(self, include_signature: bool = False) -> Dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_signature:
            payload.pop("signature", None)
        return payload

    def canonical_json(self, include_signature: bool = False) -> str:
        return canonical_json(self.canonical_payload(include_signature=include_signature))

    @model_validator(mode="after")
    def _validate_integrity(self) -> "PortfolioDecision":
        payload = self.model_dump(mode="json")
        expected_hash = committee_payload_hash(payload)
        if self.payload_hash != expected_hash:
            raise ValueError("payload_hash does not match committee decision payload")

        expected_decision_id = committee_decision_id(payload)
        if self.decision_id != expected_decision_id:
            raise ValueError("decision_id does not match committee decision payload")
        return self


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def committee_payload_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("decision_id", None)
    clean.pop("payload_hash", None)
    clean.pop("signature", None)
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def committee_decision_id(payload: Mapping[str, Any]) -> str:
    seed = "|".join(
        [
            str(payload["schema_version"]),
            str(payload["nonce"]),
            str(payload["event_key"]),
            str(payload["platform"]),
            str(payload["market_id"]),
            str(payload["side"]),
            committee_payload_hash(payload),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def sign_committee_decision(decision: PortfolioDecision, signing_key: str) -> str:
    if not signing_key:
        raise ValueError("committee signing_key is required")
    return hmac.new(
        signing_key.encode("utf-8"),
        decision.canonical_json(include_signature=False).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
