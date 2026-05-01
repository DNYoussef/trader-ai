import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.integration.mieza_quant_bridge import (
    MiezaReplayError,
    MiezaSignalBridge,
    MiezaSignalValidationError,
    MiezaSignatureError,
    alpha_events_from_envelope,
    sign_mieza_envelope,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mieza_signal_envelope.json"
SIGNING_KEY = "unit-test-mieza-signing-key"
BASE_TIME = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


def _iso(timestamp: datetime) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _signed_fixture(nonce: str = "nonce-1", generated_at: datetime = BASE_TIME) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["generated_at"] = _iso(generated_at)
    payload["nonce"] = nonce
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)
    return payload


@pytest.mark.unit
@pytest.mark.security
def test_contract_fixture_signature_is_valid():
    envelope = MiezaSignalBridge(signing_key=SIGNING_KEY).load_file(FIXTURE_PATH, now=BASE_TIME)

    assert envelope.nonce == "contract-fixture-nonce"
    assert envelope.signature == "e575ea11ee76c13b333f3dba6c33c97133803c56009f4b70c31e45c962c80394"


@pytest.mark.unit
@pytest.mark.security
def test_valid_signed_mieza_envelope_maps_to_alpha_events():
    payload = _signed_fixture()
    envelope = MiezaSignalBridge(signing_key=SIGNING_KEY).validate_envelope(
        payload,
        now=BASE_TIME + timedelta(seconds=15),
    )

    assert envelope.schema_version == "mieza.signal.v1"
    assert envelope.source == "mieza-quant"
    assert len(envelope.signals) == 1

    events = envelope.to_alpha_events()
    assert events == [
        {
            "source": "mieza-quant",
            "asset_class": "prediction_market",
            "platform": "polymarket",
            "market_id": "polymarket-election-001",
            "question": "Will the event resolve yes?",
            "signal_type": "equilibrium",
            "side": "yes",
            "direction": 1,
            "edge": 0.08,
            "confidence": 0.72,
            "recommended_size": 3,
            "market_price": 0.42,
            "estimated_fair_price": 0.5,
            "generated_at": "2026-04-30T12:00:00Z",
            "nonce": "nonce-1",
        }
    ]


@pytest.mark.unit
@pytest.mark.security
def test_alpha_events_helper_accepts_seen_nonce_cache():
    payload = _signed_fixture(nonce="helper-nonce")
    seen_nonces = set()

    events = alpha_events_from_envelope(
        payload,
        signing_key=SIGNING_KEY,
        seen_nonces=seen_nonces,
        now=BASE_TIME,
    )

    assert events[0]["asset_class"] == "prediction_market"
    assert seen_nonces == {"helper-nonce"}


@pytest.mark.unit
@pytest.mark.security
def test_tampered_envelope_is_rejected():
    payload = _signed_fixture(nonce="tamper-nonce")
    tampered = copy.deepcopy(payload)
    tampered["signals"][0]["edge"] = 0.2

    with pytest.raises(MiezaSignatureError, match="invalid"):
        MiezaSignalBridge(signing_key=SIGNING_KEY).validate_envelope(tampered, now=BASE_TIME)


@pytest.mark.unit
@pytest.mark.security
def test_invalid_probability_is_rejected_even_with_valid_signature():
    payload = _signed_fixture(nonce="bad-probability")
    payload["signals"][0]["estimated_fair_price"] = 1.4
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)

    with pytest.raises(MiezaSignalValidationError, match="estimated_fair_price"):
        MiezaSignalBridge(signing_key=SIGNING_KEY).validate_envelope(payload, now=BASE_TIME)


@pytest.mark.unit
@pytest.mark.security
def test_stale_envelope_is_rejected():
    payload = _signed_fixture(
        nonce="stale-nonce",
        generated_at=BASE_TIME - timedelta(seconds=301),
    )

    with pytest.raises(MiezaSignalValidationError, match="stale"):
        MiezaSignalBridge(signing_key=SIGNING_KEY, max_age_seconds=300).validate_envelope(
            payload,
            now=BASE_TIME,
        )


@pytest.mark.unit
@pytest.mark.security
def test_nonce_replay_is_rejected():
    payload = _signed_fixture(nonce="replay-nonce")
    bridge = MiezaSignalBridge(signing_key=SIGNING_KEY)

    bridge.validate_envelope(payload, now=BASE_TIME)
    with pytest.raises(MiezaReplayError, match="duplicate"):
        bridge.validate_envelope(payload, now=BASE_TIME)


@pytest.mark.unit
@pytest.mark.security
def test_signing_key_is_required(monkeypatch):
    monkeypatch.delenv("MIEZA_SIGNAL_SIGNING_KEY", raising=False)

    with pytest.raises(MiezaSignalValidationError, match="MIEZA_SIGNAL_SIGNING_KEY"):
        MiezaSignalBridge()
