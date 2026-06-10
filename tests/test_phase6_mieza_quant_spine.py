import copy
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.integration.mieza_quant_bridge import (
    MiezaSignalBridge,
    MiezaSignalValidationError,
    MiezaSignatureError,
    sign_mieza_envelope,
)
from src.integration.mieza_signal_ingestion import MiezaSignalIngestionService
from src.integration.mieza_signal_store import MiezaSQLiteStore
from src.intelligence.prediction_markets.moo_allocation import MiezaMOOAllocationService
from src.risk.prediction_market_risk import (
    PredictionMarketExposure,
    PredictionMarketRiskConfig,
    PredictionMarketRiskGate,
)
from src.trading.prediction_market_executor import DryRunPredictionMarketExecutor


SIGNING_KEY = "phase6-mieza-spine-signing-key"
GENERATED_AT_TEXT = "2026-06-05T12:00:00Z"
GENERATED_AT = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
NOW = GENERATED_AT + timedelta(seconds=30)
TRADER_AI_ROOT = Path(__file__).resolve().parents[1]
MIEZA_ENGINE_ROOT = TRADER_AI_ROOT.parent / "mieza-quant" / "engine"


@pytest.fixture(scope="module")
def producer_payload(tmp_path_factory):
    if shutil.which("clojure") is None:
        pytest.skip("clojure CLI is required for the cross-repo Mieza producer test")
    if not MIEZA_ENGINE_ROOT.exists():
        pytest.skip(f"Mieza Quant engine repo is not available at {MIEZA_ENGINE_ROOT}")

    tmp_path = tmp_path_factory.mktemp("mieza-quant-export")
    input_path = tmp_path / "signals.edn"
    output_path = tmp_path / "envelope.json"
    input_path.write_text(
        """
        [{:market_id "p6-polymarket-001"
          :platform :polymarket
          :question "Will the P6 Mieza Trader AI spine accept a real exported signal?"
          :signal_type :equilibrium
          :side :yes
          :edge 0.08
          :confidence 0.72
          :recommended_size 3
          :market_price 0.42
          :estimated_fair_price 0.50}]
        """.strip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["MIEZA_SIGNAL_SIGNING_KEY"] = SIGNING_KEY
    result = subprocess.run(
        [
            "clojure",
            "-M:export-signals",
            str(input_path),
            str(output_path),
            "--nonce",
            "phase6-e2e-nonce",
            "--generated-at",
            GENERATED_AT_TEXT,
        ],
        cwd=MIEZA_ENGINE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "mieza.signal.v1"
    assert payload["source"] == "mieza-quant"
    assert payload["generated_at"] == GENERATED_AT_TEXT
    assert payload["nonce"] == "phase6-e2e-nonce"
    assert payload["signature"] == sign_mieza_envelope(payload, SIGNING_KEY)
    return payload


@pytest.mark.integration
@pytest.mark.security
def test_phase6_mieza_quant_export_ingests_through_trader_ai_dry_run_without_live_order(
    tmp_path,
    producer_payload,
):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    risk_gate = RecordingRiskGate()
    executor = RecordingDryRunExecutor()
    service = MiezaSignalIngestionService(
        bridge=MiezaSignalBridge(
            signing_key=SIGNING_KEY,
            nonce_store=store,
            max_age_seconds=600,
        ),
        store=store,
        risk_gate=risk_gate,
        executor=executor,
    )

    result = service.ingest_payload(producer_payload, now=NOW, execution_mode="dry_run")

    assert result.status == "accepted"
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert len(risk_gate.events_seen) == 1
    assert risk_gate.events_seen[0]["market_id"] == "p6-polymarket-001"
    assert risk_gate.events_seen[0]["source"] == "mieza-quant"
    assert len(executor.orders) == 1
    assert executor.orders[0].dry_run is True
    assert result.execution_results[0]["status"] == "dry_run_accepted"
    assert result.execution_results[0]["dry_run"] is True
    assert result.execution_results[0]["venue_order_id"] is None
    [execution_row] = store.list_execution_results()
    assert execution_row["status"] == "dry_run_accepted"
    assert execution_row["dry_run"] == 1
    assert execution_row["venue_order_id"] is None
    [audit_row] = store.list_audits()
    assert audit_row["status"] == "accepted"
    assert store.has_nonce("phase6-e2e-nonce")


@pytest.mark.integration
@pytest.mark.security
def test_phase6_mieza_quant_envelope_rejects_signature_key_schema_and_staleness(
    monkeypatch,
    producer_payload,
):
    tampered = copy.deepcopy(producer_payload)
    tampered["signals"][0]["edge"] = 0.12
    with pytest.raises(MiezaSignatureError, match="invalid"):
        MiezaSignalBridge(signing_key=SIGNING_KEY).validate_envelope(tampered, now=NOW)

    monkeypatch.delenv("MIEZA_SIGNAL_SIGNING_KEY", raising=False)
    with pytest.raises(MiezaSignalValidationError, match="MIEZA_SIGNAL_SIGNING_KEY"):
        MiezaSignalBridge()

    malformed = copy.deepcopy(producer_payload)
    malformed["signals"][0]["side"] = "buy"
    malformed["signature"] = sign_mieza_envelope(malformed, SIGNING_KEY)
    with pytest.raises(MiezaSignalValidationError, match="side"):
        MiezaSignalBridge(signing_key=SIGNING_KEY).validate_envelope(malformed, now=NOW)

    with pytest.raises(MiezaSignalValidationError, match="stale"):
        MiezaSignalBridge(signing_key=SIGNING_KEY, max_age_seconds=5).validate_envelope(
            producer_payload,
            now=GENERATED_AT + timedelta(seconds=10),
        )


@pytest.mark.integration
@pytest.mark.security
def test_phase6_over_limit_prediction_market_exposure_reaches_risk_gate_and_blocks_execution(
    tmp_path,
    producer_payload,
):
    store = MiezaSQLiteStore(tmp_path / "mieza.db")
    risk_gate = RecordingRiskGate(
        PredictionMarketRiskConfig(
            max_market_exposure=1.0,
            max_dollars_per_signal=1000.0,
        )
    )
    loose_allocator = MiezaMOOAllocationService(
        PredictionMarketRiskConfig(
            max_market_exposure=1000.0,
            max_dollars_per_signal=1000.0,
        )
    )
    service = MiezaSignalIngestionService(
        bridge=MiezaSignalBridge(
            signing_key=SIGNING_KEY,
            nonce_store=store,
            max_age_seconds=600,
        ),
        store=store,
        risk_gate=risk_gate,
        moo_allocator=loose_allocator,
        executor=RecordingDryRunExecutor(),
    )

    result = service.ingest_payload(
        producer_payload,
        now=NOW,
        execution_mode="dry_run",
        exposure=PredictionMarketExposure(
            market_exposure={"p6-polymarket-001": 0.9},
        ),
    )

    assert len(risk_gate.events_seen) == 1
    assert risk_gate.events_seen[0]["recommended_size"] == 3
    assert result.status == "risk_rejected"
    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert result.execution_results == []
    assert "market exposure limit exceeded" in result.errors
    assert store.list_execution_results() == []
    assert store.list_audits()[-1]["status"] == "risk_rejected"


class RecordingRiskGate(PredictionMarketRiskGate):
    def __init__(self, config=None):
        super().__init__(config)
        self.events_seen = []

    def validate_batch(self, events, exposure=None):
        events_list = list(events)
        self.events_seen.extend(copy.deepcopy(events_list))
        return super().validate_batch(events_list, exposure=exposure)


class RecordingDryRunExecutor:
    def __init__(self):
        self.orders = []
        self._inner = DryRunPredictionMarketExecutor()

    def execute(self, order):
        self.orders.append(order)
        assert order.dry_run is True
        return self._inner.execute(order)
