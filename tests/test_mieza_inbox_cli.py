import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.integration.mieza_inbox_cli import main
from src.integration.mieza_quant_bridge import sign_mieza_envelope
from src.integration.mieza_signal_store import MiezaSQLiteStore


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mieza_signal_envelope.json"
SIGNING_KEY = "unit-test-mieza-signing-key"
FIXTURE_MAX_AGE_SECONDS = "2000000"


def _run_inbox(inbox: Path, db: Path, capsys) -> tuple[int, dict]:
    code = main(
        [
            "--inbox",
            str(inbox),
            "--db",
            str(db),
            "--max-age-seconds",
            FIXTURE_MAX_AGE_SECONDS,
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def _current_signed_payload(nonce: str | None = None) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if nonce is not None:
        payload["nonce"] = nonce
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)
    return payload


def _copy_fixture(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _current_signed_payload()), encoding="utf-8")


def _write_tampered(path: Path) -> None:
    payload = _current_signed_payload()
    payload["signals"][0]["confidence"] = 0.99
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_valid_with_nonce(path: Path, nonce: str) -> None:
    payload = _current_signed_payload(nonce)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _envelopes(path: Path) -> list[Path]:
    return [
        item for item in path.glob("*.json")
        if not item.name.endswith(".manifest.json")
        and not item.name.endswith(".error.json")
    ]


@pytest.mark.integration
@pytest.mark.security
def test_inbox_ingests_valid_file_and_archives_processed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "ready.json")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 0
    assert summary["accepted"] == 1
    assert summary["duplicates"] == 0
    assert summary["rejected"] == 0
    assert summary["executed"] == 1
    assert not (inbox / "ready.json").exists()
    processed = _envelopes(inbox / "processed")
    assert len(processed) == 1
    assert Path(str(processed[0]) + ".manifest.json").exists()

    store = MiezaSQLiteStore(db)
    try:
        assert len(store.list_alpha_events()) == 1
        assert store.list_execution_results()[0]["status"] == "dry_run_accepted"
    finally:
        store.close()


@pytest.mark.integration
@pytest.mark.security
def test_inbox_rejects_invalid_signature_with_error_sidecar(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _write_tampered(inbox / "bad.json")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 2
    assert summary["accepted"] == 0
    assert summary["rejected"] == 1
    rejected = _envelopes(inbox / "rejected")
    assert len(rejected) == 1
    error = json.loads(Path(str(rejected[0]) + ".error.json").read_text(encoding="utf-8"))
    assert "invalid Mieza signal envelope signature" in error["errors"][0]


@pytest.mark.integration
@pytest.mark.security
def test_inbox_mixed_batch_processes_valid_and_invalid_independently(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "a-valid.json")
    _write_tampered(inbox / "b-invalid.json")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["executed"] == 1
    assert len(_envelopes(inbox / "processed")) == 1
    assert len(_envelopes(inbox / "rejected")) == 1


@pytest.mark.integration
@pytest.mark.security
def test_inbox_duplicate_replay_archives_duplicate_without_new_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    payload = _current_signed_payload()
    _copy_fixture(inbox / "first.json", payload)
    assert _run_inbox(inbox, db, capsys)[0] == 0

    _copy_fixture(inbox / "second.json", payload)
    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 0
    assert summary["accepted"] == 0
    assert summary["duplicates"] == 1
    assert summary["rejected"] == 0
    assert summary["executed"] == 0
    assert len(_envelopes(inbox / "processed" / "duplicates")) == 1

    store = MiezaSQLiteStore(db)
    try:
        assert len(store.list_execution_results()) == 1
    finally:
        store.close()


@pytest.mark.integration
@pytest.mark.security
def test_inbox_nonce_collision_with_new_content_is_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "first.json")
    assert _run_inbox(inbox, db, capsys)[0] == 0

    payload = _current_signed_payload()
    payload["signals"][0]["edge"] = 0.09
    payload["signals"][0]["estimated_fair_price"] = 0.51
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)
    (inbox / "collision.json").write_text(json.dumps(payload), encoding="utf-8")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 2
    assert summary["duplicates"] == 0
    assert summary["rejected"] == 1
    assert len(_envelopes(inbox / "rejected")) == 1


@pytest.mark.integration
@pytest.mark.security
def test_inbox_ignores_temp_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "partial.json.tmp")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 0
    assert summary["files"] == 0
    assert (inbox / "partial.json.tmp").exists()


@pytest.mark.integration
@pytest.mark.security
def test_inbox_recovers_existing_processing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _write_valid_with_nonce(inbox / "processing" / "crashed.json", "processing-recovery")

    code, summary = _run_inbox(inbox, db, capsys)

    assert code == 0
    assert summary["accepted"] == 1
    assert not (inbox / "processing" / "crashed.json").exists()
    assert len(_envelopes(inbox / "processed")) == 1
