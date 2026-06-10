import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.integration.mieza_inbox_cli import main as inbox_main
from src.integration.mieza_quant_bridge import sign_mieza_envelope
from src.integration.mieza_signal_store import MiezaSQLiteStore
from src.integration.mieza_spine_status_cli import main as status_main


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mieza_signal_envelope.json"
SIGNING_KEY = "unit-test-mieza-signing-key"


def _init_db(path: Path) -> None:
    store = MiezaSQLiteStore(path)
    store.close()


def _run_status(inbox: Path, db: Path, capsys, *extra: str) -> tuple[int, dict]:
    code = status_main(["--inbox", str(inbox), "--db", str(db), *extra])
    return code, json.loads(capsys.readouterr().out)


def _run_inbox(inbox: Path, db: Path, capsys) -> tuple[int, dict]:
    code = inbox_main(
        [
            "--inbox",
            str(inbox),
            "--db",
            str(db),
            "--max-age-seconds",
            "2000000",
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def _copy_fixture(path: Path) -> None:
    payload = _current_signed_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _current_signed_payload() -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["signature"] = sign_mieza_envelope(payload, SIGNING_KEY)
    return payload


def _write_tampered(path: Path) -> None:
    payload = _current_signed_payload()
    payload["signals"][0]["confidence"] = 0.99
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _age(path: Path, seconds: int) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _envelopes(path: Path) -> list[Path]:
    return [
        item for item in path.glob("*.json")
        if item.is_file()
        and not item.name.endswith(".manifest.json")
        and not item.name.endswith(".error.json")
    ]


@pytest.mark.integration
@pytest.mark.security
def test_status_empty_initialized_spine_is_ok(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    inbox.mkdir()
    _init_db(db)

    code, summary = _run_status(inbox, db, capsys)

    assert code == 0
    assert summary["status"] == "ok"
    assert summary["inbox"]["ready"] == 0
    assert summary["db"]["reachable"] is True
    assert summary["db"]["counts"]["alpha_events"] == 0


@pytest.mark.integration
@pytest.mark.security
def test_status_counts_ready_files_without_mutating(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    inbox.mkdir()
    _init_db(db)
    ready = inbox / "ready.json"
    _copy_fixture(ready)

    code, summary = _run_status(inbox, db, capsys)

    assert code == 0
    assert summary["status"] == "ok"
    assert summary["inbox"]["ready"] == 1
    assert ready.exists()


@pytest.mark.integration
@pytest.mark.security
def test_status_fresh_processing_file_is_not_stale(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _init_db(db)
    _copy_fixture(inbox / "processing" / "fresh.json")

    code, summary = _run_status(inbox, db, capsys, "--stale-minutes", "15")

    assert code == 0
    assert summary["status"] == "ok"
    assert summary["inbox"]["processing"] == 1
    assert summary["inbox"]["stale_processing"] == 0


@pytest.mark.integration
@pytest.mark.security
def test_status_old_processing_file_blocks_spine(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _init_db(db)
    stalled = inbox / "processing" / "stalled.json"
    _copy_fixture(stalled)
    _age(stalled, 3600)

    code, summary = _run_status(inbox, db, capsys, "--stale-minutes", "1")

    assert code == 2
    assert summary["status"] == "blocked"
    assert summary["inbox"]["stale_processing"] == 1
    assert "stale processing" in summary["problems"][0]


@pytest.mark.integration
@pytest.mark.security
def test_recovery_moves_eligible_stale_processing_file_back_to_inbox(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _init_db(db)
    stalled = inbox / "processing" / "stalled.json"
    _copy_fixture(stalled)
    _age(stalled, 3600)

    code, summary = _run_status(
        inbox,
        db,
        capsys,
        "--stale-minutes",
        "1",
        "--recover-stale-processing",
    )

    assert code == 0
    assert summary["status"] == "ok"
    assert len(summary["recovery"]["moved"]) == 1
    assert summary["recovery"]["refused"] == []
    assert not stalled.exists()
    assert len(_envelopes(inbox)) == 1


@pytest.mark.integration
@pytest.mark.security
def test_recovery_refuses_when_archive_for_same_sha_exists(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _init_db(db)
    stalled = inbox / "processing" / "stalled.json"
    _copy_fixture(stalled)
    digest = _sha256(stalled)
    archived = inbox / "processed" / f"archived.{digest}.json"
    _copy_fixture(archived)
    Path(str(archived) + ".manifest.json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "final_path": str(archived),
                "status": "accepted",
                "created_at": "2026-05-11T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    _age(stalled, 3600)

    code, summary = _run_status(
        inbox,
        db,
        capsys,
        "--stale-minutes",
        "1",
        "--recover-stale-processing",
    )

    assert code == 2
    assert summary["status"] == "blocked"
    assert summary["recovery"]["moved"] == []
    assert summary["recovery"]["refused"][0]["reason"] == "archive already exists for sha256"
    assert stalled.exists()


@pytest.mark.integration
@pytest.mark.security
def test_status_rejected_archive_is_degraded(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _write_tampered(inbox / "bad.json")
    assert _run_inbox(inbox, db, capsys)[0] == 2

    code, summary = _run_status(inbox, db, capsys)

    assert code == 0
    assert summary["status"] == "degraded"
    assert summary["inbox"]["rejected"] == 1


@pytest.mark.integration
@pytest.mark.security
def test_status_corrupt_manifest_blocks_spine(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "ready.json")
    assert _run_inbox(inbox, db, capsys)[0] == 0
    processed = _envelopes(inbox / "processed")[0]
    Path(str(processed) + ".manifest.json").write_text("{broken", encoding="utf-8")

    code, summary = _run_status(inbox, db, capsys)

    assert code == 2
    assert summary["status"] == "blocked"
    assert "invalid manifest sidecar" in summary["problems"][0]


@pytest.mark.integration
@pytest.mark.security
def test_status_missing_db_blocks_without_creating_db(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    db = tmp_path / "missing.db"
    inbox.mkdir()

    code, summary = _run_status(inbox, db, capsys)

    assert code == 2
    assert summary["status"] == "blocked"
    assert summary["db"]["reachable"] is False
    assert "db missing" in summary["problems"][0]
    assert not db.exists()


@pytest.mark.integration
@pytest.mark.security
def test_status_reports_db_counts_after_dry_run_ingest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)
    inbox = tmp_path / "inbox"
    db = tmp_path / "mieza.db"
    _copy_fixture(inbox / "ready.json")
    assert _run_inbox(inbox, db, capsys)[0] == 0

    code, summary = _run_status(inbox, db, capsys)

    assert code == 0
    assert summary["status"] == "ok"
    assert summary["db"]["counts"]["alpha_events"] == 1
    assert summary["db"]["counts"]["execution_results"] == 1
    assert summary["db"]["latest_execution"]["status"] == "dry_run_accepted"
