import json
from pathlib import Path

import pytest

from src.integration.mieza_signal_cli import main
from src.integration.mieza_signal_store import MiezaSQLiteStore


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mieza_signal_envelope.json"
SIGNING_KEY = "unit-test-mieza-signing-key"


@pytest.mark.integration
@pytest.mark.security
def test_cli_ingests_contract_fixture_as_dry_run(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "mieza.db"
    monkeypatch.setenv("MIEZA_SIGNAL_SIGNING_KEY", SIGNING_KEY)

    code = main(
        [
            str(FIXTURE_PATH),
            "--db",
            str(db_path),
            "--max-age-seconds",
            "2000000",
            "--execution-mode",
            "dry_run",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "accepted_count": 1,
        "errors": [],
        "execution_count": 1,
        "rejected_count": 0,
        "status": "accepted",
    }

    store = MiezaSQLiteStore(db_path)
    try:
        [execution] = store.list_execution_results()
        assert execution["status"] == "dry_run_accepted"
        assert execution["dry_run"] == 1
        assert store.has_nonce("contract-fixture-nonce")
    finally:
        store.close()
