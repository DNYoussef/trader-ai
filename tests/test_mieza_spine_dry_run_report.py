import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mieza_spine_dry_run_report.py"


def _load_report_script():
    spec = importlib.util.spec_from_file_location("mieza_spine_dry_run_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.security
def test_dry_run_report_exercises_mieza_spine(tmp_path, monkeypatch):
    monkeypatch.delenv("MIEZA_SIGNAL_SIGNING_KEY", raising=False)
    module = _load_report_script()
    out = tmp_path / "daily-report.json"

    code = module.main(["--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert report["status"] == "ok"
    assert report["assertions"]["accepted_one_file"] is True
    assert report["assertions"]["one_dry_run_execution"] is True
    assert report["assertions"]["spine_status_ok"] is True
    assert report["spine"]["db"]["counts"]["alpha_events"] == 1
