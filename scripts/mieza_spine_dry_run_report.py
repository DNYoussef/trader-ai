"""Generate a dry-run health report for the Mieza signal spine."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integration.mieza_inbox_cli import main as inbox_main
from src.integration.mieza_quant_bridge import (
    SCHEMA_VERSION,
    SOURCE,
    SIGNING_KEY_ENV,
    sign_mieza_envelope,
)
from src.integration.mieza_spine_status_cli import inspect_spine


DEFAULT_OUT = "reports/mieza-spine/daily-report.json"
DEFAULT_SIGNING_KEY = "ci-dry-run-mieza-signing-key"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic Mieza dry-run spine report")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Report JSON destination")
    parser.add_argument("--work-dir", help="Optional work directory for inbox/db artifacts")
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=300,
        help="Maximum age accepted by the inbox bridge",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    signing_key = os.getenv(SIGNING_KEY_ENV, DEFAULT_SIGNING_KEY)

    report: Dict[str, Any]
    with _work_dir(args.work_dir) as work_dir:
        inbox = work_dir / "inbox"
        db_path = work_dir / "mieza-signals.db"
        envelope_path = inbox / "ci-dry-run.json"
        envelope = _signed_envelope(signing_key)
        _atomic_write_json(envelope_path, envelope)

        previous_key = os.environ.get(SIGNING_KEY_ENV)
        os.environ[SIGNING_KEY_ENV] = signing_key
        try:
            inbox_code, ingestion = _run_inbox(inbox, db_path, args.max_age_seconds)
            spine = inspect_spine(inbox, db_path, stale_minutes=15.0)
        finally:
            if previous_key is None:
                os.environ.pop(SIGNING_KEY_ENV, None)
            else:
                os.environ[SIGNING_KEY_ENV] = previous_key

        checks = _checks(inbox_code, ingestion, spine)
        status = "ok" if all(checks.values()) else "failed"
        report = {
            "status": status,
            "generated_at": _utc_now(),
            "mode": "dry_run",
            "assertions": checks,
            "ingestion": ingestion,
            "spine": spine,
            "artifacts": {
                "inbox": str(inbox),
                "db": str(db_path),
                "envelope": str(envelope_path),
            },
        }

    _atomic_write_json(Path(args.out), report)
    return 0 if report["status"] == "ok" else 1


def _signed_envelope(signing_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "generated_at": _utc_now(),
        "nonce": f"ci-dry-run-{uuid.uuid4()}",
        "signals": [
            {
                "market_id": "ci-dry-run-market-001",
                "platform": "polymarket",
                "question": "CI dry-run signal should remain dry-run only",
                "signal_type": "equilibrium",
                "side": "yes",
                "edge": 0.08,
                "confidence": 0.72,
                "recommended_size": 3,
                "market_price": 0.42,
                "estimated_fair_price": 0.50,
            }
        ],
    }
    payload["signature"] = sign_mieza_envelope(payload, signing_key)
    return payload


def _run_inbox(inbox: Path, db_path: Path, max_age_seconds: int) -> tuple[int, Dict[str, Any]]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = inbox_main(
            [
                "--inbox",
                str(inbox),
                "--db",
                str(db_path),
                "--max-age-seconds",
                str(max_age_seconds),
            ]
        )
    return code, json.loads(stdout.getvalue())


def _checks(inbox_code: int, ingestion: Dict[str, Any], spine: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "inbox_exit_zero": inbox_code == 0,
        "accepted_one_file": ingestion.get("accepted") == 1,
        "no_duplicates": ingestion.get("duplicates") == 0,
        "no_rejections": ingestion.get("rejected") == 0,
        "one_dry_run_execution": ingestion.get("executed") == 1,
        "spine_status_ok": spine.get("status") == "ok",
        "db_reachable": spine.get("db", {}).get("reachable") is True,
        "alpha_event_recorded": spine.get("db", {}).get("counts", {}).get("alpha_events") == 1,
        "execution_recorded": spine.get("db", {}).get("counts", {}).get("execution_results") == 1,
    }


@contextlib.contextmanager
def _work_dir(path: Optional[str]):
    if path:
        work_dir = Path(path)
        work_dir.mkdir(parents=True, exist_ok=True)
        yield work_dir
        return

    with tempfile.TemporaryDirectory(prefix="mieza-spine-report-") as temp_dir:
        yield Path(temp_dir)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
