"""Read-only health check and explicit recovery for the Mieza signal spine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .mieza_inbox_cli import DEFAULT_DB, DEFAULT_INBOX


ARCHIVE_DIRS = ("processed", "duplicates", "rejected")
DB_TABLES = {
    "nonces": "mieza_signal_nonces",
    "alpha_events": "mieza_alpha_events",
    "committee_decisions": "mieza_committee_decisions",
    "moo_decisions": "mieza_moo_decisions",
    "execution_results": "mieza_execution_results",
    "audits": "mieza_ingest_audit",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the Mieza Quant signal spine")
    parser.add_argument("--inbox", default=DEFAULT_INBOX, help="Mieza Quant inbox directory")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path for ingestion state")
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Processing files older than this are considered stale",
    )
    parser.add_argument(
        "--recover-stale-processing",
        action="store_true",
        help="Move eligible stale processing files back to the ready inbox",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_spine(
        Path(args.inbox),
        Path(args.db),
        stale_minutes=args.stale_minutes,
        recover_stale_processing=args.recover_stale_processing,
    )
    print(json.dumps(summary, sort_keys=True))
    return 2 if summary["status"] == "blocked" else 0


def inspect_spine(
    inbox: Path,
    db_path: Path,
    *,
    stale_minutes: float = 15.0,
    recover_stale_processing: bool = False,
) -> Dict[str, Any]:
    recovery: Dict[str, Any] = {"enabled": recover_stale_processing, "moved": [], "refused": []}

    if recover_stale_processing:
        recovery = _recover_stale_files(inbox, stale_minutes)

    inbox_summary, inbox_problems = _inspect_inbox(inbox, stale_minutes)
    db_summary, db_problems = _inspect_db(db_path)
    problems = inbox_problems + db_problems

    if problems:
        status = "blocked"
    elif inbox_summary["rejected"] or inbox_summary["duplicates"]:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "generated_at": _utc_now(),
        "inbox": inbox_summary,
        "db": db_summary,
        "problems": problems,
        "recovery": recovery,
    }


def _inspect_inbox(inbox: Path, stale_minutes: float) -> tuple[Dict[str, Any], List[str]]:
    layout = _layout(inbox)
    problems: List[str] = []

    summary: Dict[str, Any] = {
        "path": str(inbox),
        "ready": 0,
        "processing": 0,
        "stale_processing": 0,
        "stale_processing_files": [],
        "processed": 0,
        "duplicates": 0,
        "rejected": 0,
        "manifest_errors": [],
    }

    if not inbox.exists():
        problems.append(f"inbox root missing: {inbox}")
        return summary, problems
    if not inbox.is_dir():
        problems.append(f"inbox path is not a directory: {inbox}")
        return summary, problems

    summary["ready"] = len(_envelope_files(layout["inbox"]))
    processing_files = _envelope_files(layout["processing"])
    summary["processing"] = len(processing_files)

    stale_files = _stale_files(processing_files, stale_minutes)
    summary["stale_processing"] = len(stale_files)
    summary["stale_processing_files"] = [_file_record(path) for path in stale_files]
    if stale_files:
        problems.append(f"{len(stale_files)} stale processing file(s)")

    summary["processed"] = len(_envelope_files(layout["processed"]))
    summary["duplicates"] = len(_envelope_files(layout["duplicates"]))
    summary["rejected"] = len(_envelope_files(layout["rejected"]))

    manifest_errors = _validate_archives(layout)
    summary["manifest_errors"] = manifest_errors
    problems.extend(manifest_errors)

    return summary, problems


def _inspect_db(db_path: Path) -> tuple[Dict[str, Any], List[str]]:
    summary: Dict[str, Any] = {
        "path": str(db_path),
        "reachable": False,
        "counts": {key: 0 for key in DB_TABLES},
        "latest_audit": None,
        "latest_execution": None,
    }
    problems: List[str] = []

    if not db_path.exists():
        return summary, [f"db missing: {db_path}"]

    try:
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            summary["counts"] = {
                key: _count_rows(conn, table)
                for key, table in DB_TABLES.items()
            }
            summary["latest_audit"] = _fetch_one_dict(
                conn,
                """
                SELECT id, status, reason, envelope_ref, created_at
                FROM mieza_ingest_audit
                ORDER BY id DESC
                LIMIT 1
                """,
            )
            summary["latest_execution"] = _fetch_one_dict(
                conn,
                """
                SELECT id, order_id, status, dry_run, venue_order_id,
                       reconciliation_status, error, created_at, updated_at
                FROM mieza_execution_results
                ORDER BY id DESC
                LIMIT 1
                """,
            )
            summary["reachable"] = True
        finally:
            conn.close()
    except Exception as exc:
        problems.append(f"db unreadable: {db_path}: {exc}")

    return summary, problems


def _recover_stale_files(inbox: Path, stale_minutes: float) -> Dict[str, Any]:
    layout = _layout(inbox)
    recovery: Dict[str, Any] = {"enabled": True, "moved": [], "refused": []}

    if not inbox.exists() or not inbox.is_dir():
        recovery["refused"].append({"path": str(inbox), "reason": "inbox root missing or not a directory"})
        return recovery

    for path in _stale_files(_envelope_files(layout["processing"]), stale_minutes):
        digest = _sha256_file(path)
        refusal = _recovery_refusal(path, digest, layout)
        if refusal is not None:
            recovery["refused"].append(refusal)
            continue

        destination = _unique_path(layout["inbox"] / f"recovered-{path.stem}.{digest[:12]}.json")
        os.replace(path, destination)
        recovery["moved"].append({
            "from": str(path),
            "to": str(destination),
            "sha256": digest,
        })

    return recovery


def _recovery_refusal(path: Path, digest: str, layout: Dict[str, Path]) -> Optional[Dict[str, str]]:
    archive = _archive_for_sha(layout, digest)
    if archive is not None:
        return {
            "path": str(path),
            "reason": "archive already exists for sha256",
            "sha256": digest,
            "archive": str(archive),
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "reason": f"invalid json: {exc}", "sha256": digest}

    if not isinstance(payload, dict):
        return {"path": str(path), "reason": "envelope is not a json object", "sha256": digest}

    required = {"schema_version", "source", "generated_at", "nonce", "signals", "signature"}
    missing = sorted(required - set(payload))
    if missing:
        return {
            "path": str(path),
            "reason": "missing envelope keys: " + ",".join(missing),
            "sha256": digest,
        }

    return None


def _validate_archives(layout: Dict[str, Path]) -> List[str]:
    errors: List[str] = []
    for archive_name in ARCHIVE_DIRS:
        for path in _envelope_files(layout[archive_name]):
            digest = _sha256_file(path)
            manifest_path = Path(str(path) + ".manifest.json")
            manifest = _read_json_sidecar(manifest_path, errors, "manifest")
            if isinstance(manifest, dict):
                if manifest.get("sha256") != digest:
                    errors.append(f"manifest sha256 mismatch: {manifest_path}")
                if manifest.get("final_path") and Path(str(manifest["final_path"])).name != path.name:
                    errors.append(f"manifest final_path mismatch: {manifest_path}")

            if archive_name == "rejected":
                error_path = Path(str(path) + ".error.json")
                error_sidecar = _read_json_sidecar(error_path, errors, "error")
                if isinstance(error_sidecar, dict) and error_sidecar.get("sha256") != digest:
                    errors.append(f"error sidecar sha256 mismatch: {error_path}")
    return errors


def _read_json_sidecar(path: Path, errors: List[str], label: str) -> Optional[Any]:
    if not path.exists():
        errors.append(f"missing {label} sidecar: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid {label} sidecar: {path}: {exc}")
        return None


def _layout(inbox: Path) -> Dict[str, Path]:
    return {
        "inbox": inbox,
        "processing": inbox / "processing",
        "processed": inbox / "processed",
        "duplicates": inbox / "processed" / "duplicates",
        "rejected": inbox / "rejected",
    }


def _envelope_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item for item in path.glob("*.json")
        if item.is_file()
        and not item.name.endswith(".manifest.json")
        and not item.name.endswith(".error.json")
        and not item.name.endswith(".tmp")
        and ".tmp-" not in item.name
    )


def _stale_files(files: Iterable[Path], stale_minutes: float) -> List[Path]:
    cutoff = time.time() - (stale_minutes * 60.0)
    return [path for path in files if path.stat().st_mtime < cutoff]


def _archive_for_sha(layout: Dict[str, Path], digest: str) -> Optional[Path]:
    for name in ARCHIVE_DIRS:
        root = layout[name]
        if not root.exists():
            continue
        for candidate in root.rglob(f"*.{digest}.json*"):
            return candidate
    return None


def _file_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def _fetch_one_dict(conn: sqlite3.Connection, query: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(query).fetchone()
    return dict(row) if row is not None else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}.{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique recovery path for {path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
