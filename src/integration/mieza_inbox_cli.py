"""Atomic inbox processor for signed Mieza Quant envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .mieza_quant_bridge import MiezaSignalBridge
from .mieza_signal_ingestion import MiezaIngestionResult, MiezaSignalIngestionService
from .mieza_signal_store import MiezaSQLiteStore


DEFAULT_INBOX = "data/inbox/mieza-quant"
DEFAULT_DB = "data/mieza_signals.db"


@dataclass
class InboxFileResult:
    source_path: str
    claimed_path: str
    final_path: str
    sha256: str
    status: str
    accepted_count: int = 0
    rejected_count: int = 0
    execution_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "claimed_path": self.claimed_path,
            "final_path": self.final_path,
            "sha256": self.sha256,
            "status": self.status,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "execution_count": self.execution_count,
            "errors": self.errors,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process signed Mieza Quant inbox envelopes")
    parser.add_argument("--inbox", default=DEFAULT_INBOX, help="Inbox directory containing ready *.json envelopes")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path for ingestion state")
    parser.add_argument("--max-age-seconds", type=int, default=300, help="Maximum accepted envelope age")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    inbox = Path(args.inbox)
    layout = _ensure_layout(inbox)
    store = MiezaSQLiteStore(Path(args.db))
    results: List[InboxFileResult] = []
    operational_errors: List[str] = []

    try:
        bridge = MiezaSignalBridge(nonce_store=store, max_age_seconds=args.max_age_seconds)
        service = MiezaSignalIngestionService(bridge=bridge, store=store)

        for source in _ready_files(layout):
            try:
                claimed, original_path, digest = _claim_file(source, layout)
                result = _process_claimed_file(claimed, original_path, digest, layout, service)
            except Exception as exc:
                operational_errors.append(f"{source}: {exc}")
                continue
            results.append(result)
    finally:
        store.close()

    summary = _summary(results, operational_errors)
    print(json.dumps(summary, sort_keys=True))

    if operational_errors:
        return 1
    if summary["rejected"] > 0:
        return 2
    return 0


def _ensure_layout(inbox: Path) -> Dict[str, Path]:
    paths = {
        "inbox": inbox,
        "processing": inbox / "processing",
        "processed": inbox / "processed",
        "duplicates": inbox / "processed" / "duplicates",
        "rejected": inbox / "rejected",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _ready_files(layout: Dict[str, Path]) -> List[Path]:
    processing = sorted(layout["processing"].glob("*.json"))
    inbox = sorted(
        path for path in layout["inbox"].glob("*.json")
        if path.is_file() and ".tmp-" not in path.name and not path.name.endswith(".tmp")
    )
    return processing + inbox


def _claim_file(source: Path, layout: Dict[str, Path]) -> tuple[Path, Path, str]:
    digest = _sha256_file(source)
    if source.parent == layout["processing"]:
        return source, source, digest

    claimed = _unique_path(layout["processing"] / f"{source.stem}.{digest[:12]}.json")
    os.replace(source, claimed)
    return claimed, source, digest


def _process_claimed_file(
    claimed: Path,
    original_path: Path,
    digest: str,
    layout: Dict[str, Path],
    service: MiezaSignalIngestionService,
) -> InboxFileResult:
    try:
        ingestion = service.ingest_file(claimed, execution_mode="dry_run")
        status = _transport_status(ingestion, digest, layout)
        errors = list(ingestion.errors)
    except Exception as exc:
        ingestion = None
        status = "rejected"
        errors = [str(exc)]

    final_dir = {
        "accepted": layout["processed"],
        "duplicate": layout["duplicates"],
        "rejected": layout["rejected"],
    }[status]
    final_path = _archive_file(claimed, final_dir, digest)
    result = InboxFileResult(
        source_path=str(original_path),
        claimed_path=str(claimed),
        final_path=str(final_path),
        sha256=digest,
        status=status,
        accepted_count=ingestion.accepted_count if ingestion is not None else 0,
        rejected_count=ingestion.rejected_count if ingestion is not None else 1,
        execution_count=len(ingestion.execution_results) if ingestion is not None else 0,
        errors=errors,
    )
    _write_manifest(result)
    if status == "rejected":
        _write_error_sidecar(final_path, result)
    return result


def _transport_status(
    ingestion: MiezaIngestionResult,
    digest: str,
    layout: Dict[str, Path],
) -> str:
    if ingestion.status in {"accepted", "validated", "partially_accepted"}:
        return "accepted"
    if _is_replay(ingestion) and _sha_seen(layout, digest):
        return "duplicate"
    return "rejected"


def _is_replay(ingestion: MiezaIngestionResult) -> bool:
    text = "\n".join(ingestion.errors)
    return ingestion.status in {"rejected", "duplicate_rejected"} and (
        "duplicate envelope nonce" in text
        or "duplicate alpha event already persisted" in text
    )


def _sha_seen(layout: Dict[str, Path], digest: str) -> bool:
    for root in (layout["processed"], layout["duplicates"]):
        if any(root.glob(f"*.{digest}.json")):
            return True
    return False


def _archive_file(path: Path, destination_dir: Path, digest: str) -> Path:
    destination = _unique_path(destination_dir / f"{path.stem}.{digest}.json")
    os.replace(path, destination)
    return destination


def _write_manifest(result: InboxFileResult) -> None:
    payload = result.to_dict()
    payload["created_at"] = _utc_now()
    _atomic_write_json(Path(str(result.final_path) + ".manifest.json"), payload)


def _write_error_sidecar(path: Path, result: InboxFileResult) -> None:
    _atomic_write_json(
        Path(str(path) + ".error.json"),
        {
            "status": result.status,
            "errors": result.errors,
            "sha256": result.sha256,
            "source_path": result.source_path,
            "final_path": result.final_path,
            "created_at": _utc_now(),
        },
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _summary(results: List[InboxFileResult], operational_errors: List[str]) -> Dict[str, Any]:
    return {
        "accepted": sum(1 for result in results if result.status == "accepted"),
        "duplicates": sum(1 for result in results if result.status == "duplicate"),
        "rejected": sum(1 for result in results if result.status == "rejected"),
        "executed": sum(result.execution_count for result in results),
        "files": len(results),
        "errors": operational_errors + [
            error for result in results for error in result.errors if result.status == "rejected"
        ],
        "results": [result.to_dict() for result in results],
    }


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
    raise RuntimeError(f"could not find unique archive path for {path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
