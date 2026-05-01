"""Durable storage for Mieza Quant ingestion state."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class MiezaSQLiteStore:
    """SQLite-backed nonce, alpha-event, execution, and audit store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def reserve_nonce(self, nonce: str, generated_at: datetime, source: str) -> bool:
        """Reserve a nonce. Returns False if it was already reserved."""
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO mieza_signal_nonces
                        (nonce, generated_at, source, accepted_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        nonce,
                        _datetime_to_text(generated_at),
                        source,
                        _datetime_to_text(_now()),
                    ),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                if not self._nonce_has_side_effects(nonce):
                    return True
                return False

    def has_nonce(self, nonce: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM mieza_signal_nonces WHERE nonce = ?",
                (nonce,),
            ).fetchone()
            return row is not None

    def record_alpha_events(self, events: Iterable[Mapping[str, Any]]) -> List[int]:
        row_ids: List[int] = []
        with self._lock:
            for event in events:
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO mieza_alpha_events
                        (nonce, event_key, platform, market_id, side, signal_type,
                         edge, confidence, market_price, estimated_fair_price,
                         generated_at, payload_json, accepted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["nonce"],
                        alpha_event_key(event),
                        event["platform"],
                        event["market_id"],
                        event["side"],
                        event["signal_type"],
                        float(event["edge"]),
                        float(event["confidence"]),
                        float(event["market_price"]),
                        float(event["estimated_fair_price"]),
                        event["generated_at"],
                        _canonical_json(event),
                        _datetime_to_text(_now()),
                    ),
                )
                if cursor.rowcount:
                    row_ids.append(int(cursor.lastrowid))
            self._conn.commit()
        return row_ids

    def reserve_execution_order(self, order: Any) -> bool:
        """Reserve an execution idempotency key before any live venue call."""
        payload = asdict(order) if is_dataclass(order) else dict(order)
        now = _datetime_to_text(_now())
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO mieza_execution_results
                        (order_id, idempotency_key, platform, market_id, side,
                         status, dry_run, venue_order_id, venue_status,
                         reconciliation_status, error, raw_response_json,
                         payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "intent_" + str(payload["idempotency_key"])[:16],
                        payload["idempotency_key"],
                        payload["platform"],
                        payload["market_id"],
                        payload["side"],
                        "intent_reserved",
                        1 if payload.get("dry_run") else 0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        _canonical_json(payload),
                        now,
                        now,
                    ),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def record_execution_results(self, results: Iterable[Mapping[str, Any]]) -> List[int]:
        row_ids: List[int] = []
        with self._lock:
            for result in results:
                now = _datetime_to_text(_now())
                cursor = self._conn.execute(
                    """
                    INSERT INTO mieza_execution_results
                        (order_id, idempotency_key, platform, market_id, side,
                         status, dry_run, venue_order_id, venue_status,
                         reconciliation_status, error, raw_response_json,
                         payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO UPDATE SET
                        order_id = excluded.order_id,
                        platform = excluded.platform,
                        market_id = excluded.market_id,
                        side = excluded.side,
                        status = excluded.status,
                        dry_run = excluded.dry_run,
                        venue_order_id = excluded.venue_order_id,
                        venue_status = excluded.venue_status,
                        reconciliation_status = excluded.reconciliation_status,
                        error = excluded.error,
                        raw_response_json = excluded.raw_response_json,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        result["order_id"],
                        result["idempotency_key"],
                        result["platform"],
                        result["market_id"],
                        result["side"],
                        result["status"],
                        1 if result.get("dry_run") else 0,
                        result.get("venue_order_id"),
                        result.get("venue_status"),
                        result.get("reconciliation_status"),
                        result.get("error"),
                        _canonical_json(result["raw_response"]) if result.get("raw_response") else None,
                        _canonical_json(result),
                        now,
                        now,
                    ),
                )
                row_ids.append(int(cursor.lastrowid))
            self._conn.commit()
        return row_ids

    def record_committee_decision(self, decision: Any) -> int:
        """Persist one prediction-market committee decision."""
        payload = _object_to_payload(decision)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO mieza_committee_decisions
                    (decision_id, nonce, event_key, platform, market_id, side,
                     approved, rating, payload_hash, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["decision_id"],
                    payload["nonce"],
                    payload["event_key"],
                    payload["platform"],
                    payload["market_id"],
                    payload["side"],
                    1 if payload.get("approved") else 0,
                    payload["rating"],
                    payload["payload_hash"],
                    _canonical_json(payload),
                    payload["created_at"],
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def record_committee_decisions(self, decisions: Iterable[Any]) -> List[int]:
        row_ids: List[int] = []
        with self._lock:
            for decision in decisions:
                payload = _object_to_payload(decision)
                cursor = self._conn.execute(
                    """
                    INSERT INTO mieza_committee_decisions
                        (decision_id, nonce, event_key, platform, market_id, side,
                         approved, rating, payload_hash, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["decision_id"],
                        payload["nonce"],
                        payload["event_key"],
                        payload["platform"],
                        payload["market_id"],
                        payload["side"],
                        1 if payload.get("approved") else 0,
                        payload["rating"],
                        payload["payload_hash"],
                        _canonical_json(payload),
                        payload["created_at"],
                    ),
                )
                row_ids.append(int(cursor.lastrowid))
            self._conn.commit()
        return row_ids

    def record_moo_decision(self, decision: Any) -> int:
        payload = _object_to_payload(decision)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO mieza_moo_decisions
                    (decision_id, decision_type, nonce, event_key,
                     committee_decision_id, platform, market_id, side,
                     approved, optimizer_source, inputs_hash, payload_hash,
                     payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["decision_id"],
                    payload["decision_type"],
                    payload["nonce"],
                    payload["event_key"],
                    payload.get("committee_decision_id"),
                    payload["platform"],
                    payload["market_id"],
                    payload["side"],
                    1 if payload.get("approved") else 0,
                    payload["optimizer_source"],
                    payload["inputs_hash"],
                    payload["payload_hash"],
                    _canonical_json(payload),
                    payload["created_at"],
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def record_moo_decisions(self, decisions: Iterable[Any]) -> List[int]:
        row_ids: List[int] = []
        with self._lock:
            for decision in decisions:
                payload = _object_to_payload(decision)
                cursor = self._conn.execute(
                    """
                    INSERT INTO mieza_moo_decisions
                        (decision_id, decision_type, nonce, event_key,
                         committee_decision_id, platform, market_id, side,
                         approved, optimizer_source, inputs_hash, payload_hash,
                         payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["decision_id"],
                        payload["decision_type"],
                        payload["nonce"],
                        payload["event_key"],
                        payload.get("committee_decision_id"),
                        payload["platform"],
                        payload["market_id"],
                        payload["side"],
                        1 if payload.get("approved") else 0,
                        payload["optimizer_source"],
                        payload["inputs_hash"],
                        payload["payload_hash"],
                        _canonical_json(payload),
                        payload["created_at"],
                    ),
                )
                row_ids.append(int(cursor.lastrowid))
            self._conn.commit()
        return row_ids

    def record_market_outcome(self, outcome: Any) -> int:
        payload = _object_to_payload(outcome)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO mieza_market_outcomes
                    (outcome_id, decision_id, venue_order_id, platform, market_id,
                     resolved_outcome, entry_price, exit_or_resolution_value,
                     contracts, fees, slippage, pnl, thesis_accuracy, reflection_json,
                     resolved_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(outcome_id) DO UPDATE SET
                    venue_order_id = excluded.venue_order_id,
                    resolved_outcome = excluded.resolved_outcome,
                    entry_price = excluded.entry_price,
                    exit_or_resolution_value = excluded.exit_or_resolution_value,
                    contracts = excluded.contracts,
                    fees = excluded.fees,
                    slippage = excluded.slippage,
                    pnl = excluded.pnl,
                    thesis_accuracy = excluded.thesis_accuracy,
                    reflection_json = excluded.reflection_json,
                    resolved_at = excluded.resolved_at
                """,
                (
                    payload["outcome_id"],
                    payload["decision_id"],
                    payload.get("venue_order_id"),
                    payload["platform"],
                    payload["market_id"],
                    payload["resolved_outcome"],
                    float(payload["entry_price"]),
                    float(payload["exit_or_resolution_value"]),
                    int(payload["contracts"]),
                    float(payload.get("fees", 0.0)),
                    float(payload.get("slippage", 0.0)),
                    float(payload["pnl"]),
                    payload["thesis_accuracy"],
                    _canonical_json(payload.get("reflection", {})),
                    payload["resolved_at"],
                    payload["created_at"],
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def record_audit(
        self,
        status: str,
        reason: str,
        envelope_ref: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO mieza_ingest_audit
                    (status, reason, envelope_ref, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    status,
                    reason,
                    envelope_ref,
                    _canonical_json(payload) if payload is not None else None,
                    _datetime_to_text(_now()),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_alpha_events(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_alpha_events ORDER BY id")

    def list_execution_results(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_execution_results ORDER BY id")

    def list_committee_decisions(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_committee_decisions ORDER BY id")

    def list_moo_decisions(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_moo_decisions ORDER BY id")

    def list_market_outcomes(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_market_outcomes ORDER BY id")

    def list_pending_execution_intents(self) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM mieza_execution_results
            WHERE status IN ('intent_reserved', 'live_reconcile_failed')
            ORDER BY id
            """
        )

    def list_audits(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM mieza_ingest_audit ORDER BY id")

    def purge_nonces_older_than(self, cutoff: datetime) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM mieza_signal_nonces
                WHERE generated_at < ?
                  AND nonce NOT IN (SELECT DISTINCT nonce FROM mieza_alpha_events)
                """,
                (_datetime_to_text(cutoff),),
            )
            self._conn.commit()
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _fetch_all(self, query: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def _initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mieza_signal_nonces (
                    nonce TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    accepted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mieza_alpha_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nonce TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    edge REAL NOT NULL,
                    confidence REAL NOT NULL,
                    market_price REAL NOT NULL,
                    estimated_fair_price REAL NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    accepted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mieza_execution_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    venue_order_id TEXT,
                    venue_status TEXT,
                    reconciliation_status TEXT,
                    error TEXT,
                    raw_response_json TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS mieza_ingest_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    envelope_ref TEXT,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mieza_committee_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    rating TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mieza_market_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    outcome_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    venue_order_id TEXT,
                    platform TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    resolved_outcome TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_or_resolution_value REAL NOT NULL,
                    contracts INTEGER NOT NULL,
                    fees REAL NOT NULL,
                    slippage REAL NOT NULL,
                    pnl REAL NOT NULL,
                    thesis_accuracy TEXT NOT NULL,
                    reflection_json TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mieza_moo_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    committee_decision_id TEXT,
                    platform TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    optimizer_source TEXT NOT NULL,
                    inputs_hash TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_mieza_execution_idempotency
                    ON mieza_execution_results (idempotency_key);
                """
            )
            self._ensure_column("mieza_execution_results", "venue_order_id", "TEXT")
            self._ensure_column("mieza_execution_results", "venue_status", "TEXT")
            self._ensure_column("mieza_execution_results", "reconciliation_status", "TEXT")
            self._ensure_column("mieza_execution_results", "error", "TEXT")
            self._ensure_column("mieza_execution_results", "raw_response_json", "TEXT")
            self._ensure_column("mieza_execution_results", "updated_at", "TEXT")
            self._ensure_column("mieza_market_outcomes", "contracts", "INTEGER")
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mieza_alpha_event_key
                    ON mieza_alpha_events (event_key)
                """
            )
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mieza_committee_decision_id
                    ON mieza_committee_decisions (decision_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mieza_committee_event_key
                    ON mieza_committee_decisions (event_key)
                """
            )
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mieza_market_outcome_id
                    ON mieza_market_outcomes (outcome_id)
                """
            )
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mieza_moo_decision_id
                    ON mieza_moo_decisions (decision_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mieza_moo_event_key
                    ON mieza_moo_decisions (event_key)
                """
            )
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        existing = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _nonce_has_side_effects(self, nonce: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM mieza_alpha_events WHERE nonce = ? LIMIT 1",
            (nonce,),
        ).fetchone()
        return row is not None


def alpha_event_key(event: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(event["nonce"]),
            str(event["platform"]),
            str(event["market_id"]),
            str(event["side"]),
            str(event["signal_type"]),
        ]
    )


def _object_to_payload(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(timezone.utc)
