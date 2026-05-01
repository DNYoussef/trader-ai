"""CLI for ingesting signed Mieza Quant signal files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .mieza_quant_bridge import MiezaBridgeError, MiezaSignalBridge
from .mieza_signal_ingestion import EXECUTION_MODES, MiezaSignalIngestionService
from .mieza_signal_store import MiezaSQLiteStore
from ..trading.prediction_market_executor import (
    KalshiVenueClient,
    LivePredictionMarketExecutor,
    PredictionMarketExecutionError,
)


LIVE_ENABLE_ENV = "TRADER_AI_ENABLE_LIVE_PREDICTION_MARKETS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a signed Mieza Quant signal envelope")
    parser.add_argument("envelope", help="Path to signed Mieza JSON envelope")
    parser.add_argument(
        "--db",
        default="data/mieza_signals.db",
        help="SQLite database path for nonce, audit, alpha, and execution records",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=300,
        help="Maximum accepted envelope age in seconds",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Alias for --execution-mode validate_only",
    )
    parser.add_argument(
        "--execution-mode",
        choices=sorted(EXECUTION_MODES),
        default=None,
        help="Execution mode: validate_only, dry_run, or live",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = MiezaSQLiteStore(Path(args.db))

    try:
        bridge = MiezaSignalBridge(
            nonce_store=store,
            max_age_seconds=args.max_age_seconds,
        )
        execution_mode = args.execution_mode
        if execution_mode is None and args.no_execute:
            execution_mode = "validate_only"
        executor = _build_executor(execution_mode, store)
        service = MiezaSignalIngestionService(
            bridge=bridge,
            store=store,
            executor=executor,
        )
        result = service.ingest_file(args.envelope, execution_mode=execution_mode)
    except (MiezaBridgeError, PredictionMarketExecutionError, ValueError) as exc:
        output = {"status": "rejected", "errors": [str(exc)]}
        print(json.dumps(output, sort_keys=True))
        return 1
    finally:
        store.close()

    output = {
        "status": result.status,
        "accepted_count": result.accepted_count,
        "rejected_count": result.rejected_count,
        "execution_count": len(result.execution_results),
        "errors": result.errors,
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if result.status in {"accepted", "partially_accepted"} else 2


def _build_executor(execution_mode: str | None, store: MiezaSQLiteStore):
    if execution_mode != "live":
        return None

    if not _truthy(os.getenv(LIVE_ENABLE_ENV)):
        raise PredictionMarketExecutionError(
            f"live mode requires {LIVE_ENABLE_ENV}=true"
        )

    clients = {}
    kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID")
    kalshi_private_key = _read_secret_or_file(
        value_env="KALSHI_PRIVATE_KEY_PEM",
        path_env="KALSHI_PRIVATE_KEY_PATH",
    )
    if kalshi_api_key_id and kalshi_private_key:
        clients["kalshi"] = KalshiVenueClient(
            api_key_id=kalshi_api_key_id,
            private_key_pem=kalshi_private_key,
            base_url=os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2"),
        )

    if not clients:
        raise PredictionMarketExecutionError(
            "live mode requires at least one configured venue client; "
            "set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PEM or KALSHI_PRIVATE_KEY_PATH"
        )

    return LivePredictionMarketExecutor(
        clients,
        enable_live=True,
        journal=store,
    )


def _read_secret_or_file(*, value_env: str, path_env: str) -> str | bytes | None:
    value = os.getenv(value_env)
    if value:
        return value
    path = os.getenv(path_env)
    if not path:
        return None
    return Path(path).read_bytes()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
