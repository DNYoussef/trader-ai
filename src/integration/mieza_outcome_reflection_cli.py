"""CLI for recording resolved prediction-market outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..intelligence.prediction_markets.outcome_reflection import (
    CompositePredictionMarketResolutionProvider,
    FilePredictionMarketResolutionProvider,
    KalshiMarketResolutionProvider,
    PolymarketGammaResolutionProvider,
    PredictionMarketOutcomeReflectionJob,
    PredictionMarketResolutionError,
)
from .mieza_signal_store import MiezaSQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record prediction-market outcomes from resolution metadata"
    )
    parser.add_argument(
        "--db",
        default="data/mieza_signals.db",
        help="SQLite database path with committee, execution, and outcome records",
    )
    parser.add_argument(
        "--resolutions",
        help="Path to JSON resolution metadata",
    )
    parser.add_argument(
        "--resolution-source",
        choices=["file", "venue"],
        default="file",
        help="Resolution source: JSON file or read-only venue APIs",
    )
    parser.add_argument(
        "--venue-platforms",
        default="kalshi,polymarket",
        help="Comma-separated venue providers to enable for --resolution-source venue",
    )
    parser.add_argument(
        "--kalshi-base-url",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public market API base URL",
    )
    parser.add_argument(
        "--polymarket-gamma-base-url",
        default="https://gamma-api.polymarket.com",
        help="Polymarket Gamma API base URL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = MiezaSQLiteStore(Path(args.db))
    try:
        provider = _build_resolution_provider(args)
        result = PredictionMarketOutcomeReflectionJob(store, provider).run()
    except (OSError, ValueError, json.JSONDecodeError, PredictionMarketResolutionError) as exc:
        print(json.dumps({"status": "rejected", "errors": [str(exc)]}, sort_keys=True))
        return 1
    finally:
        store.close()

    output = {
        "status": "completed" if not result.errors else "completed_with_errors",
        "scanned_executions": result.scanned_executions,
        "recorded_outcomes": result.recorded_outcomes,
        "skipped_count": result.skipped_count,
        "outcome_ids": [outcome.outcome_id for outcome in result.outcomes],
        "errors": result.errors,
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if not result.errors else 2


def _build_resolution_provider(args: argparse.Namespace):
    if args.resolution_source == "file":
        if not args.resolutions:
            raise ValueError("--resolutions is required when --resolution-source=file")
        return FilePredictionMarketResolutionProvider(args.resolutions)

    platforms = {
        value.strip().lower()
        for value in str(args.venue_platforms or "").split(",")
        if value.strip()
    }
    providers = []
    if "kalshi" in platforms:
        providers.append(KalshiMarketResolutionProvider(base_url=args.kalshi_base_url))
    if "polymarket" in platforms:
        providers.append(
            PolymarketGammaResolutionProvider(base_url=args.polymarket_gamma_base_url)
        )
    if not providers:
        raise ValueError("--venue-platforms must include kalshi and/or polymarket")
    return CompositePredictionMarketResolutionProvider(providers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
