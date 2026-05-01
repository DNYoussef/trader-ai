"""Opt-in live Alpaca paper trade smoke test."""

import asyncio
import os
from decimal import Decimal

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.live_api, pytest.mark.e2e]


def _live_alpaca_enabled() -> bool:
    return os.getenv("RUN_LIVE_ALPACA_TESTS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.skipif(
    not _live_alpaca_enabled(),
    reason="Set RUN_LIVE_ALPACA_TESTS=true to submit a live Alpaca paper trade.",
)
def test_spy_five_dollar_paper_trade() -> None:
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        pytest.skip("ALPACA_API_KEY and ALPACA_SECRET_KEY are required.")

    from src.trading_engine import TradingEngine

    engine = TradingEngine()
    assert engine.initialize()

    result = asyncio.run(engine.execute_manual_trade("SPY", Decimal("5.00"), "buy"))

    assert result.get("success"), result.get("error")
    assert result.get("order_id")
    assert result.get("symbol") == "SPY"
    assert result.get("side", "").lower() == "buy"
