"""Opt-in live Alpaca connection test."""

import os

import pytest
from alpaca.trading.client import TradingClient


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
    reason="Set RUN_LIVE_ALPACA_TESTS=true to run live Alpaca paper API tests.",
)
def test_alpaca_paper_connection() -> None:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        pytest.skip("ALPACA_API_KEY and ALPACA_SECRET_KEY are required.")

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    account = client.get_account()

    assert account.account_number
    assert float(account.equity) >= 0.0
    assert float(account.cash) >= 0.0
    assert float(account.buying_power) >= 0.0
    assert float(account.portfolio_value) >= 0.0
