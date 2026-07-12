from decimal import Decimal
from types import SimpleNamespace
import sys

import pytest
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from src.trading.terminal_data_provider import MarketDataPoint, TradingTerminalDataProvider


@pytest.mark.asyncio
async def test_live_terminal_mode_does_not_emit_synthetic_insights():
    provider = TradingTerminalDataProvider(
        symbols=["SPY"],
        enable_live_data=True,
        alpaca_data_provider=object(),
    )
    provider.market_data["SPY"] = MarketDataPoint(
        symbol="SPY",
        timestamp=1.0,
        price=100.0,
        volume=1_000,
        change=0.0,
        change_percent=0.0,
    )

    await provider._generate_algorithmic_signals()
    await provider._generate_ai_inflections()

    assert provider.get_algorithmic_signals() == []
    assert provider.get_ai_inflections() == []
    assert provider.get_terminal_insight_status() == {
        "algorithmic_signals": "unavailable_no_live_signal_engine",
        "ai_inflections": "unavailable_no_live_inflection_engine",
    }
    assert provider.get_terminal_snapshot()["terminal_insight_status"][
        "algorithmic_signals"
    ] == "unavailable_no_live_signal_engine"


@pytest.mark.asyncio
async def test_demo_terminal_mode_labels_synthetic_insights():
    provider = TradingTerminalDataProvider(symbols=["SPY"], enable_live_data=False)
    provider.market_data["SPY"] = MarketDataPoint(
        symbol="SPY",
        timestamp=1.0,
        price=100.0,
        volume=1_000,
        change=0.0,
        change_percent=0.0,
    )

    await provider._generate_algorithmic_signals()
    await provider._generate_ai_inflections()

    assert provider.get_algorithmic_signals()
    assert provider.get_ai_inflections()
    assert {
        signal["evidence_status"] for signal in provider.get_algorithmic_signals()
    } == {"synthetic_demo_terminal_insight"}
    assert {
        inflection["evidence_status"] for inflection in provider.get_ai_inflections()
    } == {"synthetic_demo_terminal_insight"}


def _dashboard_module_without_optional_runtime(monkeypatch):
    constants_module = sys.modules.get("constants")
    if constants_module is not None and not hasattr(constants_module, "CORS_ORIGINS"):
        sys.modules.pop("constants", None)
        sys.modules.pop("src.dashboard.run_server_simple", None)

    from src.dashboard import run_server_simple as dashboard

    monkeypatch.setattr(dashboard, "LIVE_DATA_AVAILABLE", False)
    monkeypatch.setattr(dashboard, "AI_AVAILABLE", False)
    monkeypatch.setattr(dashboard, "JWT_AUTH_AVAILABLE", False)
    monkeypatch.setattr(dashboard, "RATE_LIMITER_AVAILABLE", False)
    return dashboard


def test_archived_dashboard_exposes_no_websocket(monkeypatch):
    dashboard = _dashboard_module_without_optional_runtime(monkeypatch)
    server = dashboard.SimpleDashboardServer(trading_engine=None)

    assert not [route for route in server.app.routes if isinstance(route, WebSocketRoute)]
    with pytest.raises(WebSocketDisconnect):
        with TestClient(server.app).websocket_connect("/ws/f018-probe"):
            pytest.fail("archived dashboard upgraded an unauthenticated WebSocket")


def test_dashboard_trade_execute_fails_closed_without_trading_engine(monkeypatch):
    dashboard = _dashboard_module_without_optional_runtime(monkeypatch)
    server = dashboard.SimpleDashboardServer(trading_engine=None)
    client = TestClient(server.app)

    response = client.post(
        dashboard.C.API_TRADING_EXECUTE,
        json={"symbol": "SPY", "action": "buy", "dollar_amount": "25.50"},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["success"] is False
    assert detail["execution_status"] == "unavailable_no_trading_engine"
    assert detail["evidence_status"] == "not_executed"


def test_dashboard_trade_execute_calls_trading_engine_safety_path(monkeypatch):
    dashboard = _dashboard_module_without_optional_runtime(monkeypatch)

    class FakeTradingEngine:
        def __init__(self):
            self.calls = []

        async def execute_manual_trade(self, symbol, dollar_amount, action, gate="MANUAL"):
            self.calls.append(
                {
                    "symbol": symbol,
                    "dollar_amount": dollar_amount,
                    "action": action,
                    "gate": gate,
                }
            )
            return SimpleNamespace(status="filled", order_id="order-123")

    engine = FakeTradingEngine()
    server = dashboard.SimpleDashboardServer(trading_engine=engine)
    client = TestClient(server.app)

    response = client.post(
        dashboard.C.API_TRADING_EXECUTE,
        json={"symbol": "spy", "side": "BUY", "amount": "25.50", "gate": "G0"},
    )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "symbol": "SPY",
            "dollar_amount": Decimal("25.50"),
            "action": "buy",
            "gate": "G0",
        }
    ]
    payload = response.json()
    assert payload["success"] is True
    assert payload["execution_status"] == "executed_through_trading_engine"
    assert payload["evidence_status"] == "trading_engine_result"
    assert payload["order_id"] == "order-123"
