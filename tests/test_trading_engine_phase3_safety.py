"""Phase 3 regression tests for trading engine startup and halt safety."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import src.trading_engine as trading_engine_module
from src.trading_engine import TradingEngine, validate_trading_mode


@pytest.mark.asyncio
async def test_start_uses_async_initializer_in_running_loop(tmp_path):
    engine = TradingEngine(config_path=str(tmp_path / "missing.json"))
    engine.initialize = Mock(side_effect=AssertionError("sync initializer must not run"))
    engine.initialize_async = AsyncMock(return_value=False)

    await engine.start()

    engine.initialize_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_async_awaits_async_dependencies(monkeypatch, tmp_path):
    class FakeBroker:
        def __init__(self, config):
            self.config = config
            self.is_connected = False

        async def connect(self):
            self.is_connected = True
            return True

    class FakeMarketData:
        def __init__(self, broker):
            self.broker = broker

    class FakePortfolioManager:
        def __init__(self, broker, market_data, initial_capital):
            self.broker = broker
            self.market_data = market_data
            self.initial_capital = initial_capital

    class FakeGateManager:
        pass

    class FakeSafetyIntegration:
        def __init__(self, config):
            self.config = config
            self.circuit_manager = object()

        async def initialize(self, engine):
            self.engine = engine
            return True

    class FakeTradeExecutor:
        def __init__(self, broker, portfolio, market_data, gate_manager, circuit_manager):
            self.broker = broker
            self.portfolio = portfolio
            self.market_data = market_data
            self.gate_manager = gate_manager
            self.circuit_manager = circuit_manager

    monkeypatch.setattr(trading_engine_module, "MemoryClient", lambda: object())
    monkeypatch.setattr(trading_engine_module, "AlpacaAdapter", FakeBroker)
    monkeypatch.setattr(trading_engine_module, "TradingSafetyIntegration", FakeSafetyIntegration)
    monkeypatch.setattr(trading_engine_module, "get_state_provider", lambda: object())
    monkeypatch.setattr(trading_engine_module, "set_trading_engine", lambda engine: None)
    monkeypatch.setattr("src.market.market_data.MarketDataProvider", FakeMarketData)
    monkeypatch.setattr("src.portfolio.portfolio_manager.PortfolioManager", FakePortfolioManager)
    monkeypatch.setattr("src.gates.gate_manager.GateManager", FakeGateManager)
    monkeypatch.setattr("src.trading.trade_executor.TradeExecutor", FakeTradeExecutor)

    engine = TradingEngine(config_path=str(tmp_path / "missing.json"))
    engine.audit_log_path = str(tmp_path / "audit.jsonl")
    engine.config.update({
        "broker": "alpaca",
        "api_key": "paper-key",
        "secret_key": "paper-secret",
        "initial_capital": 200,
        "mode": "paper",
    })

    assert await engine.initialize_async() is True
    assert engine.broker.is_connected is True
    assert isinstance(engine.trade_executor, FakeTradeExecutor)


@pytest.mark.asyncio
async def test_daily_loss_breach_trips_kill_switch_and_cancels_orders(tmp_path):
    engine = TradingEngine(config_path=str(tmp_path / "missing.json"))
    engine.config["audit_enabled"] = False
    engine.running = True
    engine.kill_switch_activated = False
    engine.memory_client = None
    engine.safety_integration = None
    engine.market_data = SimpleNamespace(get_market_status=AsyncMock(return_value=True))
    engine.portfolio_manager = SimpleNamespace(
        sync_with_broker=AsyncMock(return_value=True),
        check_daily_loss=AsyncMock(return_value={
            "triggered": True,
            "daily_change_pct": -0.025,
            "limit_pct": -0.02,
            "required_action": "kill_switch",
        }),
    )
    engine.trade_executor = SimpleNamespace(
        cancel_all_pending_orders=AsyncMock(return_value=2),
    )
    engine.broker = SimpleNamespace(
        is_connected=True,
        cancel_all_orders=AsyncMock(return_value=3),
        close_all_positions=AsyncMock(return_value=True),
        get_account_value=AsyncMock(return_value=Decimal("195.00")),
        get_positions=AsyncMock(return_value=[]),
        disconnect=AsyncMock(),
    )
    engine._execute_rebalancing = AsyncMock()

    await engine._execute_trading_cycle()

    assert engine.kill_switch_activated is True
    assert engine.running is False
    engine.trade_executor.cancel_all_pending_orders.assert_awaited_once()
    engine.broker.cancel_all_orders.assert_awaited_once()
    engine.broker.close_all_positions.assert_awaited_once()
    engine._execute_rebalancing.assert_not_awaited()


def test_live_mode_env_confirmation_fails_closed(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "YES")
    monkeypatch.setattr(
        trading_engine_module.sys,
        "stdin",
        SimpleNamespace(isatty=lambda: False),
    )

    with pytest.raises(SystemExit, match="interactive operator confirmation"):
        validate_trading_mode()
