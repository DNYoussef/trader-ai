from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.gates.gate_manager import GateLevel, GateManager
from src.trading_engine import TradingEngine


class FakeSiphonAutomator:
    def __init__(self):
        self.force_args = []

    def should_execute_siphon(self):
        return True, "phase5 test due"

    async def execute_manual_siphon(self, force=False):
        self.force_args.append(force)
        return SimpleNamespace(
            status=SimpleNamespace(value="success"),
            withdrawal_amount=Decimal("25.00"),
            withdrawal_success=True,
            errors=[],
        )


@pytest.mark.asyncio
async def test_trading_cycle_executes_due_weekly_profit_siphon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = TradingEngine(config_path=str(tmp_path / "missing-config.json"))
    audit_events = []

    engine._audit_log = lambda event: audit_events.append(event)
    engine._log_memory_event = lambda *args, **kwargs: None
    engine.market_data = SimpleNamespace(get_market_status=AsyncMock(return_value=True))
    engine.portfolio_manager = SimpleNamespace(
        sync_with_broker=AsyncMock(return_value=True),
        check_daily_loss=AsyncMock(
            return_value={"triggered": False, "daily_change_pct": 0.0, "limit_pct": 0.05}
        ),
        get_total_portfolio_value=AsyncMock(return_value=Decimal("1000.00")),
        create_daily_snapshot=AsyncMock(),
    )
    engine.broker = SimpleNamespace(get_cash_balance=AsyncMock(return_value=Decimal("500.00")))
    engine._execute_rebalancing = AsyncMock()
    engine.siphon_automator = FakeSiphonAutomator()

    await engine._execute_trading_cycle()

    engine._execute_rebalancing.assert_awaited_once_with(Decimal("1000.00"))
    engine.portfolio_manager.create_daily_snapshot.assert_awaited_once()
    assert engine.siphon_automator.force_args == [False]
    siphon_events = [event for event in audit_events if event["event"] == "weekly_profit_siphon"]
    assert siphon_events
    assert siphon_events[0]["status"] == "success"
    assert siphon_events[0]["withdrawal_amount"] == "25.00"


def test_gate_manager_rejects_buy_that_exceeds_concentration_cap(tmp_path):
    manager = GateManager(data_dir=str(tmp_path))
    manager.current_gate = GateLevel.G0
    config = manager.gate_configs[GateLevel.G0]
    config.cash_floor_pct = 0.0
    config.max_position_pct = 1.0
    config.max_concentration_pct = 0.30

    portfolio = {
        "cash": 1_000.0,
        "total_value": 1_000.0,
        "positions": {
            "AMDY": {
                "quantity": 25,
                "current_price": 10.0,
                "market_value": 250.0,
                "sector": "yield",
            }
        },
    }

    under_cap = manager.validate_trade(
        {
            "symbol": "ULTY",
            "side": "BUY",
            "quantity": 4,
            "price": 10.0,
            "trade_type": "STOCK",
            "sector": "yield",
        },
        portfolio,
    )
    over_cap = manager.validate_trade(
        {
            "symbol": "ULTY",
            "side": "BUY",
            "quantity": 10,
            "price": 10.0,
            "trade_type": "STOCK",
            "sector": "yield",
        },
        portfolio,
    )

    assert under_cap.is_valid
    assert not over_cap.is_valid
    concentration_violations = [
        violation for violation in over_cap.violations
        if violation["type"] == "concentration_exceeded"
    ]
    assert concentration_violations
    assert concentration_violations[0]["details"]["post_trade_group_value"] == 350.0
    assert concentration_violations[0]["details"]["concentration_pct"] == pytest.approx(0.35)
