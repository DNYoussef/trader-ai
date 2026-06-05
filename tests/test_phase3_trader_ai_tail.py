from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.cycles.weekly_cycle import WeeklyCycle
from src.dashboard.live_data_provider import LiveDataProvider
from src.integration.trading_state_provider import TradingStateProvider
from src.portfolio.portfolio_manager import PortfolioManager, Position
from src.risk.kelly_criterion import KellyComponents, KellyCriterionCalculator
from src.risk.kelly_enhanced import AssetRiskProfile, EnhancedKellyCriterion
from src.strategies.antifragility_engine import AntifragilityEngine
from src.trading_engine import TradingEngine


class FakeDPI:
    def calculate_dpi(self, symbol, lookback_days=None):
        return 0.9, SimpleNamespace()

    def _fetch_market_data(self, symbol, periods):
        return pd.DataFrame({"Close": np.linspace(100.0, 110.0, periods + 1)})


class FakeGate:
    def validate_trade(self, trade_details, portfolio_state):
        return SimpleNamespace(is_valid=True)


def test_antifragility_defaults_match_taleb_barbell_claim():
    engine = AntifragilityEngine(portfolio_value=100_000)

    assert engine.barbell_config.safe_allocation == 0.80
    assert engine.barbell_config.risky_allocation == 0.20
    assert "CASH" in engine.barbell_config.safe_instruments
    assert "QQQ" in engine.barbell_config.risky_instruments


@pytest.mark.asyncio
async def test_dashboard_barbell_fallbacks_match_taleb_claim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    state_provider = TradingStateProvider(trading_engine=None)
    state_fallback = await state_provider.get_barbell_allocation()
    live_fallback = LiveDataProvider(state_provider=state_provider).generate_barbell_allocation()

    for fallback in (state_fallback, live_fallback):
        assert fallback["safe_allocation"] == 80
        assert fallback["risky_allocation"] == 20
        assert "CASH" in fallback["safe_instruments"]
        assert "QQQ" in fallback["risky_instruments"]


def test_convexity_kelly_uses_annualized_mean_with_annualized_variance():
    engine = AntifragilityEngine(portfolio_value=100_000)
    log_returns = np.array([0.0001 + (0.02 if i % 2 == 0 else -0.02) for i in range(260)])
    prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])

    metrics = engine.assess_convexity("OSC", prices.tolist(), 10_000)

    assert metrics.kelly_fraction > 0.05


def test_kelly_time_to_ruin_extreme_guards_have_correct_direction():
    calc = KellyCriterionCalculator(FakeDPI(), FakeGate())

    assert calc._calculate_time_to_ruin(kelly=0.1, edge=0.1, volatility=0.01) == float("inf")
    assert calc._calculate_time_to_ruin(kelly=0.9, edge=0.001, volatility=1.0) == 1.0


def test_narrative_gap_multiplier_cannot_exceed_configured_max_kelly(monkeypatch):
    calc = KellyCriterionCalculator(FakeDPI(), FakeGate(), max_kelly=0.25)
    components = KellyComponents(
        edge=0.10,
        odds=2.0,
        win_probability=0.60,
        loss_probability=0.40,
        raw_kelly=0.40,
        capped_kelly=0.40,
        dpi_adjustment=1.0,
        final_kelly=0.25,
    )

    monkeypatch.setattr(calc, "_calculate_edge_from_dpi", lambda dpi_score, dpi_components: 0.10)
    monkeypatch.setattr(calc, "_estimate_probabilities", lambda symbol, historical_data=None: {"win_prob": 0.60, "loss_prob": 0.40})
    monkeypatch.setattr(calc, "_calculate_odds", lambda symbol, historical_data=None: 2.0)
    monkeypatch.setattr(calc, "_calculate_kelly_components", lambda edge, probabilities, odds, dpi_score: components)
    monkeypatch.setattr(calc, "_calculate_risk_metrics", lambda symbol, kelly_components, historical_data=None: calc._default_risk_metrics())
    monkeypatch.setattr(calc, "_apply_constraints", lambda kelly_components, risk_metrics: 0.20)
    monkeypatch.setattr(calc, "_calculate_narrative_gap_multiplier", lambda symbol, current_price, kelly_components, dpi_score: 10.0)
    monkeypatch.setattr(calc, "_validate_gate_compliance", lambda symbol, dollar_amount, available_capital: True)
    monkeypatch.setattr(calc, "_calculate_confidence_score", lambda kelly_components, risk_metrics, dpi_score: 0.9)

    recommendation = calc.calculate_kelly_position("ULTY", current_price=10.0, available_capital=1_000.0)

    assert recommendation.kelly_percentage == pytest.approx(0.25)
    assert recommendation.dollar_amount == pytest.approx(250.0)


def test_expected_sharpe_uses_realized_returns_not_dpi_edge():
    calc = KellyCriterionCalculator(FakeDPI(), FakeGate())
    components = KellyComponents(
        edge=1.0,
        odds=2.0,
        win_probability=0.60,
        loss_probability=0.40,
        raw_kelly=0.30,
        capped_kelly=0.30,
        dpi_adjustment=1.0,
        final_kelly=0.10,
    )
    history = pd.DataFrame({"Close": np.linspace(100.0, 90.0, 80)})

    metrics = calc._calculate_risk_metrics("LOSS", components, history)

    assert metrics.sharpe_expectation < 0


def test_portfolio_cvar_is_expected_shortfall_not_parametric_var(tmp_path):
    cfg = EnhancedKellyCriterion.__new__(EnhancedKellyCriterion)._default_config()
    cfg["data_path"] = str(tmp_path / "kelly")
    engine = EnhancedKellyCriterion(cfg)
    engine.asset_profiles = {
        "A": AssetRiskProfile("A", 0.05, 0.20, -0.1, -0.2, -0.30, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0, 0.1),
        "B": AssetRiskProfile("B", 0.04, 0.30, -0.1, -0.3, -0.45, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 1, 0.1),
    }
    engine.correlation_matrix = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = {"A": 0.5, "B": 0.5}

    sigma = np.sqrt((0.5**2 * 0.20**2) + (0.5**2 * 0.30**2))
    expected_shortfall = sigma * (norm.pdf(norm.ppf(0.05)) / 0.05)

    assert engine.calculate_portfolio_cvar(weights) == pytest.approx(expected_shortfall)
    assert engine.calculate_portfolio_cvar(weights) > sigma * abs(norm.ppf(0.05))


def test_weekly_cycle_defines_buy_allocations_for_every_gate():
    cycle = WeeklyCycle(
        portfolio_manager=SimpleNamespace(),
        trade_executor=SimpleNamespace(buy_market_order=lambda **kwargs: {"ok": True}),
        market_data=SimpleNamespace(),
        holiday_calendar=SimpleNamespace(),
        enable_dpi=False,
    )

    assert set(cycle.GATE_ALLOCATIONS) == {f"G{i}" for i in range(13)}
    for gate, allocation in cycle.GATE_ALLOCATIONS.items():
        allocation.validate()
        result = cycle.execute_buy_phase(gate, available_cash=100.0)
        assert result["success"] is True


@pytest.mark.asyncio
async def test_portfolio_total_value_does_not_sync_by_default(monkeypatch):
    manager = PortfolioManager(
        broker_adapter=SimpleNamespace(is_connected=True),
        market_data_provider=SimpleNamespace(),
        initial_capital=Decimal("50.00"),
    )
    manager.cash_balance = Decimal("50.00")
    manager.positions = {
        "SPY": Position(
            symbol="SPY",
            quantity=Decimal("1"),
            avg_cost=Decimal("20.00"),
            current_price=Decimal("25.00"),
            market_value=Decimal("25.00"),
            unrealized_pnl=Decimal("5.00"),
            unrealized_pnl_percent=Decimal("0.25"),
            gate="SAFE_HEDGE",
            last_updated=pd.Timestamp.utcnow().to_pydatetime(),
        )
    }
    sync_calls = 0

    async def fake_sync():
        nonlocal sync_calls
        sync_calls += 1
        manager.cash_balance = Decimal("60.00")
        return True

    monkeypatch.setattr(manager, "sync_with_broker", fake_sync)

    assert await manager.get_total_portfolio_value() == Decimal("75.00")
    assert await manager.get_total_portfolio_value() == Decimal("75.00")
    assert sync_calls == 0
    assert await manager.get_total_portfolio_value(refresh=True) == Decimal("85.00")
    assert sync_calls == 1


@pytest.mark.asyncio
async def test_trading_engine_rebalance_path_applies_kelly_caps_and_blocks():
    engine = TradingEngine.__new__(TradingEngine)

    class Market:
        async def get_current_price(self, symbol):
            return {"ULTY": 10.0, "AMDY": 20.0, "MISSING": None}.get(symbol)

    class Kelly:
        def calculate_kelly_position(self, symbol, current_price, available_capital):
            if symbol == "ULTY":
                return SimpleNamespace(kelly_percentage=0.04, dollar_amount=40.0, gate_compliant=True)
            return SimpleNamespace(kelly_percentage=0.03, dollar_amount=60.0, gate_compliant=False)

    engine.market_data = Market()
    engine.kelly_calculator = Kelly()

    adjusted = await engine._apply_kelly_position_sizing(
        {
            "ULTY": Decimal("100.00"),
            "AMDY": Decimal("100.00"),
            "MISSING": Decimal("100.00"),
            "CASH": Decimal("50.00"),
        },
        total_value=Decimal("1000.00"),
    )

    assert adjusted["ULTY"] == Decimal("40.0")
    assert adjusted["AMDY"] == Decimal("0")
    assert adjusted["MISSING"] == Decimal("0")
    assert adjusted["CASH"] == Decimal("50.00")


def test_removed_dead_loss_rate_noop_expression():
    source = Path("src/risk/kelly_criterion.py").read_text(encoding="utf-8")

    assert "\n            1 - win_rate" not in source


def test_repo_root_documentation_sprawl_is_archived():
    allowed_root_docs = {
        "README.md",
        "CLAUDE.md",
        "SETUP_GUIDE.md",
        "REMEDIATION-PLAN.md",
        "RAILWAY-DEPLOY.md",
        "RESTART-SERVER.md",
        "training_commands.md",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-full.txt",
    }

    root_docs = {path.name for path in Path(".").iterdir() if path.is_file() and path.suffix in {".md", ".txt"}}
    archived_docs = list((Path("archive") / "phase3-root-docs").glob("*"))

    assert root_docs == allowed_root_docs
    assert len(archived_docs) >= 70
    assert not any("COMPLETE" in name or "SUMMARY" in name or "REPORT" in name for name in root_docs)


def test_removed_modules_tree_is_archived_not_active():
    archive = Path("archive") / "phase3-removed-modules"

    assert not Path(".removed-modules").exists()
    assert archive.exists()
    assert (archive / "enterprise").exists()
    assert (archive / "theater-detection").exists()
