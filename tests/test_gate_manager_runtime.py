import pytest

from src.gates.gate_manager import GateLevel, GateManager


@pytest.mark.unit
def test_gate_manager_defines_full_g0_to_g12_runtime_ladder(tmp_path):
    manager = GateManager(data_dir=str(tmp_path))

    assert list(manager.gate_configs) == list(GateLevel)
    assert manager.gate_configs[GateLevel.G0].capital_min == 200.0
    assert manager.gate_configs[GateLevel.G12].capital_min == 10_000_000.0
    assert manager.gate_configs[GateLevel.G12].capital_max == float("inf")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("capital", "expected_gate"),
    [
        (199.99, GateLevel.G0),
        (200.00, GateLevel.G0),
        (500.00, GateLevel.G1),
        (1_000.00, GateLevel.G2),
        (2_500.00, GateLevel.G3),
        (5_000.00, GateLevel.G4),
        (10_000.00, GateLevel.G5),
        (25_000.00, GateLevel.G6),
        (50_000.00, GateLevel.G7),
        (100_000.00, GateLevel.G8),
        (250_000.00, GateLevel.G9),
        (500_000.00, GateLevel.G10),
        (1_000_000.00, GateLevel.G11),
        (10_000_000.00, GateLevel.G12),
        (25_000_000.00, GateLevel.G12),
    ],
)
def test_gate_manager_maps_capital_to_full_gate_ladder(tmp_path, capital, expected_gate):
    manager = GateManager(data_dir=str(tmp_path))

    manager.update_capital(capital)

    assert manager.current_gate == expected_gate


@pytest.mark.unit
def test_gate_manager_graduation_and_downgrade_are_generic_past_g3(tmp_path):
    manager = GateManager(data_dir=str(tmp_path))
    manager.current_gate = GateLevel.G3

    assert manager.execute_graduation()
    assert manager.current_gate == GateLevel.G4

    assert manager.execute_downgrade()
    assert manager.current_gate == GateLevel.G3


@pytest.mark.unit
def test_gate_manager_has_graduation_criteria_through_g11(tmp_path):
    manager = GateManager(data_dir=str(tmp_path))

    g11_criteria = manager.get_graduation_criteria(GateLevel.G11)

    assert g11_criteria is not None
    assert g11_criteria["min_capital"] == 10_000_000.0
    assert manager.get_graduation_criteria(GateLevel.G12) is None


@pytest.mark.unit
def test_gate_manager_high_gate_guardrails_remain_conservative(tmp_path):
    manager = GateManager(data_dir=str(tmp_path))
    manager.current_gate = GateLevel.G9

    valid = manager.validate_trade(
        {"symbol": "BIL", "side": "BUY", "quantity": 1, "price": 1_000, "trade_type": "STOCK"},
        {"cash": 250_000, "positions": {}, "total_value": 250_000},
    )
    oversized = manager.validate_trade(
        {"symbol": "BIL", "side": "BUY", "quantity": 20, "price": 1_000, "trade_type": "STOCK"},
        {"cash": 250_000, "positions": {}, "total_value": 250_000},
    )
    unknown = manager.validate_trade(
        {"symbol": "PRIVATE_SWAP", "side": "BUY", "quantity": 1, "price": 1_000, "trade_type": "STOCK"},
        {"cash": 250_000, "positions": {}, "total_value": 250_000},
    )

    assert valid.is_valid
    assert not oversized.is_valid
    assert "position_size_exceeded" in {violation["type"] for violation in oversized.violations}
    assert not unknown.is_valid
    assert "asset_not_allowed" in {violation["type"] for violation in unknown.violations}
