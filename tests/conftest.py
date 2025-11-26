"""
ISS-008: Pytest configuration and shared fixtures for Gary x Taleb trading system.
Provides common test infrastructure, fixtures, and utilities.
"""
import pytest
import asyncio
import threading
import time
import tempfile
import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock

# Add tests directory to path for mock imports
# ISS-015: Fix import paths for mock modules
_tests_dir = os.path.dirname(__file__)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Import all mock components (using package-style imports from tests/mocks/)
from mocks.mock_broker import MockBroker, create_mock_broker
from mocks.mock_gate_manager import MockGateManager, create_mock_gate_manager
from mocks.mock_weekly_cycle import MockWeeklyCycleManager, create_mock_weekly_cycle_manager


# ISS-008: Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    # These markers are also defined in pytest.ini for IDE support
    markers = [
        "integration: mark test as integration test",
        "slow: mark test as slow running",
        "broker: mark test as broker-related",
        "gates: mark test as gate-related",
        "cycle: mark test as weekly cycle-related",
        "concurrent: mark test as concurrency test",
        "error_handling: mark test as error handling test",
        # ISS-008: Additional markers
        "unit: mark test as unit test",
        "e2e: mark test as end-to-end test",
        "trading: mark test as trading engine test",
        "portfolio: mark test as portfolio management test",
        "safety: mark test as safety system test",
        "market_data: mark test as market data test",
        "security: mark test as security test",
        "performance: mark test as performance test",
        "mock: mark test as using mocked dependencies",
        "live_api: mark test as requiring live API access",
        "smoke: mark test as smoke test",
        "regression: mark test as regression test",
        "authentication: mark test as authentication test",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """ISS-008: Modify test collection to add default markers based on paths and names"""
    for item in items:
        fspath_str = str(item.fspath).lower()
        item_name = item.name.lower()

        # Add markers based on test file location
        path_markers = {
            "broker": pytest.mark.broker,
            "gate": pytest.mark.gates,
            "cycle": pytest.mark.cycle,
            "integration": pytest.mark.integration,
            # ISS-008: Additional path-based markers
            "safety": pytest.mark.safety,
            "trading": pytest.mark.trading,
            "portfolio": pytest.mark.portfolio,
            "market": pytest.mark.market_data,
            "security": pytest.mark.security,
            "performance": pytest.mark.performance,
            "unit": pytest.mark.unit,
            "e2e": pytest.mark.e2e,
            "end_to_end": pytest.mark.e2e,
            "auth": pytest.mark.authentication,
        }
        for keyword, marker in path_markers.items():
            if keyword in fspath_str:
                item.add_marker(marker)

        # Add markers based on test function name
        name_markers = {
            "concurrent": pytest.mark.concurrent,
            "thread": pytest.mark.concurrent,
            "error": pytest.mark.error_handling,
            "exception": pytest.mark.error_handling,
            "failure": pytest.mark.error_handling,
            "scenario": pytest.mark.slow,
            "workflow": pytest.mark.slow,
            "end_to_end": pytest.mark.slow,
            # ISS-008: Additional name-based markers
            "smoke": pytest.mark.smoke,
            "regression": pytest.mark.regression,
            "perf": pytest.mark.performance,
            "benchmark": pytest.mark.performance,
            "security": pytest.mark.security,
            "kill_switch": pytest.mark.safety,
            "circuit_breaker": pytest.mark.safety,
        }
        for keyword, marker in name_markers.items():
            if keyword in item_name:
                item.add_marker(marker)


# Fixtures for test isolation
@pytest.fixture
def isolated_test_env():
    """Provide isolated test environment"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="foundation_test_")
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    yield {
        "temp_dir": temp_dir,
        "original_cwd": original_cwd
    }
    
    # Cleanup
    os.chdir(original_cwd)
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_time():
    """Mock time functions for deterministic testing"""
    import time
    from unittest.mock import patch
    
    # Start with fixed time
    fixed_time = 1640995200.0  # 2022-01-01 00:00:00 UTC
    
    def mock_time_func():
        return fixed_time
        
    def advance_time(seconds):
        nonlocal fixed_time
        fixed_time += seconds
        
    with patch('time.time', side_effect=mock_time_func):
        yield advance_time


# Component fixtures
@pytest.fixture
def fast_broker():
    """Provide fast broker for testing (no delays)"""
    broker = create_mock_broker(
        connection_delay=0.0,
        order_fill_delay=0.001,  # Very fast fills
        error_rate=0.0
    )
    yield broker
    broker.reset_for_testing()


@pytest.fixture
def reliable_broker():
    """Provide reliable broker with realistic delays"""
    broker = create_mock_broker(
        connection_delay=0.1,
        order_fill_delay=0.05,
        error_rate=0.0
    )
    yield broker
    broker.reset_for_testing()


@pytest.fixture
def unreliable_broker():
    """Provide unreliable broker for error testing"""
    broker = create_mock_broker(
        connection_delay=0.1,
        order_fill_delay=0.1,
        error_rate=0.3  # 30% error rate
    )
    yield broker
    broker.reset_for_testing()


@pytest.fixture
def fast_gate_manager():
    """Provide fast gate manager for testing"""
    manager = create_mock_gate_manager(
        validation_delay=0.001,
        error_rate=0.0
    )
    yield manager
    manager.reset_for_testing()


@pytest.fixture
def strict_gate_manager():
    """Provide strict gate manager with tight thresholds"""
    manager = create_mock_gate_manager(
        validation_delay=0.01,
        error_rate=0.0
    )
    
    # Override default thresholds to be stricter
    for gate in manager.get_gates():
        if gate.gate_type.value == "risk_check":
            gate.threshold_config["max_risk"] = 0.5  # Stricter than default 0.8
        elif gate.gate_type.value == "position_limit":
            gate.threshold_config["max_positions"] = 10  # Stricter than default 20
        elif gate.gate_type.value == "loss_limit":
            gate.threshold_config["max_loss"] = -2000.0  # Stricter than default -5000
            
    yield manager
    manager.reset_for_testing()


@pytest.fixture
def lenient_gate_manager():
    """Provide lenient gate manager with loose thresholds"""
    manager = create_mock_gate_manager(
        validation_delay=0.01,
        error_rate=0.0
    )
    
    # Override default thresholds to be more lenient
    for gate in manager.get_gates():
        if gate.gate_type.value == "risk_check":
            gate.threshold_config["max_risk"] = 0.95  # More lenient
        elif gate.gate_type.value == "position_limit":
            gate.threshold_config["max_positions"] = 50  # More lenient
        elif gate.gate_type.value == "loss_limit":
            gate.threshold_config["max_loss"] = -20000.0  # More lenient
            
    yield manager
    manager.reset_for_testing()


@pytest.fixture
def fast_cycle_manager():
    """Provide fast cycle manager for testing"""
    from mocks.mock_weekly_cycle import CyclePhase
    
    manager = create_mock_weekly_cycle_manager(
        cycle_duration=timedelta(seconds=10),  # Very short cycles
        phase_delays={phase.value: 0.001 for phase in CyclePhase},
        auto_advance_phases=False,
        error_rate=0.0
    )
    yield manager
    manager.reset_for_testing()


@pytest.fixture
def auto_advance_cycle_manager():
    """Provide cycle manager with auto-advancing phases"""
    from mocks.mock_weekly_cycle import CyclePhase
    
    manager = create_mock_weekly_cycle_manager(
        phase_delays={phase.value: 0.05 for phase in CyclePhase},
        auto_advance_phases=True,
        error_rate=0.0
    )
    yield manager
    manager.reset_for_testing()


# Integration fixtures
@pytest.fixture
def foundation_system_fast():
    """Provide fast Foundation system for testing"""
    from foundation.test_integration import FoundationSystem
    from mocks.mock_weekly_cycle import CyclePhase
    
    system = FoundationSystem(
        broker_config={
            "connection_delay": 0.0,
            "order_fill_delay": 0.001,
            "error_rate": 0.0
        },
        gate_config={
            "validation_delay": 0.001,
            "error_rate": 0.0
        },
        cycle_config={
            "phase_delays": {phase.value: 0.001 for phase in CyclePhase},
            "auto_advance_phases": False,
            "error_rate": 0.0
        }
    )
    
    yield system
    system.shutdown_system()


@pytest.fixture
def foundation_system_realistic():
    """Provide realistic Foundation system for integration testing"""
    from foundation.test_integration import FoundationSystem
    from mocks.mock_weekly_cycle import CyclePhase
    
    system = FoundationSystem(
        broker_config={
            "connection_delay": 0.1,
            "order_fill_delay": 0.05,
            "error_rate": 0.02  # 2% error rate
        },
        gate_config={
            "validation_delay": 0.01,
            "error_rate": 0.01  # 1% error rate
        },
        cycle_config={
            "phase_delays": {phase.value: 0.02 for phase in CyclePhase},
            "auto_advance_phases": False,
            "error_rate": 0.01
        }
    )
    
    yield system
    system.shutdown_system()


# Test data fixtures
@pytest.fixture
def sample_portfolio_data():
    """Provide sample portfolio data for testing"""
    return {
        "total_value": 1000000.0,
        "positions": {
            "AAPL": {"value": 200000.0, "weight": 0.20, "shares": 1143},
            "GOOGL": {"value": 150000.0, "weight": 0.15, "shares": 54},
            "MSFT": {"value": 180000.0, "weight": 0.18, "shares": 433},
            "TSLA": {"value": 100000.0, "weight": 0.10, "shares": 402},
            "CASH": {"value": 370000.0, "weight": 0.37, "shares": 1}
        },
        "market_prices": {
            "AAPL": 175.0,
            "GOOGL": 2780.0,
            "MSFT": 415.0,
            "TSLA": 249.0,
            "CASH": 1.0
        }
    }


@pytest.fixture
def sample_allocation_targets():
    """Provide sample allocation targets for testing"""
    return {
        "conservative": {
            "AAPL": 0.15,
            "GOOGL": 0.10,
            "MSFT": 0.15,
            "SPY": 0.30,
            "BND": 0.20,
            "CASH": 0.10
        },
        "aggressive": {
            "AAPL": 0.25,
            "GOOGL": 0.20,
            "MSFT": 0.15,
            "TSLA": 0.15,
            "NVDA": 0.15,
            "QQQ": 0.10
        },
        "concentrated": {
            "AAPL": 0.60,
            "GOOGL": 0.25,
            "MSFT": 0.15
        }
    }


# Utility fixtures
@pytest.fixture
def event_collector():
    """Collect events from callbacks for testing"""
    events = []
    
    def collect_event(event_type, data):
        events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })
        
    return events, collect_event


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    import psutil
    import threading
    
    metrics = {
        "start_time": time.time(),
        "peak_memory_mb": 0,
        "cpu_samples": [],
        "thread_count_samples": []
    }
    
    stop_monitoring = threading.Event()
    
    def monitor():
        process = psutil.Process()
        while not stop_monitoring.wait(0.1):
            # Memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], memory_mb)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            metrics["cpu_samples"].append(cpu_percent)
            
            # Thread count
            thread_count = threading.active_count()
            metrics["thread_count_samples"].append(thread_count)
            
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    yield metrics
    
    # Stop monitoring and calculate final metrics
    stop_monitoring.set()
    metrics["end_time"] = time.time()
    metrics["duration"] = metrics["end_time"] - metrics["start_time"]
    
    if metrics["cpu_samples"]:
        metrics["avg_cpu_percent"] = sum(metrics["cpu_samples"]) / len(metrics["cpu_samples"])
        metrics["max_cpu_percent"] = max(metrics["cpu_samples"])
        
    if metrics["thread_count_samples"]:
        metrics["avg_thread_count"] = sum(metrics["thread_count_samples"]) / len(metrics["thread_count_samples"])
        metrics["max_thread_count"] = max(metrics["thread_count_samples"])


# Parametrized fixtures
@pytest.fixture(params=[0.0, 0.1, 0.3])
def error_rate(request):
    """Parametrized error rate for testing reliability"""
    return request.param


@pytest.fixture(params=["fast", "realistic", "slow"])
def timing_profile(request):
    """Parametrized timing profiles"""
    profiles = {
        "fast": {
            "connection_delay": 0.0,
            "order_fill_delay": 0.001,
            "validation_delay": 0.001,
            "phase_delay": 0.001
        },
        "realistic": {
            "connection_delay": 0.1,
            "order_fill_delay": 0.05,
            "validation_delay": 0.01,
            "phase_delay": 0.02
        },
        "slow": {
            "connection_delay": 0.5,
            "order_fill_delay": 0.2,
            "validation_delay": 0.1,
            "phase_delay": 0.1
        }
    }
    return profiles[request.param]


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_threads():
    """Automatically cleanup threads after each test"""
    initial_thread_count = threading.active_count()
    yield
    
    # Wait a bit for threads to cleanup naturally
    time.sleep(0.1)
    
    # Check for thread leaks
    final_thread_count = threading.active_count()
    if final_thread_count > initial_thread_count + 2:  # Allow some tolerance
        import warnings
        warnings.warn(
            f"Potential thread leak detected: {initial_thread_count} -> {final_thread_count}",
            ResourceWarning
        )


@pytest.fixture
def no_network():
    """Disable network access for isolated testing"""
    import socket
    from unittest.mock import patch
    
    def mock_getaddrinfo(*args, **kwargs):
        raise socket.gaierror("Network disabled for testing")
        
    with patch('socket.getaddrinfo', side_effect=mock_getaddrinfo):
        yield


# Test skip conditions
def pytest_runtest_setup(item):
    """Setup conditions for skipping tests"""
    # Skip slow tests unless specifically requested
    if "slow" in item.keywords:
        if not item.config.getoption("--runslow", default=False):
            pytest.skip("slow tests skipped (use --runslow to run)")
            
    # Skip integration tests in unit test mode
    if "integration" in item.keywords:
        if item.config.getoption("--unit-only", default=False):
            pytest.skip("integration tests skipped in unit-only mode")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--unit-only",
        action="store_true", 
        default=False,
        help="run only unit tests, skip integration tests"
    )
    parser.addoption(
        "--coverage-target",
        action="store",
        default="80",
        help="minimum coverage percentage required"
    )


# Coverage reporting
def pytest_sessionfinish(session, exitstatus):
    """Session finish hook for coverage reporting"""
    if hasattr(session.config, '_coverage'):
        coverage_target = float(session.config.getoption("--coverage-target"))
        # Coverage reporting would be handled by pytest-cov plugin
        pass


# Custom markers for test organization
pytestmark = [
    pytest.mark.foundation,  # All tests are part of foundation phase
]