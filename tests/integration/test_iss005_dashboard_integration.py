"""
ISS-005: Integration tests for Dashboard-TradingEngine connection.

Tests the complete data flow from TradingEngine through TradingStateProvider
to LiveDataProvider and the dashboard server.
"""
import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestTradingStateProvider:
    """Test TradingStateProvider functionality."""

    def test_provider_creation(self):
        """Test that TradingStateProvider can be created without engine."""
        from src.integration.trading_state_provider import TradingStateProvider

        provider = TradingStateProvider()
        assert provider is not None
        assert provider.is_connected is False

    def test_provider_with_mock_engine(self):
        """Test TradingStateProvider with a mock trading engine."""
        from src.integration.trading_state_provider import TradingStateProvider, set_trading_engine

        # Create mock engine
        mock_engine = Mock()
        mock_engine.mode = 'paper'
        mock_engine.kill_switch_active = False
        mock_engine.is_running = True
        mock_engine.portfolio_manager = Mock()
        mock_engine.portfolio_manager.get_all_positions.return_value = []
        mock_engine.risk_manager = Mock()
        mock_engine.risk_manager.get_metrics.return_value = {
            'p_ruin': 0.05,
            'var_95': 1000,
            'sharpe_ratio': 1.5
        }
        mock_engine.antifragility_engine = None

        provider = TradingStateProvider(mock_engine)
        assert provider.is_connected is True

    def test_state_file_operations(self):
        """Test state publishing and reading via file."""
        from src.integration.trading_state_provider import TradingStateProvider
        import tempfile
        import os

        # Create temp file for test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            provider = TradingStateProvider()
            provider._state_file = Path(temp_path)

            # Test publishing
            test_state = {
                'metrics': {'portfolio_value': 1000, 'p_ruin': 0.05},
                'positions': [],
                'source': 'test'
            }

            result = provider.publish_state(test_state)
            assert result is True

            # Verify file was written
            assert Path(temp_path).exists()

            # Test reading
            state = provider.read_published_state()
            assert state is not None
            assert state.get('metrics', {}).get('portfolio_value') == 1000

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_async_get_metrics(self):
        """Test async metrics retrieval."""
        from src.integration.trading_state_provider import TradingStateProvider

        provider = TradingStateProvider()
        metrics = await provider.get_metrics()

        # Should return DashboardMetrics with defaults
        assert metrics is not None
        assert hasattr(metrics, 'portfolio_value')
        assert hasattr(metrics, 'p_ruin')


class TestLiveDataProvider:
    """Test LiveDataProvider functionality."""

    def test_provider_creation(self):
        """Test that LiveDataProvider can be created."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        assert provider is not None

    def test_generate_metrics_no_data(self):
        """Test metrics generation with no real data."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        metrics = provider.generate_metrics()

        assert isinstance(metrics, dict)
        assert 'portfolio_value' in metrics
        assert 'p_ruin' in metrics
        assert 'source' in metrics

    def test_generate_positions_empty(self):
        """Test positions generation with no real data."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        positions = provider.generate_positions()

        assert isinstance(positions, list)

    def test_generate_alerts(self):
        """Test alerts generation."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        alerts = provider.generate_alerts()

        assert isinstance(alerts, list)

    def test_generate_barbell_allocation(self):
        """Test barbell allocation generation."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        barbell = provider.generate_barbell_allocation()

        assert isinstance(barbell, dict)
        assert 'safe_allocation' in barbell
        assert 'risky_allocation' in barbell

    def test_generate_engine_status(self):
        """Test engine status generation."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        status = provider.generate_engine_status()

        assert isinstance(status, dict)
        assert 'connected' in status
        assert 'status' in status

    @pytest.mark.asyncio
    async def test_async_metrics(self):
        """Test async metrics generation."""
        from src.dashboard.live_data_provider import LiveDataProvider

        provider = LiveDataProvider()
        metrics = await provider.generate_metrics_async()

        assert isinstance(metrics, dict)
        assert 'portfolio_value' in metrics


class TestStateBridge:
    """Test StateBridge file-based communication."""

    def test_bridge_creation(self):
        """Test StateBridge creation."""
        from src.integration.state_bridge import StateBridge
        import tempfile

        bridge = StateBridge(tempfile.mktemp(suffix='.json'))
        assert bridge is not None

    def test_publish_and_read(self):
        """Test publishing and reading state."""
        from src.integration.state_bridge import StateBridge
        import tempfile
        import os

        temp_path = tempfile.mktemp(suffix='.json')

        try:
            bridge = StateBridge(temp_path)

            # Publish
            test_state = {'test': 'data', 'value': 123}
            result = bridge.publish_state(test_state)
            assert result is True

            # Read
            state = bridge.read_state()
            assert state is not None
            assert state.get('test') == 'data'
            assert state.get('value') == 123

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_positions_publish(self):
        """Test publishing positions."""
        from src.integration.state_bridge import StateBridge
        import tempfile
        import os

        temp_path = tempfile.mktemp(suffix='.json')

        try:
            bridge = StateBridge(temp_path)

            positions = [
                {'symbol': 'SPY', 'quantity': 10, 'market_value': 5000},
                {'symbol': 'ULTY', 'quantity': 100, 'market_value': 1000}
            ]

            result = bridge.publish_positions(positions)
            assert result is True

            # Verify
            read_positions = bridge.get_positions()
            assert len(read_positions) == 2
            assert read_positions[0]['symbol'] == 'SPY'

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDashboardServerIntegration:
    """Test dashboard server with LiveDataProvider."""

    def test_server_creation_without_engine(self):
        """Test that dashboard server can be created without trading engine."""
        from src.dashboard.run_server_simple import SimpleDashboardServer

        server = SimpleDashboardServer()
        assert server is not None
        assert server.data_provider is not None

    def test_server_creation_with_mock_engine(self):
        """Test dashboard server with mock trading engine."""
        from src.dashboard.run_server_simple import SimpleDashboardServer, create_dashboard_server

        mock_engine = Mock()
        mock_engine.mode = 'paper'

        server = create_dashboard_server(trading_engine=mock_engine)
        assert server is not None

    def test_health_endpoint_format(self):
        """Test that health endpoint returns expected format."""
        from src.dashboard.run_server_simple import SimpleDashboardServer
        from fastapi.testclient import TestClient

        server = SimpleDashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'live_data' in data
        assert 'data_source' in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns valid data."""
        from src.dashboard.run_server_simple import SimpleDashboardServer
        from fastapi.testclient import TestClient

        server = SimpleDashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/metrics/current")
        assert response.status_code == 200

        data = response.json()
        assert 'portfolio_value' in data or 'p_ruin' in data

    def test_engine_status_endpoint(self):
        """Test engine status endpoint."""
        from src.dashboard.run_server_simple import SimpleDashboardServer
        from fastapi.testclient import TestClient

        server = SimpleDashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/engine/status")
        assert response.status_code == 200

        data = response.json()
        assert 'connected' in data
        assert 'status' in data


class TestEndToEndDataFlow:
    """Test complete data flow from engine to dashboard."""

    @pytest.mark.asyncio
    async def test_complete_data_flow(self):
        """Test data flows from mock engine through to dashboard endpoints."""
        from src.integration.trading_state_provider import TradingStateProvider, set_trading_engine
        from src.dashboard.live_data_provider import LiveDataProvider
        import tempfile
        import os

        # Create mock engine with realistic data
        mock_engine = Mock()
        mock_engine.mode = 'paper'
        mock_engine.kill_switch_active = False
        mock_engine.is_running = True

        # Mock portfolio manager
        mock_engine.portfolio_manager = Mock()
        mock_engine.portfolio_manager.total_value = 10000.0
        mock_engine.portfolio_manager.cash = 3000.0
        mock_engine.portfolio_manager.get_all_positions.return_value = [
            Mock(symbol='SPY', quantity=10, entry_price=450.0, current_price=455.0,
                 market_value=4550.0, unrealized_pnl=50.0)
        ]

        # Mock risk manager
        mock_engine.risk_manager = Mock()
        mock_engine.risk_manager.get_metrics.return_value = {
            'p_ruin': 0.03,
            'var_95': 500,
            'var_99': 800,
            'expected_shortfall': 600,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.8,
            'volatility': 0.12
        }

        # Mock antifragility engine
        mock_engine.antifragility_engine = Mock()
        mock_engine.antifragility_engine.get_current_allocation.return_value = {
            'safe': 0.65,
            'risky': 0.35
        }

        temp_path = tempfile.mktemp(suffix='.json')

        try:
            # Create state provider with mock engine
            provider = TradingStateProvider(mock_engine)
            provider._state_file = Path(temp_path)

            # Get and publish state
            state = await provider.get_full_state()
            provider.publish_state(state)

            # Create live data provider reading from same file
            live_provider = LiveDataProvider(state_provider=provider)

            # Verify metrics flow through
            metrics = live_provider.generate_metrics()
            assert metrics['portfolio_value'] > 0

            # Verify barbell allocation flows through
            barbell = live_provider.generate_barbell_allocation()
            assert 'safe_allocation' in barbell

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
