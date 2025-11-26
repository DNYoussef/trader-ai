#!/usr/bin/env python3
"""
Test suite for the Gary×Taleb Risk Dashboard.
Tests WebSocket server, API endpoints, and real-time functionality.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.websocket_server import RiskDashboardServer

# Test fixtures
@pytest.fixture
def dashboard_server():
    """Create a test dashboard server instance."""
    server = RiskDashboardServer()
    return server

@pytest.fixture
def test_client(dashboard_server):
    """Create a test client for API testing."""
    return TestClient(dashboard_server.app)

@pytest.fixture
def sample_risk_metrics():
    """Sample risk metrics for testing."""
    return {
        'portfolio_value': 10000.0,
        'max_drawdown': 0.05,
        'volatility': 0.15,
        'beta': 1.0,
        'positions_count': 5,
        'cash_available': 2000.0,
        'margin_used': 0.3,
        'unrealized_pnl': 150.0,
        'daily_pnl': 75.0,
    }

@pytest.fixture
def sample_position():
    """Sample position data for testing."""
    return {
        'quantity': 100,
        'market_value': 2000.0,
        'unrealized_pnl': 50.0,
        'entry_price': 150.0,
        'current_price': 155.0,
        'weight': 0.2,
    }

class TestDashboardAPI:
    """Test the REST API endpoints."""

    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "connections" in data

    def test_metrics_endpoint_empty(self, test_client):
        """Test metrics endpoint when no data is available."""
        response = test_client.get("/api/metrics/current")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_positions_endpoint_empty(self, test_client):
        """Test positions endpoint when no data is available."""
        response = test_client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_alerts_endpoint_empty(self, test_client):
        """Test alerts endpoint when no data is available."""
        response = test_client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_acknowledge_alert_not_found(self, test_client):
        """Test acknowledging a non-existent alert."""
        response = test_client.post("/api/alerts/nonexistent/acknowledge")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

class TestRiskCalculator:
    """Test the risk calculation functions."""

    def test_p_ruin_calculation(self, dashboard_server):
        """Test P(ruin) calculation."""
        calculator = dashboard_server.risk_calculator

        # Test with empty returns
        p_ruin = calculator.calculate_p_ruin(10000.0, [])
        assert p_ruin == 0.5  # Default conservative estimate

        # Test with sample returns
        returns = [0.01, -0.02, 0.015, -0.01, 0.005] * 20  # 100 returns
        p_ruin = calculator.calculate_p_ruin(10000.0, returns)
        assert 0.0 <= p_ruin <= 1.0

    def test_var_calculation(self, dashboard_server):
        """Test VaR calculation."""
        calculator = dashboard_server.risk_calculator

        # Test with empty returns
        var = calculator.calculate_var(10000.0, [])
        assert var == 500.0  # Default 5% VaR

        # Test with sample returns
        returns = [0.01, -0.02, 0.015, -0.01, 0.005] * 20
        var_95 = calculator.calculate_var(10000.0, returns, 0.95)
        var_99 = calculator.calculate_var(10000.0, returns, 0.99)
        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher

    def test_expected_shortfall_calculation(self, dashboard_server):
        """Test Expected Shortfall calculation."""
        calculator = dashboard_server.risk_calculator

        # Test with sample returns
        returns = [0.01, -0.02, 0.015, -0.01, 0.005] * 20
        es = calculator.calculate_expected_shortfall(10000.0, returns)
        assert es > 0

    def test_sharpe_ratio_calculation(self, dashboard_server):
        """Test Sharpe ratio calculation."""
        calculator = dashboard_server.risk_calculator

        # Test with empty returns
        sharpe = calculator.calculate_sharpe_ratio([])
        assert sharpe == 0.0

        # Test with positive returns
        returns = [0.01, 0.02, 0.015, 0.01, 0.005] * 20
        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe > 0

        # Test with negative returns
        returns = [-0.01, -0.02, -0.015, -0.01, -0.005] * 20
        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe < 0

class TestDataUpdates:
    """Test data update functionality."""

    def test_update_risk_metrics(self, dashboard_server, sample_risk_metrics):
        """Test updating risk metrics."""
        dashboard_server.update_risk_metrics(sample_risk_metrics)

        assert dashboard_server.latest_metrics is not None
        assert dashboard_server.latest_metrics.portfolio_value == 10000.0
        assert len(dashboard_server.historical_data['portfolio_values']) == 1

    def test_update_position(self, dashboard_server, sample_position):
        """Test updating position data."""
        symbol = "AAPL"
        dashboard_server.update_position(symbol, sample_position)

        assert symbol in dashboard_server.positions
        position = dashboard_server.positions[symbol]
        assert position.symbol == symbol
        assert position.quantity == 100
        assert position.market_value == 2000.0

    def test_alert_creation(self, dashboard_server):
        """Test alert creation and thresholds."""
        # Create metrics that should trigger alerts
        metrics_data = {
            'portfolio_value': 10000.0,
            'max_drawdown': 0.25,  # Above critical threshold
            'volatility': 0.15,
            'beta': 1.0,
            'positions_count': 5,
            'cash_available': 2000.0,
            'margin_used': 0.95,  # Above critical threshold
            'unrealized_pnl': 150.0,
            'daily_pnl': 75.0,
        }

        # Update metrics (should create alerts)
        dashboard_server.update_risk_metrics(metrics_data)

        # Check that alerts were created
        assert len(dashboard_server.alerts) > 0

        # Find critical alerts
        critical_alerts = [a for a in dashboard_server.alerts if a.severity == 'critical']
        assert len(critical_alerts) > 0

class TestWebSocketFunctionality:
    """Test WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, dashboard_server):
        """Test WebSocket connection and basic communication."""
        # This would require running the server in a separate thread
        # For now, we test the connection manager directly

        manager = dashboard_server.connection_manager

        # Simulate connection
        client_id = "test_client"
        manager.connection_metadata[client_id] = {
            'connected_at': 1234567890,
            'last_ping': 1234567890,
            'subscriptions': set()
        }

        # Test subscription management
        manager.add_subscription(client_id, 'risk_metrics')
        assert 'risk_metrics' in manager.connection_metadata[client_id]['subscriptions']

        manager.remove_subscription(client_id, 'risk_metrics')
        assert 'risk_metrics' not in manager.connection_metadata[client_id]['subscriptions']

    def test_message_handling(self, dashboard_server):
        """Test WebSocket message handling."""
        # Test client message handling
        manager = dashboard_server.connection_manager
        client_id = "test_client"

        # Initialize client
        manager.connection_metadata[client_id] = {
            'connected_at': 1234567890,
            'last_ping': 1234567890,
            'subscriptions': set()
        }

        # This would test the actual message handling
        # In a real test, we'd send messages through a WebSocket connection

class TestPerformance:
    """Test performance characteristics."""

    def test_large_position_update(self, dashboard_server):
        """Test handling many position updates."""
        # Create many positions
        symbols = [f"STOCK{i}" for i in range(100)]

        for symbol in symbols:
            position_data = {
                'quantity': 100,
                'market_value': 2000.0,
                'unrealized_pnl': 50.0,
                'entry_price': 150.0,
                'current_price': 155.0,
                'weight': 0.01,
            }
            dashboard_server.update_position(symbol, position_data)

        assert len(dashboard_server.positions) == 100

    def test_historical_data_limit(self, dashboard_server):
        """Test that historical data is limited to prevent memory issues."""
        # Add many data points
        for i in range(1500):  # More than the 1000 limit
            metrics_data = {
                'portfolio_value': 10000.0 + i,
                'max_drawdown': 0.05,
                'volatility': 0.15,
                'beta': 1.0,
                'positions_count': 5,
                'cash_available': 2000.0,
                'margin_used': 0.3,
                'unrealized_pnl': 150.0,
                'daily_pnl': 75.0,
            }
            dashboard_server.update_risk_metrics(metrics_data)

        # Check that historical data is limited
        assert len(dashboard_server.historical_data['portfolio_values']) <= 1000

class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_metrics_data(self, dashboard_server):
        """Test handling of invalid metrics data."""
        # Test with missing data
        try:
            dashboard_server.update_risk_metrics({})
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Should handle missing data gracefully: {e}")

    def test_invalid_position_data(self, dashboard_server):
        """Test handling of invalid position data."""
        try:
            dashboard_server.update_position("INVALID", {})
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Should handle invalid position data gracefully: {e}")

def run_integration_test():
    """Run a full integration test with both server and client."""
    print("Running integration test...")

    # This would start the server and test with a real client
    # For now, we'll just print a message
    print("Integration test would start server and test with WebSocket client")
    print("✅ Integration test placeholder completed")

if __name__ == "__main__":
    # Run tests
    print("🧪 Running Gary×Taleb Dashboard Tests")

    # Run unit tests
    pytest.main([__file__, "-v"])

    # Run integration test
    run_integration_test()

    print("🎉 All tests completed!")