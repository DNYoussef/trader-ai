"""
Test circuit breaker integration with trade execution.

Verifies that circuit breakers properly block trades when triggered.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal
from src.trading.trade_executor import TradeExecutor


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with trade executor."""

    @pytest.fixture
    def mock_circuit_manager(self):
        """Create mock circuit manager."""
        manager = Mock()
        manager.get_system_status = Mock()
        return manager

    @pytest.fixture
    def trade_executor(self, mock_circuit_manager):
        """Create trade executor with mock dependencies."""
        broker = AsyncMock()
        broker.is_connected = True
        broker.is_market_open = AsyncMock(return_value=True)
        broker.get_buying_power = AsyncMock(return_value=Decimal("1000"))
        broker.submit_order = AsyncMock()

        portfolio = AsyncMock()
        portfolio.get_total_portfolio_value = AsyncMock(return_value=Decimal("1000"))
        portfolio.positions = AsyncMock()
        portfolio.positions.get = AsyncMock(return_value=None)
        portfolio.record_transaction = AsyncMock()

        market_data = AsyncMock()
        market_data.get_current_price = AsyncMock(return_value=100.0)

        executor = TradeExecutor(broker, portfolio, market_data)
        executor.circuit_manager = mock_circuit_manager

        return executor

    @pytest.mark.asyncio
    async def test_buy_order_blocked_when_circuit_breaker_open(self, trade_executor, mock_circuit_manager):
        """Test that buy orders are blocked when circuit breaker is open."""
        # Configure circuit manager to return open breaker
        mock_circuit_manager.get_system_status.return_value = {
            'open_breakers': 1,
            'circuit_breakers': {
                'trading_loss': {
                    'state': 'open',
                    'reason': 'Loss limit exceeded'
                }
            }
        }

        # Attempt to place buy order
        result = await trade_executor.buy_market_order('SPY', Decimal('100'), 'TEST')

        # Verify trade was blocked
        assert result.status == 'error'
        assert 'Circuit breakers active' in result.broker_response.get('error', '')

    @pytest.mark.asyncio
    async def test_sell_order_blocked_when_circuit_breaker_open(self, trade_executor, mock_circuit_manager):
        """Test that sell orders are blocked when circuit breaker is open."""
        # Configure circuit manager to return open breaker
        mock_circuit_manager.get_system_status.return_value = {
            'open_breakers': 1,
            'circuit_breakers': {
                'trading_loss': {
                    'state': 'open',
                    'reason': 'Loss limit exceeded'
                }
            }
        }

        # Mock position for sell
        trade_executor.broker.get_position = AsyncMock(return_value=Mock(qty=10, market_value=Decimal('1000')))

        # Attempt to place sell order
        result = await trade_executor.sell_market_order('SPY', Decimal('100'), 'TEST')

        # Verify trade was blocked
        assert result.status == 'error'
        assert 'Circuit breakers active' in result.broker_response.get('error', '')

    @pytest.mark.asyncio
    async def test_trade_allowed_when_circuit_breaker_closed(self, trade_executor, mock_circuit_manager):
        """Test that trades proceed normally when circuit breakers are closed."""
        # Configure circuit manager to return closed breakers
        mock_circuit_manager.get_system_status.return_value = {
            'open_breakers': 0,
            'circuit_breakers': {
                'trading_loss': {
                    'state': 'closed',
                    'reason': None
                }
            }
        }

        # Mock successful order submission
        mock_order = Mock()
        mock_order.id = 'ORDER123'
        mock_order.qty = Decimal('1')
        mock_order.filled_qty = Decimal('1')
        mock_order.filled_avg_price = Decimal('100')
        mock_order.status = Mock(value='filled')
        mock_order.submitted_at = None

        trade_executor.broker.submit_order.return_value = mock_order

        # Attempt to place buy order
        result = await trade_executor.buy_market_order('SPY', Decimal('100'), 'TEST')

        # Verify trade was allowed
        assert result.status == 'filled'
        assert result.order_id == 'ORDER123'
        assert trade_executor.broker.submit_order.called

    @pytest.mark.asyncio
    async def test_specific_trading_loss_breaker_blocks_trade(self, trade_executor, mock_circuit_manager):
        """Test that specific trading loss breaker blocks trades."""
        # Configure only trading_loss breaker as open
        mock_circuit_manager.get_system_status.return_value = {
            'open_breakers': 0,  # No general breakers
            'circuit_breakers': {
                'trading_loss': {
                    'state': 'open',
                    'reason': 'Daily loss limit of -5% exceeded'
                }
            }
        }

        # Attempt to place buy order
        result = await trade_executor.buy_market_order('SPY', Decimal('100'), 'TEST')

        # Verify trade was blocked by specific breaker
        assert result.status == 'error'
        assert 'Loss limit circuit breaker OPEN' in result.broker_response.get('error', '')

    @pytest.mark.asyncio
    async def test_trade_proceeds_without_circuit_manager(self, mock_circuit_manager):
        """Test that trades proceed normally if circuit_manager is not set."""
        # Create executor without circuit_manager
        broker = AsyncMock()
        broker.is_connected = True
        broker.is_market_open = AsyncMock(return_value=True)
        broker.get_buying_power = AsyncMock(return_value=Decimal("1000"))

        portfolio = AsyncMock()
        portfolio.get_total_portfolio_value = AsyncMock(return_value=Decimal("1000"))
        portfolio.positions = AsyncMock()
        portfolio.positions.get = AsyncMock(return_value=None)
        portfolio.record_transaction = AsyncMock()

        market_data = AsyncMock()
        market_data.get_current_price = AsyncMock(return_value=100.0)

        executor = TradeExecutor(broker, portfolio, market_data)
        # Note: No circuit_manager assigned

        # Mock successful order submission
        mock_order = Mock()
        mock_order.id = 'ORDER123'
        mock_order.qty = Decimal('1')
        mock_order.filled_qty = Decimal('1')
        mock_order.filled_avg_price = Decimal('100')
        mock_order.status = Mock(value='filled')
        mock_order.submitted_at = None

        executor.broker.submit_order.return_value = mock_order

        # Trade should proceed normally without circuit manager
        result = await executor.buy_market_order('SPY', Decimal('100'), 'TEST')

        # Verify trade was allowed
        assert result.status == 'filled'
        assert result.order_id == 'ORDER123'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
