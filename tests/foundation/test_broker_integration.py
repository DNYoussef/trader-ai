"""
Comprehensive tests for broker integration in Foundation phase.
Tests connection management, order processing, position tracking, and error handling.
"""
import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import mock objects
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mocks.mock_broker import (
    MockBroker, MockOrder, MockPosition, MockBrokerError,
    OrderStatus, OrderType, create_mock_broker,
    create_test_position, create_test_order
)


class TestBrokerConnection:
    """Test broker connection management"""
    
    def test_successful_connection(self):
        """Test successful broker connection"""
        broker = create_mock_broker(connection_delay=0.01)
        
        assert not broker.is_connected()
        result = broker.connect()
        
        assert result is True
        assert broker.is_connected()
        assert broker.connection_count == 1
        assert broker.last_connection_time is not None

    def test_connection_with_delay(self):
        """Test connection with realistic delay"""
        broker = create_mock_broker(connection_delay=0.1)
        
        start_time = time.time()
        broker.connect()
        end_time = time.time()
        
        assert broker.is_connected()
        assert (end_time - start_time) >= 0.1

    def test_connection_failure(self):
        """Test connection failure handling"""
        broker = create_mock_broker(error_rate=1.0)  # Force errors
        
        with pytest.raises(MockBrokerError) as exc_info:
            broker.connect()
            
        assert exc_info.value.error_code == 1001
        assert not broker.is_connected()

    def test_disconnect(self):
        """Test broker disconnection"""
        broker = create_mock_broker()
        broker.connect()
        
        assert broker.is_connected()
        result = broker.disconnect()
        
        assert result is True
        assert not broker.is_connected()

    def test_multiple_connections(self):
        """Test multiple connection attempts"""
        broker = create_mock_broker()
        
        # First connection
        broker.connect()
        assert broker.connection_count == 1
        
        # Second connection (should increment counter)
        broker.disconnect()
        broker.connect()
        assert broker.connection_count == 2

    def test_connection_callback(self):
        """Test connection state change callbacks"""
        broker = create_mock_broker()
        callback_results = []
        
        def connection_callback(connected: bool):
            callback_results.append(connected)
            
        broker.on_connection_change = connection_callback
        
        broker.connect()
        broker.disconnect()
        
        assert callback_results == [True, False]


class TestOrderManagement:
    """Test order placement, tracking, and management"""
    
    @pytest.fixture
    def connected_broker(self):
        """Fixture providing connected broker"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        return broker

    def test_market_buy_order(self, connected_broker):
        """Test placing market buy order"""
        order = connected_broker.place_order("AAPL", 100, OrderType.MARKET)
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.id in connected_broker.orders

    def test_market_sell_order(self, connected_broker):
        """Test placing market sell order"""
        # First create a position
        connected_broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)  # Wait for order to fill
        
        # Then sell
        sell_order = connected_broker.place_order("AAPL", -50, OrderType.MARKET)
        
        assert sell_order.quantity == -50
        assert sell_order.status == OrderStatus.PENDING

    def test_limit_order(self, connected_broker):
        """Test placing limit order"""
        limit_price = 150.0
        order = connected_broker.place_order(
            "AAPL", 100, OrderType.LIMIT, limit_price=limit_price
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == limit_price

    def test_order_without_connection(self):
        """Test placing order without connection"""
        broker = create_mock_broker()
        
        with pytest.raises(MockBrokerError) as exc_info:
            broker.place_order("AAPL", 100)
            
        assert exc_info.value.error_code == 1002

    def test_invalid_quantity_order(self, connected_broker):
        """Test placing order with invalid quantity"""
        with pytest.raises(MockBrokerError) as exc_info:
            connected_broker.place_order("AAPL", 0)
            
        assert exc_info.value.error_code == 2001

    def test_insufficient_buying_power(self, connected_broker):
        """Test order with insufficient buying power"""
        # Try to buy more than available buying power
        expensive_quantity = 10000  # Should exceed buying power
        
        with pytest.raises(MockBrokerError) as exc_info:
            connected_broker.place_order("AAPL", expensive_quantity)
            
        assert exc_info.value.error_code == 2002

    def test_order_filling(self, connected_broker):
        """Test order filling process"""
        order = connected_broker.place_order("AAPL", 100, OrderType.MARKET)
        
        # Wait for order to fill
        time.sleep(0.05)
        
        updated_order = connected_broker.get_order(order.id)
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == 100
        assert updated_order.avg_fill_price > 0

    def test_partial_fills(self):
        """Test partial order filling"""
        broker = create_mock_broker(simulate_partial_fills=True, order_fill_delay=0.01)
        broker.connect()
        
        order = broker.place_order("AAPL", 200, OrderType.MARKET)
        time.sleep(0.05)
        
        updated_order = broker.get_order(order.id)
        assert updated_order.status == OrderStatus.PARTIAL
        assert 0 < updated_order.filled_quantity < 200

    def test_order_cancellation(self, connected_broker):
        """Test order cancellation"""
        # Place order but don't let it fill immediately
        broker = create_mock_broker(order_fill_delay=1.0)  # Long delay
        broker.connect()
        
        order = broker.place_order("AAPL", 100, OrderType.LIMIT, limit_price=50.0)
        
        # Cancel before fill
        result = broker.cancel_order(order.id)
        assert result is True
        
        updated_order = broker.get_order(order.id)
        assert updated_order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self, connected_broker):
        """Test cancelling non-existent order"""
        fake_order_id = "fake-order-id"
        
        with pytest.raises(MockBrokerError) as exc_info:
            connected_broker.cancel_order(fake_order_id)
            
        assert exc_info.value.error_code == 2003

    def test_cancel_filled_order(self, connected_broker):
        """Test cancelling already filled order"""
        order = connected_broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)  # Let it fill
        
        with pytest.raises(MockBrokerError) as exc_info:
            connected_broker.cancel_order(order.id)
            
        assert exc_info.value.error_code == 2004

    def test_get_orders_by_symbol(self, connected_broker):
        """Test retrieving orders filtered by symbol"""
        # Place orders for different symbols
        order1 = connected_broker.place_order("AAPL", 100)
        order2 = connected_broker.place_order("GOOGL", 50)
        order3 = connected_broker.place_order("AAPL", -25)
        
        aapl_orders = connected_broker.get_orders("AAPL")
        assert len(aapl_orders) == 2
        assert all(o.symbol == "AAPL" for o in aapl_orders)

    def test_order_callbacks(self, connected_broker):
        """Test order update callbacks"""
        callback_results = []
        
        def order_callback(order):
            callback_results.append(order.status)
            
        connected_broker.on_order_update = order_callback
        
        order = connected_broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        assert OrderStatus.FILLED in callback_results


class TestPositionTracking:
    """Test position tracking and management"""
    
    @pytest.fixture
    def broker_with_position(self):
        """Fixture providing broker with established position"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)  # Wait for fill
        return broker

    def test_position_creation(self, broker_with_position):
        """Test position creation from order fill"""
        position = broker_with_position.get_position("AAPL")
        
        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_cost > 0

    def test_position_accumulation(self, broker_with_position):
        """Test accumulating position with multiple orders"""
        # Add to existing position
        broker_with_position.place_order("AAPL", 50, OrderType.MARKET)
        time.sleep(0.05)
        
        position = broker_with_position.get_position("AAPL")
        assert position.quantity == 150

    def test_position_reduction(self, broker_with_position):
        """Test reducing position with sell order"""
        # Reduce position
        broker_with_position.place_order("AAPL", -30, OrderType.MARKET)
        time.sleep(0.05)
        
        position = broker_with_position.get_position("AAPL")
        assert position.quantity == 70

    def test_position_closure(self, broker_with_position):
        """Test closing position completely"""
        # Close entire position
        broker_with_position.place_order("AAPL", -100, OrderType.MARKET)
        time.sleep(0.05)
        
        position = broker_with_position.get_position("AAPL")
        assert position is None  # Position should be removed

    def test_position_reversal(self, broker_with_position):
        """Test reversing position (long to short)"""
        # Sell more than current position
        broker_with_position.place_order("AAPL", -150, OrderType.MARKET)
        time.sleep(0.05)
        
        position = broker_with_position.get_position("AAPL")
        assert position.quantity == -50  # Now short 50 shares

    def test_multiple_positions(self):
        """Test managing multiple positions"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        # Create positions in different symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            broker.place_order(symbol, 100, OrderType.MARKET)
            
        time.sleep(0.1)
        
        positions = broker.get_positions()
        assert len(positions) == 3
        assert set(p.symbol for p in positions) == set(symbols)

    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        # Set initial price
        broker.set_market_price("AAPL", 100.0)
        
        # Buy at current price
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        # Change market price
        broker.set_market_price("AAPL", 110.0)
        
        position = broker.get_position("AAPL")
        assert position.current_price == 110.0
        assert position.unrealized_pnl == 1000.0  # (110-100) * 100

    def test_position_callbacks(self):
        """Test position update callbacks"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        callback_results = []
        
        def position_callback(position):
            callback_results.append(position.symbol)
            
        broker.on_position_update = position_callback
        
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        assert "AAPL" in callback_results


class TestAccountManagement:
    """Test account balance and buying power management"""
    
    def test_initial_account_state(self):
        """Test initial account balance and buying power"""
        broker = create_mock_broker()
        
        assert broker.get_account_balance() == 100000.0
        assert broker.get_buying_power() == 100000.0

    def test_account_balance_after_trade(self):
        """Test account balance changes after trades"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        initial_balance = broker.get_account_balance()
        
        # Set known price for predictable calculation
        broker.set_market_price("AAPL", 100.0)
        
        # Buy 100 shares at $100 each
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        # Balance should decrease by trade cost + commission
        expected_cost = 100 * 100 + 1.0  # price * quantity + commission
        expected_balance = initial_balance - expected_cost
        
        assert abs(broker.get_account_balance() - expected_balance) < 1.0

    def test_buying_power_management(self):
        """Test buying power constraints"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        # Set high price to test buying power limits
        broker.set_market_price("EXPENSIVE", 1000.0)
        
        # This should work
        broker.place_order("EXPENSIVE", 50, OrderType.MARKET)
        time.sleep(0.05)
        
        # This should fail due to insufficient buying power
        with pytest.raises(MockBrokerError):
            broker.place_order("EXPENSIVE", 100, OrderType.MARKET)


class TestMarketData:
    """Test market data functionality"""
    
    def test_current_price_retrieval(self):
        """Test getting current market prices"""
        broker = create_mock_broker()
        
        # Price should be auto-generated if not set
        price1 = broker.get_current_price("AAPL")
        assert price1 > 0
        
        # Should return consistent price for same symbol
        price2 = broker.get_current_price("AAPL")
        assert price1 == price2

    def test_price_setting(self):
        """Test manually setting market prices"""
        broker = create_mock_broker()
        
        test_price = 150.0
        broker.set_market_price("AAPL", test_price)
        
        assert broker.get_current_price("AAPL") == test_price

    def test_price_volatility_simulation(self):
        """Test price volatility simulation"""
        broker = create_mock_broker()
        broker.price_volatility = 0.1  # 10% volatility
        
        base_price = 100.0
        broker.set_market_price("AAPL", base_price)
        
        # Get multiple price samples
        prices = [broker.get_current_price("AAPL") for _ in range(10)]
        
        # Prices should vary but stay within reasonable bounds
        min_price = min(prices)
        max_price = max(prices)
        
        assert min_price >= base_price * 0.8  # Within volatility range
        assert max_price <= base_price * 1.2


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_error_rate_simulation(self):
        """Test configurable error rate"""
        broker = create_mock_broker(error_rate=0.5, order_fill_delay=0.01)
        broker.connect()
        
        # Place multiple orders and expect some to fail
        failed_count = 0
        total_orders = 10
        
        for i in range(total_orders):
            try:
                order = broker.place_order("AAPL", 10, OrderType.MARKET)
                time.sleep(0.05)
                
                if broker.get_order(order.id).status == OrderStatus.REJECTED:
                    failed_count += 1
            except MockBrokerError:
                failed_count += 1
                
        # Some orders should have failed
        assert failed_count > 0

    def test_concurrent_operations(self):
        """Test thread safety with concurrent operations"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        def place_orders():
            for i in range(5):
                try:
                    broker.place_order("AAPL", 10, OrderType.MARKET)
                except MockBrokerError:
                    pass  # Expected with concurrent access
                    
        # Start multiple threads
        threads = [threading.Thread(target=place_orders) for _ in range(3)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Broker should still be in consistent state
        assert broker.is_connected()
        assert len(broker.orders) <= 15  # Maximum possible orders

    def test_reset_functionality(self):
        """Test broker reset for testing"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        # Create some state
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        assert len(broker.orders) > 0
        assert len(broker.positions) > 0
        
        # Reset
        broker.reset_for_testing()
        
        assert len(broker.orders) == 0
        assert len(broker.positions) == 0
        assert not broker.is_connected()
        assert broker.get_account_balance() == 100000.0


class TestPerformanceMetrics:
    """Test performance tracking and metrics"""
    
    def test_trade_counting(self):
        """Test trade performance counting"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        initial_trades = broker.total_trades
        
        # Place successful trade
        broker.place_order("AAPL", 100, OrderType.MARKET)
        time.sleep(0.05)
        
        assert broker.total_trades == initial_trades + 1
        assert broker.successful_trades > 0

    def test_connection_metrics(self):
        """Test connection tracking metrics"""
        broker = create_mock_broker()
        
        initial_count = broker.connection_count
        
        broker.connect()
        assert broker.connection_count == initial_count + 1
        
        broker.disconnect()
        broker.connect()
        assert broker.connection_count == initial_count + 2


@pytest.mark.integration
class TestBrokerIntegrationScenarios:
    """Integration test scenarios combining multiple broker features"""
    
    def test_full_trading_session(self):
        """Test complete trading session workflow"""
        broker = create_mock_broker(order_fill_delay=0.01)
        
        # 1. Connect
        broker.connect()
        assert broker.is_connected()
        
        # 2. Set market prices
        broker.set_market_price("AAPL", 150.0)
        broker.set_market_price("GOOGL", 2500.0)
        
        # 3. Place initial orders
        aapl_order = broker.place_order("AAPL", 100, OrderType.MARKET)
        googl_order = broker.place_order("GOOGL", 10, OrderType.LIMIT, limit_price=2400.0)
        
        # 4. Wait for market order to fill
        time.sleep(0.05)
        
        # 5. Verify position created
        aapl_position = broker.get_position("AAPL")
        assert aapl_position is not None
        assert aapl_position.quantity == 100
        
        # 6. Modify position
        broker.place_order("AAPL", -25, OrderType.MARKET)  # Partial sell
        time.sleep(0.05)
        
        # 7. Verify position updated
        updated_position = broker.get_position("AAPL")
        assert updated_position.quantity == 75
        
        # 8. Check account balance changed
        assert broker.get_account_balance() < 100000.0
        
        # 9. Cancel pending order
        broker.cancel_order(googl_order.id)
        assert broker.get_order(googl_order.id).status == OrderStatus.CANCELLED
        
        # 10. Disconnect
        broker.disconnect()
        assert not broker.is_connected()

    def test_risk_scenario_position_limits(self):
        """Test risk management scenario with position limits"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        
        # Set low account balance to test limits
        broker.account_balance = 5000.0
        broker.buying_power = 5000.0
        
        broker.set_market_price("AAPL", 100.0)
        
        # Should succeed
        broker.place_order("AAPL", 40, OrderType.MARKET)
        time.sleep(0.05)
        
        # Should fail due to insufficient buying power
        with pytest.raises(MockBrokerError):
            broker.place_order("AAPL", 50, OrderType.MARKET)

    def test_market_volatility_scenario(self):
        """Test trading during market volatility"""
        broker = create_mock_broker(order_fill_delay=0.01)
        broker.connect()
        broker.price_volatility = 0.05  # 5% volatility
        
        symbol = "VOLATILE_STOCK"
        broker.set_market_price(symbol, 100.0)
        
        # Place order
        broker.place_order(symbol, 100, OrderType.MARKET)
        time.sleep(0.05)
        
        # Simulate price movement
        for _ in range(5):
            new_price = broker.get_current_price(symbol)
            broker.set_market_price(symbol, new_price * 1.02)  # 2% increase
            
        position = broker.get_position(symbol)
        assert position.unrealized_pnl != 0  # Should have some P&L

    def test_high_frequency_scenario(self):
        """Test high-frequency trading scenario"""
        broker = create_mock_broker(order_fill_delay=0.001)  # Very fast fills
        broker.connect()
        
        # Place many small orders rapidly
        orders = []
        for i in range(20):
            order = broker.place_order("AAPL", 5, OrderType.MARKET)
            orders.append(order)
            
        time.sleep(0.1)  # Wait for all to fill
        
        # Check that all orders filled
        filled_orders = [
            broker.get_order(order.id) for order in orders
            if broker.get_order(order.id).status == OrderStatus.FILLED
        ]
        
        assert len(filled_orders) >= 15  # Most should fill successfully
        
        # Check final position
        position = broker.get_position("AAPL")
        assert position.quantity == len(filled_orders) * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])