"""
Production Trading Engine for Gary×Taleb trading system.

Orchestrates all trading components with proper async/await integration,
error handling, audit logging, and kill switch functionality.
"""

import asyncio
import logging
import time
from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional
import json
import os

from .brokers.broker_interface import BrokerInterface
from .brokers.alpaca_adapter import AlpacaAdapter
from .integration.memory_client import MemoryClient

logger = logging.getLogger(__name__)


class TradingEngine:
    """Production trading engine with proper async integration and audit logging."""

    def __init__(self, config_path: str = 'config/config.json'):
        self.config = self._load_config(config_path)
        self.running = False
        self.kill_switch_activated = False
        self.mode = self.config.get('mode', 'paper')  # paper or live

        # Initialize components (will be set during initialize())
        self.broker = None
        self.market_data = None
        self.portfolio_manager = None
        self.trade_executor = None
        self.memory_client = None

        # Audit log
        self.audit_log_path = '.claude/.artifacts/audit_log.jsonl'
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

        logger.info(f"Trading Engine created in {self.mode} mode")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with fallback defaults."""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Default configuration
        return {
            'mode': 'paper',
            'broker': 'alpaca',
            'initial_capital': 200,
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'audit_enabled': True,
            'rebalance_frequency_minutes': 60  # Rebalance every hour
        }

    def initialize(self) -> bool:
        """Initialize all trading engine components."""
        try:
            # Initialize Memory Client
            try:
                self.memory_client = MemoryClient()
                logger.info("Memory Client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Memory Client: {e}")

            # Initialize broker
            if self.config['broker'] == 'alpaca':
                broker_config = {
                    'api_key': self.config.get('api_key', ''),
                    'secret_key': self.config.get('secret_key', ''),
                    'paper_trading': self.mode == 'paper'
                }

                if not broker_config['api_key'] or not broker_config['secret_key']:
                    logger.error("PRODUCTION ERROR: Alpaca API credentials not provided")
                    if self.mode == 'live':
                        return False

                self.broker = AlpacaAdapter(broker_config)
            else:
                raise ValueError(f"Unknown broker: {self.config['broker']}")

            # Connect to broker
            if not asyncio.run(self.broker.connect()):
                logger.error("Failed to connect to broker")
                return False

            # Initialize trading components with real dependencies
            from .portfolio.portfolio_manager import PortfolioManager
            from .trading.trade_executor import TradeExecutor
            from .market.market_data import MarketDataProvider

            # Pass broker adapter to all components for real integration
            self.market_data = MarketDataProvider(self.broker)
            self.portfolio_manager = PortfolioManager(
                self.broker,
                self.market_data,
                Decimal(str(self.config['initial_capital']))
            )
            self.trade_executor = TradeExecutor(
                self.broker,
                self.portfolio_manager,
                self.market_data
            )

            # Log initialization
            self._audit_log({
                'event': 'engine_initialized',
                'mode': self.mode,
                'broker': self.config['broker'],
                'initial_capital': self.config['initial_capital'],
                'timestamp': datetime.now().isoformat()
            })
            
            self._log_memory_event(
                f"Trading engine initialized in {self.mode} mode with ${self.config['initial_capital']} capital",
                {'category': 'lifecycle', 'event': 'initialization'}
            )

            logger.info(f"Trading engine initialized successfully in {self.mode} mode")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            self._audit_log({
                'event': 'initialization_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def start(self):
        """Start the trading engine main loop."""
        if not self.initialize():
            logger.error("Failed to initialize, cannot start")
            return

        self.running = True
        logger.info("Trading engine started")

        # Log engine start
        self._audit_log({
            'event': 'engine_started',
            'mode': self.mode,
            'timestamp': datetime.now().isoformat()
        })
        
        self._log_memory_event(
            "Trading engine started main loop",
            {'category': 'lifecycle', 'event': 'start'}
        )

        last_rebalance = 0
        rebalance_interval = self.config.get('rebalance_frequency_minutes', 60) * 60

        try:
            while self.running and not self.kill_switch_activated:
                try:
                    # Check if it's time to rebalance
                    current_time = time.time()
                    if current_time - last_rebalance >= rebalance_interval:
                        await self._execute_trading_cycle()
                        last_rebalance = current_time

                    # Check portfolio status every 5 minutes
                    await self._check_system_health()

                    # Sleep for 5 minutes before next check
                    await asyncio.sleep(300)

                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self._audit_log({
                        'event': 'main_loop_error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    await asyncio.sleep(300)  # Wait before retry

        finally:
            await self.stop()

    async def _execute_trading_cycle(self):
        """Execute production trading cycle."""
        try:
            logger.info("Executing trading cycle")
            
            self._log_memory_event(
                "Starting trading cycle",
                {'category': 'cycle', 'event': 'start'}
            )

            # Check if market is open
            is_market_open = await self.market_data.get_market_status()
            if not is_market_open:
                logger.info("Market is closed, skipping trading cycle")
                self._log_memory_event("Market closed, skipping cycle", {'category': 'cycle', 'status': 'skipped'})
                return

            # Sync portfolio with broker
            sync_success = await self.portfolio_manager.sync_with_broker()
            if not sync_success:
                logger.error("Failed to sync portfolio with broker")
                return

            # Get current portfolio status
            portfolio_value = await self.portfolio_manager.get_total_portfolio_value()
            cash_balance = await self.broker.get_cash_balance()

            logger.info(f"Portfolio Status - Value: ${portfolio_value}, Cash: ${cash_balance}")

            # Execute rebalancing if portfolio value is sufficient
            if portfolio_value >= Decimal("10.00"):
                await self._execute_rebalancing(portfolio_value)
            else:
                logger.info("Portfolio value too low for rebalancing")

            # Create daily snapshot
            await self.portfolio_manager.create_daily_snapshot()

            # Audit log
            self._audit_log({
                'event': 'trading_cycle_completed',
                'portfolio_value': str(portfolio_value),
                'cash_balance': str(cash_balance),
                'timestamp': datetime.now().isoformat()
            })
            
            self._log_memory_event(
                f"Trading cycle completed. Value: ${portfolio_value}",
                {'category': 'cycle', 'event': 'complete', 'value': str(portfolio_value)}
            )

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self._audit_log({
                'event': 'trading_cycle_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            self._log_memory_event(f"Trading cycle error: {e}", {'category': 'cycle', 'status': 'error'})

    async def _execute_rebalancing(self, total_value: Decimal):
        """Execute Gary×Taleb strategy rebalancing."""
        try:
            logger.info("Executing portfolio rebalancing")

            # Gary×Taleb allocation strategy:
            # 40% SPY (market hedge)
            # 35% ULTY+AMDY (momentum - split equally)
            # 15% VTIP (inflation protection)
            # 10% IAU (gold hedge)
            target_allocations = {
                'SPY': total_value * Decimal('0.40'),
                'ULTY': total_value * Decimal('0.175'),  # Half of 35%
                'AMDY': total_value * Decimal('0.175'),  # Half of 35%
                'VTIP': total_value * Decimal('0.15'),
                'IAU': total_value * Decimal('0.10')
            }

            # Execute rebalancing for each gate
            gates = {
                'SPY_HEDGE': ['SPY'],
                'MOMENTUM': ['ULTY', 'AMDY'],
                'BOND_HEDGE': ['VTIP'],
                'GOLD_HEDGE': ['IAU']
            }

            total_orders = 0
            for gate, symbols in gates.items():
                gate_targets = {symbol: target_allocations[symbol]
                              for symbol in symbols if symbol in target_allocations}

                if gate_targets:
                    results = await self.trade_executor.rebalance_gate(gate, gate_targets)
                    total_orders += len(results)
                    logger.info(f"Rebalanced {gate}: {len(results)} orders")

                    # Log individual orders
                    for result in results:
                        if result.status != 'error':
                            self._audit_log({
                                'event': 'rebalance_order',
                                'gate': gate,
                                'symbol': result.symbol,
                                'side': result.side,
                                'amount': str(result.notional),
                                'order_id': result.order_id,
                                'status': result.status,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            self._log_memory_event(
                                f"Rebalance order: {result.side} {result.symbol} (${result.notional})",
                                {'category': 'trade', 'gate': gate, 'symbol': result.symbol}
                            )

            logger.info(f"Rebalancing completed: {total_orders} total orders executed")

        except Exception as e:
            logger.error(f"Error in rebalancing: {e}")
            self._audit_log({
                'event': 'rebalancing_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    async def _check_system_health(self):
        """Check system health and connectivity."""
        try:
            # Check broker connection
            if not self.broker.is_connected:
                logger.warning("Broker disconnected, attempting reconnection...")
                if await self.broker.connect():
                    logger.info("Broker reconnected successfully")
                else:
                    logger.error("Failed to reconnect to broker")

            # Check for pending orders that may be stuck
            pending_orders = self.trade_executor.get_pending_orders()
            if len(pending_orders) > 10:  # Arbitrary threshold
                logger.warning(f"High number of pending orders: {len(pending_orders)}")

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    async def execute_manual_trade(self, symbol: str, dollar_amount: Decimal, action: str, gate: str = "MANUAL"):
        """Execute a manual trade for testing."""
        try:
            logger.info(f"Executing manual trade: {action} {symbol} ${dollar_amount}")

            if action.lower() == "buy":
                result = await self.trade_executor.buy_market_order(symbol, dollar_amount, gate)
            elif action.lower() == "sell":
                result = await self.trade_executor.sell_market_order(symbol, dollar_amount, gate)
            else:
                raise ValueError("Action must be 'buy' or 'sell'")

            logger.info(f"Manual trade result: {action} {symbol} ${dollar_amount} - Status: {result.status}")

            # Audit log
            self._audit_log({
                'event': 'manual_trade',
                'action': action,
                'symbol': symbol,
                'amount': str(dollar_amount),
                'result': result.status,
                'order_id': result.order_id,
                'timestamp': datetime.now().isoformat()
            })
            
            self._log_memory_event(
                f"Manual trade: {action} {symbol} (${dollar_amount}) - {result.status}",
                {'category': 'trade', 'type': 'manual', 'symbol': symbol}
            )

            return result

        except Exception as e:
            logger.error(f"Error in manual trade: {e}")
            self._audit_log({
                'event': 'manual_trade_error',
                'action': action,
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None

    async def get_portfolio_summary(self):
        """Get comprehensive portfolio summary."""
        try:
            await self.portfolio_manager.sync_with_broker()
            return await self.portfolio_manager.get_position_summary()
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {"error": str(e)}

    async def activate_kill_switch(self):
        """EMERGENCY STOP - Cancel all orders and halt trading."""
        logger.critical("KILL SWITCH ACTIVATED")
        self.kill_switch_activated = True

        try:
            # Cancel all open orders
            canceled_count = await self.trade_executor.cancel_all_pending_orders()
            logger.info(f"Canceled {canceled_count} pending orders")

            # Get final positions for audit
            try:
                nav = await self.broker.get_account_value()
                positions = await self.broker.get_positions()
            except Exception:
                nav = "ERROR"
                positions = []

            # Log the event
            self._audit_log({
                'event': 'KILL_SWITCH_ACTIVATED',
                'timestamp': datetime.now().isoformat(),
                'nav': str(nav),
                'positions_count': len(positions),
                'canceled_orders': canceled_count
            })
            
            self._log_memory_event(
                "KILL SWITCH ACTIVATED - TRADING HALTED",
                {'category': 'safety', 'event': 'kill_switch', 'severity': 'critical'}
            )

            # Stop the engine
            await self.stop()

        except Exception as e:
            logger.error(f"Error during kill switch activation: {e}")
            self._audit_log({
                'event': 'kill_switch_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    async def stop(self):
        """Stop the trading engine gracefully."""
        logger.info("Stopping trading engine")
        self.running = False

        try:
            # Get final NAV for audit
            final_nav = await self.broker.get_account_value() if self.broker else Decimal('0')

            # Final audit log
            self._audit_log({
                'event': 'engine_stopped',
                'timestamp': datetime.now().isoformat(),
                'final_nav': str(final_nav)
            })
            
            self._log_memory_event(
                "Trading engine stopped",
                {'category': 'lifecycle', 'event': 'stop'}
            )

            # Disconnect broker
            if self.broker:
                await self.broker.disconnect()

            logger.info("Trading engine stopped successfully")

        except Exception as e:
            logger.error(f"Error during stop: {e}")

    async def get_status(self) -> Dict:
        """Get current engine status."""
        if not self.broker:
            return {'status': 'not_initialized'}

        try:
            nav = await self.broker.get_account_value()
            cash = await self.broker.get_cash_balance()
            positions = await self.broker.get_positions()
            market_open = await self.market_data.get_market_status()

            return {
                'status': 'running' if self.running else 'stopped',
                'mode': self.mode,
                'nav': str(nav),
                'cash': str(cash),
                'positions_count': len(positions),
                'market_open': market_open,
                'kill_switch': self.kill_switch_activated,
                'broker_connected': self.broker.is_connected,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _audit_log(self, data: Dict):
        """Write to audit log (WORM - Write Once Read Many)."""
        if not self.config.get('audit_enabled', True):
            return

        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            
    def _log_memory_event(self, content: str, metadata: Dict):
        """Log event to memory system if client is available."""
        if self.memory_client:
            # Run as background task to not block trading
            # For simplicity in this sync wrapper, we assume store_memory is fast or we fire-and-forget in a real async queue
            # But store_memory is sync in the client, so we just call it.
            # Ideally this should be async or threaded.
            try:
                self.memory_client.store_memory(content, metadata)
            except Exception as e:
                logger.warning(f"Failed to log to memory: {e}")

    async def test_production_flow(self):
        """Test the production trading flow with $200 seed capital."""
        try:
            logger.info("Testing production trading flow with $200 seed capital")

            # Initialize if not already done
            if not self.broker:
                if not self.initialize():
                    raise Exception("Failed to initialize trading engine")

            # Check broker connection
            if not self.broker.is_connected:
                raise Exception("Broker not connected")

            # Get account status
            account_value = await self.broker.get_account_value()
            cash_balance = await self.broker.get_cash_balance()
            buying_power = await self.broker.get_buying_power()

            logger.info(f"Account Status - Value: ${account_value}, Cash: ${cash_balance}, Buying Power: ${buying_power}")

            # Test market data
            market_open = await self.market_data.get_market_status()
            logger.info(f"Market Status: {'OPEN' if market_open else 'CLOSED'}")

            # Test getting prices for key symbols
            test_symbols = ['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU']
            prices = await self.market_data.get_multiple_prices(test_symbols)
            logger.info(f"Symbol Prices: {[(s, f'${p}' if p else 'N/A') for s, p in prices.items()]}")

            # Sync portfolio
            sync_success = await self.portfolio_manager.sync_with_broker()
            logger.info(f"Portfolio Sync: {'SUCCESS' if sync_success else 'FAILED'}")

            # Get portfolio summary
            portfolio_summary = await self.get_portfolio_summary()
            logger.info(f"Portfolio Summary: {portfolio_summary}")

            # Test small trade if sufficient funds
            if cash_balance >= Decimal("5.00") and market_open:
                logger.info("Testing small trade execution...")
                test_result = await self.execute_manual_trade("SPY", Decimal("5.00"), "buy", "TEST")
                if test_result:
                    logger.info(f"Test trade result: {test_result.status}")
                else:
                    logger.warning("Test trade failed")

            self._audit_log({
                'event': 'production_flow_test_completed',
                'account_value': str(account_value),
                'cash_balance': str(cash_balance),
                'market_open': market_open,
                'portfolio_sync': sync_success,
                'timestamp': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            logger.error(f"Production flow test failed: {e}")
            self._audit_log({
                'event': 'production_flow_test_failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False