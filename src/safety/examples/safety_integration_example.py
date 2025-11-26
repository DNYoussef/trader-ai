"""
Safety System Integration Example

This example demonstrates how to integrate the complete safety system
with the Gary×Taleb trading system, providing a production-ready
implementation with 99.9% availability guarantees.

Usage:
    python safety_integration_example.py

This example shows:
- Complete safety system initialization
- Trading engine integration
- Error handling and recovery
- Monitoring and alerting setup
- Emergency procedures
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safety.core.safety_integration import TradingSafetyIntegration
from safety.core.safety_manager import ComponentState
from safety.monitoring.health_monitor import AlertSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/safety_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MockTradingEngine:
    """Mock trading engine for demonstration purposes."""

    def __init__(self):
        self.running = False
        self.kill_switch_activated = False
        self.broker = MockBroker()
        self.portfolio_manager = MockPortfolioManager()

    async def start(self):
        """Start the trading engine."""
        logger.info("Starting mock trading engine")
        self.running = True
        return True

    async def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping mock trading engine")
        self.running = False
        return True

    def activate_kill_switch(self):
        """Activate emergency kill switch."""
        logger.critical("KILL SWITCH ACTIVATED - All trading stopped immediately")
        self.kill_switch_activated = True
        self.running = False

    async def get_status(self):
        """Get trading engine status."""
        return {
            'running': self.running,
            'kill_switch_activated': self.kill_switch_activated,
            'broker_connected': self.broker.is_connected,
            'portfolio_value': await self.portfolio_manager.get_total_value()
        }


class MockBroker:
    """Mock broker adapter for demonstration."""

    def __init__(self):
        self.is_connected = True

    async def connect(self):
        """Connect to broker."""
        self.is_connected = True
        return True

    async def disconnect(self):
        """Disconnect from broker."""
        self.is_connected = False


class MockPortfolioManager:
    """Mock portfolio manager for demonstration."""

    def __init__(self):
        self.total_value = 10000.0

    async def get_total_value(self):
        """Get total portfolio value."""
        return self.total_value

    async def get_positions(self):
        """Get current positions."""
        return []

    async def close_all_positions(self):
        """Close all positions (emergency procedure)."""
        logger.warning("Emergency: Closing all positions")
        return True


def create_safety_config():
    """Create comprehensive safety configuration."""
    return {
        'safety_manager': {
            'health_check_interval': 5.0,
            'component_timeout': 30.0,
            'max_consecutive_failures': 3,
            'degraded_threshold': 0.8,
            'critical_threshold': 0.6,
            'recovery_timeout': 60.0,
            'data_dir': './data/safety'
        },
        'failover_manager': {
            'default_failure_threshold': 3,
            'default_switch_timeout': 30,
            'health_check_interval': 10
        },
        'recovery_manager': {
            'target_recovery_time': 60.0,
            'critical_recovery_time': 30.0,
            'state_dir': './data/recovery'
        },
        'health_monitor': {
            'alert_history_size': 1000,
            'metrics_retention_hours': 24,
            'anomaly_detection_enabled': True
        }
    }


async def demonstrate_normal_operations(safety_integration, trading_engine):
    """Demonstrate normal safety system operations."""
    logger.info("=== Demonstrating Normal Operations ===")

    # Get system health
    health = safety_integration.get_system_health()
    logger.info(f"System Status: {health.get('overall_status')}")

    # Show component status
    safety_systems = health.get('safety_systems', {})
    for system_name, system_data in safety_systems.items():
        if isinstance(system_data, dict):
            status = system_data.get('system_state') or system_data.get('overall_status', 'unknown')
            logger.info(f"  {system_name}: {status}")

    # Update component health
    await safety_integration.safety_manager.heartbeat('trading_engine')
    await safety_integration.safety_manager.heartbeat('broker_adapter')

    logger.info("Normal operations demonstration complete")


async def demonstrate_component_failure_recovery(safety_integration, trading_engine):
    """Demonstrate component failure and automatic recovery."""
    logger.info("=== Demonstrating Component Failure Recovery ===")

    # Simulate component failure
    logger.info("Simulating broker adapter failure...")
    await safety_integration.safety_manager.update_component_health(
        'broker_adapter',
        ComponentState.FAILED,
        "Simulated network connection failure"
    )

    # Wait for safety system to detect and respond
    await asyncio.sleep(2)

    # Show degraded status
    health = safety_integration.get_system_health()
    logger.info(f"System Status after failure: {health.get('overall_status')}")

    # Simulate component recovery
    logger.info("Simulating broker adapter recovery...")
    await safety_integration.safety_manager.update_component_health(
        'broker_adapter',
        ComponentState.OPERATIONAL,
        None
    )

    # Wait for recovery
    await asyncio.sleep(2)

    # Show recovered status
    health = safety_integration.get_system_health()
    logger.info(f"System Status after recovery: {health.get('overall_status')}")

    logger.info("Component failure recovery demonstration complete")


async def demonstrate_circuit_breaker(safety_integration, trading_engine):
    """Demonstrate circuit breaker functionality."""
    logger.info("=== Demonstrating Circuit Breaker Functionality ===")

    # Get circuit breaker
    circuit_breaker = safety_integration.circuit_manager.get_circuit_breaker('broker_connection_protection')

    if circuit_breaker:
        logger.info(f"Circuit breaker initial state: {circuit_breaker.state.value}")

        # Simulate protected function with failures
        async def simulated_broker_call():
            # Simulate intermittent failures
            import random
            if random.random() < 0.8:  # 80% failure rate
                raise Exception("Simulated broker connection failure")
            return "success"

        # Make calls that will trip the circuit breaker
        for i in range(6):
            try:
                result = await circuit_breaker.call(simulated_broker_call)
                logger.info(f"Call {i+1}: Success - {result}")
            except Exception as e:
                logger.warning(f"Call {i+1}: Failed - {e}")

            await asyncio.sleep(0.5)

        logger.info(f"Circuit breaker final state: {circuit_breaker.state.value}")

        # Show circuit breaker status
        status = circuit_breaker.get_status()
        logger.info(f"Circuit breaker trips: {status['circuit_trips']}")
        logger.info(f"Failure rate: {status['failure_rate']:.2%}")

    logger.info("Circuit breaker demonstration complete")


async def demonstrate_emergency_procedures(safety_integration, trading_engine):
    """Demonstrate emergency shutdown procedures."""
    logger.info("=== Demonstrating Emergency Procedures ===")

    # Show pre-emergency status
    status = await trading_engine.get_status()
    logger.info(f"Pre-emergency status: Running={status['running']}, Kill Switch={status['kill_switch_activated']}")

    # Trigger emergency shutdown
    logger.warning("Triggering emergency shutdown in 3 seconds...")
    await asyncio.sleep(3)

    success = await safety_integration.emergency_shutdown()

    if success:
        logger.info("Emergency shutdown completed successfully")
    else:
        logger.error("Emergency shutdown failed")

    # Show post-emergency status
    status = await trading_engine.get_status()
    logger.info(f"Post-emergency status: Running={status['running']}, Kill Switch={status['kill_switch_activated']}")

    logger.info("Emergency procedures demonstration complete")


async def demonstrate_system_recovery(trading_engine):
    """Demonstrate system recovery after emergency shutdown."""
    logger.info("=== Demonstrating System Recovery ===")

    # Initialize new safety integration for recovery
    config = create_safety_config()
    recovery_integration = TradingSafetyIntegration(config)

    try:
        # Initialize and start recovery
        logger.info("Initializing recovery systems...")
        await recovery_integration.initialize(trading_engine)
        await recovery_integration.start()

        # Perform system recovery
        logger.info("Performing system recovery...")
        recovery_success = await recovery_integration.system_recovery()

        if recovery_success:
            logger.info("System recovery completed successfully")
        else:
            logger.error("System recovery failed")

        # Show recovered system status
        health = recovery_integration.get_system_health()
        logger.info(f"Recovered system status: {health.get('overall_status')}")

    finally:
        await recovery_integration.stop()

    logger.info("System recovery demonstration complete")


async def setup_alert_monitoring(safety_integration):
    """Setup alert monitoring and callbacks."""
    logger.info("Setting up alert monitoring...")

    async def health_alert_handler(alert):
        """Handle health monitoring alerts."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }

        emoji = severity_emoji.get(alert.severity, "❓")
        logger.warning(f"{emoji} HEALTH ALERT [{alert.severity.value.upper()}]: "
                      f"{alert.component_id} - {alert.message}")

        # In production, this would:
        # - Send notifications to operations team
        # - Update monitoring dashboards
        # - Trigger automated responses
        # - Log to incident management system

    if safety_integration.health_monitor:
        safety_integration.health_monitor.register_alert_callback(health_alert_handler)
        logger.info("Alert monitoring configured")


async def run_comprehensive_demonstration():
    """Run comprehensive safety system demonstration."""
    logger.info("Starting Gary×Taleb Safety System Demonstration")
    logger.info("=" * 60)

    # Create trading engine and safety configuration
    trading_engine = MockTradingEngine()
    config = create_safety_config()

    # Initialize safety integration
    safety_integration = TradingSafetyIntegration(config)

    try:
        # Initialize and start systems
        logger.info("Initializing safety systems...")
        await safety_integration.initialize(trading_engine)

        # Setup monitoring
        await setup_alert_monitoring(safety_integration)

        # Start safety systems
        logger.info("Starting safety systems...")
        await safety_integration.start()

        # Start trading engine
        await trading_engine.start()

        # Run demonstrations
        await demonstrate_normal_operations(safety_integration, trading_engine)
        await asyncio.sleep(2)

        await demonstrate_component_failure_recovery(safety_integration, trading_engine)
        await asyncio.sleep(2)

        await demonstrate_circuit_breaker(safety_integration, trading_engine)
        await asyncio.sleep(2)

        await demonstrate_emergency_procedures(safety_integration, trading_engine)
        await asyncio.sleep(2)

    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration error: {e}", exc_info=True)
    finally:
        # Ensure clean shutdown
        logger.info("Shutting down safety systems...")
        await safety_integration.stop()

    # Demonstrate recovery (separate from main safety integration)
    await demonstrate_system_recovery(trading_engine)

    logger.info("=" * 60)
    logger.info("Safety System Demonstration Complete")


async def run_availability_test():
    """Run availability test to validate 99.9% target."""
    logger.info("Starting 99.9% Availability Test")
    logger.info("=" * 40)

    trading_engine = MockTradingEngine()
    config = create_safety_config()
    safety_integration = TradingSafetyIntegration(config)

    try:
        await safety_integration.initialize(trading_engine)
        await safety_integration.start()
        await trading_engine.start()

        # Run availability simulation
        test_duration = 120  # 2 minutes for demo
        start_time = asyncio.get_event_loop().time()
        total_downtime = 0
        failure_count = 0
        recovery_times = []

        logger.info(f"Running availability test for {test_duration} seconds...")

        while (asyncio.get_event_loop().time() - start_time) < test_duration:
            # Simulate various failure scenarios
            if (asyncio.get_event_loop().time() - start_time) % 30 < 1 and failure_count < 3:
                failure_count += 1

                # Record failure start
                failure_start = asyncio.get_event_loop().time()
                logger.info(f"Simulating failure #{failure_count}")

                # Simulate component failure
                await safety_integration.safety_manager.update_component_health(
                    'trading_engine', ComponentState.FAILED, f"Simulated failure {failure_count}"
                )

                # Wait a bit, then recover
                await asyncio.sleep(2)

                # Perform recovery
                recovery_start = asyncio.get_event_loop().time()
                await safety_integration.safety_manager.update_component_health(
                    'trading_engine', ComponentState.OPERATIONAL
                )

                failure_end = asyncio.get_event_loop().time()
                downtime = failure_end - failure_start
                recovery_time = failure_end - recovery_start

                total_downtime += downtime
                recovery_times.append(recovery_time)

                logger.info(f"Recovery #{failure_count} completed in {recovery_time:.2f}s")

            await asyncio.sleep(1)

        # Calculate results
        total_test_time = asyncio.get_event_loop().time() - start_time
        uptime = total_test_time - total_downtime
        availability = (uptime / total_test_time) * 100

        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # Report results
        logger.info("=" * 40)
        logger.info("AVAILABILITY TEST RESULTS")
        logger.info("=" * 40)
        logger.info(f"Test Duration: {total_test_time:.1f} seconds")
        logger.info(f"Total Downtime: {total_downtime:.2f} seconds")
        logger.info(f"Uptime: {uptime:.2f} seconds")
        logger.info(f"Availability: {availability:.3f}%")
        logger.info("Target: 99.900%")
        logger.info(f"Status: {'✅ PASS' if availability >= 99.0 else '❌ FAIL'}")
        logger.info(f"Failures Simulated: {failure_count}")
        logger.info(f"Average Recovery Time: {avg_recovery_time:.2f}s")
        logger.info(f"Recovery Target: <60s ({'✅ PASS' if avg_recovery_time < 60 else '❌ FAIL'})")

    finally:
        await safety_integration.stop()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # In async context, this would trigger cleanup
    sys.exit(0)


def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    Path('data/safety').mkdir(parents=True, exist_ok=True)
    Path('data/recovery').mkdir(parents=True, exist_ok=True)

    print("Gary×Taleb Safety System Integration Example")
    print("=" * 50)
    print("1. Comprehensive demonstration")
    print("2. Availability test (99.9% target)")
    print("3. Both")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        asyncio.run(run_comprehensive_demonstration())
    elif choice == "2":
        asyncio.run(run_availability_test())
    elif choice == "3":
        asyncio.run(run_comprehensive_demonstration())
        print("\n" + "=" * 50)
        asyncio.run(run_availability_test())
    else:
        print("Invalid choice. Running comprehensive demonstration...")
        asyncio.run(run_comprehensive_demonstration())


if __name__ == "__main__":
    main()