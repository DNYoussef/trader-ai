"""
Comprehensive tests for kill switch system
Tests response time, authentication, triggers, and audit trail
"""

import asyncio
import time
import json
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.kill_switch_system import (
    KillSwitchSystem, TriggerType, KillSwitchEvent
)
from src.safety.hardware_auth_manager import (
    HardwareAuthManager, AuthMethod, AuthResult
)
from src.safety.multi_trigger_system import (
    MultiTriggerSystem, TriggerCondition, TriggerSeverity
)
from src.safety.audit_trail_system import (
    AuditTrailSystem, EventType, EventSeverity
)

class TestKillSwitchSystem:
    """Test core kill switch functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.mock_broker = Mock()
        self.mock_broker.get_positions = AsyncMock(return_value=[
            Mock(symbol='AAPL', qty=100, market_value=15000),
            Mock(symbol='MSFT', qty=50, market_value=12000)
        ])
        self.mock_broker.close_position = AsyncMock(return_value={'success': True})
        self.mock_broker.get_account = AsyncMock(return_value=Mock(equity=50000))

        self.config = {
            'loss_limit': -1000,
            'position_limit': 10000,
            'heartbeat_timeout': 30,
            'hardware_auth': {
                'master_keys': {
                    'default': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'  # SHA256 of empty string
                },
                'emergency_override_hash': 'd4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35'
            }
        }

        self.temp_dir = tempfile.mkdtemp()
        self.kill_switch = KillSwitchSystem(self.mock_broker, self.config)

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'kill_switch'):
            self.kill_switch.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_kill_switch_response_time(self):
        """Test kill switch meets <500ms response time requirement"""
        start_time = time.time()

        event = await self.kill_switch.trigger_kill_switch(
            TriggerType.MANUAL_PANIC,
            {'test': True}
        )

        response_time = (time.time() - start_time) * 1000

        assert response_time < 500, f"Response time {response_time:.1f}ms exceeds 500ms target"
        assert event.response_time_ms < 500
        assert event.success is True

    @pytest.mark.asyncio
    async def test_manual_panic_with_authentication(self):
        """Test manual panic button with hardware authentication"""
        auth_data = {'method': 'master', 'key': ''}  # Empty string for test hash

        event = await self.kill_switch.manual_panic_button(auth_data)

        assert event.trigger_type == TriggerType.MANUAL_PANIC
        assert event.authentication_method == 'master'
        assert event.success is True

    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test kill switch rejection on authentication failure"""
        auth_data = {'method': 'master', 'key': 'wrong_key'}

        event = await self.kill_switch.trigger_kill_switch(
            TriggerType.MANUAL_PANIC,
            {'test': True},
            auth_data
        )

        assert event.success is False
        assert event.error == "Authentication failed"

    @pytest.mark.asyncio
    async def test_position_flattening(self):
        """Test emergency position flattening"""
        # Setup positions to flatten
        self.mock_broker.get_positions.return_value = [
            Mock(symbol='AAPL', qty=100, market_value=15000),
            Mock(symbol='MSFT', qty=-50, market_value=12000)
        ]

        event = await self.kill_switch.trigger_kill_switch(
            TriggerType.LOSS_LIMIT,
            {'loss': -1500}
        )

        # Verify close_position was called for each position
        assert self.mock_broker.close_position.call_count >= 2
        assert event.success is True

    def test_system_status(self):
        """Test kill switch system status"""
        status = self.kill_switch.get_system_status()

        assert isinstance(status, dict)
        assert 'active' in status
        assert 'armed' in status
        assert 'monitoring_active' in status
        assert 'last_heartbeat' in status

    @pytest.mark.asyncio
    async def test_kill_switch_test_mode(self):
        """Test kill switch in test mode (no actual position flattening)"""
        test_result = await self.kill_switch.test_kill_switch()

        assert test_result['test_successful'] is True
        assert test_result['response_time_ms'] < 500
        assert test_result['meets_target'] is True

class TestHardwareAuthentication:
    """Test hardware authentication system"""

    def setup_method(self):
        """Setup for each test"""
        self.config = {
            'allowed_methods': ['master_key', 'emergency_override'],
            'master_keys': {
                'default': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',  # Empty string
                'admin': '6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b'    # "hello"
            },
            'emergency_override_hash': 'd4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35'  # "hello"
        }
        self.auth_manager = HardwareAuthManager(self.config)

    @pytest.mark.asyncio
    async def test_master_key_authentication(self):
        """Test master key authentication"""
        result = await self.auth_manager.authenticate({
            'method': 'master_key',
            'key': ''  # Empty string matches test hash
        })

        assert result.success is True
        assert result.method == AuthMethod.MASTER_KEY
        assert result.user_id == 'default'

    @pytest.mark.asyncio
    async def test_master_key_authentication_failure(self):
        """Test master key authentication failure"""
        result = await self.auth_manager.authenticate({
            'method': 'master_key',
            'key': 'wrong_key'
        })

        assert result.success is False
        assert result.method == AuthMethod.MASTER_KEY

    @pytest.mark.asyncio
    async def test_emergency_override(self):
        """Test emergency override authentication"""
        result = await self.auth_manager.authenticate({
            'method': 'emergency_override',
            'key': 'hello'  # Matches emergency hash
        })

        assert result.success is True
        assert result.method == AuthMethod.EMERGENCY_OVERRIDE
        assert result.user_id == 'EMERGENCY'

    def test_available_methods(self):
        """Test getting available authentication methods"""
        methods = self.auth_manager.get_available_methods()

        assert AuthMethod.MASTER_KEY in methods
        assert AuthMethod.EMERGENCY_OVERRIDE in methods

    @pytest.mark.asyncio
    async def test_authentication_performance(self):
        """Test authentication meets performance requirements"""
        start_time = time.time()

        result = await self.auth_manager.authenticate({
            'method': 'master_key',
            'key': ''
        })

        auth_time = (time.time() - start_time) * 1000

        assert auth_time < 100, f"Authentication took {auth_time:.1f}ms, should be <100ms"
        assert result.duration_ms < 100

class TestMultiTriggerSystem:
    """Test multi-trigger monitoring system"""

    def setup_method(self):
        """Setup for each test"""
        self.mock_broker = Mock()
        self.mock_broker.get_positions = AsyncMock(return_value=[])
        self.mock_broker.get_account = AsyncMock(return_value=Mock(equity=10000))

        self.mock_kill_switch = Mock()
        self.mock_kill_switch.trigger_kill_switch = AsyncMock()

        self.config = {
            'triggers': {
                'loss_limits': {
                    'session_loss_percent': -5.0,
                    'drawdown_percent': -3.0
                },
                'position_limits': {
                    'max_exposure_ratio': 0.8,
                    'max_single_position_ratio': 0.3
                },
                'api_failure': {
                    'response_timeout': 3000,
                    'connection_timeout': 20
                }
            }
        }

        self.multi_trigger = MultiTriggerSystem(
            self.mock_kill_switch,
            self.mock_broker,
            self.config
        )

    def test_trigger_conditions_loading(self):
        """Test loading of trigger conditions from config"""
        assert len(self.multi_trigger.trigger_conditions) > 0

        # Check loss limit triggers
        loss_triggers = [t for t in self.multi_trigger.trigger_conditions
                        if t.trigger_type == TriggerType.LOSS_LIMIT]
        assert len(loss_triggers) >= 2

        # Check position limit triggers
        pos_triggers = [t for t in self.multi_trigger.trigger_conditions
                       if t.trigger_type == TriggerType.POSITION_LIMIT]
        assert len(pos_triggers) >= 2

    def test_trigger_status(self):
        """Test trigger status tracking"""
        status = self.multi_trigger.get_trigger_status()

        assert isinstance(status, dict)
        assert 'monitoring' in status
        assert 'trigger_count' in status
        assert 'active_triggers' in status
        assert 'triggers' in status

    @pytest.mark.asyncio
    async def test_loss_limit_evaluation(self):
        """Test loss limit trigger evaluation"""
        # Create a trigger condition
        condition = TriggerCondition(
            name="Test Loss Limit",
            trigger_type=TriggerType.LOSS_LIMIT,
            severity=TriggerSeverity.CRITICAL,
            threshold_value=-5.0,
            time_window_seconds=60,
            consecutive_failures=1
        )

        from src.safety.multi_trigger_system import TriggerStatus
        status = TriggerStatus(
            condition=condition,
            current_value=0.0,
            last_check=0.0,
            failure_count=0,
            consecutive_failures=0,
            triggered=False
        )

        # Test threshold breach (loss exceeds limit)
        should_trigger = await self.multi_trigger._evaluate_trigger(status, -6.0)

        assert should_trigger is True
        assert status.triggered is True
        assert status.consecutive_failures == 1

class TestAuditTrailSystem:
    """Test audit trail system"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'storage_path': self.temp_dir,
            'version': '1.0.0',
            'environment': 'test'
        }
        self.audit = AuditTrailSystem(self.config)

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'audit'):
            self.audit.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_event_logging(self):
        """Test basic event logging"""
        self.audit.start()

        await self.audit.log_event(
            EventType.SYSTEM_ERROR,
            EventSeverity.ERROR,
            "Test error message",
            details={'test': True}
        )

        # Wait for async processing
        await asyncio.sleep(0.1)

        events = self.audit.query_events(limit=10)
        assert len(events) >= 1

        test_event = next((e for e in events if e.message == "Test error message"), None)
        assert test_event is not None
        assert test_event.event_type == EventType.SYSTEM_ERROR
        assert test_event.severity == EventSeverity.ERROR

    @pytest.mark.asyncio
    async def test_kill_switch_event_logging(self):
        """Test kill switch event logging"""
        self.audit.start()

        # Create mock kill switch event
        kill_event = KillSwitchEvent(
            timestamp=time.time(),
            trigger_type=TriggerType.MANUAL_PANIC,
            trigger_data={'test': True},
            response_time_ms=250.0,
            positions_flattened=2,
            authentication_method='master_key',
            success=True
        )

        await self.audit.log_kill_switch_activation(kill_event)

        # Wait for processing
        await asyncio.sleep(0.1)

        events = self.audit.query_events(
            event_types=[EventType.KILL_SWITCH_ACTIVATION],
            limit=10
        )

        assert len(events) >= 1
        event = events[0]
        assert event.event_type == EventType.KILL_SWITCH_ACTIVATION
        assert event.severity == EventSeverity.CRITICAL

    def test_event_integrity(self):
        """Test event integrity verification"""
        from src.safety.audit_trail_system import AuditEvent

        event = AuditEvent(
            event_type=EventType.SYSTEM_ERROR,
            severity=EventSeverity.ERROR,
            message="Test message",
            details={'test': True}
        )

        # Event should pass integrity check
        assert event.verify_integrity() is True

        # Tamper with event
        event.message = "Tampered message"

        # Should fail integrity check
        assert event.verify_integrity() is False

    @pytest.mark.asyncio
    async def test_audit_report_generation(self):
        """Test audit report generation"""
        self.audit.start()

        # Log some events
        await self.audit.log_event(
            EventType.KILL_SWITCH_ACTIVATION,
            EventSeverity.CRITICAL,
            "Test kill switch",
            details={'test': True}
        )

        await self.audit.log_event(
            EventType.AUTHENTICATION_ATTEMPT,
            EventSeverity.INFO,
            "Test auth",
            details={'method': 'master_key', 'success': True}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Generate report
        report = self.audit.generate_report()

        assert isinstance(report, dict)
        assert 'report_period' in report
        assert 'event_summary' in report
        assert 'kill_switch_details' in report

        # Should have at least one event
        assert report['event_summary']['total_events'] >= 1

class TestIntegratedKillSwitchSystem:
    """Integration tests for complete kill switch system"""

    def setup_method(self):
        """Setup integrated system"""
        self.mock_broker = Mock()
        self.mock_broker.get_positions = AsyncMock(return_value=[
            Mock(symbol='AAPL', qty=100, market_value=15000)
        ])
        self.mock_broker.close_position = AsyncMock(return_value={'success': True})
        self.mock_broker.get_account = AsyncMock(return_value=Mock(equity=50000))

        self.temp_dir = tempfile.mkdtemp()

        # Load kill switch config
        config_path = Path(__file__).parent.parent / 'config' / 'kill_switch_config.json'
        with open(config_path) as f:
            self.config = json.load(f)

        # Override paths for testing
        self.config['kill_switch']['audit']['storage_path'] = self.temp_dir

        self.kill_switch = KillSwitchSystem(self.mock_broker, self.config['kill_switch'])
        self.audit = AuditTrailSystem({
            'storage_path': self.temp_dir,
            'version': '1.0.0',
            'environment': 'test'
        })

    def teardown_method(self):
        """Cleanup integrated system"""
        if hasattr(self, 'kill_switch'):
            self.kill_switch.stop_monitoring()
        if hasattr(self, 'audit'):
            self.audit.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_end_to_end_kill_switch_activation(self):
        """Test complete kill switch activation flow"""
        self.audit.start()
        self.kill_switch.start_monitoring()

        # Trigger kill switch with authentication
        auth_data = {'method': 'master_key', 'key': ''}  # Empty string for default test hash

        start_time = time.time()
        event = await self.kill_switch.manual_panic_button(auth_data)
        total_time = (time.time() - start_time) * 1000

        # Verify response time
        assert total_time < 500, f"Total time {total_time:.1f}ms exceeds target"
        assert event.success is True
        assert event.authentication_method == 'master'

        # Wait for audit processing
        await asyncio.sleep(0.1)

        # Verify audit trail
        audit_events = self.audit.query_events(
            event_types=[EventType.KILL_SWITCH_ACTIVATION],
            limit=10
        )
        assert len(audit_events) >= 1

        # Verify position flattening was called
        assert self.mock_broker.close_position.called

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test kill switch performance under concurrent load"""
        self.kill_switch.start_monitoring()

        # Simulate multiple concurrent triggers
        tasks = []
        for i in range(10):
            task = self.kill_switch.trigger_kill_switch(
                TriggerType.SYSTEM_ERROR,
                {'concurrent_test': i}
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000

        # All should complete within reasonable time
        assert total_time < 2000  # 2 seconds for 10 concurrent triggers

        # At least some should succeed (first one definitely should)
        successes = [r for r in results if isinstance(r, KillSwitchEvent) and r.success]
        assert len(successes) >= 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])