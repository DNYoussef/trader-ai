"""
Phase 3 Audit Script - Gate Progression Psychology

Tests the gate progression psychology system to ensure genuine functionality
and proper integration with the existing gate management system.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gates.gate_manager import GateManager, GateLevel, GraduationMetrics
from gates.gate_psychology import GatePsychology, CelebrationStyle, MotivationType
from gates.enhanced_gate_manager import EnhancedGateManager
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gate_psychology_initialization():
    """Test gate psychology system initializes correctly."""
    print("\n=== Testing Gate Psychology Initialization ===")

    try:
        # Create base manager
        gate_manager = GateManager(data_dir="./test_data/gates")

        # Create psychology system
        psychology = GatePsychology(gate_manager)

        # Verify initialization
        assert hasattr(psychology, 'unlocks')
        assert hasattr(psychology, 'celebration_flows')
        assert hasattr(psychology, 'motivation_templates')

        print("Gate psychology system initialized successfully")

        # Test unlocks are configured
        for gate in [GateLevel.G1, GateLevel.G2, GateLevel.G3]:
            unlock = psychology.unlocks.get(gate)
            assert unlock is not None, f"Missing unlock configuration for {gate.value}"
            assert len(unlock.new_features) > 0, f"No new features defined for {gate.value}"
            assert len(unlock.expanded_capabilities) > 0, f"No capabilities defined for {gate.value}"
            print(f"Gate {gate.value}: {len(unlock.new_features)} features, {len(unlock.expanded_capabilities)} capabilities")

        return True

    except Exception as e:
        print(f"FAILED: Gate psychology initialization error: {e}")
        return False


def test_celebration_flow_creation():
    """Test celebration flow creation for gate progressions."""
    print("\n=== Testing Celebration Flow Creation ===")

    try:
        gate_manager = GateManager(data_dir="./test_data/gates")
        psychology = GatePsychology(gate_manager)

        # Test metrics
        test_metrics = GraduationMetrics(
            consecutive_compliant_days=15,
            total_violations_30d=0,
            performance_score=0.75,
            sharpe_ratio_30d=1.5
        )

        # Test all gate progressions
        progressions = [
            (GateLevel.G0, GateLevel.G1),
            (GateLevel.G1, GateLevel.G2),
            (GateLevel.G2, GateLevel.G3)
        ]

        for from_gate, to_gate in progressions:
            celebration = psychology.create_graduation_celebration(from_gate, to_gate, test_metrics)

            assert celebration is not None, f"No celebration created for {from_gate.value} -> {to_gate.value}"
            assert celebration.gate_from == from_gate
            assert celebration.gate_to == to_gate
            assert len(celebration.celebration_title) > 0
            assert len(celebration.motivational_message) > 0
            assert len(celebration.next_steps) > 0
            assert celebration.unlocks is not None

            print(f"Celebration created: {from_gate.value} -> {to_gate.value}")
            print(f"  Title: {celebration.celebration_title}")
            print(f"  Style: {celebration.celebration_style.value}")
            print(f"  Features unlocked: {len(celebration.unlocks.new_features)}")

        return True

    except Exception as e:
        print(f"FAILED: Celebration flow creation error: {e}")
        return False


def test_progress_motivation_generation():
    """Test progress motivation message generation."""
    print("\n=== Testing Progress Motivation Generation ===")

    try:
        gate_manager = GateManager(data_dir="./test_data/gates")
        psychology = GatePsychology(gate_manager)

        # Test different progress scenarios
        progress_scenarios = [
            {
                'current_gate': 'G0',
                'progress_percentage': 25.0,
                'consecutive_days': 5,
                'performance_score': 0.4
            },
            {
                'current_gate': 'G1',
                'progress_percentage': 60.0,
                'consecutive_days': 12,
                'performance_score': 0.7
            },
            {
                'current_gate': 'G2',
                'progress_percentage': 85.0,
                'consecutive_days': 25,
                'performance_score': 0.8
            }
        ]

        for scenario in progress_scenarios:
            motivation = psychology.generate_progress_motivation(scenario)

            assert motivation is not None, f"No motivation generated for scenario: {scenario}"
            assert len(motivation.primary_message) > 0
            assert len(motivation.supporting_points) > 0
            assert len(motivation.call_to_action) > 0
            assert motivation.motivation_type in MotivationType

            print(f"Motivation for {scenario['current_gate']} at {scenario['progress_percentage']}%:")
            print(f"  Type: {motivation.motivation_type.value}")
            print(f"  Message: {motivation.primary_message[:60]}...")
            print(f"  CTA: {motivation.call_to_action}")

        return True

    except Exception as e:
        print(f"FAILED: Progress motivation generation error: {e}")
        return False


def test_milestone_celebration_creation():
    """Test milestone celebration creation."""
    print("\n=== Testing Milestone Celebration Creation ===")

    try:
        gate_manager = GateManager(data_dir="./test_data/gates")
        psychology = GatePsychology(gate_manager)

        # Test different milestone types
        milestones = [
            ('first_profit', {'profit': 5.50}),
            ('7_day_streak', {'days': 7}),
            ('30_day_streak', {'days': 30}),
            ('first_rebalance', {'rebalance_count': 1}),
            ('risk_management_win', {'risk_saved': 50.0})
        ]

        for milestone_type, milestone_data in milestones:
            celebration = psychology.create_milestone_celebration(milestone_type, milestone_data)

            assert celebration is not None, f"No celebration created for {milestone_type}"
            assert 'title' in celebration
            assert 'message' in celebration
            assert 'style' in celebration
            assert celebration['milestone_type'] == milestone_type

            print(f"Milestone celebration: {milestone_type}")
            print(f"  Title: {celebration['title']}")
            print(f"  Style: {celebration['style'].value}")

        return True

    except Exception as e:
        print(f"FAILED: Milestone celebration creation error: {e}")
        return False


def test_enhanced_gate_manager_integration():
    """Test enhanced gate manager integration."""
    print("\n=== Testing Enhanced Gate Manager Integration ===")

    try:
        # Create enhanced gate manager
        enhanced_manager = EnhancedGateManager(data_dir="./test_data/gates", enable_psychology=True)

        # Test callback registration
        celebration_triggered = []
        progress_updated = []
        milestone_achieved = []

        def celebration_callback(event_type, data):
            celebration_triggered.append((event_type, data))

        def progress_callback(event_type, data):
            progress_updated.append((event_type, data))

        def milestone_callback(event_type, data):
            milestone_achieved.append((event_type, data))

        enhanced_manager.add_celebration_callback(celebration_callback)
        enhanced_manager.add_progress_callback(progress_callback)
        enhanced_manager.add_milestone_callback(milestone_callback)

        print("Callback registration successful")

        # Test enhanced status report
        status = enhanced_manager.get_enhanced_status_report()

        assert 'psychology_enabled' in status
        assert status['psychology_enabled'] is True
        assert 'progress_tracking' in status
        assert 'next_gate_preview' in status

        print("Enhanced status report generated")
        print(f"  Current gate: {status['current_gate']}")
        print(f"  Psychology enabled: {status['psychology_enabled']}")

        # Test trade validation with guidance
        test_trade = {
            'symbol': 'SPY',  # Not allowed in G0
            'side': 'BUY',
            'quantity': 10,
            'price': 440.00,
            'trade_type': 'STOCK'
        }

        test_portfolio = {
            'cash': 200.00,
            'total_value': 200.00,
            'positions': {}
        }

        validation_result = enhanced_manager.validate_trade_with_guidance(test_trade, test_portfolio)

        assert 'human_guidance' in validation_result
        assert 'learning_opportunity' in validation_result

        if not validation_result['is_valid']:
            print("Trade validation with guidance:")
            print(f"  Valid: {validation_result['is_valid']}")
            if validation_result['human_guidance']:
                print(f"  Guidance: {validation_result['human_guidance']['title']}")

        return True

    except Exception as e:
        print(f"FAILED: Enhanced gate manager integration error: {e}")
        return False


def test_psychology_principles_implementation():
    """Test implementation of mobile app psychology principles."""
    print("\n=== Testing Psychology Principles Implementation ===")

    try:
        gate_manager = GateManager(data_dir="./test_data/gates")
        psychology = GatePsychology(gate_manager)

        principles_verified = []

        # 1. Test Achievement Focus (unlocking features)
        unlock_g1 = psychology.unlocks.get(GateLevel.G1)
        if unlock_g1 and len(unlock_g1.psychological_rewards) > 0:
            principles_verified.append("Achievement focus with psychological rewards")

        # 2. Test Progress Momentum (celebration styles)
        test_celebration = psychology.celebration_flows.get((GateLevel.G0, GateLevel.G1))
        if test_celebration and test_celebration.celebration_style in [CelebrationStyle.CONFETTI, CelebrationStyle.FIREWORKS]:
            principles_verified.append("Progress momentum with visual celebrations")

        # 3. Test Social Proof (sharing messages)
        if test_celebration and len(test_celebration.social_share_message) > 0:
            principles_verified.append("Social proof with sharing capabilities")

        # 4. Test Loss Aversion (risk protection messaging)
        enhanced_manager = EnhancedGateManager(data_dir="./test_data/gates", enable_psychology=True)
        test_trade = {
            'symbol': 'ULTY',
            'side': 'BUY',
            'quantity': 50,  # Large quantity to trigger position limit
            'price': 5.50,
            'trade_type': 'STOCK'
        }

        test_portfolio = {
            'cash': 100.00,  # Low cash to trigger cash floor
            'total_value': 200.00,
            'positions': {}
        }

        validation = enhanced_manager.validate_trade_with_guidance(test_trade, test_portfolio)
        if validation.get('human_guidance') and 'protect' in validation['human_guidance'].get('message', '').lower():
            principles_verified.append("Loss aversion with protective messaging")

        # 5. Test Commitment Escalation (next steps)
        if test_celebration and len(test_celebration.next_steps) > 0:
            principles_verified.append("Commitment escalation with next steps")

        print("Psychology principles implemented:")
        for principle in principles_verified:
            print(f"  - {principle}")

        if len(principles_verified) >= 3:
            print("Psychology principles successfully integrated")
            return True
        else:
            print("FAILED: Insufficient psychology principles implemented")
            return False

    except Exception as e:
        print(f"FAILED: Psychology principles test error: {e}")
        return False


def test_integration_with_existing_system():
    """Test integration with existing gate management system."""
    print("\n=== Testing Integration with Existing System ===")

    try:
        # Test that enhanced manager preserves base functionality
        enhanced_manager = EnhancedGateManager(data_dir="./test_data/gates", enable_psychology=True)

        # Test delegation to base manager
        original_gate = enhanced_manager.current_gate
        original_capital = enhanced_manager.current_capital

        print(f"Current gate: {original_gate.value}")
        print(f"Current capital: ${original_capital}")

        # Test base functionality still works
        status = enhanced_manager.get_status_report()
        assert 'current_gate' in status
        assert 'gate_config' in status

        # Test enhanced functionality is additive
        enhanced_status = enhanced_manager.get_enhanced_status_report()
        assert 'psychology_enabled' in enhanced_status
        assert all(key in enhanced_status for key in status.keys())  # Base keys still present

        print("Base functionality preserved")
        print("Enhanced functionality added")

        # Test psychology can be disabled
        disabled_manager = EnhancedGateManager(data_dir="./test_data/gates", enable_psychology=False)
        assert disabled_manager.psychology_enabled is False
        assert disabled_manager.psychology is None

        print("Psychology system can be disabled")

        return True

    except Exception as e:
        print(f"FAILED: Integration test error: {e}")
        return False


def run_phase_3_audit():
    """Run complete Phase 3 audit."""
    print("PHASE 3 AUDIT: Gate Progression Psychology")
    print("=" * 60)

    # Ensure test data directory exists
    test_data_dir = Path("./test_data/gates")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    all_tests_passed = True

    tests = [
        test_gate_psychology_initialization,
        test_celebration_flow_creation,
        test_progress_motivation_generation,
        test_milestone_celebration_creation,
        test_enhanced_gate_manager_integration,
        test_psychology_principles_implementation,
        test_integration_with_existing_system
    ]

    for test_func in tests:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"CRITICAL ERROR in {test_func.__name__}: {e}")
            all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("PHASE 3 AUDIT PASSED - Gate progression psychology is genuine and complete")
        print("   - Gate psychology system initializes correctly")
        print("   - Celebration flows create compelling experiences")
        print("   - Progress motivation generates personalized content")
        print("   - Milestone celebrations reward achievements")
        print("   - Enhanced gate manager integrates seamlessly")
        print("   - Mobile app psychology principles implemented")
        print("   - Integration preserves existing functionality")
    else:
        print("PHASE 3 AUDIT FAILED - Issues detected")
        print("   - Review test failures above")
        print("   - Fix issues before proceeding to Phase 4")

    return all_tests_passed


if __name__ == "__main__":
    success = run_phase_3_audit()
    sys.exit(0 if success else 1)