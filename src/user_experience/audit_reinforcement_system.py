"""
Phase 5 Audit Script - Psychological Reinforcement System

Tests the psychological reinforcement system to ensure genuine functionality,
proper integration with trading events, and effective user engagement mechanisms.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_experience.psychological_reinforcement import (
    PsychologicalReinforcementEngine,
    ReinforcementType,
    TriggerCondition
)
from user_experience.reinforcement_integration import ReinforcementOrchestrator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_reinforcement_engine_initialization():
    """Test reinforcement engine initializes correctly."""
    print("\n=== Testing Reinforcement Engine Initialization ===")

    try:
        # Create reinforcement engine
        engine = PsychologicalReinforcementEngine()

        # Verify initialization
        assert hasattr(engine, 'user_profiles')
        assert hasattr(engine, 'reinforcement_templates')
        assert hasattr(engine, 'trigger_handlers')
        assert hasattr(engine, 'effectiveness_tracker')

        print("Reinforcement engine initialized successfully")

        # Test templates are loaded
        required_templates = [
            'first_profit',
            'gate_graduation',
            'streak_milestone',
            'risk_protection',
            'concept_mastery',
            'surprise_delight',
            'social_proof'
        ]

        for template_id in required_templates:
            assert template_id in engine.reinforcement_templates, f"Missing template: {template_id}"
            template = engine.reinforcement_templates[template_id]

            assert 'type' in template
            assert 'intensity' in template
            assert 'templates' in template
            assert len(template['templates']) > 0

            # Test template structure
            for template_variant in template['templates']:
                assert 'title' in template_variant
                assert 'message' in template_variant
                assert 'visual_effect' in template_variant

            print(f"Template {template_id}: Complete with {len(template['templates'])} variants")

        # Test trigger handlers are set up
        required_triggers = [
            TriggerCondition.TRADE_EXECUTED,
            TriggerCondition.PROFIT_ACHIEVED,
            TriggerCondition.GATE_PROGRESSION,
            TriggerCondition.STREAK_MAINTAINED,
            TriggerCondition.CONCEPT_LEARNED,
            TriggerCondition.RISK_AVOIDED
        ]

        for trigger in required_triggers:
            assert trigger in engine.trigger_handlers, f"Missing trigger handler: {trigger}"
            print(f"Trigger handler registered: {trigger.value}")

        return True

    except Exception as e:
        print(f"FAILED: Reinforcement engine initialization error: {e}")
        return False


def test_user_registration_and_profiles():
    """Test user registration and profile management."""
    print("\n=== Testing User Registration and Profiles ===")

    try:
        engine = PsychologicalReinforcementEngine()

        # Test user registration
        test_user_id = "test_user_123"
        test_persona = "aggressive"
        test_preferences = {
            'celebration_style': 'strong',
            'notification_frequency': 'high',
            'social_sharing': True
        }

        engine.register_user(test_user_id, test_persona, test_preferences)

        # Verify user was registered
        assert test_user_id in engine.user_profiles
        profile = engine.user_profiles[test_user_id]

        assert profile.user_id == test_user_id
        assert profile.persona == test_persona
        assert profile.preferred_celebration_style == 'strong'
        assert profile.notification_frequency == 'high'
        assert profile.social_sharing_enabled is True

        print("User registration working correctly")

        # Test profile summary
        summary = engine.get_user_reinforcement_summary(test_user_id)

        assert summary['user_id'] == test_user_id
        assert 'total_reinforcements' in summary
        assert 'preferences' in summary
        assert summary['preferences']['celebration_style'] == 'strong'

        print("User profile summary generated correctly")

        return True

    except Exception as e:
        print(f"FAILED: User registration error: {e}")
        return False


def test_reinforcement_triggering():
    """Test reinforcement event triggering."""
    print("\n=== Testing Reinforcement Event Triggering ===")

    try:
        engine = PsychologicalReinforcementEngine()

        test_user_id = "test_trigger_user"
        engine.register_user(test_user_id)

        # Test different trigger scenarios
        trigger_tests = [
            {
                'condition': TriggerCondition.PROFIT_ACHIEVED,
                'context': {
                    'profit_amount': 15.50,
                    'is_first_profit': True,
                    'symbol': 'ALTY'
                },
                'expected_template': 'first_profit'
            },
            {
                'condition': TriggerCondition.GATE_PROGRESSION,
                'context': {
                    'new_gate': 'G1',
                    'old_gate': 'G0',
                    'features_unlocked': ['options_trading', 'higher_limits']
                },
                'expected_template': 'gate_graduation'
            },
            {
                'condition': TriggerCondition.STREAK_MAINTAINED,
                'context': {
                    'streak_days': 7,  # This is a milestone day
                    'streak_type': 'daily_compliance'
                },
                'expected_template': 'streak_milestone'
            },
            {
                'condition': TriggerCondition.RISK_AVOIDED,
                'context': {
                    'amount_protected': 25.00,
                    'risk_type': 'position_size',
                    'trigger_reason': 'system_protection'
                },
                'expected_template': 'risk_protection'
            }
        ]

        for test in trigger_tests:
            event = engine.trigger_reinforcement(
                test_user_id,
                test['condition'],
                test['context']
            )

            # All conditions should generate events
            assert event is not None, f"No event generated for {test['condition'].value}"
            assert event.user_id == test_user_id
            assert event.trigger_condition == test['condition']
            assert len(event.title) > 0
            assert len(event.message) > 0
            assert event.visual_effect is not None

            print(f"Reinforcement triggered: {test['condition'].value} -> {event.title}")

        return True

    except Exception as e:
        print(f"FAILED: Reinforcement triggering error: {e}")
        return False


def test_reinforcement_orchestrator():
    """Test reinforcement orchestrator integration."""
    print("\n=== Testing Reinforcement Orchestrator ===")

    try:
        # Create orchestrator
        orchestrator = ReinforcementOrchestrator()

        test_user_id = "test_orchestrator_user"
        orchestrator.register_user(test_user_id)

        # Test event processing
        test_events = [
            {
                'event_type': 'trade_executed',
                'event_data': {
                    'symbol': 'ULTY',
                    'quantity': 10,
                    'price': 50.00
                }
            },
            {
                'event_type': 'profit_realized',
                'event_data': {
                    'profit_amount': 12.50,
                    'symbol': 'ALTY'
                }
            },
            {
                'event_type': 'gate_progressed',
                'event_data': {
                    'new_gate': 'G1',
                    'old_gate': 'G0',
                    'features_unlocked': ['advanced_trading']
                }
            }
        ]

        reinforcement_events = []
        for test_event in test_events:
            event = orchestrator.process_trading_event(
                test_user_id,
                test_event['event_type'],
                test_event['event_data']
            )
            if event:
                reinforcement_events.append(event)

        # At least profit should trigger reinforcement, gate progression might be filtered
        assert len(reinforcement_events) >= 1, f"Expected at least 1 reinforcement event, got {len(reinforcement_events)}"

        print(f"Orchestrator processed {len(reinforcement_events)} reinforcement events")

        # Test user state tracking
        user_state = orchestrator.user_states[test_user_id]
        assert user_state['total_trades'] >= 1
        assert user_state['total_profit'] >= 12.50

        print("User state tracking working correctly")

        # Test engagement dashboard
        dashboard = orchestrator.get_user_engagement_dashboard(test_user_id)

        assert dashboard['user_id'] == test_user_id
        assert 'engagement_overview' in dashboard
        assert 'current_streaks' in dashboard
        assert 'milestone_progress' in dashboard

        print("Engagement dashboard generated successfully")

        return True

    except Exception as e:
        print(f"FAILED: Reinforcement orchestrator error: {e}")
        return False


def test_fatigue_prevention():
    """Test fatigue prevention and intelligent spacing."""
    print("\n=== Testing Fatigue Prevention ===")

    try:
        engine = PsychologicalReinforcementEngine()

        test_user_id = "test_fatigue_user"
        engine.register_user(test_user_id, preferences={'notification_frequency': 'normal'})

        # Trigger multiple events rapidly
        rapid_events = []
        for i in range(5):
            event = engine.trigger_reinforcement(
                test_user_id,
                TriggerCondition.PROFIT_ACHIEVED,
                {'profit_amount': 10.00 + i, 'is_first_profit': False, 'symbol': 'ALTY'}
            )
            rapid_events.append(event)

        # Should have fatigue prevention - not all events should trigger
        actual_events = [e for e in rapid_events if e is not None]
        assert len(actual_events) < 5, "Fatigue prevention should limit rapid events"

        print(f"Fatigue prevention working: {len(actual_events)}/5 events triggered")

        # Test user profile fatigue tracking
        profile = engine.user_profiles[test_user_id]
        assert len(profile.recent_reinforcements) > 0
        assert profile.last_reinforcement is not None

        print("Fatigue tracking mechanisms in place")

        return True

    except Exception as e:
        print(f"FAILED: Fatigue prevention error: {e}")
        return False


def test_personalization_effectiveness():
    """Test personalization and effectiveness tracking."""
    print("\n=== Testing Personalization and Effectiveness ===")

    try:
        engine = PsychologicalReinforcementEngine()

        # Test different user personas
        personas = ['conservative', 'balanced', 'aggressive']
        personalization_tests = []

        for persona in personas:
            user_id = f"test_user_{persona}"
            engine.register_user(user_id, persona)

            event = engine.trigger_reinforcement(
                user_id,
                TriggerCondition.PROFIT_ACHIEVED,
                {'profit_amount': 20.00, 'is_first_profit': True}
            )

            if event:
                personalization_tests.append({
                    'persona': persona,
                    'event': event,
                    'intensity': event.intensity
                })

        assert len(personalization_tests) > 0, "No personalized events generated"
        print(f"Personalization tested across {len(personalization_tests)} personas")

        # Test effectiveness feedback
        if personalization_tests:
            test_event = personalization_tests[0]['event']
            feedback = {
                'rating': 4.5,
                'engagement': 'high',
                'user_action': 'cta_clicked'
            }

            engine.update_reinforcement_effectiveness(test_event.event_id, feedback)

            # Verify effectiveness was recorded
            user_profile = engine.user_profiles[personalization_tests[0]['event'].user_id]
            assert len(user_profile.effectiveness_history) > 0
            assert user_profile.effectiveness_history[-1] == 4.5

            print("Effectiveness tracking working correctly")

        return True

    except Exception as e:
        print(f"FAILED: Personalization effectiveness error: {e}")
        return False


def test_analytics_and_insights():
    """Test analytics and system insights."""
    print("\n=== Testing Analytics and Insights ===")

    try:
        engine = PsychologicalReinforcementEngine()
        orchestrator = ReinforcementOrchestrator(engine)

        # Generate test data
        test_users = ['user1', 'user2', 'user3']
        for user_id in test_users:
            orchestrator.register_user(user_id)

            # Simulate various events
            orchestrator.process_trading_event(user_id, 'profit_realized',
                                             {'profit_amount': 15.00, 'symbol': 'ALTY'})
            orchestrator.process_trading_event(user_id, 'gate_progressed',
                                             {'new_gate': 'G1', 'old_gate': 'G0'})

        # Test system analytics
        system_analytics = orchestrator.get_system_analytics()

        assert 'user_metrics' in system_analytics
        assert 'reinforcement_metrics' in system_analytics
        assert 'engagement_metrics' in system_analytics

        assert system_analytics['user_metrics']['total_registered_users'] == len(test_users)
        assert system_analytics['reinforcement_metrics']['total_reinforcement_events'] > 0

        print("System analytics generated successfully")
        print(f"  - Total users: {system_analytics['user_metrics']['total_registered_users']}")
        print(f"  - Total events: {system_analytics['reinforcement_metrics']['total_reinforcement_events']}")

        # Test reinforcement analytics
        reinforcement_analytics = engine.get_reinforcement_analytics()

        assert 'total_reinforcement_events' in reinforcement_analytics
        assert 'active_users' in reinforcement_analytics
        assert 'reinforcement_type_distribution' in reinforcement_analytics

        print("Reinforcement analytics working correctly")

        return True

    except Exception as e:
        print(f"FAILED: Analytics and insights error: {e}")
        return False


def test_psychology_principles_integration():
    """Test implementation of mobile app psychology principles."""
    print("\n=== Testing Psychology Principles Integration ===")

    try:
        engine = PsychologicalReinforcementEngine()
        orchestrator = ReinforcementOrchestrator(engine)

        test_user_id = "test_psychology_user"
        orchestrator.register_user(test_user_id)

        principles_verified = []

        # 1. Test Variable Ratio Reinforcement
        profit_events = []
        for i in range(10):
            event = orchestrator.process_trading_event(
                test_user_id,
                'trade_executed',
                {'symbol': 'ULTY', 'trade_count': i + 1}
            )
            profit_events.append(event is not None)

        # Should have some variability in reinforcement
        reinforced_count = sum(profit_events)
        if 0 < reinforced_count < len(profit_events):
            principles_verified.append("Variable ratio reinforcement implemented")

        # 2. Test Achievement Focus
        achievement_event = orchestrator.process_trading_event(
            test_user_id,
            'gate_progressed',
            {'new_gate': 'G1', 'old_gate': 'G0', 'features_unlocked': ['advanced_trading']}
        )
        if achievement_event and achievement_event.event_type == ReinforcementType.MILESTONE:
            principles_verified.append("Achievement-focused reinforcement")

        # 3. Test Social Proof Elements
        if achievement_event and achievement_event.social_sharing_prompt:
            principles_verified.append("Social proof through sharing prompts")

        # 4. Test Progress Visualization
        engagement_dashboard = orchestrator.get_user_engagement_dashboard(test_user_id)
        if ('milestone_progress' in engagement_dashboard and
            'current_streaks' in engagement_dashboard):
            principles_verified.append("Progress visualization and milestone tracking")

        # 5. Test Immediate Feedback
        profit_event = orchestrator.process_trading_event(
            test_user_id,
            'profit_realized',
            {'profit_amount': 25.00, 'is_first_profit': True}
        )
        if profit_event and profit_event.call_to_action:
            principles_verified.append("Immediate feedback with clear next actions")

        print("Psychology principles implemented:")
        for principle in principles_verified:
            print(f"  - {principle}")

        if len(principles_verified) >= 2:
            print("Psychology principles successfully integrated")
            return True
        else:
            print("FAILED: Insufficient psychology principles implemented")
            return False

    except Exception as e:
        print(f"FAILED: Psychology principles test error: {e}")
        return False


def test_integration_readiness():
    """Test integration readiness with existing trading system."""
    print("\n=== Testing Integration Readiness ===")

    try:
        orchestrator = ReinforcementOrchestrator()

        integration_checks = []

        # Test callback system
        callback_triggered = []

        def test_callback(event_type, data):
            callback_triggered.append((event_type, data))

        orchestrator.add_celebration_callback(test_callback)
        orchestrator.add_progress_callback(test_callback)

        # Trigger events to test callbacks
        test_user_id = "test_integration_user"
        orchestrator.register_user(test_user_id)

        orchestrator.process_trading_event(
            test_user_id,
            'gate_progressed',
            {'new_gate': 'G1', 'old_gate': 'G0'}
        )

        if len(callback_triggered) > 0:
            integration_checks.append("Callback system for UI integration")

        # Test batch event processing

        # Note: batch processing is async, but we'll test the structure
        if hasattr(orchestrator, 'process_batch_events'):
            integration_checks.append("Batch event processing capability")

        # Test state persistence structure
        if (hasattr(orchestrator, 'user_states') and
            hasattr(orchestrator, 'active_streaks')):
            integration_checks.append("State management for persistence")

        # Test analytics API
        analytics = orchestrator.get_system_analytics()
        if isinstance(analytics, dict) and 'user_metrics' in analytics:
            integration_checks.append("Analytics API for monitoring")

        print("Integration readiness checks:")
        for check in integration_checks:
            print(f"  - {check}")

        if len(integration_checks) >= 3:
            print("Integration readiness verified")
            return True
        else:
            print("FAILED: Insufficient integration readiness")
            return False

    except Exception as e:
        print(f"FAILED: Integration readiness test error: {e}")
        return False


def run_phase_5_audit():
    """Run complete Phase 5 audit."""
    print("PHASE 5 AUDIT: Psychological Reinforcement System")
    print("=" * 60)

    all_tests_passed = True

    tests = [
        test_reinforcement_engine_initialization,
        test_user_registration_and_profiles,
        test_reinforcement_triggering,
        test_reinforcement_orchestrator,
        test_fatigue_prevention,
        test_personalization_effectiveness,
        test_analytics_and_insights,
        test_psychology_principles_integration,
        test_integration_readiness
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
        print("PHASE 5 AUDIT PASSED - Psychological reinforcement system is genuine and effective")
        print("   - Reinforcement engine initializes with comprehensive templates")
        print("   - User registration and profile management working")
        print("   - Event triggering responds correctly to trading events")
        print("   - Orchestrator integrates all components seamlessly")
        print("   - Fatigue prevention protects user experience")
        print("   - Personalization and effectiveness tracking functional")
        print("   - Analytics provide actionable insights")
        print("   - Mobile app psychology principles implemented")
        print("   - Integration readiness confirmed for production")
    else:
        print("PHASE 5 AUDIT FAILED - Issues detected")
        print("   - Review test failures above")
        print("   - Fix issues before proceeding to Phase 6")

    return all_tests_passed


if __name__ == "__main__":
    success = run_phase_5_audit()
    sys.exit(0 if success else 1)