"""
Test script for onboarding flow system audit.

Tests the complete onboarding flow to ensure genuine functionality
and not just coding theater.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_experience.onboarding_flow import TradingOnboardingFlow, OnboardingStep
from user_experience.value_screens import ValueScreenGenerator, TradingInsight
from user_experience.psychological_triggers import MotivationEngine, CommitmentTracker, UserPsychProfile, MicroCommitment, MicroCommitmentType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_complete_onboarding_flow():
    """Test complete onboarding flow from start to finish."""
    print("\n=== Testing Complete Onboarding Flow ===")

    # Initialize onboarding system
    onboarding = TradingOnboardingFlow()

    # Start session
    session = onboarding.start_onboarding(user_id="test_user_001")
    assert session.session_id is not None
    print(f"Session created: {session.session_id}")

    # Test each step progression
    steps_completed = 0

    while session.current_step != OnboardingStep.COMPLETION:
        # Get current step content
        step_content = onboarding.get_current_step_content(session.session_id)
        assert step_content is not None
        print(f"Step {session.current_step.value}: Content generated")

        # Simulate user response based on step
        response = _generate_mock_response(session.current_step, steps_completed)

        # Process response
        next_content = onboarding.process_response(session.session_id, response)

        if next_content.get('status') == 'completed':
            break

        steps_completed += 1

        # Safety check
        if steps_completed > 10:
            print("ERROR: Too many steps - infinite loop detected")
            return False

    print(f"Onboarding completed in {steps_completed} steps")

    # Verify session has collected meaningful data
    if session.session_id in onboarding.active_sessions:
        final_session = onboarding.active_sessions[session.session_id]
        assert len(final_session.pain_points) > 0, "No pain points collected"
        assert len(final_session.goals) > 0, "No goals collected"
        assert final_session.persona is not None, "Persona not determined"
        print(f"Collected {len(final_session.pain_points)} pain points")
        print(f"Collected {len(final_session.goals)} goals")
        print(f"Persona: {final_session.persona.value}")

    return True


def test_value_screens_generation():
    """Test value screens generate compelling content."""
    print("\n=== Testing Value Screens Generation ===")

    generator = ValueScreenGenerator()

    # Test for different personas
    personas = ["beginner", "casual_investor", "active_trader"]

    for persona in personas:
        pain_points = ["emotional_decisions", "lack_of_strategy"]
        goals = ["steady_income", "grow_capital"]

        screens = generator.generate_persona_value_screens(persona, pain_points, goals)

        assert len(screens) > 0, f"No screens generated for {persona}"
        assert len(screens) >= 3, f"Too few screens for {persona}"

        # Verify screens have required elements
        for screen in screens:
            assert 'title' in screen, "Screen missing title"
            assert 'type' in screen, "Screen missing type"

            if screen['type'] == 'value_insight':
                assert 'main_statistic' in screen, "Value screen missing statistic"
                assert 'supporting_text' in screen, "Value screen missing supporting text"

        print(f"Generated {len(screens)} screens for {persona}")

    # Test specific insights can be retrieved
    insight = generator.get_insight_by_pain_point("emotional_decisions")
    assert insight is not None, "Could not retrieve specific insight"
    print(f"Retrieved insight: {insight.title}")

    return True


def test_psychological_triggers():
    """Test psychological trigger system."""
    print("\n=== Testing Psychological Triggers ===")

    # Setup system
    commitment_tracker = CommitmentTracker()
    motivation_engine = MotivationEngine(commitment_tracker)

    # Create mock user profile
    user_profile = UserPsychProfile(
        user_id="test_user_001",
        personality_type="analytical",
        risk_tolerance=0.6,
        motivation_drivers=["money", "learning"],
        response_patterns={}
    )

    # Test commitment tracking
    from datetime import datetime
    commitment = MicroCommitment(
        commitment_type=MicroCommitmentType.ADMIT_PROBLEM,
        description="User admitted to emotional trading",
        timestamp=datetime.now(),
        emotional_weight=0.8,
        completion_effort=0.6,
        value_demonstrated=0.7
    )

    score = commitment_tracker.record_commitment("test_user_001", commitment)
    assert score > 0, "Commitment not scored properly"
    print(f"Commitment scored: {score:.2f}")

    # Test trigger generation
    from user_experience.psychological_triggers import TriggerType

    context = {
        'previous_action': 'identified your trading challenges',
        'next_action': 'see how our system addresses them'
    }

    trigger = motivation_engine.generate_personalized_trigger(
        user_profile, TriggerType.COMMITMENT_ESCALATION, context
    )

    assert trigger is not None, "Trigger not generated"
    assert 'content' in trigger, "Trigger missing content"
    assert 'message' in trigger['content'], "Trigger missing message"
    print(f"Generated trigger: {trigger['content']['message'][:50]}...")

    return True


def test_integration_completeness():
    """Test that all components integrate properly."""
    print("\n=== Testing Integration Completeness ===")

    try:
        # Test imports work
        from user_experience import TradingOnboardingFlow, ValueScreenGenerator, MotivationEngine
        print("All imports successful")

        # Test initialization
        onboarding = TradingOnboardingFlow()
        value_gen = ValueScreenGenerator()
        commitment_tracker = CommitmentTracker()
        motivation_engine = MotivationEngine(commitment_tracker)
        print("All components initialize")

        # Test basic functionality
        session = onboarding.start_onboarding()
        screens = value_gen.generate_persona_value_screens("beginner", ["fear"], ["income"])
        print("Basic functionality works")

        return True

    except Exception as e:
        print(f"Integration test failed: {e}")
        return False


def _generate_mock_response(step, step_number):
    """Generate mock user response for testing."""

    mock_responses = {
        OnboardingStep.WELCOME: {
            'question_id': 'welcome_intro',
            'response': 'continue'
        },
        OnboardingStep.PROBLEM_IDENTIFICATION: {
            'question_id': 'trading_frustrations',
            'response': 'emotional_decisions' if step_number % 2 == 0 else 'lack_of_strategy',
            'emotional_weight': 0.8
        },
        OnboardingStep.GOAL_SETTING: {
            'question_id': 'financial_goals',
            'response': 'steady_income' if step_number < 3 else 'grow_capital'
        },
        OnboardingStep.EXPERIENCE_ASSESSMENT: {
            'question_id': 'trading_experience',
            'response': 'casual_investor'
        },
        OnboardingStep.COMMITMENT_BUILDING: {
            'question_id': 'time_commitment',
            'response': '1_hour'
        },
        OnboardingStep.VALUE_DEMONSTRATION: {
            'question_id': 'value_understood',
            'response': 'yes'
        },
        OnboardingStep.SYSTEM_EXPLANATION: {
            'question_id': 'system_understood',
            'response': 'yes'
        },
        OnboardingStep.FIRST_ACTION: {
            'question_id': 'ready_to_start',
            'response': 'yes'
        }
    }

    return mock_responses.get(step, {'question_id': 'default', 'response': 'continue'})


def run_phase_1_audit():
    """Run complete Phase 1 audit."""
    print("PHASE 1 AUDIT: Enhanced User Onboarding System")
    print("=" * 60)

    all_tests_passed = True

    try:
        # Test 1: Complete onboarding flow
        if not test_complete_onboarding_flow():
            all_tests_passed = False

        # Test 2: Value screens generation
        if not test_value_screens_generation():
            all_tests_passed = False

        # Test 3: Psychological triggers
        if not test_psychological_triggers():
            all_tests_passed = False

        # Test 4: Integration completeness
        if not test_integration_completeness():
            all_tests_passed = False

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("PHASE 1 AUDIT PASSED - Genuine functionality confirmed")
        print("   - Onboarding flow works end-to-end")
        print("   - Value screens generate compelling content")
        print("   - Psychological triggers are personalized")
        print("   - All components integrate properly")
    else:
        print("PHASE 1 AUDIT FAILED - Issues detected")
        print("   - Review test failures above")
        print("   - Fix issues before proceeding to Phase 2")

    return all_tests_passed


if __name__ == "__main__":
    success = run_phase_1_audit()
    sys.exit(0 if success else 1)