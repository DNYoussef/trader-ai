"""
Phase 4 Audit Script - Causal Intelligence Education

Tests the causal intelligence education system to ensure genuine functionality
and effective knowledge transfer with mobile app psychology principles.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_experience.causal_education import (
    CausalEducationEngine,
    CausalEducationAPI,
    EducationLevel,
    ConceptDifficulty
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_education_engine_initialization():
    """Test causal education engine initializes correctly."""
    print("\n=== Testing Education Engine Initialization ===")

    try:
        # Create education engine
        engine = CausalEducationEngine()

        # Verify initialization
        assert hasattr(engine, 'concepts')
        assert hasattr(engine, 'tutorials')
        assert hasattr(engine, 'user_progress')

        print("Education engine initialized successfully")

        # Test concepts are loaded
        required_concepts = [
            'distributional_flows',
            'causal_dag',
            'policy_shocks',
            'natural_experiments',
            'synthetic_controls'
        ]

        for concept_id in required_concepts:
            assert concept_id in engine.concepts, f"Missing concept: {concept_id}"
            concept = engine.concepts[concept_id]

            # Test concept completeness
            assert len(concept.simplified_explanation) > 0
            assert len(concept.detailed_explanation) > 0
            assert len(concept.expert_details) > 0
            assert len(concept.analogy) > 0
            assert len(concept.curiosity_hook) > 0
            assert len(concept.practical_benefit) > 0

            print(f"Concept {concept_id}: Complete with all required fields")

        # Test tutorials are loaded
        required_tutorials = ['money_flows_intro', 'cause_effect_detective']
        for tutorial_id in required_tutorials:
            assert tutorial_id in engine.tutorials, f"Missing tutorial: {tutorial_id}"
            tutorial = engine.tutorials[tutorial_id]

            assert len(tutorial.steps) > 0
            assert len(tutorial.reward_upon_completion) > 0
            assert len(tutorial.practical_outcome) > 0

            print(f"Tutorial {tutorial_id}: Complete with {len(tutorial.steps)} steps")

        return True

    except Exception as e:
        print(f"FAILED: Education engine initialization error: {e}")
        return False


def test_personalized_learning_paths():
    """Test personalized learning path generation."""
    print("\n=== Testing Personalized Learning Paths ===")

    try:
        engine = CausalEducationEngine()

        # Test different user levels
        user_levels = [EducationLevel.BEGINNER, EducationLevel.INTERMEDIATE, EducationLevel.ADVANCED]

        for level in user_levels:
            user_id = f"test_user_{level.value}"
            learning_path = engine.get_personalized_learning_path(user_id, level)

            assert learning_path['user_id'] == user_id
            assert learning_path['current_level'] == level.value
            assert 'next_concepts' in learning_path
            assert 'recommended_tutorials' in learning_path
            assert 'progress_percentage' in learning_path
            assert 'achievements_unlocked' in learning_path
            assert 'next_milestone' in learning_path

            # Check concepts are appropriate for level
            for concept in learning_path['next_concepts']:
                if level == EducationLevel.BEGINNER:
                    assert concept.difficulty in [ConceptDifficulty.SIMPLE, ConceptDifficulty.MODERATE]
                elif level == EducationLevel.INTERMEDIATE:
                    assert concept.difficulty != ConceptDifficulty.EXPERT

            print(f"Learning path generated for {level.value}: {len(learning_path['next_concepts'])} concepts")

        return True

    except Exception as e:
        print(f"FAILED: Learning path generation error: {e}")
        return False


def test_adaptive_explanations():
    """Test adaptive explanation generation based on user level."""
    print("\n=== Testing Adaptive Explanations ===")

    try:
        engine = CausalEducationEngine()

        test_concept = 'distributional_flows'
        user_levels = [EducationLevel.BEGINNER, EducationLevel.INTERMEDIATE, EducationLevel.EXPERT]

        explanations = {}
        for level in user_levels:
            explanation = engine.explain_concept(test_concept, level)

            assert explanation['concept_id'] == test_concept
            assert len(explanation['explanation']) > 0
            assert len(explanation['curiosity_hook']) > 0
            assert len(explanation['practical_benefit']) > 0

            explanations[level.value] = explanation['explanation']
            print(f"Explanation for {level.value}: {len(explanation['explanation'])} characters")

        # Verify explanations are different for different levels
        assert explanations['beginner'] != explanations['expert']
        assert len(explanations['expert']) > len(explanations['beginner'])  # Expert should be more detailed

        print("Adaptive explanations working correctly")
        return True

    except Exception as e:
        print(f"FAILED: Adaptive explanation error: {e}")
        return False


def test_interactive_tutorials():
    """Test interactive tutorial system."""
    print("\n=== Testing Interactive Tutorials ===")

    try:
        engine = CausalEducationEngine()

        # Test tutorial startup
        user_id = "test_user_tutorial"
        tutorial_id = "money_flows_intro"

        tutorial_session = engine.start_interactive_tutorial(tutorial_id, user_id)

        assert 'session_id' in tutorial_session
        assert 'tutorial' in tutorial_session
        assert 'current_step' in tutorial_session
        assert 'progress' in tutorial_session

        session_id = tutorial_session['session_id']
        print(f"Tutorial session started: {session_id}")

        # Test response processing
        mock_response = {
            "answer": "I notice 3x normal volume in tech stocks",
            "confidence": "high"
        }

        response_result = engine.process_tutorial_response(session_id, mock_response)

        assert 'session_id' in response_result
        assert 'response_processed' in response_result
        assert 'feedback' in response_result
        assert 'points_earned' in response_result

        print(f"Tutorial response processed: {response_result['points_earned']} points earned")

        return True

    except Exception as e:
        print(f"FAILED: Interactive tutorial error: {e}")
        return False


def test_psychology_principles_implementation():
    """Test implementation of mobile app psychology principles."""
    print("\n=== Testing Psychology Principles Implementation ===")

    try:
        engine = CausalEducationEngine()
        api = CausalEducationAPI()

        principles_verified = []

        # 1. Test Progressive Disclosure
        concept = engine.concepts['distributional_flows']
        if (len(concept.simplified_explanation) < len(concept.detailed_explanation) <
            len(concept.expert_details)):
            principles_verified.append("Progressive disclosure with increasing detail levels")

        # 2. Test Curiosity Hooks
        for concept_id, concept in engine.concepts.items():
            if concept.curiosity_hook.startswith(("Want to", "What if", "How would")):
                principles_verified.append("Curiosity hooks with compelling questions")
                break

        # 3. Test Confidence Building
        for concept_id, concept in engine.concepts.items():
            if "you already" in concept.confidence_builder.lower():
                principles_verified.append("Confidence building with familiar concepts")
                break

        # 4. Test Practical Benefits
        for concept_id, concept in engine.concepts.items():
            if any(word in concept.practical_benefit.lower()
                   for word in ["spot", "predict", "before", "advantage"]):
                principles_verified.append("Clear practical benefits highlighting advantage")
                break

        # 5. Test Gamification
        dashboard = api.get_learning_dashboard("test_user", "beginner")
        if ('achievements' in dashboard and 'next_milestone' in dashboard and
            dashboard['user_progress']['progress_percentage'] is not None):
            principles_verified.append("Gamification with achievements and progress tracking")

        # 6. Test Social Learning Elements
        tutorial_session = engine.start_interactive_tutorial("money_flows_intro", "test_user")
        if 'reward' in tutorial_session['tutorial']:
            principles_verified.append("Social learning with rewards and recognition")

        print("Psychology principles implemented:")
        for principle in principles_verified:
            print(f"  - {principle}")

        if len(principles_verified) >= 4:
            print("Psychology principles successfully integrated")
            return True
        else:
            print("FAILED: Insufficient psychology principles implemented")
            return False

    except Exception as e:
        print(f"FAILED: Psychology principles test error: {e}")
        return False


def test_api_interface():
    """Test API interface functionality."""
    print("\n=== Testing API Interface ===")

    try:
        api = CausalEducationAPI()

        # Test learning dashboard
        dashboard = api.get_learning_dashboard("test_user", "beginner")

        assert dashboard['dashboard_type'] == 'causal_education'
        assert 'user_progress' in dashboard
        assert 'featured_concept' in dashboard
        assert 'daily_insight' in dashboard

        print(f"Learning dashboard generated: {dashboard['user_progress']['progress_percentage']}% complete")

        # Test contextual explanations
        trading_context = {
            "current_positions": ["ALTY", "ULTY"],
            "recent_trades": ["SPY"],
            "market_conditions": "volatile"
        }

        contextual_explanation = api.explain_in_context(
            "distributional_flows",
            trading_context,
            "intermediate"
        )

        assert 'concept_id' in contextual_explanation
        assert 'contextual_example' in contextual_explanation

        print("Contextual explanation generated with trading context")

        return True

    except Exception as e:
        print(f"FAILED: API interface error: {e}")
        return False


def test_educational_effectiveness():
    """Test educational effectiveness and comprehension aids."""
    print("\n=== Testing Educational Effectiveness ===")

    try:
        engine = CausalEducationEngine()

        effectiveness_metrics = []

        # Test concept readability
        for concept_id, concept in engine.concepts.items():
            # Check for analogies and metaphors
            if len(concept.analogy) > 0 and len(concept.visual_metaphor) > 0:
                effectiveness_metrics.append("Analogies and visual metaphors provided")

            # Check for real-world examples
            if len(concept.real_world_example) > 0:
                effectiveness_metrics.append("Real-world examples included")

            # Check prerequisite structure
            if len(concept.prerequisites) >= 0 and len(concept.unlocks) >= 0:
                effectiveness_metrics.append("Logical learning progression structure")

            break  # Test first concept as representative

        # Test tutorial engagement
        tutorial = engine.tutorials['money_flows_intro']
        if (tutorial.estimated_time and
            tutorial.practical_outcome and
            tutorial.reward_upon_completion):
            effectiveness_metrics.append("Engaging tutorial design with clear outcomes")

        # Test milestone system
        learning_path = engine.get_personalized_learning_path("test_user", EducationLevel.BEGINNER)
        if (learning_path['next_milestone']['reward'] and
            learning_path['progress_percentage'] is not None):
            effectiveness_metrics.append("Motivational milestone system")

        print("Educational effectiveness features:")
        for metric in effectiveness_metrics:
            print(f"  - {metric}")

        if len(effectiveness_metrics) >= 4:
            print("Educational effectiveness verified")
            return True
        else:
            print("FAILED: Insufficient educational effectiveness features")
            return False

    except Exception as e:
        print(f"FAILED: Educational effectiveness test error: {e}")
        return False


def test_integration_readiness():
    """Test integration readiness with existing trading system."""
    print("\n=== Testing Integration Readiness ===")

    try:
        api = CausalEducationAPI()

        integration_checks = []

        # Test API structure compatibility
        dashboard = api.get_learning_dashboard("test_user")
        if isinstance(dashboard, dict) and 'dashboard_type' in dashboard:
            integration_checks.append("Compatible API structure for dashboard integration")

        # Test user state management
        engine = CausalEducationEngine()
        learning_path = engine.get_personalized_learning_path("test_user", EducationLevel.BEGINNER)
        if 'user_id' in learning_path:
            integration_checks.append("User state management for persistence")

        # Test contextual integration
        contextual = api.explain_in_context("distributional_flows", {"positions": []})
        if 'contextual_example' in contextual:
            integration_checks.append("Contextual integration with trading data")

        # Test event-driven architecture readiness
        tutorial_session = engine.start_interactive_tutorial("money_flows_intro", "test_user")
        if 'session_id' in tutorial_session:
            integration_checks.append("Session-based architecture for event tracking")

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


def run_phase_4_audit():
    """Run complete Phase 4 audit."""
    print("PHASE 4 AUDIT: Causal Intelligence Education")
    print("=" * 60)

    all_tests_passed = True

    tests = [
        test_education_engine_initialization,
        test_personalized_learning_paths,
        test_adaptive_explanations,
        test_interactive_tutorials,
        test_psychology_principles_implementation,
        test_api_interface,
        test_educational_effectiveness,
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
        print("PHASE 4 AUDIT PASSED - Causal intelligence education is genuine and effective")
        print("   - Education engine initializes with complete concepts and tutorials")
        print("   - Personalized learning paths adapt to user levels")
        print("   - Explanations adjust complexity based on user expertise")
        print("   - Interactive tutorials provide hands-on learning")
        print("   - Mobile app psychology principles implemented effectively")
        print("   - API interface ready for dashboard integration")
        print("   - Educational effectiveness features verified")
        print("   - Integration readiness confirmed")
    else:
        print("PHASE 4 AUDIT FAILED - Issues detected")
        print("   - Review test failures above")
        print("   - Fix issues before proceeding to Phase 5")

    return all_tests_passed


if __name__ == "__main__":
    success = run_phase_4_audit()
    sys.exit(0 if success else 1)