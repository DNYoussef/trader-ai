"""
Psychological Triggers and Motivation Engine

Implements psychological principles to build user commitment,
track micro-investments, and increase engagement with the trading system.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of psychological triggers"""
    COMMITMENT_ESCALATION = "commitment_escalation"
    SOCIAL_PROOF = "social_proof"
    LOSS_AVERSION = "loss_aversion"
    PROGRESS_MOMENTUM = "progress_momentum"
    AUTHORITY_TRUST = "authority_trust"
    SCARCITY_URGENCY = "scarcity_urgency"


class MicroCommitmentType(Enum):
    """Types of micro-commitments users can make"""
    ANSWER_QUESTION = "answer_question"
    SET_GOAL = "set_goal"
    ADMIT_PROBLEM = "admit_problem"
    MAKE_CHOICE = "make_choice"
    VIEW_EDUCATION = "view_education"
    COMPLETE_SETUP = "complete_setup"
    FIRST_DEPOSIT = "first_deposit"
    FIRST_TRADE = "first_trade"


@dataclass
class MicroCommitment:
    """Individual micro-commitment made by user"""
    commitment_type: MicroCommitmentType
    description: str
    timestamp: datetime
    emotional_weight: float
    completion_effort: float  # How much effort user invested
    value_demonstrated: float  # How much value was shown
    follow_through_required: bool = False
    follow_through_completed: bool = False


@dataclass
class UserPsychProfile:
    """User psychological profile for motivation"""
    user_id: str
    personality_type: str  # analytical, emotional, social, etc.
    risk_tolerance: float
    motivation_drivers: List[str]  # money, security, status, learning, etc.
    response_patterns: Dict[str, float]
    commitment_history: List[MicroCommitment] = field(default_factory=list)
    trust_level: float = 0.5
    engagement_momentum: float = 0.5
    last_interaction: Optional[datetime] = None


class CommitmentTracker:
    """
    Tracks user micro-commitments and calculates investment level.

    Each small commitment (answering questions, setting goals, etc.)
    increases psychological investment in the system.
    """

    def __init__(self):
        """Initialize commitment tracker."""
        self.commitment_weights = self._initialize_commitment_weights()
        self.escalation_sequences = self._build_escalation_sequences()

    def _initialize_commitment_weights(self) -> Dict[MicroCommitmentType, float]:
        """Initialize weights for different commitment types."""
        return {
            MicroCommitmentType.ANSWER_QUESTION: 0.1,
            MicroCommitmentType.SET_GOAL: 0.3,
            MicroCommitmentType.ADMIT_PROBLEM: 0.4,  # High weight - admitting weakness
            MicroCommitmentType.MAKE_CHOICE: 0.2,
            MicroCommitmentType.VIEW_EDUCATION: 0.1,
            MicroCommitmentType.COMPLETE_SETUP: 0.5,
            MicroCommitmentType.FIRST_DEPOSIT: 0.8,  # Very high commitment
            MicroCommitmentType.FIRST_TRADE: 1.0     # Maximum commitment
        }

    def _build_escalation_sequences(self) -> List[List[MicroCommitmentType]]:
        """Build sequences that escalate commitment."""
        return [
            # Awareness -> Problem -> Solution -> Action
            [
                MicroCommitmentType.ANSWER_QUESTION,
                MicroCommitmentType.ADMIT_PROBLEM,
                MicroCommitmentType.SET_GOAL,
                MicroCommitmentType.VIEW_EDUCATION,
                MicroCommitmentType.COMPLETE_SETUP,
                MicroCommitmentType.FIRST_DEPOSIT,
                MicroCommitmentType.FIRST_TRADE
            ]
        ]

    def record_commitment(self, user_id: str, commitment: MicroCommitment) -> float:
        """Record a micro-commitment and return new investment level."""
        # This would integrate with user profile storage
        commitment_score = self.commitment_weights.get(commitment.commitment_type, 0.1)

        # Adjust for emotional weight and effort
        adjusted_score = commitment_score * (1 + commitment.emotional_weight) * (1 + commitment.completion_effort)

        logger.info(f"Recorded commitment for {user_id}: {commitment.commitment_type.value} (score: {adjusted_score:.2f})")

        return adjusted_score

    def calculate_investment_level(self, commitments: List[MicroCommitment]) -> float:
        """Calculate total psychological investment level."""
        if not commitments:
            return 0.0

        total_score = 0.0
        sequence_bonus = 0.0

        # Base commitment scoring
        for commitment in commitments:
            weight = self.commitment_weights.get(commitment.commitment_type, 0.1)
            effort_multiplier = 1 + commitment.completion_effort
            emotional_multiplier = 1 + commitment.emotional_weight

            commitment_score = weight * effort_multiplier * emotional_multiplier
            total_score += commitment_score

        # Bonus for following escalation sequence
        commitment_types = [c.commitment_type for c in commitments]
        for sequence in self.escalation_sequences:
            sequence_progress = self._calculate_sequence_progress(commitment_types, sequence)
            sequence_bonus += sequence_progress * 0.5  # Up to 50% bonus

        # Recency bonus - recent commitments count more
        recent_bonus = self._calculate_recency_bonus(commitments)

        final_score = total_score + sequence_bonus + recent_bonus
        return min(1.0, final_score)  # Cap at 1.0

    def _calculate_sequence_progress(self, user_commitments: List[MicroCommitmentType],
                                   target_sequence: List[MicroCommitmentType]) -> float:
        """Calculate progress through escalation sequence."""
        progress = 0.0

        for i, step in enumerate(target_sequence):
            if step in user_commitments:
                # Earlier steps in sequence get higher weight
                step_weight = (len(target_sequence) - i) / len(target_sequence)
                progress += step_weight
            else:
                break  # Sequence broken

        return progress / len(target_sequence)

    def _calculate_recency_bonus(self, commitments: List[MicroCommitment]) -> float:
        """Calculate bonus for recent commitments."""
        if not commitments:
            return 0.0

        now = datetime.now()
        recent_bonus = 0.0

        for commitment in commitments:
            days_ago = (now - commitment.timestamp).days
            if days_ago <= 1:
                recent_bonus += 0.1  # 10% bonus for same day
            elif days_ago <= 7:
                recent_bonus += 0.05  # 5% bonus for same week

        return min(0.3, recent_bonus)  # Cap at 30% bonus


class MotivationEngine:
    """
    Generates personalized motivation and psychological triggers
    based on user profile and behavior patterns.
    """

    def __init__(self, commitment_tracker: CommitmentTracker):
        """Initialize motivation engine."""
        self.commitment_tracker = commitment_tracker
        self.trigger_templates = self._build_trigger_templates()
        self.motivation_strategies = self._build_motivation_strategies()

    def _build_trigger_templates(self) -> Dict[TriggerType, List[Dict]]:
        """Build templates for psychological triggers."""
        return {
            TriggerType.COMMITMENT_ESCALATION: [
                {
                    'pattern': 'You\'ve already {previous_action}, now let\'s {next_action}',
                    'examples': {
                        'previous_action': ['identified your trading challenges', 'set your financial goals'],
                        'next_action': ['see how our system addresses them', 'configure your personalized strategy']
                    }
                },
                {
                    'pattern': 'Since you {commitment}, you\'re ready for {reward}',
                    'examples': {
                        'commitment': ['completed your profile', 'demonstrated commitment'],
                        'reward': ['advanced features', 'personalized recommendations']
                    }
                }
            ],

            TriggerType.SOCIAL_PROOF: [
                {
                    'pattern': 'Traders like you typically {behavior} and see {result}',
                    'examples': {
                        'behavior': ['start with Gate 0', 'focus on momentum strategies'],
                        'result': ['23% better risk-adjusted returns', 'faster capital growth']
                    }
                },
                {
                    'pattern': '{percentage}% of {group} report {positive_outcome}',
                    'examples': {
                        'percentage': ['89', '73', '67'],
                        'group': ['beginners', 'systematic traders', 'our users'],
                        'positive_outcome': ['increased confidence', 'better sleep', 'consistent profits']
                    }
                }
            ],

            TriggerType.LOSS_AVERSION: [
                {
                    'pattern': 'Without {solution}, most traders {negative_outcome}',
                    'examples': {
                        'solution': ['systematic risk management', 'emotional discipline'],
                        'negative_outcome': ['lose 67% within 2 years', 'make costly emotional mistakes']
                    }
                },
                {
                    'pattern': 'Don\'t let {fear} cost you {opportunity}',
                    'examples': {
                        'fear': ['analysis paralysis', 'perfectionism'],
                        'opportunity': ['steady monthly income', 'compound growth']
                    }
                }
            ],

            TriggerType.PROGRESS_MOMENTUM: [
                {
                    'pattern': 'You\'re {progress_description} - keep the momentum going!',
                    'examples': {
                        'progress_description': ['67% through setup', 'making great progress', 'almost ready to trade']
                    }
                },
                {
                    'pattern': 'You\'ve completed {milestone}, next up: {next_step}',
                    'examples': {
                        'milestone': ['risk assessment', 'goal setting', 'education module'],
                        'next_step': ['strategy configuration', 'first deposit', 'first trade']
                    }
                }
            ],

            TriggerType.AUTHORITY_TRUST: [
                {
                    'pattern': 'Based on {authority_source}, we recommend {recommendation}',
                    'examples': {
                        'authority_source': ['Nobel Prize research', 'institutional trading studies'],
                        'recommendation': ['starting with momentum strategies', 'using systematic rebalancing']
                    }
                },
                {
                    'pattern': 'Our system uses the same {methodology} as {authority_figure}',
                    'examples': {
                        'methodology': ['risk management principles', 'causal analysis'],
                        'authority_figure': ['hedge funds', 'institutional traders', 'Nassim Taleb\'s research']
                    }
                }
            ],

            TriggerType.SCARCITY_URGENCY: [
                {
                    'pattern': '{opportunity} is {time_limited} - {action_needed}',
                    'examples': {
                        'opportunity': ['Current market conditions', 'Volatility opportunities'],
                        'time_limited': ['temporary', 'won\'t last forever'],
                        'action_needed': ['start now to benefit', 'position yourself today']
                    }
                }
            ]
        }

    def _build_motivation_strategies(self) -> Dict[str, Dict]:
        """Build motivation strategies for different personality types."""
        return {
            'analytical': {
                'preferred_triggers': [TriggerType.AUTHORITY_TRUST, TriggerType.SOCIAL_PROOF],
                'content_style': 'data_heavy',
                'decision_factors': ['statistics', 'research', 'logic'],
                'persuasion_approach': 'evidence_based'
            },
            'emotional': {
                'preferred_triggers': [TriggerType.LOSS_AVERSION, TriggerType.PROGRESS_MOMENTUM],
                'content_style': 'story_driven',
                'decision_factors': ['feelings', 'outcomes', 'security'],
                'persuasion_approach': 'emotional_resonance'
            },
            'social': {
                'preferred_triggers': [TriggerType.SOCIAL_PROOF, TriggerType.COMMITMENT_ESCALATION],
                'content_style': 'community_focused',
                'decision_factors': ['peer_behavior', 'social_validation', 'belonging'],
                'persuasion_approach': 'social_validation'
            },
            'achievement': {
                'preferred_triggers': [TriggerType.PROGRESS_MOMENTUM, TriggerType.SCARCITY_URGENCY],
                'content_style': 'goal_oriented',
                'decision_factors': ['progress', 'status', 'accomplishment'],
                'persuasion_approach': 'achievement_focused'
            }
        }

    def generate_personalized_trigger(self, user_profile: UserPsychProfile,
                                    trigger_type: TriggerType,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized psychological trigger."""

        # Get user's personality strategy
        personality = user_profile.personality_type
        strategy = self.motivation_strategies.get(personality, self.motivation_strategies['analytical'])

        # Check if this trigger type is preferred for this personality
        if trigger_type not in strategy['preferred_triggers']:
            # Use backup trigger
            trigger_type = strategy['preferred_triggers'][0]

        # Get appropriate template
        templates = self.trigger_templates.get(trigger_type, [])
        if not templates:
            return self._generate_generic_trigger(context)

        # Select template based on context and user profile
        template = self._select_best_template(templates, user_profile, context)

        # Personalize the trigger content
        personalized_content = self._personalize_trigger_content(template, user_profile, context)

        return {
            'trigger_type': trigger_type.value,
            'personality_match': personality,
            'content': personalized_content,
            'effectiveness_prediction': self._predict_effectiveness(trigger_type, user_profile),
            'recommended_timing': self._get_optimal_timing(trigger_type, user_profile),
            'follow_up_required': self._requires_follow_up(trigger_type)
        }

    def _select_best_template(self, templates: List[Dict],
                            user_profile: UserPsychProfile,
                            context: Dict[str, Any]) -> Dict:
        """Select best template for user and context."""
        # For now, use first template
        # Could implement sophisticated matching logic here
        return templates[0]

    def _personalize_trigger_content(self, template: Dict,
                                   user_profile: UserPsychProfile,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize trigger content for user."""
        pattern = template['pattern']
        examples = template['examples']

        # Replace placeholders with personalized content
        personalized_pattern = pattern

        # Get user-specific values from context
        for placeholder, options in examples.items():
            if placeholder in context:
                value = context[placeholder]
            else:
                # Select most appropriate option based on user profile
                value = self._select_contextual_value(options, user_profile, context)

            personalized_pattern = personalized_pattern.replace(f'{{{placeholder}}}', value)

        return {
            'message': personalized_pattern,
            'tone': self._get_appropriate_tone(user_profile),
            'urgency_level': context.get('urgency_level', 'medium'),
            'call_to_action': self._generate_cta(user_profile, context)
        }

    def _select_contextual_value(self, options: List[str],
                               user_profile: UserPsychProfile,
                               context: Dict[str, Any]) -> str:
        """Select most appropriate value from options."""
        # Simple selection - could be made more sophisticated
        return options[0]

    def _get_appropriate_tone(self, user_profile: UserPsychProfile) -> str:
        """Get appropriate communication tone for user."""
        tone_mapping = {
            'analytical': 'professional',
            'emotional': 'empathetic',
            'social': 'friendly',
            'achievement': 'motivational'
        }
        return tone_mapping.get(user_profile.personality_type, 'professional')

    def _generate_cta(self, user_profile: UserPsychProfile, context: Dict[str, Any]) -> str:
        """Generate appropriate call-to-action."""
        cta_styles = {
            'analytical': 'Learn More',
            'emotional': 'Get Started Safely',
            'social': 'Join Others',
            'achievement': 'Take Action Now'
        }
        return cta_styles.get(user_profile.personality_type, 'Continue')

    def _predict_effectiveness(self, trigger_type: TriggerType, user_profile: UserPsychProfile) -> float:
        """Predict effectiveness of trigger for user."""
        personality = user_profile.personality_type
        strategy = self.motivation_strategies.get(personality, self.motivation_strategies['analytical'])

        if trigger_type in strategy['preferred_triggers']:
            base_effectiveness = 0.8
        else:
            base_effectiveness = 0.5

        # Adjust for user's trust level and engagement
        trust_adjustment = user_profile.trust_level * 0.3
        engagement_adjustment = user_profile.engagement_momentum * 0.2

        return min(1.0, base_effectiveness + trust_adjustment + engagement_adjustment)

    def _get_optimal_timing(self, trigger_type: TriggerType, user_profile: UserPsychProfile) -> str:
        """Get optimal timing for trigger."""
        if user_profile.last_interaction:
            hours_since = (datetime.now() - user_profile.last_interaction).total_seconds() / 3600

            if hours_since < 1:
                return 'immediate'
            elif hours_since < 24:
                return 'same_day'
            elif hours_since < 168:  # 7 days
                return 'within_week'
            else:
                return 'reengagement_needed'

        return 'immediate'

    def _requires_follow_up(self, trigger_type: TriggerType) -> bool:
        """Check if trigger requires follow-up."""
        follow_up_required = {
            TriggerType.COMMITMENT_ESCALATION: True,
            TriggerType.PROGRESS_MOMENTUM: True,
            TriggerType.SCARCITY_URGENCY: False,
            TriggerType.SOCIAL_PROOF: False,
            TriggerType.LOSS_AVERSION: True,
            TriggerType.AUTHORITY_TRUST: False
        }
        return follow_up_required.get(trigger_type, False)

    def _generate_generic_trigger(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic trigger as fallback."""
        return {
            'trigger_type': 'generic',
            'content': {
                'message': 'Continue your journey to systematic trading success',
                'tone': 'encouraging',
                'call_to_action': 'Next Step'
            },
            'effectiveness_prediction': 0.5
        }

    def create_commitment_sequence(self, user_profile: UserPsychProfile,
                                 target_action: str) -> List[Dict[str, Any]]:
        """Create sequence of escalating commitments leading to target action."""

        # Map target actions to commitment sequences
        action_sequences = {
            'first_deposit': [
                MicroCommitmentType.ANSWER_QUESTION,
                MicroCommitmentType.ADMIT_PROBLEM,
                MicroCommitmentType.SET_GOAL,
                MicroCommitmentType.VIEW_EDUCATION,
                MicroCommitmentType.COMPLETE_SETUP,
                MicroCommitmentType.FIRST_DEPOSIT
            ],
            'first_trade': [
                MicroCommitmentType.FIRST_DEPOSIT,
                MicroCommitmentType.VIEW_EDUCATION,
                MicroCommitmentType.COMPLETE_SETUP,
                MicroCommitmentType.FIRST_TRADE
            ]
        }

        sequence = action_sequences.get(target_action, [])

        # Generate triggers for each step
        trigger_sequence = []
        for i, commitment_type in enumerate(sequence):
            context = {
                'step': i + 1,
                'total_steps': len(sequence),
                'commitment_type': commitment_type.value,
                'target_action': target_action
            }

            # Vary trigger types for engagement
            trigger_type = [
                TriggerType.PROGRESS_MOMENTUM,
                TriggerType.COMMITMENT_ESCALATION,
                TriggerType.SOCIAL_PROOF
            ][i % 3]

            trigger = self.generate_personalized_trigger(user_profile, trigger_type, context)
            trigger_sequence.append(trigger)

        return trigger_sequence