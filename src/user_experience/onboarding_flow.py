"""
Trading Onboarding Flow System

Implements self-selling onboarding technique where users admit to problems
that our trading system solves, creating psychological investment and
micro-commitments that increase conversion and retention.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class OnboardingStep(Enum):
    """Onboarding flow steps"""
    WELCOME = "welcome"
    PROBLEM_IDENTIFICATION = "problem_identification"
    GOAL_SETTING = "goal_setting"
    EXPERIENCE_ASSESSMENT = "experience_assessment"
    COMMITMENT_BUILDING = "commitment_building"
    VALUE_DEMONSTRATION = "value_demonstration"
    SYSTEM_EXPLANATION = "system_explanation"
    FIRST_ACTION = "first_action"
    COMPLETION = "completion"


class TradingPersona(Enum):
    """User trading personas"""
    BEGINNER = "beginner"
    CASUAL_INVESTOR = "casual_investor"
    ACTIVE_TRADER = "active_trader"
    EXPERIENCED_TRADER = "experienced_trader"


@dataclass
class OnboardingResponse:
    """User response to onboarding question"""
    step: OnboardingStep
    question_id: str
    response: Any
    timestamp: datetime
    emotional_weight: float = 0.0  # How emotionally charged the response is


@dataclass
class OnboardingSession:
    """Complete onboarding session data"""
    session_id: str
    user_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    current_step: OnboardingStep = OnboardingStep.WELCOME
    responses: List[OnboardingResponse] = field(default_factory=list)
    persona: Optional[TradingPersona] = None
    pain_points: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    commitment_score: float = 0.0
    abandoned: bool = False
    conversion_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class OnboardingResult:
    """Result of completed onboarding"""
    session: OnboardingSession
    recommended_starting_gate: str
    recommended_allocation: Dict[str, float]
    priority_education_topics: List[str]
    predicted_success_factors: List[str]
    risk_tolerance_score: float
    engagement_likelihood: float


class TradingOnboardingFlow:
    """
    Self-selling onboarding system for trading platform.

    Uses psychological principles to guide users through admitting
    trading challenges and building commitment to the system.
    """

    def __init__(self, data_dir: str = "./data/onboarding"):
        """Initialize onboarding flow system."""
        self.data_dir = data_dir
        self.active_sessions: Dict[str, OnboardingSession] = {}

        # Question bank for different steps
        self.questions = self._initialize_question_bank()

        # Value propositions for different personas
        self.value_props = self._initialize_value_propositions()

        logger.info("Trading onboarding flow initialized")

    def start_onboarding(self, user_id: Optional[str] = None) -> OnboardingSession:
        """Start a new onboarding session."""
        session = OnboardingSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            started_at=datetime.now()
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"Started onboarding session {session.session_id}")

        return session

    def get_current_step_content(self, session_id: str) -> Dict[str, Any]:
        """Get content for current onboarding step."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        step_content = self._get_step_content(session.current_step, session)
        return {
            'session_id': session_id,
            'current_step': session.current_step.value,
            'progress': self._calculate_progress(session),
            'content': step_content
        }

    def process_response(self, session_id: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user response and advance to next step."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Record the response
        response = OnboardingResponse(
            step=session.current_step,
            question_id=response_data.get('question_id', ''),
            response=response_data.get('response'),
            timestamp=datetime.now(),
            emotional_weight=response_data.get('emotional_weight', 0.0)
        )

        session.responses.append(response)

        # Process response based on current step
        self._process_step_response(session, response)

        # Advance to next step
        next_step = self._get_next_step(session)
        if next_step:
            session.current_step = next_step
            return self.get_current_step_content(session_id)
        else:
            # Onboarding complete
            return self._complete_onboarding(session)

    def _initialize_question_bank(self) -> Dict[OnboardingStep, List[Dict[str, Any]]]:
        """Initialize question bank for each onboarding step."""
        return {
            OnboardingStep.WELCOME: [
                {
                    'id': 'welcome_intro',
                    'type': 'info',
                    'title': 'Welcome to Gary×Taleb Trading System',
                    'content': 'Let\'s create a trading strategy that works for you.',
                    'cta': 'Get Started'
                }
            ],

            OnboardingStep.PROBLEM_IDENTIFICATION: [
                {
                    'id': 'trading_frustrations',
                    'type': 'multiple_choice',
                    'question': 'What frustrates you most about investing right now?',
                    'options': [
                        {'value': 'unpredictable_returns', 'text': 'Unpredictable returns that keep me up at night'},
                        {'value': 'time_consuming', 'text': 'Too much time researching what to buy'},
                        {'value': 'emotional_decisions', 'text': 'Making emotional decisions I regret later'},
                        {'value': 'lack_of_strategy', 'text': 'No clear strategy or system to follow'},
                        {'value': 'market_timing', 'text': 'Never knowing when to buy or sell'}
                    ],
                    'emotional_weight': 0.8
                },
                {
                    'id': 'current_challenges',
                    'type': 'multiple_select',
                    'question': 'Which of these challenges do you face? (Select all that apply)',
                    'options': [
                        {'value': 'small_account', 'text': 'Starting with a small account'},
                        {'value': 'risk_management', 'text': 'Don\'t know how much risk to take'},
                        {'value': 'information_overload', 'text': 'Overwhelmed by market information'},
                        {'value': 'consistency', 'text': 'Can\'t stick to a consistent approach'},
                        {'value': 'fear_of_loss', 'text': 'Fear of losing money paralyzes me'}
                    ],
                    'emotional_weight': 0.7
                }
            ],

            OnboardingStep.GOAL_SETTING: [
                {
                    'id': 'financial_goals',
                    'type': 'single_choice',
                    'question': 'What would an extra $500 per month mean to you?',
                    'options': [
                        {'value': 'bills', 'text': 'Help cover monthly bills and expenses'},
                        {'value': 'savings', 'text': 'Build up my emergency savings fund'},
                        {'value': 'family', 'text': 'Provide better for my family'},
                        {'value': 'freedom', 'text': 'More financial freedom and choices'},
                        {'value': 'retirement', 'text': 'Accelerate my retirement savings'}
                    ],
                    'emotional_weight': 0.9
                },
                {
                    'id': 'trading_goals',
                    'type': 'single_choice',
                    'question': 'What\'s your primary trading goal?',
                    'options': [
                        {'value': 'steady_income', 'text': 'Generate steady monthly income'},
                        {'value': 'grow_capital', 'text': 'Grow my capital over time'},
                        {'value': 'learn_trading', 'text': 'Learn professional trading skills'},
                        {'value': 'beat_market', 'text': 'Beat market returns consistently'},
                        {'value': 'financial_independence', 'text': 'Achieve financial independence'}
                    ]
                }
            ],

            OnboardingStep.EXPERIENCE_ASSESSMENT: [
                {
                    'id': 'trading_experience',
                    'type': 'single_choice',
                    'question': 'How would you describe your trading experience?',
                    'options': [
                        {'value': 'complete_beginner', 'text': 'Complete beginner - never traded before'},
                        {'value': 'some_stocks', 'text': 'Bought some stocks, not much else'},
                        {'value': 'casual_investor', 'text': 'Casual investor with basic knowledge'},
                        {'value': 'active_trader', 'text': 'Active trader with some experience'},
                        {'value': 'experienced', 'text': 'Experienced trader looking for better system'}
                    ]
                },
                {
                    'id': 'current_approach',
                    'type': 'multiple_select',
                    'question': 'How do you currently make trading decisions?',
                    'options': [
                        {'value': 'gut_feeling', 'text': 'Gut feeling and intuition'},
                        {'value': 'news_tips', 'text': 'Financial news and tips'},
                        {'value': 'technical_analysis', 'text': 'Technical analysis and charts'},
                        {'value': 'fundamental_analysis', 'text': 'Company fundamentals'},
                        {'value': 'random', 'text': 'Honestly, it\'s pretty random'},
                        {'value': 'no_system', 'text': 'I don\'t have a real system'}
                    ]
                }
            ],

            OnboardingStep.COMMITMENT_BUILDING: [
                {
                    'id': 'time_commitment',
                    'type': 'single_choice',
                    'question': 'How much time can you realistically dedicate to trading each week?',
                    'options': [
                        {'value': '30_minutes', 'text': '30 minutes or less (I want it mostly automated)'},
                        {'value': '1_hour', 'text': '1 hour (quick daily check-ins)'},
                        {'value': '3_hours', 'text': '2-3 hours (moderate involvement)'},
                        {'value': '5_hours', 'text': '5+ hours (active participation)'},
                        {'value': 'full_time', 'text': 'This could be my full-time focus'}
                    ]
                },
                {
                    'id': 'learning_commitment',
                    'type': 'single_choice',
                    'question': 'Are you willing to learn our systematic approach to trading?',
                    'options': [
                        {'value': 'absolutely', 'text': 'Absolutely - I want to learn the right way'},
                        {'value': 'probably', 'text': 'Probably - if it\'s not too complicated'},
                        {'value': 'maybe', 'text': 'Maybe - depends on how much time it takes'},
                        {'value': 'prefer_automated', 'text': 'I\'d prefer something mostly automated'}
                    ]
                }
            ]
        }

    def _initialize_value_propositions(self) -> Dict[TradingPersona, List[Dict[str, str]]]:
        """Initialize value propositions for different personas."""
        return {
            TradingPersona.BEGINNER: [
                {
                    'title': 'Start With Just $200',
                    'subtitle': 'Our Gate System Protects Beginners',
                    'content': 'New traders using gate-based systems see 67% fewer devastating losses in their first year.',
                    'visual': 'safety_net'
                },
                {
                    'title': 'No Complex Analysis Required',
                    'subtitle': 'AI Does The Heavy Lifting',
                    'content': 'Our causal intelligence system processes thousands of data points so you don\'t have to.',
                    'visual': 'ai_brain'
                }
            ],

            TradingPersona.CASUAL_INVESTOR: [
                {
                    'title': 'Systematic Approach Beats Guesswork',
                    'subtitle': 'Data-Driven Decisions',
                    'content': 'Systematic traders outperform discretionary traders 73% of the time over 3+ years.',
                    'visual': 'performance_chart'
                }
            ],

            TradingPersona.EXPERIENCED_TRADER: [
                {
                    'title': 'Advanced Causal Intelligence',
                    'subtitle': 'Beyond Technical Analysis',
                    'content': 'Our system integrates policy analysis and natural experiments - tools institutional traders use.',
                    'visual': 'advanced_analytics'
                }
            ]
        }

    def _get_step_content(self, step: OnboardingStep, session: OnboardingSession) -> Dict[str, Any]:
        """Get content for a specific onboarding step."""
        base_questions = self.questions.get(step, [])

        if step == OnboardingStep.VALUE_DEMONSTRATION:
            # Show personalized value props based on responses
            persona = self._determine_persona(session)
            value_props = self.value_props.get(persona, [])
            return {
                'type': 'value_screens',
                'screens': value_props,
                'persona': persona.value
            }

        elif step == OnboardingStep.SYSTEM_EXPLANATION:
            return {
                'type': 'system_overview',
                'content': self._generate_personalized_explanation(session)
            }

        return {
            'type': 'questions',
            'questions': base_questions
        }

    def _process_step_response(self, session: OnboardingSession, response: OnboardingResponse) -> None:
        """Process response and update session state."""
        if response.step == OnboardingStep.PROBLEM_IDENTIFICATION:
            # Extract pain points
            if response.question_id == 'trading_frustrations':
                session.pain_points.append(response.response)
            elif response.question_id == 'current_challenges':
                if isinstance(response.response, list):
                    session.pain_points.extend(response.response)
                else:
                    session.pain_points.append(response.response)

        elif response.step == OnboardingStep.GOAL_SETTING:
            # Extract goals
            session.goals.append(response.response)

        elif response.step == OnboardingStep.EXPERIENCE_ASSESSMENT:
            # Determine persona
            if response.question_id == 'trading_experience':
                session.persona = self._map_experience_to_persona(response.response)

        elif response.step == OnboardingStep.COMMITMENT_BUILDING:
            # Calculate commitment score
            commitment_weight = {
                '30_minutes': 0.3,
                '1_hour': 0.5,
                '3_hours': 0.8,
                '5_hours': 0.9,
                'full_time': 1.0,
                'absolutely': 1.0,
                'probably': 0.7,
                'maybe': 0.4,
                'prefer_automated': 0.2
            }

            weight = commitment_weight.get(response.response, 0.5)
            session.commitment_score = (session.commitment_score + weight) / 2

    def _determine_persona(self, session: OnboardingSession) -> TradingPersona:
        """Determine user persona from responses."""
        if session.persona:
            return session.persona

        # Default based on pain points if no explicit experience given
        if 'small_account' in session.pain_points or 'fear_of_loss' in session.pain_points:
            return TradingPersona.BEGINNER
        elif 'lack_of_strategy' in session.pain_points:
            return TradingPersona.CASUAL_INVESTOR
        else:
            return TradingPersona.ACTIVE_TRADER

    def _map_experience_to_persona(self, experience: str) -> TradingPersona:
        """Map experience level to persona."""
        mapping = {
            'complete_beginner': TradingPersona.BEGINNER,
            'some_stocks': TradingPersona.BEGINNER,
            'casual_investor': TradingPersona.CASUAL_INVESTOR,
            'active_trader': TradingPersona.ACTIVE_TRADER,
            'experienced': TradingPersona.EXPERIENCED_TRADER
        }
        return mapping.get(experience, TradingPersona.CASUAL_INVESTOR)

    def _generate_personalized_explanation(self, session: OnboardingSession) -> Dict[str, Any]:
        """Generate personalized system explanation based on user responses."""
        persona = self._determine_persona(session)

        explanations = {
            TradingPersona.BEGINNER: {
                'title': 'Your Personalized Trading Journey',
                'main_message': 'Based on your responses, we\'ll start you in Gate 0 with our safest approach.',
                'key_points': [
                    'Start with just $200 and proven momentum strategies',
                    '50% cash protection so you can\'t lose everything',
                    'AI guides every decision - no guesswork required',
                    'Graduate to more advanced strategies as you grow'
                ],
                'next_steps': 'Let\'s set up your first automated trade...'
            },

            TradingPersona.CASUAL_INVESTOR: {
                'title': 'Systematic Approach for Better Results',
                'main_message': 'Perfect! You\'ll love how our system removes emotion from trading.',
                'key_points': [
                    'Gary×Taleb allocation: 40% SPY, 35% momentum, 25% hedges',
                    'Weekly rebalancing removes timing stress',
                    'Causal AI prevents many common mistakes',
                    '50/50 profit split: grow account + take profits'
                ],
                'next_steps': 'Ready to see how it works with real money?'
            },

            TradingPersona.EXPERIENCED_TRADER: {
                'title': 'Professional-Grade Systematic Trading',
                'main_message': 'You\'ll appreciate the institutional-level risk management.',
                'key_points': [
                    'Causal intelligence beyond technical analysis',
                    'Policy shock detection and adaptation',
                    'Natural experiment validation of strategies',
                    'Professional gate progression system'
                ],
                'next_steps': 'Let\'s configure your advanced settings...'
            }
        }

        return explanations.get(persona, explanations[TradingPersona.CASUAL_INVESTOR])

    def _get_next_step(self, session: OnboardingSession) -> Optional[OnboardingStep]:
        """Determine next step in onboarding flow."""
        step_sequence = [
            OnboardingStep.WELCOME,
            OnboardingStep.PROBLEM_IDENTIFICATION,
            OnboardingStep.GOAL_SETTING,
            OnboardingStep.EXPERIENCE_ASSESSMENT,
            OnboardingStep.COMMITMENT_BUILDING,
            OnboardingStep.VALUE_DEMONSTRATION,
            OnboardingStep.SYSTEM_EXPLANATION,
            OnboardingStep.FIRST_ACTION,
            OnboardingStep.COMPLETION
        ]

        current_index = step_sequence.index(session.current_step)
        if current_index < len(step_sequence) - 1:
            return step_sequence[current_index + 1]

        return None

    def _calculate_progress(self, session: OnboardingSession) -> float:
        """Calculate onboarding progress percentage."""
        step_weights = {
            OnboardingStep.WELCOME: 10,
            OnboardingStep.PROBLEM_IDENTIFICATION: 20,
            OnboardingStep.GOAL_SETTING: 30,
            OnboardingStep.EXPERIENCE_ASSESSMENT: 40,
            OnboardingStep.COMMITMENT_BUILDING: 50,
            OnboardingStep.VALUE_DEMONSTRATION: 70,
            OnboardingStep.SYSTEM_EXPLANATION: 85,
            OnboardingStep.FIRST_ACTION: 95,
            OnboardingStep.COMPLETION: 100
        }

        return step_weights.get(session.current_step, 0)

    def _complete_onboarding(self, session: OnboardingSession) -> Dict[str, Any]:
        """Complete onboarding and generate recommendations."""
        session.completed_at = datetime.now()

        # Generate onboarding result
        result = OnboardingResult(
            session=session,
            recommended_starting_gate="G0",  # Most users start here
            recommended_allocation=self._calculate_recommended_allocation(session),
            priority_education_topics=self._identify_education_priorities(session),
            predicted_success_factors=self._predict_success_factors(session),
            risk_tolerance_score=self._calculate_risk_tolerance(session),
            engagement_likelihood=session.commitment_score
        )

        # Clean up active session
        if session.session_id in self.active_sessions:
            del self.active_sessions[session.session_id]

        logger.info(f"Onboarding completed for session {session.session_id}")

        return {
            'status': 'completed',
            'result': result,
            'next_action': 'setup_account'
        }

    def _calculate_recommended_allocation(self, session: OnboardingSession) -> Dict[str, float]:
        """Calculate recommended starting allocation."""
        persona = self._determine_persona(session)

        if persona == TradingPersona.BEGINNER:
            return {
                'ULTY': 0.25,
                'AMDY': 0.25,
                'cash': 0.50
            }
        else:
            return {
                'SPY': 0.20,
                'ULTY': 0.15,
                'AMDY': 0.15,
                'VTIP': 0.10,
                'IAU': 0.05,
                'cash': 0.35
            }

    def _identify_education_priorities(self, session: OnboardingSession) -> List[str]:
        """Identify priority education topics based on responses."""
        priorities = []

        if 'risk_management' in session.pain_points:
            priorities.append('risk_management_basics')

        if 'emotional_decisions' in session.pain_points:
            priorities.append('systematic_trading_psychology')

        if 'lack_of_strategy' in session.pain_points:
            priorities.append('gary_taleb_methodology')

        if 'market_timing' in session.pain_points:
            priorities.append('causal_intelligence_basics')

        return priorities[:3]  # Top 3 priorities

    def _predict_success_factors(self, session: OnboardingSession) -> List[str]:
        """Predict factors that will contribute to user success."""
        factors = []

        if session.commitment_score > 0.7:
            factors.append('high_engagement_commitment')

        if 'steady_income' in session.goals:
            factors.append('realistic_income_expectations')

        persona = self._determine_persona(session)
        if persona in [TradingPersona.CASUAL_INVESTOR, TradingPersona.ACTIVE_TRADER]:
            factors.append('sufficient_trading_experience')

        if 'small_account' not in session.pain_points:
            factors.append('adequate_starting_capital')

        return factors

    def _calculate_risk_tolerance(self, session: OnboardingSession) -> float:
        """Calculate risk tolerance score (0-1)."""
        risk_score = 0.5  # Default moderate

        if 'fear_of_loss' in session.pain_points:
            risk_score -= 0.2

        if 'small_account' in session.pain_points:
            risk_score -= 0.1

        persona = self._determine_persona(session)
        if persona == TradingPersona.EXPERIENCED_TRADER:
            risk_score += 0.2
        elif persona == TradingPersona.BEGINNER:
            risk_score -= 0.1

        return max(0.1, min(1.0, risk_score))

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for an onboarding session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}

        return {
            'session_id': session_id,
            'duration_minutes': (datetime.now() - session.started_at).total_seconds() / 60,
            'steps_completed': len(session.responses),
            'current_step': session.current_step.value,
            'progress_percentage': self._calculate_progress(session),
            'commitment_score': session.commitment_score,
            'pain_points_identified': len(session.pain_points),
            'goals_identified': len(session.goals),
            'persona': session.persona.value if session.persona else None
        }

    def abandon_session(self, session_id: str, reason: str = "unknown") -> None:
        """Mark session as abandoned."""
        session = self.active_sessions.get(session_id)
        if session:
            session.abandoned = True
            logger.info(f"Onboarding session {session_id} abandoned at step {session.current_step.value}: {reason}")