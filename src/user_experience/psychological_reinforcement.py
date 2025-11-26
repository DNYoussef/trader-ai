"""
Psychological Reinforcement System

Comprehensive system that integrates all psychological elements to create
continuous engagement, motivation, and positive reinforcement throughout
the user's trading journey. Implements mobile app psychology principles
for sustained user retention and behavior modification.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class ReinforcementType(Enum):
    """Types of psychological reinforcement"""
    ACHIEVEMENT = "achievement"
    PROGRESS = "progress"
    STREAK = "streak"
    MILESTONE = "milestone"
    SURPRISE = "surprise"
    SOCIAL = "social"
    MASTERY = "mastery"


class TriggerCondition(Enum):
    """Conditions that trigger reinforcement events"""
    TRADE_EXECUTED = "trade_executed"
    PROFIT_ACHIEVED = "profit_achieved"
    GATE_PROGRESSION = "gate_progression"
    STREAK_MAINTAINED = "streak_maintained"
    CONCEPT_LEARNED = "concept_learned"
    TUTORIAL_COMPLETED = "tutorial_completed"
    RISK_AVOIDED = "risk_avoided"
    SYSTEM_COMPLIANCE = "system_compliance"


class IntensityLevel(Enum):
    """Intensity levels for reinforcement"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    EPIC = "epic"


@dataclass
class ReinforcementEvent:
    """A psychological reinforcement event"""
    event_id: str
    event_type: ReinforcementType
    trigger_condition: TriggerCondition
    intensity: IntensityLevel
    title: str
    message: str
    visual_effect: str
    user_id: str
    timestamp: datetime

    # Optional fields with defaults
    sound_effect: Optional[str] = None
    personalization_data: Dict[str, Any] = field(default_factory=dict)
    call_to_action: Optional[str] = None
    next_goal: Optional[str] = None
    social_sharing_prompt: Optional[str] = None
    engagement_score: float = 0.0
    effectiveness_rating: Optional[float] = None


@dataclass
class UserReinforcementProfile:
    """User's reinforcement preferences and history"""
    user_id: str
    persona: str = "balanced"

    # Preferences
    preferred_celebration_style: str = "moderate"
    notification_frequency: str = "normal"  # low, normal, high
    social_sharing_enabled: bool = True

    # History tracking
    total_reinforcements: int = 0
    recent_reinforcements: List[str] = field(default_factory=list)
    effectiveness_history: List[float] = field(default_factory=list)
    last_reinforcement: Optional[datetime] = None

    # Behavioral patterns
    response_times: List[float] = field(default_factory=list)
    engagement_patterns: Dict[str, float] = field(default_factory=dict)
    fatigue_indicators: List[str] = field(default_factory=list)


class PsychologicalReinforcementEngine:
    """
    Core engine that orchestrates psychological reinforcement events
    based on user behavior, preferences, and optimal timing
    """

    def __init__(self):
        """Initialize the reinforcement engine"""
        self.user_profiles = {}
        self.reinforcement_templates = self._initialize_templates()
        self.trigger_handlers = self._setup_trigger_handlers()
        self.effectiveness_tracker = {}

        logger.info("Psychological Reinforcement Engine initialized")

    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize reinforcement event templates"""

        templates = {
            "first_profit": {
                "type": ReinforcementType.ACHIEVEMENT,
                "intensity": IntensityLevel.STRONG,
                "templates": [
                    {
                        "title": "Your First Profit!",
                        "message": "You just made ${amount} following the system - this proves it works!",
                        "visual_effect": "confetti_burst",
                        "cta": "Keep following the system for more wins",
                        "social_prompt": "I just made my first systematic trading profit!"
                    }
                ]
            },

            "gate_graduation": {
                "type": ReinforcementType.MILESTONE,
                "intensity": IntensityLevel.EPIC,
                "templates": [
                    {
                        "title": "Gate {gate_number} Unlocked!",
                        "message": "You've proven your trading discipline! New features unlocked.",
                        "visual_effect": "epic_celebration",
                        "cta": "Explore your new trading capabilities",
                        "social_prompt": "Just advanced to Gate {gate_number} in systematic trading!"
                    }
                ]
            },

            "streak_milestone": {
                "type": ReinforcementType.STREAK,
                "intensity": IntensityLevel.MODERATE,
                "templates": [
                    {
                        "title": "{days} Day Streak!",
                        "message": "Your discipline is building real trading habits!",
                        "visual_effect": "streak_animation",
                        "cta": "Keep the momentum going!",
                        "social_prompt": "{days} days of disciplined systematic trading!"
                    }
                ]
            },

            "risk_protection": {
                "type": ReinforcementType.MASTERY,
                "intensity": IntensityLevel.MODERATE,
                "templates": [
                    {
                        "title": "Risk Management Win!",
                        "message": "Your position sizing just protected you from a ${amount} loss!",
                        "visual_effect": "shield_effect",
                        "cta": "Trust your system - it's working",
                        "next_goal": "Continue building your protective habits"
                    }
                ]
            },

            "concept_mastery": {
                "type": ReinforcementType.MASTERY,
                "intensity": IntensityLevel.MODERATE,
                "templates": [
                    {
                        "title": "Concept Mastered!",
                        "message": "You now understand {concept_name} - this gives you an edge!",
                        "visual_effect": "knowledge_burst",
                        "cta": "Apply this insight to your trading",
                        "next_goal": "Master the next concept"
                    }
                ]
            },

            "surprise_delight": {
                "type": ReinforcementType.SURPRISE,
                "intensity": IntensityLevel.SUBTLE,
                "templates": [
                    {
                        "title": "Bonus Insight!",
                        "message": "Here's why your {symbol} trade is working out perfectly...",
                        "visual_effect": "sparkle_effect",
                        "cta": "See the analysis",
                        "educational_hook": True
                    }
                ]
            },

            "social_proof": {
                "type": ReinforcementType.SOCIAL,
                "intensity": IntensityLevel.MODERATE,
                "templates": [
                    {
                        "title": "You're In Good Company!",
                        "message": "{percentage}% of successful traders use systematic approaches like yours.",
                        "visual_effect": "community_highlight",
                        "cta": "Keep following proven methods",
                        "credibility_source": "Academic research on trading success"
                    }
                ]
            }
        }

        return templates

    def _setup_trigger_handlers(self) -> Dict[TriggerCondition, Callable]:
        """Setup handlers for different trigger conditions"""

        handlers = {
            TriggerCondition.TRADE_EXECUTED: self._handle_trade_execution,
            TriggerCondition.PROFIT_ACHIEVED: self._handle_profit_achievement,
            TriggerCondition.GATE_PROGRESSION: self._handle_gate_progression,
            TriggerCondition.STREAK_MAINTAINED: self._handle_streak_maintenance,
            TriggerCondition.CONCEPT_LEARNED: self._handle_concept_learning,
            TriggerCondition.TUTORIAL_COMPLETED: self._handle_tutorial_completion,
            TriggerCondition.RISK_AVOIDED: self._handle_risk_avoidance,
            TriggerCondition.SYSTEM_COMPLIANCE: self._handle_system_compliance
        }

        return handlers

    def register_user(self, user_id: str, persona: str = "balanced", preferences: Dict[str, Any] = None):
        """Register a user for personalized reinforcement"""

        if user_id not in self.user_profiles:
            profile = UserReinforcementProfile(
                user_id=user_id,
                persona=persona
            )

            if preferences:
                profile.preferred_celebration_style = preferences.get(
                    'celebration_style', 'moderate'
                )
                profile.notification_frequency = preferences.get(
                    'notification_frequency', 'normal'
                )
                profile.social_sharing_enabled = preferences.get(
                    'social_sharing', True
                )

            self.user_profiles[user_id] = profile
            logger.info(f"User {user_id} registered for psychological reinforcement")

    def trigger_reinforcement(self,
                            user_id: str,
                            trigger_condition: TriggerCondition,
                            context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Trigger a reinforcement event based on user action"""

        if user_id not in self.user_profiles:
            self.register_user(user_id)

        profile = self.user_profiles[user_id]

        # Check if reinforcement should be triggered (avoid fatigue)
        if not self._should_trigger_reinforcement(profile, trigger_condition):
            return None

        # Get appropriate handler
        handler = self.trigger_handlers.get(trigger_condition)
        if not handler:
            logger.warning(f"No handler for trigger condition: {trigger_condition}")
            return None

        # Generate reinforcement event
        reinforcement_event = handler(user_id, context_data)

        if reinforcement_event:
            # Update user profile
            self._update_user_profile(profile, reinforcement_event)

            # Track effectiveness
            self._track_reinforcement(reinforcement_event)

            logger.info(f"Reinforcement triggered for {user_id}: {reinforcement_event.title}")

        return reinforcement_event

    def _should_trigger_reinforcement(self,
                                    profile: UserReinforcementProfile,
                                    trigger_condition: TriggerCondition) -> bool:
        """Determine if reinforcement should be triggered to avoid fatigue"""

        # Always allow major milestones (gate progression, first profit, etc.)
        major_milestones = [
            TriggerCondition.GATE_PROGRESSION,
            TriggerCondition.RISK_AVOIDED,
            TriggerCondition.CONCEPT_LEARNED,
            TriggerCondition.STREAK_MAINTAINED
        ]

        if trigger_condition in major_milestones:
            return True

        # Check recent reinforcement frequency
        if profile.last_reinforcement:
            time_since_last = datetime.now() - profile.last_reinforcement
            min_interval = self._get_minimum_interval(profile.notification_frequency)

            if time_since_last < min_interval:
                return False

        # Check for fatigue indicators
        if len(profile.fatigue_indicators) > 3:
            return False

        # Check if same type of reinforcement was recent
        recent_count = len([event for event in profile.recent_reinforcements[-5:]
                          if trigger_condition.value in event])

        if recent_count >= 2:
            return False

        return True

    def _get_minimum_interval(self, frequency: str) -> timedelta:
        """Get minimum interval between reinforcements"""
        intervals = {
            'low': timedelta(hours=4),
            'normal': timedelta(hours=1),
            'high': timedelta(minutes=15)
        }
        return intervals.get(frequency, timedelta(hours=1))

    def _handle_trade_execution(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle trade execution reinforcement"""

        # Simple trade execution acknowledgment
        if context_data.get('trade_count', 0) % 5 == 1:  # Every 5th trade
            return self._create_reinforcement_event(
                user_id=user_id,
                template_key="surprise_delight",
                context_data={
                    'symbol': context_data.get('symbol', 'your position'),
                    'reason': 'systematic approach'
                },
                trigger_condition=TriggerCondition.TRADE_EXECUTED
            )

        return None

    def _handle_profit_achievement(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle profit achievement reinforcement"""

        profit_amount = context_data.get('profit_amount', 0)
        is_first_profit = context_data.get('is_first_profit', False)

        if is_first_profit:
            template_key = "first_profit"
        else:
            template_key = "surprise_delight"

        context = {'amount': f"{profit_amount:.2f}"}

        # Add symbol for surprise_delight template
        if template_key == "surprise_delight":
            context['symbol'] = context_data.get('symbol', 'your position')

        return self._create_reinforcement_event(
            user_id=user_id,
            template_key=template_key,
            context_data=context,
            trigger_condition=TriggerCondition.PROFIT_ACHIEVED
        )

    def _handle_gate_progression(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle gate progression reinforcement"""

        new_gate = context_data.get('new_gate', 'G1')
        features_unlocked = context_data.get('features_unlocked', [])

        return self._create_reinforcement_event(
            user_id=user_id,
            template_key="gate_graduation",
            context_data={'gate_number': new_gate, 'features': features_unlocked},
            trigger_condition=TriggerCondition.GATE_PROGRESSION
        )

    def _handle_streak_maintenance(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle streak maintenance reinforcement"""

        streak_days = context_data.get('streak_days', 7)

        if streak_days in [7, 14, 30, 60, 100]:  # Milestone days
            return self._create_reinforcement_event(
                user_id=user_id,
                template_key="streak_milestone",
                context_data={'days': streak_days},
                trigger_condition=TriggerCondition.STREAK_MAINTAINED
            )

        return None

    def _handle_concept_learning(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle concept learning reinforcement"""

        concept_name = context_data.get('concept_name', 'trading concept')

        return self._create_reinforcement_event(
            user_id=user_id,
            template_key="concept_mastery",
            context_data={'concept_name': concept_name},
            trigger_condition=TriggerCondition.CONCEPT_LEARNED
        )

    def _handle_tutorial_completion(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle tutorial completion reinforcement"""

        # Tutorial completion is handled by the tutorial system itself
        # This is for additional reinforcement if needed
        return None

    def _handle_risk_avoidance(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle risk avoidance reinforcement"""

        amount_protected = context_data.get('amount_protected', 0)

        if amount_protected > 10:  # Significant protection
            return self._create_reinforcement_event(
                user_id=user_id,
                template_key="risk_protection",
                context_data={'amount': f"{amount_protected:.2f}"},
                trigger_condition=TriggerCondition.RISK_AVOIDED
            )

        return None

    def _handle_system_compliance(self, user_id: str, context_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle system compliance reinforcement"""

        compliance_streak = context_data.get('compliance_streak', 0)

        if compliance_streak > 0 and compliance_streak % 10 == 0:  # Every 10 compliant actions
            return self._create_reinforcement_event(
                user_id=user_id,
                template_key="social_proof",
                context_data={'percentage': 85},  # Static for now
                trigger_condition=TriggerCondition.SYSTEM_COMPLIANCE
            )

        return None

    def _create_reinforcement_event(self,
                                  user_id: str,
                                  template_key: str,
                                  context_data: Dict[str, Any],
                                  trigger_condition: TriggerCondition) -> ReinforcementEvent:
        """Create a reinforcement event from template"""

        template_config = self.reinforcement_templates[template_key]
        template = random.choice(template_config['templates'])

        # Personalize content
        title = template['title'].format(**context_data)
        message = template['message'].format(**context_data)

        # Create event
        event = ReinforcementEvent(
            event_id=f"{user_id}_{trigger_condition.value}_{int(datetime.now().timestamp())}",
            event_type=template_config['type'],
            trigger_condition=trigger_condition,
            intensity=template_config['intensity'],
            title=title,
            message=message,
            visual_effect=template['visual_effect'],
            sound_effect=template.get('sound_effect'),
            user_id=user_id,
            timestamp=datetime.now(),
            personalization_data=context_data,
            call_to_action=template.get('cta'),
            next_goal=template.get('next_goal'),
            social_sharing_prompt=template.get('social_prompt', '').format(**context_data) if template.get('social_prompt') else None
        )

        return event

    def _update_user_profile(self, profile: UserReinforcementProfile, event: ReinforcementEvent):
        """Update user profile after reinforcement event"""

        profile.total_reinforcements += 1
        profile.recent_reinforcements.append(f"{event.trigger_condition.value}_{event.timestamp}")
        profile.last_reinforcement = event.timestamp

        # Keep only recent reinforcements
        if len(profile.recent_reinforcements) > 20:
            profile.recent_reinforcements = profile.recent_reinforcements[-20:]

    def _track_reinforcement(self, event: ReinforcementEvent):
        """Track reinforcement for effectiveness analysis"""

        if event.user_id not in self.effectiveness_tracker:
            self.effectiveness_tracker[event.user_id] = []

        self.effectiveness_tracker[event.user_id].append({
            'event_id': event.event_id,
            'type': event.event_type.value,
            'intensity': event.intensity.value,
            'timestamp': event.timestamp.isoformat(),
            'engagement_score': event.engagement_score
        })

    def get_user_reinforcement_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's reinforcement history"""

        if user_id not in self.user_profiles:
            return {"error": "User not found"}

        profile = self.user_profiles[user_id]

        return {
            "user_id": user_id,
            "total_reinforcements": profile.total_reinforcements,
            "recent_reinforcements": len(profile.recent_reinforcements),
            "last_reinforcement": profile.last_reinforcement.isoformat() if profile.last_reinforcement else None,
            "average_effectiveness": sum(profile.effectiveness_history) / len(profile.effectiveness_history) if profile.effectiveness_history else 0,
            "preferences": {
                "celebration_style": profile.preferred_celebration_style,
                "notification_frequency": profile.notification_frequency,
                "social_sharing_enabled": profile.social_sharing_enabled
            },
            "engagement_patterns": profile.engagement_patterns
        }

    def update_reinforcement_effectiveness(self,
                                        event_id: str,
                                        user_feedback: Dict[str, Any]):
        """Update effectiveness rating based on user feedback"""

        for user_id, events in self.effectiveness_tracker.items():
            for event in events:
                if event['event_id'] == event_id:
                    event['user_feedback'] = user_feedback
                    event['effectiveness_rating'] = user_feedback.get('rating', 0)

                    # Update user profile
                    if user_id in self.user_profiles:
                        profile = self.user_profiles[user_id]
                        profile.effectiveness_history.append(user_feedback.get('rating', 0))

                        # Keep only recent effectiveness data
                        if len(profile.effectiveness_history) > 50:
                            profile.effectiveness_history = profile.effectiveness_history[-50:]

                    logger.info(f"Updated effectiveness for event {event_id}: {user_feedback.get('rating', 0)}")
                    return

    def get_reinforcement_analytics(self) -> Dict[str, Any]:
        """Get analytics on reinforcement system performance"""

        total_events = sum(len(events) for events in self.effectiveness_tracker.values())
        total_users = len(self.user_profiles)

        type_distribution = {}
        intensity_distribution = {}
        effectiveness_scores = []

        for events in self.effectiveness_tracker.values():
            for event in events:
                # Count by type
                event_type = event['type']
                type_distribution[event_type] = type_distribution.get(event_type, 0) + 1

                # Count by intensity
                intensity = event['intensity']
                intensity_distribution[intensity] = intensity_distribution.get(intensity, 0) + 1

                # Collect effectiveness scores
                if event.get('effectiveness_rating'):
                    effectiveness_scores.append(event['effectiveness_rating'])

        return {
            "total_reinforcement_events": total_events,
            "active_users": total_users,
            "average_events_per_user": total_events / total_users if total_users > 0 else 0,
            "reinforcement_type_distribution": type_distribution,
            "intensity_distribution": intensity_distribution,
            "average_effectiveness": sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0,
            "effectiveness_count": len(effectiveness_scores)
        }