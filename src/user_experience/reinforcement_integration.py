"""
Reinforcement Integration Layer

Integrates the psychological reinforcement system with existing trading system
components to automatically trigger appropriate reinforcements based on
real trading events and user behavior.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .psychological_reinforcement import (
    PsychologicalReinforcementEngine,
    ReinforcementEvent,
    TriggerCondition,
    ReinforcementType
)

# Import existing system components (placeholder imports)
try:
    from ..gates.enhanced_gate_manager import EnhancedGateManager
    from ..user_experience.onboarding_flow import TradingOnboardingFlow
    from ..user_experience.causal_education import CausalEducationAPI
except ImportError:
    # Handle import errors gracefully for testing
    pass

logger = logging.getLogger(__name__)


class ReinforcementOrchestrator:
    """
    Central orchestrator that connects all user experience components
    with the psychological reinforcement system
    """

    def __init__(self,
                 reinforcement_engine: Optional[PsychologicalReinforcementEngine] = None,
                 gate_manager = None,
                 education_api = None):
        """Initialize the reinforcement orchestrator"""

        self.reinforcement_engine = reinforcement_engine or PsychologicalReinforcementEngine()
        self.gate_manager = gate_manager
        self.education_api = education_api

        # Event handlers
        self.event_handlers = {}
        self.celebration_callbacks = []
        self.progress_callbacks = []

        # User state tracking
        self.user_states = {}
        self.active_streaks = {}

        # Setup integration hooks
        self._setup_integration_hooks()

        logger.info("Reinforcement Orchestrator initialized")

    def _setup_integration_hooks(self):
        """Setup hooks to integrate with existing systems"""

        # Setup trading event handlers
        self.event_handlers.update({
            'trade_executed': self._handle_trade_execution,
            'profit_realized': self._handle_profit_realization,
            'gate_progressed': self._handle_gate_progression,
            'concept_completed': self._handle_concept_completion,
            'tutorial_finished': self._handle_tutorial_completion,
            'risk_event_avoided': self._handle_risk_avoidance,
            'compliance_maintained': self._handle_compliance,
            'streak_updated': self._handle_streak_update
        })

    def register_user(self, user_id: str, user_profile: Dict[str, Any] = None):
        """Register a user for comprehensive reinforcement tracking"""

        # Register with reinforcement engine
        persona = user_profile.get('persona', 'balanced') if user_profile else 'balanced'
        preferences = user_profile.get('preferences', {}) if user_profile else {}

        self.reinforcement_engine.register_user(user_id, persona, preferences)

        # Initialize user state tracking
        self.user_states[user_id] = {
            'registration_date': datetime.now(),
            'total_trades': 0,
            'total_profit': 0.0,
            'current_gate': 'G0',
            'concepts_learned': [],
            'tutorials_completed': [],
            'compliance_streak': 0,
            'last_activity': datetime.now(),
            'milestone_history': []
        }

        self.active_streaks[user_id] = {
            'daily_compliance': 0,
            'profit_streak': 0,
            'learning_streak': 0
        }

        logger.info(f"User {user_id} registered for comprehensive reinforcement")

    def process_trading_event(self, user_id: str, event_type: str, event_data: Dict[str, Any]):
        """Process a trading system event for potential reinforcement"""

        if user_id not in self.user_states:
            self.register_user(user_id)

        # Update user state
        self._update_user_state(user_id, event_type, event_data)

        # Check for reinforcement triggers
        reinforcement_event = self._check_reinforcement_triggers(user_id, event_type, event_data)

        if reinforcement_event:
            # Notify callbacks
            self._notify_callbacks(reinforcement_event)

            # Return event for immediate handling
            return reinforcement_event

        return None

    def _update_user_state(self, user_id: str, event_type: str, event_data: Dict[str, Any]):
        """Update user state based on trading event"""

        state = self.user_states[user_id]
        streaks = self.active_streaks[user_id]

        state['last_activity'] = datetime.now()

        if event_type == 'trade_executed':
            state['total_trades'] += 1

        elif event_type == 'profit_realized':
            profit = event_data.get('profit_amount', 0)
            state['total_profit'] += profit

            if profit > 0:
                streaks['profit_streak'] += 1
            else:
                streaks['profit_streak'] = 0

        elif event_type == 'gate_progressed':
            state['current_gate'] = event_data.get('new_gate', state['current_gate'])
            state['milestone_history'].append({
                'type': 'gate_progression',
                'gate': event_data.get('new_gate'),
                'timestamp': datetime.now()
            })

        elif event_type == 'concept_completed':
            concept_id = event_data.get('concept_id')
            if concept_id and concept_id not in state['concepts_learned']:
                state['concepts_learned'].append(concept_id)
                streaks['learning_streak'] += 1

        elif event_type == 'tutorial_finished':
            tutorial_id = event_data.get('tutorial_id')
            if tutorial_id and tutorial_id not in state['tutorials_completed']:
                state['tutorials_completed'].append(tutorial_id)

        elif event_type == 'compliance_maintained':
            state['compliance_streak'] += 1
            streaks['daily_compliance'] += 1

    def _check_reinforcement_triggers(self,
                                    user_id: str,
                                    event_type: str,
                                    event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Check if event should trigger reinforcement"""

        handler = self.event_handlers.get(event_type)
        if handler:
            return handler(user_id, event_data)

        return None

    def _handle_trade_execution(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle trade execution events"""

        state = self.user_states[user_id]

        # Special handling for milestone trades
        if state['total_trades'] in [1, 5, 10, 25, 50, 100]:
            context_data = {
                'trade_count': state['total_trades'],
                'symbol': event_data.get('symbol', 'position'),
                'milestone': True
            }

            return self.reinforcement_engine.trigger_reinforcement(
                user_id,
                TriggerCondition.TRADE_EXECUTED,
                context_data
            )

        return None

    def _handle_profit_realization(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle profit realization events"""

        state = self.user_states[user_id]
        profit_amount = event_data.get('profit_amount', 0)

        if profit_amount <= 0:
            return None

        # Check if this is first profit
        is_first_profit = state['total_profit'] - profit_amount <= 0

        context_data = {
            'profit_amount': profit_amount,
            'is_first_profit': is_first_profit,
            'total_profit': state['total_profit'],
            'symbol': event_data.get('symbol', 'position')
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.PROFIT_ACHIEVED,
            context_data
        )

    def _handle_gate_progression(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle gate progression events"""

        context_data = {
            'new_gate': event_data.get('new_gate', 'G1'),
            'old_gate': event_data.get('old_gate', 'G0'),
            'features_unlocked': event_data.get('features_unlocked', []),
            'capital_increase': event_data.get('capital_increase', 0)
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.GATE_PROGRESSION,
            context_data
        )

    def _handle_concept_completion(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle concept learning completion"""

        context_data = {
            'concept_name': event_data.get('concept_name', 'trading concept'),
            'concept_id': event_data.get('concept_id'),
            'difficulty': event_data.get('difficulty', 'moderate')
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.CONCEPT_LEARNED,
            context_data
        )

    def _handle_tutorial_completion(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle tutorial completion"""

        context_data = {
            'tutorial_name': event_data.get('tutorial_name', 'interactive tutorial'),
            'tutorial_id': event_data.get('tutorial_id'),
            'score': event_data.get('score', 0),
            'time_taken': event_data.get('time_taken', 0)
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.TUTORIAL_COMPLETED,
            context_data
        )

    def _handle_risk_avoidance(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle risk avoidance events"""

        context_data = {
            'amount_protected': event_data.get('amount_protected', 0),
            'risk_type': event_data.get('risk_type', 'position_size'),
            'trigger_reason': event_data.get('trigger_reason', 'system_protection')
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.RISK_AVOIDED,
            context_data
        )

    def _handle_compliance(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle system compliance events"""

        state = self.user_states[user_id]

        context_data = {
            'compliance_streak': state['compliance_streak'],
            'compliance_type': event_data.get('compliance_type', 'general'),
            'total_compliant_actions': event_data.get('total_compliant_actions', state['compliance_streak'])
        }

        return self.reinforcement_engine.trigger_reinforcement(
            user_id,
            TriggerCondition.SYSTEM_COMPLIANCE,
            context_data
        )

    def _handle_streak_update(self, user_id: str, event_data: Dict[str, Any]) -> Optional[ReinforcementEvent]:
        """Handle streak milestone updates"""

        streak_type = event_data.get('streak_type', 'daily_compliance')
        streak_days = event_data.get('streak_days', 0)

        if streak_type == 'daily_compliance' and streak_days > 0:
            context_data = {
                'streak_days': streak_days,
                'streak_type': streak_type
            }

            return self.reinforcement_engine.trigger_reinforcement(
                user_id,
                TriggerCondition.STREAK_MAINTAINED,
                context_data
            )

        return None

    def add_celebration_callback(self, callback: Callable):
        """Add callback for celebration events"""
        self.celebration_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable):
        """Add callback for progress events"""
        self.progress_callbacks.append(callback)

    def _notify_callbacks(self, reinforcement_event: ReinforcementEvent):
        """Notify registered callbacks about reinforcement event"""

        event_data = {
            'event_type': 'reinforcement_triggered',
            'reinforcement_type': reinforcement_event.event_type.value,
            'user_id': reinforcement_event.user_id,
            'title': reinforcement_event.title,
            'message': reinforcement_event.message,
            'visual_effect': reinforcement_event.visual_effect,
            'intensity': reinforcement_event.intensity.value,
            'timestamp': reinforcement_event.timestamp
        }

        # Notify celebration callbacks
        if reinforcement_event.event_type in [ReinforcementType.ACHIEVEMENT, ReinforcementType.MILESTONE]:
            for callback in self.celebration_callbacks:
                try:
                    callback('celebration', event_data)
                except Exception as e:
                    logger.error(f"Error in celebration callback: {e}")

        # Notify progress callbacks
        if reinforcement_event.event_type in [ReinforcementType.PROGRESS, ReinforcementType.STREAK]:
            for callback in self.progress_callbacks:
                try:
                    callback('progress_update', event_data)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    def get_user_engagement_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive engagement dashboard for user"""

        if user_id not in self.user_states:
            return {"error": "User not found"}

        state = self.user_states[user_id]
        streaks = self.active_streaks[user_id]
        reinforcement_summary = self.reinforcement_engine.get_user_reinforcement_summary(user_id)

        return {
            "user_id": user_id,
            "engagement_overview": {
                "days_active": (datetime.now() - state['registration_date']).days,
                "total_trades": state['total_trades'],
                "total_profit": state['total_profit'],
                "current_gate": state['current_gate'],
                "concepts_mastered": len(state['concepts_learned']),
                "tutorials_completed": len(state['tutorials_completed'])
            },
            "current_streaks": streaks,
            "reinforcement_stats": reinforcement_summary,
            "milestone_progress": {
                "next_trade_milestone": self._get_next_trade_milestone(state['total_trades']),
                "next_learning_milestone": self._get_next_learning_milestone(len(state['concepts_learned'])),
                "next_gate_progress": self._estimate_gate_progress(user_id)
            },
            "recent_achievements": state['milestone_history'][-5:] if state['milestone_history'] else []
        }

    def _get_next_trade_milestone(self, current_trades: int) -> Dict[str, Any]:
        """Get next trade milestone information"""

        milestones = [1, 5, 10, 25, 50, 100, 250, 500]
        next_milestone = next((m for m in milestones if m > current_trades), None)

        if next_milestone:
            return {
                "target": next_milestone,
                "progress": current_trades,
                "remaining": next_milestone - current_trades,
                "reward": f"Milestone celebration at {next_milestone} trades"
            }

        return {
            "target": "No more milestones",
            "progress": current_trades,
            "remaining": 0,
            "reward": "You've reached all trade milestones!"
        }

    def _get_next_learning_milestone(self, concepts_learned: int) -> Dict[str, Any]:
        """Get next learning milestone information"""

        milestones = [1, 3, 5, 8, 10]
        next_milestone = next((m for m in milestones if m > concepts_learned), None)

        if next_milestone:
            return {
                "target": next_milestone,
                "progress": concepts_learned,
                "remaining": next_milestone - concepts_learned,
                "reward": f"Learning achievement at {next_milestone} concepts"
            }

        return {
            "target": "Master level reached",
            "progress": concepts_learned,
            "remaining": 0,
            "reward": "You've mastered all core concepts!"
        }

    def _estimate_gate_progress(self, user_id: str) -> Dict[str, Any]:
        """Estimate progress toward next gate"""

        state = self.user_states[user_id]
        current_gate = state['current_gate']

        # Mock gate progress estimation
        gate_requirements = {
            'G0': {'capital_target': 500, 'trades_needed': 10},
            'G1': {'capital_target': 1000, 'trades_needed': 20},
            'G2': {'capital_target': 2500, 'trades_needed': 50}
        }

        if current_gate in gate_requirements:
            req = gate_requirements[current_gate]
            return {
                "current_gate": current_gate,
                "capital_progress": min(state['total_profit'] / req['capital_target'] * 100, 100),
                "trade_progress": min(state['total_trades'] / req['trades_needed'] * 100, 100),
                "estimated_days_remaining": max(7 - (datetime.now() - state['last_activity']).days, 0)
            }

        return {
            "current_gate": current_gate,
            "status": "Maximum gate reached"
        }

    async def process_batch_events(self, events: List[Dict[str, Any]]) -> List[Optional[ReinforcementEvent]]:
        """Process multiple events efficiently"""

        results = []

        for event in events:
            user_id = event.get('user_id')
            event_type = event.get('event_type')
            event_data = event.get('event_data', {})

            if user_id and event_type:
                reinforcement = self.process_trading_event(user_id, event_type, event_data)
                results.append(reinforcement)
            else:
                results.append(None)

        return results

    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""

        reinforcement_analytics = self.reinforcement_engine.get_reinforcement_analytics()

        total_users = len(self.user_states)
        active_users = len([u for u, s in self.user_states.items()
                          if (datetime.now() - s['last_activity']).days < 7])

        total_trades = sum(s['total_trades'] for s in self.user_states.values())
        total_profit = sum(s['total_profit'] for s in self.user_states.values())

        return {
            "user_metrics": {
                "total_registered_users": total_users,
                "active_users_7d": active_users,
                "total_system_trades": total_trades,
                "total_system_profit": total_profit
            },
            "reinforcement_metrics": reinforcement_analytics,
            "engagement_metrics": {
                "average_concepts_per_user": sum(len(s['concepts_learned']) for s in self.user_states.values()) / total_users if total_users > 0 else 0,
                "average_tutorials_per_user": sum(len(s['tutorials_completed']) for s in self.user_states.values()) / total_users if total_users > 0 else 0
            }
        }