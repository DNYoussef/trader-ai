"""
Enhanced Gate Manager with Psychology Integration

Extends the base GateManager with psychological elements for improved
user engagement, motivation, and commitment to systematic trading.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from .gate_manager import GateManager, GateLevel
from .gate_psychology import GatePsychology

logger = logging.getLogger(__name__)


class EnhancedGateManager:
    """
    Gate Manager enhanced with psychological elements.

    Combines technical gate management with user experience improvements
    including celebration flows, motivational messaging, and achievement systems.
    """

    def __init__(self, data_dir: str = "./data/gates", enable_psychology: bool = True):
        """Initialize enhanced gate manager."""
        # Initialize base gate manager
        self.base_manager = GateManager(data_dir)

        # Initialize psychology system
        self.psychology_enabled = enable_psychology
        self.psychology = GatePsychology(self.base_manager) if enable_psychology else None

        # Event callbacks for UX integration
        self.celebration_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        self.milestone_callbacks: List[Callable] = []

        # Enhanced tracking
        self.last_progress_check = datetime.now()
        self.milestone_history: List[Dict[str, Any]] = []

        logger.info(f"Enhanced Gate Manager initialized - Psychology: {enable_psychology}")

    # Delegate core functionality to base manager
    def __getattr__(self, name):
        """Delegate unknown attributes to base manager."""
        return getattr(self.base_manager, name)

    def check_graduation_with_celebration(self, portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check graduation and trigger celebration if appropriate."""
        # Get base graduation check
        graduation_decision = self.base_manager.check_graduation(portfolio_metrics)
        current_gate = self.base_manager.current_gate

        result = {
            'graduation_decision': graduation_decision,
            'current_gate': current_gate.value,
            'celebration': None,
            'motivation': None
        }

        if graduation_decision == 'GRADUATE' and self.psychology_enabled:
            # Determine target gate
            next_gate = self._get_next_gate(current_gate)

            if next_gate:
                # Create celebration flow
                celebration = self.psychology.create_graduation_celebration(
                    current_gate, next_gate, self.base_manager.graduation_metrics
                )

                result['celebration'] = celebration
                result['next_gate'] = next_gate.value

                # Trigger celebration callbacks
                for callback in self.celebration_callbacks:
                    try:
                        callback('gate_progression', {
                            'from_gate': current_gate.value,
                            'to_gate': next_gate.value,
                            'celebration': celebration,
                            'metrics': portfolio_metrics
                        })
                    except Exception as e:
                        logger.error(f"Error in celebration callback: {e}")

        elif self.psychology_enabled:
            # Generate progress motivation for current state
            progress_data = self._calculate_progress_data(portfolio_metrics)
            motivation = self.psychology.generate_progress_motivation(
                progress_data, persona="casual_investor"  # TODO: Get from user profile
            )
            result['motivation'] = motivation

            # Trigger progress callbacks
            for callback in self.progress_callbacks:
                try:
                    callback('progress_update', {
                        'current_gate': current_gate.value,
                        'progress_data': progress_data,
                        'motivation': motivation
                    })
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

        return result

    def validate_trade_with_guidance(self, trade_details: Dict[str, Any],
                                   current_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade and provide human-friendly guidance."""
        # Get base validation
        validation_result = self.base_manager.validate_trade(trade_details, current_portfolio)

        # Enhanced result with psychology
        enhanced_result = {
            'is_valid': validation_result.is_valid,
            'violations': validation_result.violations,
            'warnings': validation_result.warnings,
            'human_guidance': None,
            'learning_opportunity': None
        }

        if not validation_result.is_valid and self.psychology_enabled:
            # Convert technical violations to human guidance
            enhanced_result['human_guidance'] = self._create_violation_guidance(
                validation_result.violations
            )

            # Check if this is a learning opportunity
            enhanced_result['learning_opportunity'] = self._identify_learning_opportunity(
                validation_result.violations
            )

        elif validation_result.warnings and self.psychology_enabled:
            # Provide encouragement for approaching limits
            enhanced_result['human_guidance'] = self._create_warning_guidance(
                validation_result.warnings
            )

        return enhanced_result

    def execute_graduation_with_experience(self) -> Dict[str, Any]:
        """Execute graduation with full psychological experience."""
        current_gate = self.base_manager.current_gate
        success = self.base_manager.execute_graduation()

        result = {
            'success': success,
            'from_gate': current_gate.value,
            'to_gate': self.base_manager.current_gate.value if success else current_gate.value,
            'celebration': None,
            'next_steps': None
        }

        if success and self.psychology_enabled:
            new_gate = self.base_manager.current_gate

            # Create celebration experience
            celebration = self.psychology.create_graduation_celebration(
                current_gate, new_gate, self.base_manager.graduation_metrics
            )

            result['celebration'] = celebration
            result['next_steps'] = celebration.next_steps

            # Trigger milestone achievement
            self._trigger_milestone_achievement('gate_graduation', {
                'from_gate': current_gate.value,
                'to_gate': new_gate.value,
                'celebration': celebration
            })

            logger.info(f"Gate graduation completed: {current_gate.value} -> {new_gate.value}")

        return result

    def track_daily_progress(self, portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track daily progress and identify milestone achievements."""
        milestones_achieved = []

        if self.psychology_enabled:
            # Check for various milestone types
            milestones_achieved.extend(self._check_profit_milestones(portfolio_metrics))
            milestones_achieved.extend(self._check_streak_milestones())
            milestones_achieved.extend(self._check_performance_milestones(portfolio_metrics))

            # Process each milestone
            for milestone in milestones_achieved:
                self._trigger_milestone_achievement(milestone['type'], milestone['data'])

        self.last_progress_check = datetime.now()

        return {
            'milestones_achieved': milestones_achieved,
            'progress_data': self._calculate_progress_data(portfolio_metrics),
            'next_check': datetime.now() + timedelta(days=1)
        }

    def get_enhanced_status_report(self) -> Dict[str, Any]:
        """Get enhanced status report with psychological elements."""
        base_status = self.base_manager.get_status_report()

        enhanced_status = {
            **base_status,
            'psychology_enabled': self.psychology_enabled,
            'progress_tracking': None,
            'next_gate_preview': None,
            'motivation_level': None
        }

        if self.psychology_enabled:
            current_gate = self.base_manager.current_gate
            next_gate = self._get_next_gate(current_gate)

            # Add progress tracking
            enhanced_status['progress_tracking'] = {
                'current_progress': self._calculate_gate_progress(),
                'days_to_graduation': self._estimate_days_to_graduation(),
                'performance_trend': self._calculate_performance_trend()
            }

            # Add next gate preview
            if next_gate:
                enhanced_status['next_gate_preview'] = self.psychology.get_gate_preview(next_gate)

            # Add motivation assessment
            enhanced_status['motivation_level'] = self._assess_motivation_level()

            # Add recent milestones
            enhanced_status['recent_milestones'] = self.milestone_history[-5:] if self.milestone_history else []

        return enhanced_status

    def add_celebration_callback(self, callback: Callable):
        """Add callback for celebration events."""
        self.celebration_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)

    def add_milestone_callback(self, callback: Callable):
        """Add callback for milestone achievements."""
        self.milestone_callbacks.append(callback)

    def _get_next_gate(self, current_gate: GateLevel) -> Optional[GateLevel]:
        """Get the next gate level."""
        gate_order = [GateLevel.G0, GateLevel.G1, GateLevel.G2, GateLevel.G3]

        try:
            current_index = gate_order.index(current_gate)
            if current_index < len(gate_order) - 1:
                return gate_order[current_index + 1]
        except ValueError:
            logger.error(f"Unknown gate level: {current_gate}")

        return None

    def _calculate_progress_data(self, portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate progress data for motivation system."""
        current_gate = self.base_manager.current_gate
        next_gate = self._get_next_gate(current_gate)

        progress_percentage = self._calculate_gate_progress()

        return {
            'current_gate': current_gate.value,
            'next_gate': next_gate.value if next_gate else None,
            'progress_percentage': progress_percentage,
            'consecutive_days': self.base_manager.graduation_metrics.consecutive_compliant_days,
            'performance_score': self.base_manager.graduation_metrics.performance_score,
            'total_violations': self.base_manager.graduation_metrics.total_violations_30d,
            'portfolio_metrics': portfolio_metrics
        }

    def _calculate_gate_progress(self) -> float:
        """Calculate progress toward next gate graduation."""
        metrics = self.base_manager.graduation_metrics

        # Get graduation criteria for current gate
        graduation_criteria = self._get_graduation_criteria()

        if not graduation_criteria:
            return 0.0

        # Calculate progress components
        progress_components = []

        # Days component
        days_progress = min(1.0, metrics.consecutive_compliant_days / graduation_criteria['min_compliant_days'])
        progress_components.append(days_progress * 0.4)  # 40% weight

        # Performance component
        performance_progress = min(1.0, metrics.performance_score / graduation_criteria['min_performance_score'])
        progress_components.append(performance_progress * 0.3)  # 30% weight

        # Violations component (inverse)
        violation_progress = 1.0 if metrics.total_violations_30d <= graduation_criteria['max_violations_30d'] else 0.5
        progress_components.append(violation_progress * 0.2)  # 20% weight

        # Capital component
        current_capital = self.base_manager.current_capital
        capital_progress = min(1.0, current_capital / graduation_criteria.get('min_capital', current_capital))
        progress_components.append(capital_progress * 0.1)  # 10% weight

        return sum(progress_components) * 100

    def _get_graduation_criteria(self) -> Optional[Dict[str, Any]]:
        """Get graduation criteria for current gate."""
        criteria_map = {
            GateLevel.G0: {
                'min_compliant_days': 14,
                'max_violations_30d': 2,
                'min_performance_score': 0.6,
                'min_capital': 500
            },
            GateLevel.G1: {
                'min_compliant_days': 21,
                'max_violations_30d': 1,
                'min_performance_score': 0.7,
                'min_capital': 1000
            },
            GateLevel.G2: {
                'min_compliant_days': 30,
                'max_violations_30d': 0,
                'min_performance_score': 0.75,
                'min_capital': 2500
            }
        }

        return criteria_map.get(self.base_manager.current_gate)

    def _estimate_days_to_graduation(self) -> Optional[int]:
        """Estimate days until ready for graduation."""
        criteria = self._get_graduation_criteria()
        if not criteria:
            return None

        metrics = self.base_manager.graduation_metrics

        days_needed = max(0, criteria['min_compliant_days'] - metrics.consecutive_compliant_days)

        if metrics.total_violations_30d > criteria['max_violations_30d']:
            days_needed = max(days_needed, 30)  # Need to wait for violation period to expire

        if metrics.performance_score < criteria['min_performance_score']:
            days_needed = max(days_needed, 14)  # Need time to improve performance

        return days_needed if days_needed > 0 else None

    def _create_violation_guidance(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create human-friendly guidance for violations."""
        if not violations:
            return None

        # Primary violation (first one)
        primary_violation = violations[0]
        violation_type = primary_violation['type']

        guidance_map = {
            'asset_not_allowed': {
                'title': "Asset Not Available Yet",
                'message': "This asset isn't available at your current gate level. This is to protect you while you build your skills.",
                'explanation': "Gate levels unlock new assets gradually as you prove your systematic trading discipline.",
                'suggestion': "Focus on mastering the assets available to you now. Gate progression will unlock more choices.",
                'tone': "protective"
            },
            'cash_floor_violation': {
                'title': "Cash Safety Check",
                'message': "This trade would bring your cash below the safety floor. This rule prevents account wipeouts.",
                'explanation': "Cash floors are your safety net - they ensure you always have capital to trade another day.",
                'suggestion': "Consider reducing your position size or wait for more cash from profits.",
                'tone': "safety-focused"
            },
            'position_size_exceeded': {
                'title': "Position Size Protection",
                'message': "This position is larger than recommended for your current level. Size matters for risk control.",
                'explanation': "Position size limits prevent any single trade from damaging your account significantly.",
                'suggestion': "Reduce your position size to stay within limits. Consistent small wins build wealth.",
                'tone': "educational"
            }
        }

        guidance = guidance_map.get(violation_type, {
            'title': "Trading Rule Check",
            'message': "This trade doesn't meet the requirements for your current gate level.",
            'explanation': "Gate rules are designed to protect your capital while you develop trading skills.",
            'suggestion': "Review your gate requirements and adjust your trade accordingly.",
            'tone': "informative"
        })

        return {
            **guidance,
            'violation_count': len(violations),
            'all_violations': violations
        }

    def _create_warning_guidance(self, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create encouraging guidance for warnings."""
        if not warnings:
            return None

        return {
            'title': "Heads Up!",
            'message': "You're approaching some limits, but you're still in the clear.",
            'explanation': "These are friendly reminders to help you stay within your risk parameters.",
            'suggestion': "Keep an eye on your position sizes and cash levels for optimal risk management.",
            'tone': "encouraging",
            'warning_count': len(warnings),
            'all_warnings': warnings
        }

    def _identify_learning_opportunity(self, violations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify if violations present learning opportunities."""
        if not violations:
            return None

        violation_type = violations[0]['type']

        learning_opportunities = {
            'position_size_exceeded': {
                'topic': "Position Sizing Mastery",
                'lesson': "Learning proper position sizing is crucial for long-term success.",
                'benefit': "Professional traders never risk more than 2-3% of their account on any single trade.",
                'action': "Practice calculating position sizes based on risk tolerance."
            },
            'cash_floor_violation': {
                'topic': "Capital Preservation",
                'lesson': "Protecting your capital is the first rule of trading.",
                'benefit': "Traders who maintain cash reserves survive market downturns and capitalize on opportunities.",
                'action': "Study how successful traders manage their cash allocations."
            }
        }

        return learning_opportunities.get(violation_type)

    def _check_profit_milestones(self, portfolio_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for profit-related milestones."""
        milestones = []

        portfolio_metrics.get('total_return', 0)
        total_profit = portfolio_metrics.get('total_profit', 0)

        # First profit milestone
        if total_profit > 0 and not self._milestone_achieved('first_profit'):
            milestones.append({
                'type': 'first_profit',
                'data': {'profit': total_profit},
                'celebration_style': 'sparkles'
            })

        # Profit thresholds
        profit_thresholds = [10, 25, 50, 100, 250, 500, 1000]
        for threshold in profit_thresholds:
            milestone_key = f'profit_{threshold}'
            if total_profit >= threshold and not self._milestone_achieved(milestone_key):
                milestones.append({
                    'type': 'profit_milestone',
                    'data': {'profit': total_profit, 'threshold': threshold},
                    'celebration_style': 'confetti' if threshold >= 100 else 'sparkles'
                })

        return milestones

    def _check_streak_milestones(self) -> List[Dict[str, Any]]:
        """Check for streak-related milestones."""
        milestones = []

        streak_days = self.base_manager.graduation_metrics.consecutive_compliant_days
        streak_thresholds = [7, 14, 21, 30, 60, 90]

        for threshold in streak_thresholds:
            milestone_key = f'streak_{threshold}'
            if streak_days >= threshold and not self._milestone_achieved(milestone_key):
                milestones.append({
                    'type': 'trading_streak',
                    'data': {'days': streak_days, 'threshold': threshold},
                    'celebration_style': 'gentle' if threshold <= 14 else 'confetti'
                })

        return milestones

    def _check_performance_milestones(self, portfolio_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance-related milestones."""
        milestones = []

        performance_score = self.base_manager.graduation_metrics.performance_score
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio_30d', 0)

        # Performance score milestones
        if performance_score >= 0.8 and not self._milestone_achieved('performance_excellent'):
            milestones.append({
                'type': 'performance_milestone',
                'data': {'performance_score': performance_score, 'level': 'excellent'},
                'celebration_style': 'confetti'
            })

        # Sharpe ratio milestones
        if sharpe_ratio >= 2.0 and not self._milestone_achieved('sharpe_excellent'):
            milestones.append({
                'type': 'sharpe_milestone',
                'data': {'sharpe_ratio': sharpe_ratio, 'level': 'excellent'},
                'celebration_style': 'sparkles'
            })

        return milestones

    def _milestone_achieved(self, milestone_key: str) -> bool:
        """Check if milestone was already achieved."""
        for milestone in self.milestone_history:
            if milestone.get('milestone_key') == milestone_key:
                return True
        return False

    def _trigger_milestone_achievement(self, milestone_type: str, milestone_data: Dict[str, Any]):
        """Trigger milestone achievement with callbacks."""
        milestone_key = f"{milestone_type}_{milestone_data.get('threshold', '')}"

        # Create milestone record
        milestone_record = {
            'milestone_key': milestone_key,
            'type': milestone_type,
            'data': milestone_data,
            'achieved_at': datetime.now(),
            'celebration_triggered': True
        }

        self.milestone_history.append(milestone_record)

        # Create celebration if psychology enabled
        if self.psychology_enabled:
            celebration = self.psychology.create_milestone_celebration(milestone_type, milestone_data)

            # Trigger milestone callbacks
            for callback in self.milestone_callbacks:
                try:
                    callback(milestone_type, {
                        'milestone': milestone_record,
                        'celebration': celebration
                    })
                except Exception as e:
                    logger.error(f"Error in milestone callback: {e}")

        logger.info(f"Milestone achieved: {milestone_type} - {milestone_data}")

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend."""
        performance_score = self.base_manager.graduation_metrics.performance_score

        if performance_score >= 0.8:
            return "excellent"
        elif performance_score >= 0.6:
            return "good"
        elif performance_score >= 0.4:
            return "improving"
        else:
            return "developing"

    def _assess_motivation_level(self) -> str:
        """Assess current motivation level."""
        metrics = self.base_manager.graduation_metrics

        # Factors that indicate high motivation
        motivation_score = 0

        if metrics.consecutive_compliant_days > 7:
            motivation_score += 1
        if metrics.total_violations_30d == 0:
            motivation_score += 1
        if metrics.performance_score > 0.6:
            motivation_score += 1

        if motivation_score >= 3:
            return "high"
        elif motivation_score >= 2:
            return "medium"
        else:
            return "developing"