"""
Gate Progression Psychology System

Implements psychological elements for gate progression to increase user engagement,
motivation, and commitment to following the systematic trading approach.

Features:
- Celebration flows for gate graduations
- Motivational messaging for progression tracking
- Achievement unlocking and feature reveals
- Psychological barriers and incentives
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from .gate_manager import GateLevel, GateManager, GraduationMetrics

logger = logging.getLogger(__name__)


class CelebrationStyle(Enum):
    """Types of celebration styles for different achievements"""
    CONFETTI = "confetti"
    SPARKLES = "sparkles"
    FIREWORKS = "fireworks"
    GENTLE = "gentle"
    EPIC = "epic"


class MotivationType(Enum):
    """Types of motivational messaging"""
    ACHIEVEMENT_FOCUS = "achievement_focus"
    PROGRESS_MOMENTUM = "progress_momentum"
    SOCIAL_PROOF = "social_proof"
    FEAR_OF_MISSING_OUT = "fear_of_missing_out"
    MASTERY_BUILDING = "mastery_building"


@dataclass
class GateUnlock:
    """Features unlocked when graduating to a new gate"""
    gate_level: GateLevel
    new_features: List[str]
    expanded_capabilities: List[str]
    psychological_rewards: List[str]
    next_gate_preview: Optional[str] = None


@dataclass
class CelebrationFlow:
    """Complete celebration flow for gate progression"""
    gate_from: GateLevel
    gate_to: GateLevel
    celebration_title: str
    celebration_subtitle: str
    achievement_description: str
    celebration_style: CelebrationStyle
    unlocks: GateUnlock
    motivational_message: str
    next_steps: List[str]
    social_share_message: str
    estimated_time_to_next: Optional[str] = None


@dataclass
class ProgressMotivation:
    """Motivational content for current progress state"""
    current_gate: GateLevel
    progress_percentage: float
    motivation_type: MotivationType
    primary_message: str
    supporting_points: List[str]
    call_to_action: str
    urgency_level: str  # low, medium, high
    social_context: Optional[str] = None


class GatePsychology:
    """
    Psychological enhancement system for gate progression.

    Converts technical gate management into engaging user experiences
    with celebration flows, motivational messaging, and achievement systems.
    """

    def __init__(self, gate_manager: GateManager):
        """Initialize gate psychology system."""
        self.gate_manager = gate_manager

        # Initialize psychological frameworks
        self.unlocks = self._initialize_gate_unlocks()
        self.celebration_flows = self._initialize_celebration_flows()
        self.motivation_templates = self._initialize_motivation_templates()

        logger.info("Gate Psychology system initialized")

    def _initialize_gate_unlocks(self) -> Dict[GateLevel, GateUnlock]:
        """Initialize feature unlocks for each gate level."""
        return {
            GateLevel.G1: GateUnlock(
                gate_level=GateLevel.G1,
                new_features=[
                    "Gold hedging with IAU and GLDM",
                    "Inflation protection with VTIP",
                    "Increased position limits (22%)",
                    "Weekly siphon eligibility"
                ],
                expanded_capabilities=[
                    "Portfolio diversification across 5 assets",
                    "60% cash floor for enhanced safety",
                    "Real-time risk monitoring",
                    "Advanced performance analytics"
                ],
                psychological_rewards=[
                    "Proven systematic trading discipline",
                    "Graduated from beginner restrictions",
                    "Access to institutional-style hedging",
                    "Recognition as developing trader"
                ],
                next_gate_preview="G2 unlocks factor ETFs and dividend strategies"
            ),

            GateLevel.G2: GateUnlock(
                gate_level=GateLevel.G2,
                new_features=[
                    "Factor ETFs (VTI, VTV, VUG, VEA, VWO)",
                    "Dividend strategies (SCHD, DGRO, NOBL, VYM)",
                    "Enhanced position limits (20%)",
                    "Advanced rebalancing algorithms"
                ],
                expanded_capabilities=[
                    "Portfolio across 14 different assets",
                    "65% cash floor with growth focus",
                    "Factor-based diversification",
                    "Dividend income optimization"
                ],
                psychological_rewards=[
                    "Advanced trader status achieved",
                    "Access to professional strategies",
                    "Significant capital growth milestone",
                    "Recognition as systematic expert"
                ],
                next_gate_preview="G3 unlocks options trading and maximum flexibility"
            ),

            GateLevel.G3: GateUnlock(
                gate_level=GateLevel.G3,
                new_features=[
                    "Long options trading enabled",
                    "SPY, QQQ, IWM, DIA access",
                    "0.5% theta exposure limit",
                    "Maximum trading flexibility"
                ],
                expanded_capabilities=[
                    "Options for enhanced income",
                    "70% cash floor with income focus",
                    "Professional-level strategies",
                    "Maximum system capabilities"
                ],
                psychological_rewards=[
                    "Master trader status unlocked",
                    "Complete system access achieved",
                    "Elite trading community member",
                    "Maximum earning potential"
                ],
                next_gate_preview=None  # Highest gate
            )
        }

    def _initialize_celebration_flows(self) -> Dict[Tuple[GateLevel, GateLevel], CelebrationFlow]:
        """Initialize celebration flows for gate transitions."""
        flows = {}

        # G0 -> G1 Celebration
        flows[(GateLevel.G0, GateLevel.G1)] = CelebrationFlow(
            gate_from=GateLevel.G0,
            gate_to=GateLevel.G1,
            celebration_title="Welcome to Gate 1!",
            celebration_subtitle="You've proven your commitment to systematic trading",
            achievement_description="Successfully graduated from beginner constraints with consistent performance and risk discipline",
            celebration_style=CelebrationStyle.CONFETTI,
            unlocks=self.unlocks[GateLevel.G1],
            motivational_message="This is where your trading journey gets exciting! You've shown you can follow a system and manage risk responsibly. Now you get access to the hedging strategies that protect institutional portfolios.",
            next_steps=[
                "Explore gold hedging with IAU/GLDM",
                "Add inflation protection with VTIP",
                "Monitor your new 22% position limits",
                "Set up automatic weekly profit siphon"
            ],
            social_share_message="Just graduated to Gate 1 in my systematic trading journey! Now I have access to gold hedging and inflation protection strategies.",
            estimated_time_to_next="4-8 weeks with consistent execution"
        )

        # G1 -> G2 Celebration
        flows[(GateLevel.G1, GateLevel.G2)] = CelebrationFlow(
            gate_from=GateLevel.G1,
            gate_to=GateLevel.G2,
            celebration_title="Advanced Trader Status!",
            celebration_subtitle="You've mastered intermediate systematic trading",
            achievement_description="Demonstrated advanced risk management and consistent profitability with diversified portfolio",
            celebration_style=CelebrationStyle.FIREWORKS,
            unlocks=self.unlocks[GateLevel.G2],
            motivational_message="Incredible progress! You're now trading like the professionals. Factor ETFs and dividend strategies are the tools institutional investors use to build long-term wealth. You've earned access to them.",
            next_steps=[
                "Diversify with factor ETFs (VTI, VTV, VUG)",
                "Add international exposure (VEA, VWO)",
                "Implement dividend income strategy",
                "Optimize factor-based rebalancing"
            ],
            social_share_message="Advanced Trader status achieved! Now accessing factor ETFs and dividend strategies used by institutional investors.",
            estimated_time_to_next="6-12 weeks with strong performance"
        )

        # G2 -> G3 Celebration
        flows[(GateLevel.G2, GateLevel.G3)] = CelebrationFlow(
            gate_from=GateLevel.G2,
            gate_to=GateLevel.G3,
            celebration_title="Master Trader Unlocked!",
            celebration_subtitle="You've reached the highest level of systematic trading",
            achievement_description="Achieved master-level discipline, risk management, and profitability across multiple market conditions",
            celebration_style=CelebrationStyle.EPIC,
            unlocks=self.unlocks[GateLevel.G3],
            motivational_message="You are now a master of systematic trading! Options access puts you in the elite category of traders who can generate income in any market condition. You have everything needed to build serious wealth.",
            next_steps=[
                "Begin conservative options strategies",
                "Monitor theta exposure carefully",
                "Maximize income generation potential",
                "Mentor other traders in the community"
            ],
            social_share_message="MASTER TRADER STATUS ACHIEVED! Full access to options trading and maximum system capabilities unlocked!",
            estimated_time_to_next=None
        )

        return flows

    def _initialize_motivation_templates(self) -> Dict[MotivationType, Dict[str, Any]]:
        """Initialize motivational messaging templates."""
        return {
            MotivationType.ACHIEVEMENT_FOCUS: {
                "templates": [
                    "You're {progress}% of the way to {next_gate}! Each day of discipline gets you closer to {reward}.",
                    "Your consistency is paying off - {achievement} shows you're mastering systematic trading.",
                    "You've already accomplished {past_achievement}, now you're building toward {future_achievement}."
                ],
                "tone": "accomplishment-focused",
                "urgency": "medium"
            },

            MotivationType.PROGRESS_MOMENTUM: {
                "templates": [
                    "You're on a roll! {streak_days} days of following your system - keep the momentum going.",
                    "Progress is accelerating: {recent_improvement}. The compound effect is starting to work.",
                    "You're in the zone! Your last {time_period} shows you're hitting your stride."
                ],
                "tone": "energetic",
                "urgency": "medium"
            },

            MotivationType.SOCIAL_PROOF: {
                "templates": [
                    "Traders at your level typically {benchmark_behavior} - you're right on track!",
                    "You're performing better than {percentile}% of traders at the same stage.",
                    "The community celebrates graduations like yours - {social_recognition}."
                ],
                "tone": "validating",
                "urgency": "low"
            },

            MotivationType.FEAR_OF_MISSING_OUT: {
                "templates": [
                    "Gate {next_gate} traders are accessing {exclusive_features} - you're so close!",
                    "While you're building discipline, {comparison_group} is already using {advanced_features}.",
                    "The next {time_period} could be your breakthrough to {major_milestone}."
                ],
                "tone": "urgent",
                "urgency": "high"
            },

            MotivationType.MASTERY_BUILDING: {
                "templates": [
                    "Each gate teaches you something new: you're mastering {current_skill}, next is {next_skill}.",
                    "Your trading skills are evolving: {skill_progression} shows real expertise developing.",
                    "From beginner to expert: you've mastered {mastered_skills}, now building {developing_skills}."
                ],
                "tone": "educational",
                "urgency": "low"
            }
        }

    def create_graduation_celebration(self, from_gate: GateLevel, to_gate: GateLevel,
                                     metrics: GraduationMetrics) -> CelebrationFlow:
        """Create personalized celebration flow for gate graduation."""
        base_flow = self.celebration_flows.get((from_gate, to_gate))

        if not base_flow:
            logger.warning(f"No celebration flow found for {from_gate.value} -> {to_gate.value}")
            return self._create_generic_celebration(from_gate, to_gate)

        # Personalize the celebration based on user metrics
        personalized_flow = CelebrationFlow(
            gate_from=base_flow.gate_from,
            gate_to=base_flow.gate_to,
            celebration_title=base_flow.celebration_title,
            celebration_subtitle=base_flow.celebration_subtitle,
            achievement_description=self._personalize_achievement_description(
                base_flow.achievement_description, metrics
            ),
            celebration_style=base_flow.celebration_style,
            unlocks=base_flow.unlocks,
            motivational_message=self._personalize_motivational_message(
                base_flow.motivational_message, metrics
            ),
            next_steps=base_flow.next_steps,
            social_share_message=base_flow.social_share_message,
            estimated_time_to_next=base_flow.estimated_time_to_next
        )

        logger.info(f"Created graduation celebration: {from_gate.value} -> {to_gate.value}")
        return personalized_flow

    def generate_progress_motivation(self, current_progress: Dict[str, Any],
                                   persona: str = "casual_investor") -> ProgressMotivation:
        """Generate motivational content based on current progress."""
        current_gate = GateLevel(current_progress.get('current_gate', 'G0'))
        progress_percentage = current_progress.get('progress_percentage', 0.0)

        # Choose motivation type based on progress and persona
        motivation_type = self._select_motivation_type(progress_percentage, persona)

        # Generate personalized message
        motivation_content = self._generate_motivation_content(
            motivation_type, current_gate, progress_percentage, current_progress
        )

        return ProgressMotivation(
            current_gate=current_gate,
            progress_percentage=progress_percentage,
            motivation_type=motivation_type,
            primary_message=motivation_content['primary_message'],
            supporting_points=motivation_content['supporting_points'],
            call_to_action=motivation_content['call_to_action'],
            urgency_level=motivation_content['urgency_level'],
            social_context=motivation_content.get('social_context')
        )

    def create_milestone_celebration(self, milestone_type: str, milestone_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create celebration for smaller milestones within gates."""
        celebrations = {
            "first_profit": {
                "title": "First Profit Achievement!",
                "message": "You've made your first profit using systematic trading - this proves the system works!",
                "style": CelebrationStyle.SPARKLES,
                "reward": "Confidence in your trading system",
                "next_step": "Keep following the system consistently"
            },

            "7_day_streak": {
                "title": "One Week of Discipline!",
                "message": "7 days of following your system shows you're building real trading habits.",
                "style": CelebrationStyle.GENTLE,
                "reward": "Habit formation momentum",
                "next_step": "Aim for a 14-day streak next"
            },

            "30_day_streak": {
                "title": "Monthly Mastery!",
                "message": "30 days of systematic execution - you're developing true trading discipline!",
                "style": CelebrationStyle.CONFETTI,
                "reward": "Proven systematic trader status",
                "next_step": "You're ready for more advanced strategies"
            },

            "first_rebalance": {
                "title": "First Rebalance Complete!",
                "message": "You've executed your first systematic rebalance - this is how professionals manage risk.",
                "style": CelebrationStyle.GENTLE,
                "reward": "Professional trading technique mastered",
                "next_step": "Watch how rebalancing improves your returns"
            },

            "risk_management_win": {
                "title": "Risk Management Success!",
                "message": "Your position sizing and risk controls just saved you from a potential loss.",
                "style": CelebrationStyle.SPARKLES,
                "reward": "Proof that systematic risk management works",
                "next_step": "Trust the system - it's protecting your capital"
            }
        }

        celebration = celebrations.get(milestone_type, {
            "title": "Progress Made!",
            "message": "You're making steady progress in your trading journey.",
            "style": CelebrationStyle.GENTLE,
            "reward": "Continuous improvement",
            "next_step": "Keep following your systematic approach"
        })

        return {
            **celebration,
            "milestone_type": milestone_type,
            "milestone_data": milestone_data,
            "timestamp": datetime.now(),
            "dismissible": True
        }

    def _personalize_achievement_description(self, base_description: str,
                                           metrics: GraduationMetrics) -> str:
        """Personalize achievement description with user metrics."""
        personalizations = []

        if metrics.consecutive_compliant_days > 21:
            personalizations.append(f"with {metrics.consecutive_compliant_days} days of perfect execution")

        if metrics.sharpe_ratio_30d and metrics.sharpe_ratio_30d > 1.0:
            personalizations.append(f"achieving a {metrics.sharpe_ratio_30d:.1f} Sharpe ratio")

        if metrics.max_drawdown_30d < 0.05:  # Less than 5% drawdown
            personalizations.append("while maintaining excellent risk control")

        if personalizations:
            return f"{base_description} ({', '.join(personalizations)})"

        return base_description

    def _personalize_motivational_message(self, base_message: str,
                                        metrics: GraduationMetrics) -> str:
        """Personalize motivational message with user-specific achievements."""
        if metrics.total_violations_30d == 0:
            return f"{base_message} Your perfect compliance record shows you have the discipline for advanced strategies."

        if metrics.consecutive_compliant_days > 30:
            return f"{base_message} Your {metrics.consecutive_compliant_days}-day streak proves you're mastering systematic trading."

        return base_message

    def _select_motivation_type(self, progress_percentage: float, persona: str) -> MotivationType:
        """Select appropriate motivation type based on progress and persona."""
        if progress_percentage < 25:
            return MotivationType.ACHIEVEMENT_FOCUS
        elif progress_percentage < 50:
            return MotivationType.PROGRESS_MOMENTUM
        elif progress_percentage < 75:
            return MotivationType.SOCIAL_PROOF
        elif progress_percentage < 90:
            return MotivationType.FEAR_OF_MISSING_OUT
        else:
            return MotivationType.MASTERY_BUILDING

    def _generate_motivation_content(self, motivation_type: MotivationType,
                                   current_gate: GateLevel, progress_percentage: float,
                                   current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific motivational content."""
        template_data = self.motivation_templates[motivation_type]

        if motivation_type == MotivationType.ACHIEVEMENT_FOCUS:
            next_gate = self._get_next_gate(current_gate)
            return {
                "primary_message": f"You're {progress_percentage:.0f}% of the way to {next_gate.value if next_gate else 'maximum level'}!",
                "supporting_points": [
                    "Each day of systematic execution builds your discipline",
                    "Gate progression unlocks powerful new strategies",
                    "You're building skills used by professional traders"
                ],
                "call_to_action": "Keep following your system - graduation is getting closer!",
                "urgency_level": "medium"
            }

        elif motivation_type == MotivationType.PROGRESS_MOMENTUM:
            return {
                "primary_message": "Your momentum is building - you're in the zone!",
                "supporting_points": [
                    f"Gate {current_gate.value} mastery is developing steadily",
                    "Systematic execution is becoming natural",
                    "Your risk management skills are strengthening"
                ],
                "call_to_action": "Don't break the streak - keep the momentum going!",
                "urgency_level": "medium"
            }

        # Add other motivation types...
        else:
            return {
                "primary_message": "You're making steady progress in your trading journey.",
                "supporting_points": ["Systematic approach is working", "Skills are developing"],
                "call_to_action": "Continue following your system",
                "urgency_level": "low"
            }

    def _get_next_gate(self, current_gate: GateLevel) -> Optional[GateLevel]:
        """Get the next gate level."""
        gate_order = [GateLevel.G0, GateLevel.G1, GateLevel.G2, GateLevel.G3]
        current_index = gate_order.index(current_gate)

        if current_index < len(gate_order) - 1:
            return gate_order[current_index + 1]

        return None

    def _create_generic_celebration(self, from_gate: GateLevel, to_gate: GateLevel) -> CelebrationFlow:
        """Create generic celebration flow as fallback."""
        return CelebrationFlow(
            gate_from=from_gate,
            gate_to=to_gate,
            celebration_title=f"Gate {to_gate.value} Unlocked!",
            celebration_subtitle="You've advanced to the next trading level",
            achievement_description="Successfully met all requirements for gate progression",
            celebration_style=CelebrationStyle.CONFETTI,
            unlocks=self.unlocks.get(to_gate, GateUnlock(
                gate_level=to_gate,
                new_features=["Enhanced trading capabilities"],
                expanded_capabilities=["Increased flexibility"],
                psychological_rewards=["Advanced trader recognition"]
            )),
            motivational_message="Congratulations on your progression! Keep building your systematic trading skills.",
            next_steps=["Explore your new capabilities", "Continue systematic execution"],
            social_share_message=f"Just graduated to Gate {to_gate.value} in my trading journey!",
            estimated_time_to_next="Continue consistent execution"
        )

    def get_gate_preview(self, target_gate: GateLevel) -> Dict[str, Any]:
        """Get preview of what's unlocked at target gate."""
        unlocks = self.unlocks.get(target_gate)

        if not unlocks:
            return {"error": f"No information available for {target_gate.value}"}

        return {
            "gate_level": target_gate.value,
            "new_features": unlocks.new_features,
            "expanded_capabilities": unlocks.expanded_capabilities,
            "psychological_rewards": unlocks.psychological_rewards,
            "next_gate_preview": unlocks.next_gate_preview,
            "motivation": f"Reaching {target_gate.value} means you'll have access to strategies used by professional traders."
        }

    def generate_near_graduation_excitement(self, metrics: GraduationMetrics,
                                          target_gate: GateLevel) -> Dict[str, Any]:
        """Generate excitement messaging when user is close to graduation."""
        days_needed = max(0, 14 - metrics.consecutive_compliant_days)  # Minimum 14 days needed
        performance_gap = max(0, 0.6 - metrics.performance_score)  # Minimum 0.6 score needed

        if days_needed == 0 and performance_gap == 0:
            urgency = "high"
            message = f"🔥 You're ready to graduate to {target_gate.value}! All requirements met!"
        elif days_needed <= 3:
            urgency = "high"
            message = f"⚡ Just {days_needed} more days of consistency to unlock {target_gate.value}!"
        elif performance_gap < 0.1:
            urgency = "medium"
            message = f"📈 You're so close to {target_gate.value}! Keep executing your system."
        else:
            urgency = "low"
            message = f"🎯 Building toward {target_gate.value} - your discipline is paying off."

        preview = self.get_gate_preview(target_gate)

        return {
            "urgency_level": urgency,
            "motivational_message": message,
            "days_to_graduation": days_needed,
            "performance_gap": performance_gap,
            "preview": preview,
            "call_to_action": "Don't stop now - you're almost there!" if urgency == "high" else "Keep building your systematic trading skills."
        }