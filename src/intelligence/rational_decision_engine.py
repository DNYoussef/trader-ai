"""
Matt Freeman's Rational Decision Theory Engine
Implements the Guild of the Rose 5-step decision process for trading decisions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class UtilityFunction:
    """Personal utility function for risk-adjusted decision making"""

    def __init__(self, risk_aversion: float = 0.5, loss_aversion: float = 2.0):
        self.risk_aversion = risk_aversion  # 0 = risk-neutral, 1 = highly risk-averse
        self.loss_aversion = loss_aversion  # Multiplier for losses vs gains

    def calculate_utility(self, outcome: float, baseline: float = 0) -> float:
        """Calculate utility using prospect theory"""
        if outcome >= baseline:
            # Gains: Concave utility function
            return (outcome - baseline) ** (1 - self.risk_aversion)
        else:
            # Losses: Convex utility function with loss aversion
            return -self.loss_aversion * ((baseline - outcome) ** (1 - self.risk_aversion))

@dataclass
class DecisionOption:
    """Single decision option in the framework"""
    name: str
    description: str
    entry_cost: Decimal
    position_size: Decimal
    time_horizon: timedelta
    metadata: Dict[str, Any]

@dataclass
class CriticalVariable:
    """Key variable affecting decision outcomes"""
    name: str
    description: str
    impact_weight: float  # 0-1 scale
    current_value: Any
    probability_distribution: Dict[str, float]  # scenario -> probability
    uncertainty_level: float  # 0-1 scale

@dataclass
class OutcomeBin:
    """Categorized possible outcome"""
    name: str
    description: str
    probability: float
    expected_return: float
    worst_case: float
    best_case: float
    variables_causing: List[str]

@dataclass
class DecisionTree:
    """Complete decision tree with all branches"""
    root_decision: str
    options: List[DecisionOption]
    variables: List[CriticalVariable]
    outcomes: List[OutcomeBin]
    utility_scores: Dict[str, float]
    expected_utility: float
    recommendation: str
    confidence_interval: Tuple[float, float]

class CalibrationTracker:
    """Tracks user prediction accuracy for calibration training"""

    def __init__(self):
        self.predictions: List[Dict] = []
        self.accuracy_by_confidence = {}

    def record_prediction(self, confidence: float, prediction: Any,
                         actual: Optional[Any] = None) -> str:
        """Record a prediction for later scoring"""
        prediction_id = f"pred_{len(self.predictions)}_{datetime.now().isoformat()}"

        self.predictions.append({
            'id': prediction_id,
            'confidence': confidence,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now(),
            'resolved': actual is not None
        })

        return prediction_id

    def resolve_prediction(self, prediction_id: str, actual: Any) -> bool:
        """Resolve a prediction with actual outcome"""
        for pred in self.predictions:
            if pred['id'] == prediction_id:
                pred['actual'] = actual
                pred['resolved'] = True
                self._update_calibration_scores()
                return True
        return False

    def _update_calibration_scores(self):
        """Update calibration accuracy by confidence level"""
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for bin_level in confidence_bins:
            relevant_preds = [
                p for p in self.predictions
                if p['resolved'] and abs(p['confidence'] - bin_level) <= 0.05
            ]

            if relevant_preds:
                correct = sum(1 for p in relevant_preds if p['prediction'] == p['actual'])
                accuracy = correct / len(relevant_preds)
                self.accuracy_by_confidence[bin_level] = {
                    'accuracy': accuracy,
                    'sample_size': len(relevant_preds),
                    'calibration_error': abs(accuracy - bin_level)
                }

    def get_calibration_adjustment(self, stated_confidence: float) -> float:
        """Adjust confidence based on historical calibration"""
        if not self.accuracy_by_confidence:
            return stated_confidence

        # Find closest calibration bin
        closest_bin = min(self.accuracy_by_confidence.keys(),
                         key=lambda x: abs(x - stated_confidence))

        calibration_data = self.accuracy_by_confidence[closest_bin]
        if calibration_data['sample_size'] >= 10:  # Require minimum sample
            return calibration_data['accuracy']

        return stated_confidence

class RationalDecisionEngine:
    """
    Implements Matt Freeman's 5-step rational decision process:
    1. List the Options
    2. List Critical Variables
    3. Bin Possible Outcomes
    4. Rank Possible Outcomes
    5. Assign Utility Values
    """

    def __init__(self, utility_function: Optional[UtilityFunction] = None):
        self.utility_function = utility_function or UtilityFunction()
        self.calibration_tracker = CalibrationTracker()
        self.decision_history: List[DecisionTree] = []

    def analyze_contrarian_opportunity(self, opportunity: Dict[str, Any],
                                     user_context: Dict[str, Any]) -> DecisionTree:
        """
        Complete rational analysis of a contrarian trading opportunity
        """
        logger.info(f"Analyzing contrarian opportunity: {opportunity.get('topic', 'Unknown')}")

        # Step 1: List the Options
        options = self._generate_options(opportunity, user_context)

        # Step 2: List Critical Variables
        variables = self._identify_critical_variables(opportunity)

        # Step 3: Bin Possible Outcomes
        outcomes = self._bin_outcomes(opportunity, variables, options)

        # Step 4: Rank Possible Outcomes
        ranked_outcomes = self._rank_outcomes(outcomes)

        # Step 5: Assign Utility Values
        utility_scores = self._assign_utility_values(ranked_outcomes, user_context)

        # Generate final recommendation
        recommendation, expected_utility, confidence = self._generate_recommendation(
            options, outcomes, utility_scores, variables
        )

        decision_tree = DecisionTree(
            root_decision=f"Trade: {opportunity.get('topic', 'Unknown')}",
            options=options,
            variables=variables,
            outcomes=ranked_outcomes,
            utility_scores=utility_scores,
            expected_utility=expected_utility,
            recommendation=recommendation,
            confidence_interval=confidence
        )

        self.decision_history.append(decision_tree)
        return decision_tree

    def _generate_options(self, opportunity: Dict[str, Any],
                         user_context: Dict[str, Any]) -> List[DecisionOption]:
        """Step 1: Generate all possible trading options"""

        base_position_size = Decimal(str(user_context.get('max_position_size', 1000)))
        conviction = opportunity.get('conviction', 0.5)

        options = [
            # No action
            DecisionOption(
                name="No Trade",
                description="Do not enter this position",
                entry_cost=Decimal("0"),
                position_size=Decimal("0"),
                time_horizon=timedelta(days=0),
                metadata={"risk_level": "none", "rationale": "Avoid opportunity"}
            ),

            # Conservative position
            DecisionOption(
                name="Conservative Position",
                description="Small position with tight stops",
                entry_cost=base_position_size * Decimal("0.02"),
                position_size=base_position_size * Decimal("0.02"),
                time_horizon=timedelta(days=30),
                metadata={"risk_level": "low", "stop_loss": 0.05}
            ),

            # Standard position
            DecisionOption(
                name="Standard Position",
                description="Normal position size based on conviction",
                entry_cost=base_position_size * Decimal(str(conviction * 0.1)),
                position_size=base_position_size * Decimal(str(conviction * 0.1)),
                time_horizon=timedelta(days=90),
                metadata={"risk_level": "medium", "stop_loss": 0.10}
            ),

            # Aggressive position (Gary moment)
            DecisionOption(
                name="Gary Moment",
                description="Large conviction bet when consensus is extremely wrong",
                entry_cost=base_position_size * Decimal(str(conviction * 0.2)),
                position_size=base_position_size * Decimal(str(conviction * 0.2)),
                time_horizon=timedelta(days=180),
                metadata={"risk_level": "high", "stop_loss": 0.15}
            )
        ]

        return options

    def _identify_critical_variables(self, opportunity: Dict[str, Any]) -> List[CriticalVariable]:
        """Step 2: Identify key variables affecting outcomes"""

        variables = [
            CriticalVariable(
                name="Inequality Trend",
                description="Direction and speed of wealth concentration",
                impact_weight=0.4,
                current_value=opportunity.get('inequality_metrics', {}),
                probability_distribution={
                    "accelerating": 0.4,
                    "stable": 0.4,
                    "reversing": 0.2
                },
                uncertainty_level=0.3
            ),

            CriticalVariable(
                name="Consensus Strength",
                description="How strongly held is the consensus view",
                impact_weight=0.3,
                current_value=opportunity.get('consensus_strength', 0.7),
                probability_distribution={
                    "very_strong": 0.3,
                    "moderate": 0.5,
                    "weak": 0.2
                },
                uncertainty_level=0.2
            ),

            CriticalVariable(
                name="Policy Response",
                description="Government/central bank reaction to inequality",
                impact_weight=0.2,
                current_value="unknown",
                probability_distribution={
                    "pro_inequality": 0.6,
                    "neutral": 0.3,
                    "anti_inequality": 0.1
                },
                uncertainty_level=0.5
            ),

            CriticalVariable(
                name="Market Timing",
                description="Market cycle and technical conditions",
                impact_weight=0.1,
                current_value=opportunity.get('technical_score', 0.5),
                probability_distribution={
                    "favorable": 0.4,
                    "neutral": 0.4,
                    "unfavorable": 0.2
                },
                uncertainty_level=0.3
            )
        ]

        return variables

    def _bin_outcomes(self, opportunity: Dict[str, Any],
                     variables: List[CriticalVariable],
                     options: List[DecisionOption]) -> List[OutcomeBin]:
        """Step 3: Categorize possible outcomes"""

        base_return = opportunity.get('expected_return', 0.15)

        outcomes = [
            OutcomeBin(
                name="Bull Case",
                description="Gary thesis proves extremely correct",
                probability=0.25,
                expected_return=base_return * 3.0,
                worst_case=base_return * 1.5,
                best_case=base_return * 5.0,
                variables_causing=["Inequality Trend", "Consensus Strength"]
            ),

            OutcomeBin(
                name="Base Case",
                description="Thesis plays out as expected",
                probability=0.50,
                expected_return=base_return,
                worst_case=base_return * 0.5,
                best_case=base_return * 2.0,
                variables_causing=["Inequality Trend"]
            ),

            OutcomeBin(
                name="Bear Case",
                description="Consensus proves partially correct",
                probability=0.20,
                expected_return=-base_return * 0.5,
                worst_case=-base_return * 1.0,
                best_case=0.0,
                variables_causing=["Policy Response", "Market Timing"]
            ),

            OutcomeBin(
                name="Black Swan",
                description="Extreme negative outcome",
                probability=0.05,
                expected_return=-base_return * 2.0,
                worst_case=-base_return * 4.0,
                best_case=-base_return * 1.0,
                variables_causing=["Policy Response", "Market Timing"]
            )
        ]

        return outcomes

    def _rank_outcomes(self, outcomes: List[OutcomeBin]) -> List[OutcomeBin]:
        """Step 4: Rank outcomes by expected value"""
        return sorted(outcomes,
                     key=lambda x: x.expected_return * x.probability,
                     reverse=True)

    def _assign_utility_values(self, outcomes: List[OutcomeBin],
                              user_context: Dict[str, Any]) -> Dict[str, float]:
        """Step 5: Assign utility values considering personal risk tolerance"""

        utility_scores = {}
        baseline_wealth = user_context.get('current_capital', 10000)

        for outcome in outcomes:
            # Calculate utility for worst, expected, and best case
            worst_utility = self.utility_function.calculate_utility(
                outcome.worst_case * baseline_wealth, baseline_wealth
            )
            expected_utility = self.utility_function.calculate_utility(
                outcome.expected_return * baseline_wealth, baseline_wealth
            )
            best_utility = self.utility_function.calculate_utility(
                outcome.best_case * baseline_wealth, baseline_wealth
            )

            # Weight by probability
            utility_scores[outcome.name] = (
                outcome.probability * expected_utility +
                0.1 * worst_utility +  # Slight weight to worst case
                0.1 * best_utility     # Slight weight to best case
            )

        return utility_scores

    def _generate_recommendation(self, options: List[DecisionOption],
                               outcomes: List[OutcomeBin],
                               utility_scores: Dict[str, float],
                               variables: List[CriticalVariable]) -> Tuple[str, float, Tuple[float, float]]:
        """Generate final recommendation with confidence interval"""

        # Calculate expected utility for each option
        option_utilities = {}

        for option in options:
            if option.position_size == 0:
                option_utilities[option.name] = 0.0
                continue

            total_utility = 0.0
            for outcome in outcomes:
                # Scale utility by position size
                scaled_utility = utility_scores[outcome.name] * float(option.position_size) / 1000
                total_utility += scaled_utility

            option_utilities[option.name] = total_utility

        # Find best option
        best_option = max(option_utilities.items(), key=lambda x: x[1])
        expected_utility = best_option[1]

        # Calculate confidence based on variable uncertainty
        avg_uncertainty = np.mean([v.uncertainty_level for v in variables])
        confidence_width = avg_uncertainty * 0.3  # Convert to confidence interval width
        confidence_interval = (
            expected_utility - confidence_width,
            expected_utility + confidence_width
        )

        return best_option[0], expected_utility, confidence_interval

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get user's calibration accuracy report"""
        return {
            'total_predictions': len(self.calibration_tracker.predictions),
            'resolved_predictions': len([p for p in self.calibration_tracker.predictions if p['resolved']]),
            'accuracy_by_confidence': self.calibration_tracker.accuracy_by_confidence,
            'overall_calibration_error': np.mean([
                data['calibration_error']
                for data in self.calibration_tracker.accuracy_by_confidence.values()
                if data['sample_size'] >= 5
            ]) if self.calibration_tracker.accuracy_by_confidence else 0.0
        }

    def export_decision_tree(self, decision_tree: DecisionTree) -> Dict[str, Any]:
        """Export decision tree for frontend visualization"""
        return {
            'root_decision': decision_tree.root_decision,
            'recommendation': decision_tree.recommendation,
            'expected_utility': decision_tree.expected_utility,
            'confidence_interval': decision_tree.confidence_interval,
            'options': [
                {
                    'name': opt.name,
                    'description': opt.description,
                    'position_size': float(opt.position_size),
                    'entry_cost': float(opt.entry_cost),
                    'risk_level': opt.metadata.get('risk_level', 'unknown'),
                    'time_horizon_days': opt.time_horizon.days
                }
                for opt in decision_tree.options
            ],
            'outcomes': [
                {
                    'name': outcome.name,
                    'description': outcome.description,
                    'probability': outcome.probability,
                    'expected_return': outcome.expected_return,
                    'worst_case': outcome.worst_case,
                    'best_case': outcome.best_case,
                    'utility_score': decision_tree.utility_scores.get(outcome.name, 0)
                }
                for outcome in decision_tree.outcomes
            ],
            'critical_variables': [
                {
                    'name': var.name,
                    'description': var.description,
                    'impact_weight': var.impact_weight,
                    'uncertainty_level': var.uncertainty_level,
                    'current_state': var.probability_distribution
                }
                for var in decision_tree.variables
            ]
        }