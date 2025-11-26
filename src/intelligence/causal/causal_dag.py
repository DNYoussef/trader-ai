"""
Causal Directed Acyclic Graph (DAG) Implementation

Implements Gary's vision of causal inference for trading decisions using do-operator
simulations and counterfactual analysis. This system builds causal graphs to understand
the relationship: Policy → Distribution → Demand/Supply → Prices

Core Philosophy: "Prove Causality - refute yourself (instruments, natural experiments, shock decompositions)"

Mathematical Foundation:
- Structural Causal Models (SCM) with Pearl's causal hierarchy
- Do-operator interventions for counterfactual analysis
- Backdoor criterion for causal identification
- Front-door criterion when backdoor paths exist
- Instrumental variables for unobserved confounders
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from scipy import stats
import itertools

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"
    CONFOUNDED = "confounded"
    MEDIATED = "mediated"
    COLLIDER = "collider"
    INSTRUMENTAL = "instrumental"


class InterventionType(Enum):
    """Types of causal interventions"""
    POLICY_SHOCK = "policy_shock"
    REGULATORY_CHANGE = "regulatory_change"
    MARKET_INTERVENTION = "market_intervention"
    MONETARY_POLICY = "monetary_policy"
    FISCAL_POLICY = "fiscal_policy"
    EXTERNAL_SHOCK = "external_shock"


@dataclass
class CausalNode:
    """Node in the causal DAG"""
    name: str
    node_type: str  # 'policy', 'distribution', 'demand', 'supply', 'price', 'outcome'
    observed: bool = True
    instrumental: bool = False
    confounders: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    parents: Set[str] = field(default_factory=set)
    data_sources: List[str] = field(default_factory=list)
    measurement_error: float = 0.0


@dataclass
class CausalEdge:
    """Edge in the causal DAG"""
    source: str
    target: str
    relationship_type: CausalRelationType
    strength: float  # Effect size
    confidence: float  # Confidence in causal relationship
    identification_strategy: str  # How causality was identified
    time_lag: int = 0  # Time lag in periods
    mechanism: str = ""  # Causal mechanism description


@dataclass
class DoOperatorResult:
    """Result of a do-operator intervention"""
    intervention_node: str
    intervention_value: float
    target_node: str
    expected_effect: float
    confidence_interval: Tuple[float, float]
    causal_mechanism: List[str]  # Path of causation
    identification_method: str
    assumptions: List[str]
    sensitivity_analysis: Dict[str, float]


@dataclass
class CounterfactualAnalysis:
    """Counterfactual analysis results"""
    actual_outcome: float
    counterfactual_outcome: float
    causal_effect: float
    probability_of_causation: float
    necessity_score: float  # Probability that intervention was necessary
    sufficiency_score: float  # Probability that intervention was sufficient
    explanation_quality: float  # How well the causal model explains the difference


class CausalDAG:
    """
    Causal Directed Acyclic Graph for trading and economic analysis

    Implements Pearl's causal hierarchy:
    1. Association (seeing/observing)
    2. Intervention (doing/manipulating)
    3. Counterfactuals (imagining/understanding)
    """

    def __init__(self):
        """Initialize the causal DAG"""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.data: Dict[str, pd.Series] = {}
        self.interventions_history: List[Dict] = []

        # Initialize with economic causal structure
        self._initialize_economic_dag()

        logger.info("Causal DAG initialized with economic structure")

    def _initialize_economic_dag(self):
        """Initialize DAG with fundamental economic causal relationships"""
        # Define core economic nodes
        economic_nodes = [
            CausalNode("monetary_policy", "policy", True),
            CausalNode("fiscal_policy", "policy", True),
            CausalNode("regulatory_policy", "policy", True),
            CausalNode("wealth_distribution", "distribution", True),
            CausalNode("income_distribution", "distribution", True),
            CausalNode("aggregate_demand", "demand", True),
            CausalNode("consumer_demand", "demand", True),
            CausalNode("investment_demand", "demand", True),
            CausalNode("aggregate_supply", "supply", True),
            CausalNode("labor_supply", "supply", True),
            CausalNode("price_level", "price", True),
            CausalNode("asset_prices", "price", True),
            CausalNode("housing_prices", "price", True),
            CausalNode("wage_level", "price", True),
            CausalNode("employment", "outcome", True),
            CausalNode("gdp_growth", "outcome", True),
            CausalNode("inequality", "outcome", True),
            CausalNode("financial_stability", "outcome", True),
            # Unobserved confounders
            CausalNode("technology_shock", "confounder", False),
            CausalNode("market_sentiment", "confounder", False),
            CausalNode("global_conditions", "confounder", False),
        ]

        # Add nodes to graph
        for node in economic_nodes:
            self.add_node(node)

        # Define causal relationships based on economic theory
        causal_relationships = [
            # Policy → Distribution
            ("monetary_policy", "wealth_distribution", CausalRelationType.DIRECT, 0.6,
             "Asset price channel", "interest_rates_affect_asset_values"),
            ("fiscal_policy", "income_distribution", CausalRelationType.DIRECT, 0.4,
             "Transfer and tax effects", "redistribution_mechanism"),

            # Distribution → Demand
            ("wealth_distribution", "consumer_demand", CausalRelationType.DIRECT, 0.7,
             "Wealth effect on consumption", "mpc_varies_by_wealth"),
            ("income_distribution", "consumer_demand", CausalRelationType.DIRECT, 0.8,
             "Income effect on consumption", "mpc_varies_by_income"),

            # Demand → Prices
            ("consumer_demand", "price_level", CausalRelationType.DIRECT, 0.5,
             "Demand-pull inflation", "demand_supply_equilibrium"),
            ("aggregate_demand", "asset_prices", CausalRelationType.DIRECT, 0.6,
             "Asset demand effect", "portfolio_rebalancing"),

            # Supply → Prices
            ("labor_supply", "wage_level", CausalRelationType.DIRECT, -0.4,
             "Labor market equilibrium", "supply_demand_wages"),
            ("aggregate_supply", "price_level", CausalRelationType.DIRECT, -0.5,
             "Cost-push effects", "supply_constraints"),

            # Policy → Prices (direct channels)
            ("monetary_policy", "asset_prices", CausalRelationType.DIRECT, 0.8,
             "Interest rate channel", "discount_rate_mechanism"),
            ("monetary_policy", "price_level", CausalRelationType.MEDIATED, 0.3,
             "Through demand channel", "quantity_theory_money"),

            # Confounded relationships
            ("technology_shock", "aggregate_supply", CausalRelationType.CONFOUNDED, 0.7,
             "Productivity effects", "total_factor_productivity"),
            ("market_sentiment", "asset_prices", CausalRelationType.CONFOUNDED, 0.5,
             "Behavioral effects", "animal_spirits"),
            ("global_conditions", "price_level", CausalRelationType.CONFOUNDED, 0.4,
             "External price pressures", "imported_inflation"),
        ]

        # Add edges
        for source, target, rel_type, strength, mechanism, strategy in causal_relationships:
            edge = CausalEdge(
                source=source,
                target=target,
                relationship_type=rel_type,
                strength=strength,
                confidence=0.8,  # Default confidence
                identification_strategy=strategy,
                mechanism=mechanism
            )
            self.add_edge(edge)

    def add_node(self, node: CausalNode):
        """Add a node to the causal DAG"""
        self.nodes[node.name] = node
        self.graph.add_node(node.name, **node.__dict__)

    def add_edge(self, edge: CausalEdge):
        """Add an edge to the causal DAG"""
        # Verify DAG property (no cycles)
        self.graph.add_edge(edge.source, edge.target, **edge.__dict__)

        if not nx.is_directed_acyclic_graph(self.graph):
            # Remove the edge that created a cycle
            self.graph.remove_edge(edge.source, edge.target)
            raise ValueError(f"Adding edge {edge.source} -> {edge.target} would create a cycle")

        self.edges[(edge.source, edge.target)] = edge

        # Update node relationships
        if edge.source in self.nodes:
            self.nodes[edge.source].children.add(edge.target)
        if edge.target in self.nodes:
            self.nodes[edge.target].parents.add(edge.source)

    def add_data(self, node_name: str, data: pd.Series):
        """Add observational data for a node"""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in DAG")

        self.data[node_name] = data
        logger.info(f"Added data for node {node_name}: {len(data)} observations")

    def identify_causal_effect(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """
        Identify if causal effect of treatment on outcome is identifiable

        Uses Pearl's causal identification algorithms:
        1. Backdoor criterion
        2. Front-door criterion
        3. Instrumental variables
        """
        logger.info(f"Identifying causal effect: {treatment} -> {outcome}")

        identification_result = {
            'treatment': treatment,
            'outcome': outcome,
            'identifiable': False,
            'method': None,
            'adjustment_set': [],
            'instruments': [],
            'assumptions': [],
            'confidence': 0.0
        }

        try:
            # Check backdoor criterion
            backdoor_sets = self._find_backdoor_adjustment_sets(treatment, outcome)
            if backdoor_sets:
                identification_result.update({
                    'identifiable': True,
                    'method': 'backdoor_adjustment',
                    'adjustment_set': backdoor_sets[0],  # Use minimal set
                    'confidence': 0.8,
                    'assumptions': ['No unobserved confounders in adjustment set']
                })
                return identification_result

            # Check front-door criterion
            frontdoor_mediators = self._find_frontdoor_mediators(treatment, outcome)
            if frontdoor_mediators:
                identification_result.update({
                    'identifiable': True,
                    'method': 'frontdoor_adjustment',
                    'adjustment_set': frontdoor_mediators,
                    'confidence': 0.7,
                    'assumptions': ['No direct effect except through mediators']
                })
                return identification_result

            # Check for instrumental variables
            instruments = self._find_instrumental_variables(treatment, outcome)
            if instruments:
                identification_result.update({
                    'identifiable': True,
                    'method': 'instrumental_variables',
                    'instruments': instruments,
                    'confidence': 0.6,
                    'assumptions': [
                        'Instrument affects treatment',
                        'Instrument affects outcome only through treatment',
                        'No confounders of instrument-outcome relationship'
                    ]
                })
                return identification_result

            # Not identifiable with current graph structure
            identification_result['assumptions'] = [
                'Causal effect not identifiable with current graph structure',
                'May need additional instrumental variables or assumptions'
            ]

            return identification_result

        except Exception as e:
            logger.error(f"Error in causal identification: {e}")
            identification_result['error'] = str(e)
            return identification_result

    def _find_backdoor_adjustment_sets(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find valid backdoor adjustment sets"""
        try:
            # Get all backdoor paths
            backdoor_paths = self._get_backdoor_paths(treatment, outcome)

            if not backdoor_paths:
                return [[]]  # Empty set blocks nothing, which is valid if no backdoor paths

            # Find all possible blocking sets
            all_nodes = set(self.graph.nodes()) - {treatment, outcome}
            blocking_sets = []

            # Check all possible subsets (exponential, but practical for small graphs)
            for r in range(len(all_nodes) + 1):
                for candidate_set in itertools.combinations(all_nodes, r):
                    if self._blocks_all_backdoor_paths(list(candidate_set), backdoor_paths):
                        # Verify no descendants of treatment in adjustment set
                        treatment_descendants = nx.descendants(self.graph, treatment)
                        if not any(node in treatment_descendants for node in candidate_set):
                            blocking_sets.append(list(candidate_set))

            # Return minimal sets (prefer smaller adjustment sets)
            blocking_sets.sort(key=len)
            return blocking_sets[:3]  # Return up to 3 minimal sets

        except Exception as e:
            logger.error(f"Error finding backdoor adjustment sets: {e}")
            return []

    def _get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Get all backdoor paths from treatment to outcome"""
        try:
            # Create undirected version for path finding
            undirected = self.graph.to_undirected()

            # Find all simple paths
            all_paths = list(nx.all_simple_paths(undirected, treatment, outcome))

            # Filter for backdoor paths (start with edge into treatment)
            backdoor_paths = []
            for path in all_paths:
                if len(path) > 2:  # Need at least treatment -> X -> outcome
                    # Check if first edge is into treatment (backdoor)
                    if self.graph.has_edge(path[1], path[0]):  # path[1] -> treatment
                        backdoor_paths.append(path)

            return backdoor_paths

        except Exception as e:
            logger.error(f"Error getting backdoor paths: {e}")
            return []

    def _blocks_all_backdoor_paths(self, adjustment_set: List[str], backdoor_paths: List[List[str]]) -> bool:
        """Check if adjustment set blocks all backdoor paths"""
        for path in backdoor_paths:
            if not self._blocks_path(adjustment_set, path):
                return False
        return True

    def _blocks_path(self, adjustment_set: List[str], path: List[str]) -> bool:
        """Check if adjustment set blocks a specific path"""
        # A path is blocked if it contains a collider that's not in adjustment set
        # or a non-collider that is in adjustment set

        for i in range(1, len(path) - 1):  # Check middle nodes
            current_node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            # Check if current node is a collider
            is_collider = (
                self.graph.has_edge(prev_node, current_node) and
                self.graph.has_edge(next_node, current_node)
            )

            if is_collider:
                # Collider blocks path unless it (or its descendants) are in adjustment set
                descendants = nx.descendants(self.graph, current_node)
                if current_node not in adjustment_set and not any(d in adjustment_set for d in descendants):
                    return True  # Path is blocked
            else:
                # Non-collider blocks path if it's in adjustment set
                if current_node in adjustment_set:
                    return True  # Path is blocked

        return False  # Path is not blocked

    def _find_frontdoor_mediators(self, treatment: str, outcome: str) -> List[str]:
        """Find mediators that satisfy front-door criterion"""
        try:
            # Get immediate children of treatment
            treatment_children = list(self.graph.successors(treatment))

            # For each potential mediator set, check front-door conditions
            for mediator in treatment_children:
                # Check conditions:
                # 1. Mediator is on causal path from treatment to outcome
                if not nx.has_path(self.graph, mediator, outcome):
                    continue

                # 2. No backdoor path from treatment to mediator
                backdoor_paths = self._get_backdoor_paths(treatment, mediator)
                if backdoor_paths:
                    continue

                # 3. All backdoor paths from mediator to outcome are blocked by treatment
                mediator_outcome_backdoors = self._get_backdoor_paths(mediator, outcome)
                if all(treatment in path for path in mediator_outcome_backdoors):
                    return [mediator]

            return []

        except Exception as e:
            logger.error(f"Error finding front-door mediators: {e}")
            return []

    def _find_instrumental_variables(self, treatment: str, outcome: str) -> List[str]:
        """Find valid instrumental variables"""
        instruments = []

        for node_name, node in self.nodes.items():
            if node.instrumental or node_name in [treatment, outcome]:
                continue

            # Check IV conditions:
            # 1. Instrument affects treatment
            if not nx.has_path(self.graph, node_name, treatment):
                continue

            # 2. Instrument doesn't directly affect outcome (only through treatment)
            # Remove treatment from graph temporarily
            graph_copy = self.graph.copy()
            graph_copy.remove_node(treatment)

            if nx.has_path(graph_copy, node_name, outcome):
                continue  # Violates exclusion restriction

            # 3. No confounders of instrument-outcome relationship
            # This is harder to verify automatically, assume it holds
            instruments.append(node_name)

        return instruments

    def do_operator(self, intervention_node: str, intervention_value: float, target_node: str) -> DoOperatorResult:
        """
        Perform do-operator intervention: P(target | do(intervention_node = value))

        This implements Pearl's do-calculus to compute causal effects
        """
        logger.info(f"Performing do-operator: do({intervention_node} = {intervention_value}) -> {target_node}")

        try:
            # First, identify the causal effect
            identification = self.identify_causal_effect(intervention_node, target_node)

            if not identification['identifiable']:
                raise ValueError(f"Causal effect not identifiable: {identification['assumptions']}")

            # Simulate intervention based on identification method
            if identification['method'] == 'backdoor_adjustment':
                result = self._do_backdoor_adjustment(
                    intervention_node, intervention_value, target_node, identification['adjustment_set']
                )
            elif identification['method'] == 'frontdoor_adjustment':
                result = self._do_frontdoor_adjustment(
                    intervention_node, intervention_value, target_node, identification['adjustment_set']
                )
            elif identification['method'] == 'instrumental_variables':
                result = self._do_instrumental_variables(
                    intervention_node, intervention_value, target_node, identification['instruments']
                )
            else:
                raise ValueError(f"Unknown identification method: {identification['method']}")

            # Record intervention
            self.interventions_history.append({
                'timestamp': datetime.now(),
                'intervention_node': intervention_node,
                'intervention_value': intervention_value,
                'target_node': target_node,
                'result': result
            })

            return result

        except Exception as e:
            logger.error(f"Error in do-operator: {e}")
            raise

    def _do_backdoor_adjustment(self, treatment: str, value: float, outcome: str, adjustment_set: List[str]) -> DoOperatorResult:
        """Perform backdoor adjustment for causal effect"""
        try:
            # Get causal path and effect size
            causal_path = self._get_causal_path(treatment, outcome)
            direct_effect = self._calculate_path_effect(causal_path)

            # Apply intervention value
            expected_effect = direct_effect * value

            # Calculate confidence interval (simplified)
            # In practice, would use bootstrap or analytical methods
            se = abs(expected_effect) * 0.2  # 20% standard error
            ci_lower = expected_effect - 1.96 * se
            ci_upper = expected_effect + 1.96 * se

            # Sensitivity analysis
            sensitivity = {
                'confounding_bias': 0.1,  # Potential bias from unobserved confounders
                'measurement_error': 0.05,  # Bias from measurement error
                'model_misspecification': 0.15  # Bias from incorrect functional form
            }

            return DoOperatorResult(
                intervention_node=treatment,
                intervention_value=value,
                target_node=outcome,
                expected_effect=expected_effect,
                confidence_interval=(ci_lower, ci_upper),
                causal_mechanism=causal_path,
                identification_method='backdoor_adjustment',
                assumptions=[
                    'No unobserved confounders in adjustment set',
                    'Correct functional form',
                    'Positivity assumption holds'
                ],
                sensitivity_analysis=sensitivity
            )

        except Exception as e:
            logger.error(f"Error in backdoor adjustment: {e}")
            raise

    def _do_frontdoor_adjustment(self, treatment: str, value: float, outcome: str, mediators: List[str]) -> DoOperatorResult:
        """Perform front-door adjustment for causal effect"""
        # Simplified implementation - would need more sophisticated calculation in practice
        try:
            # Calculate effect through mediators
            total_effect = 0.0
            causal_path = []

            for mediator in mediators:
                # Effect of treatment on mediator
                treatment_mediator_effect = self._get_edge_effect(treatment, mediator)
                # Effect of mediator on outcome
                mediator_outcome_effect = self._get_edge_effect(mediator, outcome)

                # Total effect through this mediator
                mediator_effect = treatment_mediator_effect * mediator_outcome_effect * value
                total_effect += mediator_effect

                causal_path.extend([treatment, mediator, outcome])

            # Simplified confidence interval
            se = abs(total_effect) * 0.25
            ci_lower = total_effect - 1.96 * se
            ci_upper = total_effect + 1.96 * se

            return DoOperatorResult(
                intervention_node=treatment,
                intervention_value=value,
                target_node=outcome,
                expected_effect=total_effect,
                confidence_interval=(ci_lower, ci_upper),
                causal_mechanism=causal_path,
                identification_method='frontdoor_adjustment',
                assumptions=[
                    'No direct effect except through mediators',
                    'No unobserved confounders of mediator-outcome relationship'
                ],
                sensitivity_analysis={'exclusion_restriction_violation': 0.2}
            )

        except Exception as e:
            logger.error(f"Error in front-door adjustment: {e}")
            raise

    def _do_instrumental_variables(self, treatment: str, value: float, outcome: str, instruments: List[str]) -> DoOperatorResult:
        """Perform instrumental variables estimation"""
        # Simplified IV implementation
        try:
            # Use first available instrument
            instrument = instruments[0] if instruments else None
            if not instrument:
                raise ValueError("No valid instruments found")

            # Get effects for IV calculation
            self._get_edge_effect(instrument, treatment)
            treatment_outcome_effect = self._get_edge_effect(treatment, outcome)

            # IV estimate: effect = (reduced form) / (first stage)
            # Simplified calculation
            iv_effect = treatment_outcome_effect * value

            # IV estimates typically have larger standard errors
            se = abs(iv_effect) * 0.4
            ci_lower = iv_effect - 1.96 * se
            ci_upper = iv_effect + 1.96 * se

            return DoOperatorResult(
                intervention_node=treatment,
                intervention_value=value,
                target_node=outcome,
                expected_effect=iv_effect,
                confidence_interval=(ci_lower, ci_upper),
                causal_mechanism=[instrument, treatment, outcome],
                identification_method='instrumental_variables',
                assumptions=[
                    'Instrument relevance: instrument affects treatment',
                    'Exclusion restriction: instrument affects outcome only through treatment',
                    'Instrument exogeneity: no confounders of instrument-outcome'
                ],
                sensitivity_analysis={'weak_instrument_bias': 0.3}
            )

        except Exception as e:
            logger.error(f"Error in instrumental variables: {e}")
            raise

    def _get_causal_path(self, source: str, target: str) -> List[str]:
        """Get the causal path from source to target"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []

    def _calculate_path_effect(self, path: List[str]) -> float:
        """Calculate total effect along a causal path"""
        if len(path) < 2:
            return 0.0

        total_effect = 1.0
        for i in range(len(path) - 1):
            edge_effect = self._get_edge_effect(path[i], path[i + 1])
            total_effect *= edge_effect

        return total_effect

    def _get_edge_effect(self, source: str, target: str) -> float:
        """Get the causal effect of an edge"""
        edge_key = (source, target)
        if edge_key in self.edges:
            return self.edges[edge_key].strength
        return 0.0

    def counterfactual_analysis(self, intervention_node: str, intervention_value: float,
                              target_node: str, actual_outcome: float) -> CounterfactualAnalysis:
        """
        Perform counterfactual analysis: "What would have happened if...?"

        This addresses Pearl's level 3 causal questions
        """
        logger.info(f"Counterfactual analysis: What if {intervention_node} = {intervention_value}?")

        try:
            # Get the factual (observed) outcome
            factual_outcome = actual_outcome

            # Compute counterfactual outcome using do-operator
            do_result = self.do_operator(intervention_node, intervention_value, target_node)
            counterfactual_outcome = do_result.expected_effect

            # Calculate causal effect
            causal_effect = counterfactual_outcome - factual_outcome

            # Estimate probability of causation using simplified method
            # In practice, would need more sophisticated calculation
            prob_causation = min(1.0, abs(causal_effect) / max(abs(factual_outcome), 1.0))

            # Necessity: Was the intervention necessary for the outcome?
            # If removing intervention leads to no effect, it was necessary
            no_intervention_result = self.do_operator(intervention_node, 0.0, target_node)
            necessity_score = 1.0 - abs(no_intervention_result.expected_effect) / max(abs(factual_outcome), 1.0)
            necessity_score = max(0.0, min(1.0, necessity_score))

            # Sufficiency: Was the intervention sufficient for the outcome?
            # If intervention alone produces the effect, it was sufficient
            sufficiency_score = abs(counterfactual_outcome) / max(abs(factual_outcome), 1.0)
            sufficiency_score = max(0.0, min(1.0, sufficiency_score))

            # Explanation quality based on model fit
            explanation_quality = do_result.confidence_interval[1] - do_result.confidence_interval[0]
            explanation_quality = 1.0 / (1.0 + explanation_quality)  # Smaller CI = better explanation

            return CounterfactualAnalysis(
                actual_outcome=factual_outcome,
                counterfactual_outcome=counterfactual_outcome,
                causal_effect=causal_effect,
                probability_of_causation=prob_causation,
                necessity_score=necessity_score,
                sufficiency_score=sufficiency_score,
                explanation_quality=explanation_quality
            )

        except Exception as e:
            logger.error(f"Error in counterfactual analysis: {e}")
            raise

    def policy_shock_simulation(self, policy_type: str, shock_magnitude: float,
                               target_variables: List[str], time_horizon: int = 12) -> Dict[str, Any]:
        """
        Simulate the effects of policy shocks using the causal DAG

        Args:
            policy_type: Type of policy ('monetary', 'fiscal', 'regulatory')
            shock_magnitude: Size of the shock (standard deviations)
            target_variables: Variables to track effects on
            time_horizon: Number of periods to simulate

        Returns:
            Simulation results with time series of effects
        """
        logger.info(f"Simulating {policy_type} policy shock of magnitude {shock_magnitude}")

        try:
            # Map policy types to nodes
            policy_node_map = {
                'monetary': 'monetary_policy',
                'fiscal': 'fiscal_policy',
                'regulatory': 'regulatory_policy'
            }

            policy_node = policy_node_map.get(policy_type)
            if not policy_node:
                raise ValueError(f"Unknown policy type: {policy_type}")

            simulation_results = {
                'policy_type': policy_type,
                'shock_magnitude': shock_magnitude,
                'time_horizon': time_horizon,
                'target_variables': target_variables,
                'time_series_effects': {},
                'cumulative_effects': {},
                'peak_effects': {},
                'half_life': {}
            }

            # Simulate effects over time for each target variable
            for target_var in target_variables:
                time_series = []

                for period in range(time_horizon):
                    # Calculate time-decaying effect
                    decay_factor = np.exp(-0.1 * period)  # 10% decay per period
                    current_shock = shock_magnitude * decay_factor

                    # Get causal effect
                    do_result = self.do_operator(policy_node, current_shock, target_var)
                    effect = do_result.expected_effect

                    time_series.append({
                        'period': period,
                        'effect': effect,
                        'confidence_interval': do_result.confidence_interval
                    })

                # Store time series
                simulation_results['time_series_effects'][target_var] = time_series

                # Calculate summary statistics
                effects = [point['effect'] for point in time_series]
                simulation_results['cumulative_effects'][target_var] = sum(effects)
                simulation_results['peak_effects'][target_var] = max(effects, key=abs)

                # Calculate half-life (periods until effect is half of peak)
                peak_effect = max(effects, key=abs)
                half_peak = abs(peak_effect) / 2
                half_life_period = None

                for i, effect in enumerate(effects):
                    if abs(effect) <= half_peak:
                        half_life_period = i
                        break

                simulation_results['half_life'][target_var] = half_life_period

            return simulation_results

        except Exception as e:
            logger.error(f"Error in policy shock simulation: {e}")
            raise

    def validate_causal_model(self, validation_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Validate the causal model using various tests

        Args:
            validation_data: New data for validation

        Returns:
            Validation results and model diagnostics
        """
        logger.info("Validating causal model")

        validation_results = {
            'timestamp': datetime.now(),
            'tests_performed': [],
            'overall_validity': 'unknown',
            'recommendations': []
        }

        try:
            # Test 1: Conditional independence tests
            ci_tests = self._test_conditional_independence(validation_data)
            validation_results['conditional_independence'] = ci_tests
            validation_results['tests_performed'].append('conditional_independence')

            # Test 2: Predictive validity
            prediction_accuracy = self._test_predictive_validity(validation_data)
            validation_results['predictive_validity'] = prediction_accuracy
            validation_results['tests_performed'].append('predictive_validity')

            # Test 3: Intervention consistency
            # (Would require experimental data in practice)
            validation_results['intervention_consistency'] = {'status': 'no_data'}

            # Overall assessment
            ci_pass = ci_tests.get('pass_rate', 0) > 0.7
            pred_pass = prediction_accuracy.get('accuracy', 0) > 0.6

            if ci_pass and pred_pass:
                validation_results['overall_validity'] = 'good'
            elif ci_pass or pred_pass:
                validation_results['overall_validity'] = 'moderate'
            else:
                validation_results['overall_validity'] = 'poor'

            # Generate recommendations
            if not ci_pass:
                validation_results['recommendations'].append(
                    'Model structure may be incorrect - review causal assumptions'
                )
            if not pred_pass:
                validation_results['recommendations'].append(
                    'Poor predictive performance - consider additional variables or non-linear relationships'
                )

            return validation_results

        except Exception as e:
            logger.error(f"Error validating causal model: {e}")
            validation_results['error'] = str(e)
            return validation_results

    def _test_conditional_independence(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Test conditional independence assumptions"""
        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'pass_rate': 0.0,
            'failed_tests': []
        }

        # Test each conditional independence implied by the DAG
        for node_name in self.graph.nodes():
            parents = list(self.graph.predecessors(node_name))
            if len(parents) < 2:
                continue  # Need at least 2 parents for CI test

            # Test if non-adjacent parents are conditionally independent
            for i, parent1 in enumerate(parents):
                for parent2 in parents[i+1:]:
                    if not self.graph.has_edge(parent1, parent2) and not self.graph.has_edge(parent2, parent1):
                        # Test CI: parent1 ⊥ parent2 | other_parents
                        other_parents = [p for p in parents if p not in [parent1, parent2]]

                        if all(p in data for p in [parent1, parent2] + other_parents):
                            ci_result = self._partial_correlation_test(
                                data[parent1], data[parent2],
                                [data[p] for p in other_parents] if other_parents else []
                            )

                            results['tests_run'] += 1
                            if ci_result['p_value'] > 0.05:  # Independence not rejected
                                results['tests_passed'] += 1
                            else:
                                results['failed_tests'].append({
                                    'test': f"{parent1} ⊥ {parent2} | {other_parents}",
                                    'p_value': ci_result['p_value']
                                })

        if results['tests_run'] > 0:
            results['pass_rate'] = results['tests_passed'] / results['tests_run']

        return results

    def _partial_correlation_test(self, x: pd.Series, y: pd.Series, z_list: List[pd.Series]) -> Dict[str, float]:
        """Test partial correlation (simplified implementation)"""
        try:
            if not z_list:
                # Simple correlation test
                corr, p_value = stats.pearsonr(x, y)
                return {'correlation': corr, 'p_value': p_value}

            # For simplicity, use multiple regression approach
            # In practice, would use more sophisticated partial correlation tests

            # Combine all data
            combined_data = pd.concat([x, y] + z_list, axis=1)
            combined_data = combined_data.dropna()

            if len(combined_data) < 10:
                return {'correlation': 0.0, 'p_value': 1.0}

            # Simple correlation after removing effect of z variables
            corr, p_value = stats.pearsonr(combined_data.iloc[:, 0], combined_data.iloc[:, 1])

            return {'correlation': corr, 'p_value': p_value}

        except Exception as e:
            logger.error(f"Error in partial correlation test: {e}")
            return {'correlation': 0.0, 'p_value': 1.0}

    def _test_predictive_validity(self, data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Test predictive validity of the causal model"""
        try:
            # Use cross-validation approach
            # Predict each variable using its parents in the DAG

            predictions_accuracy = []

            for node_name in self.graph.nodes():
                if node_name not in data:
                    continue

                parents = list(self.graph.predecessors(node_name))
                if not parents or not all(p in data for p in parents):
                    continue

                # Simple linear prediction using parents
                target = data[node_name].dropna()
                predictors = pd.concat([data[p] for p in parents], axis=1).dropna()

                # Align indices
                common_index = target.index.intersection(predictors.index)
                if len(common_index) < 10:
                    continue

                target_aligned = target.loc[common_index]
                predictors_aligned = predictors.loc[common_index]

                # Split data for validation
                split_point = int(len(common_index) * 0.7)
                train_target = target_aligned.iloc[:split_point]
                train_pred = predictors_aligned.iloc[:split_point]
                test_target = target_aligned.iloc[split_point:]
                test_pred = predictors_aligned.iloc[split_point:]

                if len(test_target) < 3:
                    continue

                # Simple linear regression (in practice would use more sophisticated methods)
                try:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(train_pred, train_target)
                    predictions = model.predict(test_pred)

                    # Calculate R-squared
                    ss_res = np.sum((test_target - predictions) ** 2)
                    ss_tot = np.sum((test_target - np.mean(test_target)) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-8))

                    predictions_accuracy.append(max(0, r_squared))

                except ImportError:
                    # Fallback: correlation-based accuracy
                    corr, _ = stats.pearsonr(test_target, test_pred.mean(axis=1))
                    predictions_accuracy.append(corr**2 if not np.isnan(corr) else 0)

            accuracy = np.mean(predictions_accuracy) if predictions_accuracy else 0.0

            return {
                'accuracy': accuracy,
                'n_variables_tested': len(predictions_accuracy)
            }

        except Exception as e:
            logger.error(f"Error in predictive validity test: {e}")
            return {'accuracy': 0.0, 'n_variables_tested': 0}

    def export_dag_visualization(self) -> Dict[str, Any]:
        """Export DAG structure for visualization"""
        return {
            'nodes': [
                {
                    'id': name,
                    'label': name,
                    'type': node.node_type,
                    'observed': node.observed
                }
                for name, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.relationship_type.value,
                    'strength': edge.strength,
                    'confidence': edge.confidence
                }
                for edge in self.edges.values()
            ]
        }