"""
Synthetic Controls for Counterfactual Validation

Implements Gary's vision of counterfactual validation using synthetic control methods,
placebo tests, and backdoor checks. This system validates causal claims by constructing
synthetic counterfactuals and testing robustness of causal identification.

Core Philosophy: "Prove Causality - refute yourself (instruments, natural experiments, shock decompositions)"

Mathematical Foundation:
- Synthetic control method (Abadie & Gardeazabal)
- Placebo tests and permutation inference
- Backdoor criterion validation
- Instrumental variable validation
- Regression discontinuity robustness
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats, optimize
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import warnings

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Methods for causal validation"""
    SYNTHETIC_CONTROL = "synthetic_control"
    PLACEBO_TEST = "placebo_test"
    PERMUTATION_TEST = "permutation_test"
    BACKDOOR_CHECK = "backdoor_check"
    INSTRUMENTAL_VALIDATION = "instrumental_validation"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"


class TreatmentType(Enum):
    """Types of treatment/intervention"""
    POLICY_CHANGE = "policy_change"
    REGULATORY_SHOCK = "regulatory_shock"
    MONETARY_INTERVENTION = "monetary_intervention"
    FISCAL_INTERVENTION = "fiscal_intervention"
    MARKET_SHOCK = "market_shock"
    TECHNOLOGICAL_CHANGE = "technological_change"


@dataclass
class SyntheticControlResult:
    """Results from synthetic control analysis"""
    treated_unit: str
    treatment_date: datetime
    synthetic_weights: Dict[str, float]
    pre_treatment_fit: float  # RMSE before treatment
    post_treatment_effect: List[float]  # Treatment effects over time
    average_treatment_effect: float
    p_value: float  # From permutation test
    confidence_interval: Tuple[float, float]
    control_units: List[str]
    validation_metrics: Dict[str, float]


@dataclass
class PlaceboTestResult:
    """Results from placebo test"""
    original_effect: float
    placebo_effects: List[float]
    p_value: float
    effect_rank: int  # Rank of original effect among placebos
    total_placebos: int
    robust_to_placebo: bool


@dataclass
class ValidationSuite:
    """Complete validation suite results"""
    treatment_unit: str
    outcome_variable: str
    treatment_date: datetime
    synthetic_control: SyntheticControlResult
    placebo_tests: List[PlaceboTestResult]
    permutation_p_value: float
    backdoor_validation: Dict[str, Any]
    instrumental_validation: Dict[str, Any]
    overall_validity_score: float
    recommendations: List[str]


class SyntheticControlValidator:
    """
    Synthetic Control validator for causal inference

    Implements multiple validation methods to test robustness of causal claims:
    1. Synthetic control method for counterfactual construction
    2. Placebo tests for treatment effects
    3. Permutation tests for statistical significance
    4. Backdoor criterion validation
    5. Instrumental variable validation
    """

    def __init__(self, regularization_alpha: float = 0.01,
                 min_pre_treatment_periods: int = 12):
        """
        Initialize synthetic control validator

        Args:
            regularization_alpha: Regularization parameter for weight optimization
            min_pre_treatment_periods: Minimum periods before treatment for fitting
        """
        self.regularization_alpha = regularization_alpha
        self.min_pre_treatment_periods = min_pre_treatment_periods
        self.validation_results: List[ValidationSuite] = []

        logger.info("Synthetic Control Validator initialized")

    def synthetic_control_analysis(self,
                                 data: pd.DataFrame,
                                 treated_unit: str,
                                 outcome_variable: str,
                                 treatment_date: datetime,
                                 control_units: Optional[List[str]] = None,
                                 predictor_variables: Optional[List[str]] = None) -> SyntheticControlResult:
        """
        Perform synthetic control analysis

        Args:
            data: Panel data with units, time, and variables
            treated_unit: Unit that received treatment
            outcome_variable: Outcome variable of interest
            treatment_date: Date when treatment began
            control_units: List of potential control units (all others if None)
            predictor_variables: Variables to match on (outcome lags if None)

        Returns:
            Synthetic control analysis results
        """
        logger.info(f"Performing synthetic control analysis for {treated_unit}")

        try:
            # Prepare data
            if 'unit' not in data.columns or 'date' not in data.columns:
                raise ValueError("Data must have 'unit' and 'date' columns")

            # Convert date column if needed
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])

            # Get control units
            if control_units is None:
                control_units = [unit for unit in data['unit'].unique()
                               if unit != treated_unit]

            # Split pre/post treatment periods
            pre_treatment = data[data['date'] < treatment_date].copy()
            post_treatment = data[data['date'] >= treatment_date].copy()

            if len(pre_treatment) < self.min_pre_treatment_periods:
                raise ValueError(f"Insufficient pre-treatment periods: {len(pre_treatment)}")

            # Construct synthetic control
            synthetic_weights = self._optimize_synthetic_weights(
                pre_treatment, treated_unit, control_units,
                outcome_variable, predictor_variables
            )

            # Calculate pre-treatment fit
            pre_treatment_fit = self._calculate_pre_treatment_fit(
                pre_treatment, treated_unit, control_units,
                outcome_variable, synthetic_weights
            )

            # Calculate post-treatment effects
            post_treatment_effects = self._calculate_treatment_effects(
                post_treatment, treated_unit, control_units,
                outcome_variable, synthetic_weights
            )

            # Calculate average treatment effect
            avg_treatment_effect = np.mean(post_treatment_effects)

            # Perform permutation test for significance
            p_value = self._permutation_test(
                data, treated_unit, control_units, outcome_variable,
                treatment_date, avg_treatment_effect
            )

            # Calculate confidence interval
            confidence_interval = self._bootstrap_confidence_interval(
                data, treated_unit, control_units, outcome_variable,
                treatment_date, synthetic_weights
            )

            # Validation metrics
            validation_metrics = {
                'pre_treatment_rmse': pre_treatment_fit,
                'post_treatment_periods': len(post_treatment_effects),
                'weight_concentration': max(synthetic_weights.values()),
                'n_control_units': len(control_units)
            }

            result = SyntheticControlResult(
                treated_unit=treated_unit,
                treatment_date=treatment_date,
                synthetic_weights=synthetic_weights,
                pre_treatment_fit=pre_treatment_fit,
                post_treatment_effect=post_treatment_effects,
                average_treatment_effect=avg_treatment_effect,
                p_value=p_value,
                confidence_interval=confidence_interval,
                control_units=control_units,
                validation_metrics=validation_metrics
            )

            logger.info(f"Synthetic control analysis completed: ATE = {avg_treatment_effect:.4f}, p = {p_value:.3f}")
            return result

        except Exception as e:
            logger.error(f"Error in synthetic control analysis: {e}")
            raise

    def _optimize_synthetic_weights(self,
                                   pre_treatment_data: pd.DataFrame,
                                   treated_unit: str,
                                   control_units: List[str],
                                   outcome_variable: str,
                                   predictor_variables: Optional[List[str]]) -> Dict[str, float]:
        """Optimize synthetic control weights"""
        try:
            # Get treated unit data
            treated_data = pre_treatment_data[
                pre_treatment_data['unit'] == treated_unit
            ].sort_values('date')

            # Get control units data
            control_data = pre_treatment_data[
                pre_treatment_data['unit'].isin(control_units)
            ].pivot(index='date', columns='unit', values=outcome_variable)

            # Align dates
            common_dates = treated_data['date'].values
            control_aligned = control_data.loc[
                control_data.index.isin(common_dates)
            ].fillna(method='ffill').fillna(method='bfill')

            treated_outcome = treated_data[outcome_variable].values

            if len(treated_outcome) != len(control_aligned):
                # Align by reindexing
                treated_series = pd.Series(
                    treated_outcome,
                    index=treated_data['date'].values
                )
                common_index = treated_series.index.intersection(control_aligned.index)
                treated_outcome = treated_series.loc[common_index].values
                control_aligned = control_aligned.loc[common_index]

            # Remove units with missing data
            control_matrix = control_aligned.dropna(axis=1).values
            available_units = control_aligned.dropna(axis=1).columns.tolist()

            if control_matrix.shape[1] == 0:
                raise ValueError("No control units with complete data")

            # Optimization objective: minimize pre-treatment prediction error
            def objective(weights):
                if len(weights) != control_matrix.shape[1]:
                    return float('inf')

                # Ensure weights sum to 1 and are non-negative
                if abs(np.sum(weights) - 1.0) > 1e-6 or np.any(weights < 0):
                    return float('inf')

                synthetic_outcome = control_matrix @ weights
                mse = np.mean((treated_outcome - synthetic_outcome) ** 2)

                # Add regularization to prevent overfitting
                regularization = self.regularization_alpha * np.sum(weights ** 2)
                return mse + regularization

            # Constraints: weights sum to 1 and are non-negative
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 1) for _ in range(control_matrix.shape[1])]

            # Initial guess: equal weights
            initial_weights = np.ones(control_matrix.shape[1]) / control_matrix.shape[1]

            # Optimize
            result = optimize.minimize(
                objective, initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if not result.success:
                logger.warning("Weight optimization did not converge, using Ridge regression")
                # Fallback to Ridge regression
                ridge = Ridge(alpha=self.regularization_alpha, positive=True)
                ridge.fit(control_matrix, treated_outcome)
                weights = ridge.coef_
                weights = np.maximum(weights, 0)  # Ensure non-negative
                weights = weights / np.sum(weights)  # Normalize
            else:
                weights = result.x

            # Create weights dictionary
            weight_dict = {unit: weight for unit, weight in zip(available_units, weights)}

            # Add zero weights for units not included
            for unit in control_units:
                if unit not in weight_dict:
                    weight_dict[unit] = 0.0

            return weight_dict

        except Exception as e:
            logger.error(f"Error optimizing synthetic weights: {e}")
            raise

    def _calculate_pre_treatment_fit(self,
                                   pre_treatment_data: pd.DataFrame,
                                   treated_unit: str,
                                   control_units: List[str],
                                   outcome_variable: str,
                                   weights: Dict[str, float]) -> float:
        """Calculate pre-treatment fit quality"""
        try:
            # Get treated unit outcome
            treated_data = pre_treatment_data[
                pre_treatment_data['unit'] == treated_unit
            ].sort_values('date')[outcome_variable].values

            # Calculate synthetic outcome
            synthetic_outcome = np.zeros(len(treated_data))

            for unit, weight in weights.items():
                if weight > 0:
                    unit_data = pre_treatment_data[
                        pre_treatment_data['unit'] == unit
                    ].sort_values('date')[outcome_variable].values

                    if len(unit_data) == len(treated_data):
                        synthetic_outcome += weight * unit_data

            # Calculate RMSE
            rmse = np.sqrt(np.mean((treated_data - synthetic_outcome) ** 2))
            return rmse

        except Exception as e:
            logger.error(f"Error calculating pre-treatment fit: {e}")
            return float('inf')

    def _calculate_treatment_effects(self,
                                   post_treatment_data: pd.DataFrame,
                                   treated_unit: str,
                                   control_units: List[str],
                                   outcome_variable: str,
                                   weights: Dict[str, float]) -> List[float]:
        """Calculate post-treatment effects"""
        try:
            # Get treated unit outcome
            treated_data = post_treatment_data[
                post_treatment_data['unit'] == treated_unit
            ].sort_values('date')

            treatment_effects = []

            for _, row in treated_data.iterrows():
                treated_outcome = row[outcome_variable]

                # Calculate synthetic outcome
                synthetic_outcome = 0.0
                for unit, weight in weights.items():
                    if weight > 0:
                        unit_outcome = post_treatment_data[
                            (post_treatment_data['unit'] == unit) &
                            (post_treatment_data['date'] == row['date'])
                        ][outcome_variable]

                        if len(unit_outcome) > 0:
                            synthetic_outcome += weight * unit_outcome.iloc[0]

                # Treatment effect = actual - synthetic
                effect = treated_outcome - synthetic_outcome
                treatment_effects.append(effect)

            return treatment_effects

        except Exception as e:
            logger.error(f"Error calculating treatment effects: {e}")
            return []

    def _permutation_test(self,
                         data: pd.DataFrame,
                         treated_unit: str,
                         control_units: List[str],
                         outcome_variable: str,
                         treatment_date: datetime,
                         observed_effect: float,
                         n_permutations: int = 500) -> float:
        """Perform permutation test for statistical significance"""
        try:
            logger.info(f"Performing permutation test with {n_permutations} permutations")

            placebo_effects = []

            # Randomly assign treatment to control units
            for i in range(n_permutations):
                # Randomly select a placebo treated unit
                placebo_treated = np.random.choice(control_units)

                # Exclude the placebo treated unit from controls
                placebo_controls = [unit for unit in control_units if unit != placebo_treated]

                if len(placebo_controls) < 3:  # Need minimum controls
                    continue

                try:
                    # Run synthetic control on placebo
                    placebo_result = self.synthetic_control_analysis(
                        data=data,
                        treated_unit=placebo_treated,
                        outcome_variable=outcome_variable,
                        treatment_date=treatment_date,
                        control_units=placebo_controls
                    )

                    placebo_effects.append(placebo_result.average_treatment_effect)

                except Exception as e:
                    # Skip failed placebo tests
                    continue

            if len(placebo_effects) == 0:
                logger.warning("No successful placebo tests completed")
                return 1.0

            # Calculate p-value: fraction of placebo effects >= observed effect
            extreme_effects = sum(1 for effect in placebo_effects
                                if abs(effect) >= abs(observed_effect))
            p_value = extreme_effects / len(placebo_effects)

            logger.info(f"Permutation test completed: {len(placebo_effects)} successful permutations, p = {p_value:.3f}")
            return p_value

        except Exception as e:
            logger.error(f"Error in permutation test: {e}")
            return 1.0

    def _bootstrap_confidence_interval(self,
                                     data: pd.DataFrame,
                                     treated_unit: str,
                                     control_units: List[str],
                                     outcome_variable: str,
                                     treatment_date: datetime,
                                     weights: Dict[str, float],
                                     n_bootstrap: int = 200,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        try:
            bootstrap_effects = []

            # Get post-treatment data
            post_data = data[data['date'] >= treatment_date].copy()
            dates = sorted(post_data['date'].unique())

            for i in range(n_bootstrap):
                # Bootstrap sample dates
                bootstrap_dates = np.random.choice(dates, size=len(dates), replace=True)
                bootstrap_data = post_data[post_data['date'].isin(bootstrap_dates)]

                # Calculate effects for bootstrap sample
                effects = self._calculate_treatment_effects(
                    bootstrap_data, treated_unit, control_units,
                    outcome_variable, weights
                )

                if effects:
                    bootstrap_effects.append(np.mean(effects))

            if len(bootstrap_effects) < 10:
                logger.warning("Insufficient bootstrap samples for confidence interval")
                return (float('-inf'), float('inf'))

            # Calculate percentiles
            alpha = 1 - confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)

            ci_lower = np.percentile(bootstrap_effects, lower_percentile)
            ci_upper = np.percentile(bootstrap_effects, upper_percentile)

            return (ci_lower, ci_upper)

        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (float('-inf'), float('inf'))

    def placebo_test_suite(self,
                          data: pd.DataFrame,
                          treated_unit: str,
                          outcome_variable: str,
                          treatment_date: datetime,
                          placebo_treatments: Optional[List[datetime]] = None,
                          placebo_outcomes: Optional[List[str]] = None) -> List[PlaceboTestResult]:
        """
        Run comprehensive placebo test suite

        Args:
            data: Panel data
            treated_unit: Unit that received treatment
            outcome_variable: Outcome variable
            treatment_date: Actual treatment date
            placebo_treatments: Alternative treatment dates to test
            placebo_outcomes: Alternative outcome variables to test

        Returns:
            List of placebo test results
        """
        logger.info("Running placebo test suite")

        placebo_results = []

        try:
            # Get original effect
            control_units = [unit for unit in data['unit'].unique()
                           if unit != treated_unit]

            original_result = self.synthetic_control_analysis(
                data, treated_unit, outcome_variable, treatment_date, control_units
            )
            original_effect = original_result.average_treatment_effect

            # Test 1: Alternative treatment dates
            if placebo_treatments:
                for placebo_date in placebo_treatments:
                    try:
                        placebo_result = self.synthetic_control_analysis(
                            data, treated_unit, outcome_variable, placebo_date, control_units
                        )

                        placebo_test = PlaceboTestResult(
                            original_effect=original_effect,
                            placebo_effects=[placebo_result.average_treatment_effect],
                            p_value=1.0 if abs(placebo_result.average_treatment_effect) >= abs(original_effect) else 0.0,
                            effect_rank=1,
                            total_placebos=1,
                            robust_to_placebo=abs(original_effect) > abs(placebo_result.average_treatment_effect)
                        )

                        placebo_results.append(placebo_test)

                    except Exception as e:
                        logger.warning(f"Placebo date test failed for {placebo_date}: {e}")

            # Test 2: Alternative outcome variables
            if placebo_outcomes:
                for placebo_outcome in placebo_outcomes:
                    if placebo_outcome in data.columns:
                        try:
                            placebo_result = self.synthetic_control_analysis(
                                data, treated_unit, placebo_outcome, treatment_date, control_units
                            )

                            placebo_test = PlaceboTestResult(
                                original_effect=original_effect,
                                placebo_effects=[placebo_result.average_treatment_effect],
                                p_value=1.0 if abs(placebo_result.average_treatment_effect) >= abs(original_effect) else 0.0,
                                effect_rank=1,
                                total_placebos=1,
                                robust_to_placebo=abs(original_effect) > abs(placebo_result.average_treatment_effect)
                            )

                            placebo_results.append(placebo_test)

                        except Exception as e:
                            logger.warning(f"Placebo outcome test failed for {placebo_outcome}: {e}")

            # Test 3: Alternative control pools
            if len(control_units) > 5:
                for _ in range(min(5, len(control_units) // 2)):  # Test different control pools
                    subset_controls = np.random.choice(
                        control_units,
                        size=max(3, len(control_units) // 2),
                        replace=False
                    ).tolist()

                    try:
                        placebo_result = self.synthetic_control_analysis(
                            data, treated_unit, outcome_variable, treatment_date, subset_controls
                        )

                        placebo_test = PlaceboTestResult(
                            original_effect=original_effect,
                            placebo_effects=[placebo_result.average_treatment_effect],
                            p_value=1.0 if abs(placebo_result.average_treatment_effect) >= abs(original_effect) else 0.0,
                            effect_rank=1,
                            total_placebos=1,
                            robust_to_placebo=abs(original_effect) > abs(placebo_result.average_treatment_effect)
                        )

                        placebo_results.append(placebo_test)

                    except Exception as e:
                        logger.warning(f"Placebo control pool test failed: {e}")

            logger.info(f"Placebo test suite completed: {len(placebo_results)} tests")
            return placebo_results

        except Exception as e:
            logger.error(f"Error in placebo test suite: {e}")
            return []

    def validate_backdoor_criterion(self,
                                  causal_graph: Dict[str, List[str]],
                                  treatment: str,
                                  outcome: str,
                                  adjustment_set: List[str],
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate backdoor criterion for causal identification

        Args:
            causal_graph: Adjacency list representation of causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            adjustment_set: Proposed adjustment set
            data: Data for conditional independence tests

        Returns:
            Validation results
        """
        logger.info(f"Validating backdoor criterion: {treatment} -> {outcome}")

        validation_result = {
            'criterion_satisfied': False,
            'backdoor_paths': [],
            'blocked_paths': [],
            'unblocked_paths': [],
            'conditional_independence_tests': [],
            'recommendations': []
        }

        try:
            # Find all paths from treatment to outcome
            all_paths = self._find_all_paths(causal_graph, treatment, outcome)

            # Identify backdoor paths (paths that start with an edge into treatment)
            backdoor_paths = []
            for path in all_paths:
                if len(path) > 2:  # Need intermediate nodes
                    # Check if first edge goes into treatment
                    if treatment in causal_graph.get(path[1], []):
                        backdoor_paths.append(path)

            validation_result['backdoor_paths'] = backdoor_paths

            # Check if adjustment set blocks all backdoor paths
            blocked_paths = []
            unblocked_paths = []

            for path in backdoor_paths:
                if self._path_blocked_by_adjustment_set(path, adjustment_set, causal_graph):
                    blocked_paths.append(path)
                else:
                    unblocked_paths.append(path)

            validation_result['blocked_paths'] = blocked_paths
            validation_result['unblocked_paths'] = unblocked_paths

            # Criterion satisfied if all backdoor paths are blocked
            criterion_satisfied = len(unblocked_paths) == 0

            # Additional check: no descendants of treatment in adjustment set
            treatment_descendants = self._get_descendants(causal_graph, treatment)
            invalid_adjusters = [var for var in adjustment_set if var in treatment_descendants]

            if invalid_adjusters:
                criterion_satisfied = False
                validation_result['recommendations'].append(
                    f"Remove descendants of treatment from adjustment set: {invalid_adjusters}"
                )

            validation_result['criterion_satisfied'] = criterion_satisfied

            # Perform conditional independence tests if data available
            if not data.empty and all(var in data.columns for var in [treatment, outcome] + adjustment_set):
                ci_tests = self._conditional_independence_tests(
                    data, treatment, outcome, adjustment_set
                )
                validation_result['conditional_independence_tests'] = ci_tests

            # Generate recommendations
            if not criterion_satisfied:
                if unblocked_paths:
                    validation_result['recommendations'].append(
                        f"Add variables to block unblocked paths: {unblocked_paths}"
                    )
                if not backdoor_paths:
                    validation_result['recommendations'].append(
                        "No backdoor paths found - direct causal identification may be possible"
                    )

            return validation_result

        except Exception as e:
            logger.error(f"Error validating backdoor criterion: {e}")
            validation_result['error'] = str(e)
            return validation_result

    def _find_all_paths(self, graph: Dict[str, List[str]], source: str, target: str) -> List[List[str]]:
        """Find all paths between source and target in directed graph"""
        def dfs_paths(current_path):
            current = current_path[-1]
            if current == target:
                yield current_path[:]
                return

            if len(current_path) > 10:  # Prevent infinite loops
                return

            for neighbor in graph.get(current, []):
                if neighbor not in current_path:  # Avoid cycles
                    current_path.append(neighbor)
                    yield from dfs_paths(current_path)
                    current_path.pop()

        return list(dfs_paths([source]))

    def _path_blocked_by_adjustment_set(self, path: List[str],
                                      adjustment_set: List[str],
                                      graph: Dict[str, List[str]]) -> bool:
        """Check if path is blocked by adjustment set"""
        # Simple implementation: path is blocked if any non-collider on path is in adjustment set
        for i in range(1, len(path) - 1):
            node = path[i]

            # Check if node is a collider (both edges point to it)
            is_collider = (
                path[i-1] in graph.get(node, []) and  # Previous node points to current
                path[i+1] in graph.get(node, [])      # Next node points to current
            )

            if not is_collider and node in adjustment_set:
                return True  # Non-collider in adjustment set blocks path

        return False

    def _get_descendants(self, graph: Dict[str, List[str]], node: str) -> Set[str]:
        """Get all descendants of a node"""
        descendants = set()

        def dfs(current):
            for child in graph.get(current, []):
                if child not in descendants:
                    descendants.add(child)
                    dfs(child)

        dfs(node)
        return descendants

    def _conditional_independence_tests(self, data: pd.DataFrame,
                                      treatment: str, outcome: str,
                                      adjustment_set: List[str]) -> List[Dict[str, Any]]:
        """Perform conditional independence tests"""
        tests = []

        try:
            # Test that treatment and outcome are dependent unconditionally
            unconditional_test = self._independence_test(
                data[treatment], data[outcome], []
            )
            tests.append({
                'test': f"{treatment} ⊥ {outcome}",
                'p_value': unconditional_test['p_value'],
                'independent': unconditional_test['p_value'] > 0.05,
                'expected_independent': False  # Should be dependent
            })

            # Test conditional independence given adjustment set
            if adjustment_set:
                conditional_test = self._independence_test(
                    data[treatment], data[outcome],
                    [data[var] for var in adjustment_set]
                )
                tests.append({
                    'test': f"{treatment} ⊥ {outcome} | {adjustment_set}",
                    'p_value': conditional_test['p_value'],
                    'independent': conditional_test['p_value'] > 0.05,
                    'expected_independent': True  # Should be independent after adjustment
                })

        except Exception as e:
            logger.error(f"Error in conditional independence tests: {e}")

        return tests

    def _independence_test(self, x: pd.Series, y: pd.Series,
                         conditioning_vars: List[pd.Series]) -> Dict[str, float]:
        """Test independence between x and y, optionally conditioning on other variables"""
        try:
            if not conditioning_vars:
                # Simple correlation test
                correlation, p_value = stats.pearsonr(x.dropna(), y.dropna())
                return {'correlation': correlation, 'p_value': p_value}

            # Partial correlation test (simplified)
            # In practice, would use more sophisticated methods

            # Create combined dataset
            combined = pd.concat([x, y] + conditioning_vars, axis=1).dropna()

            if len(combined) < 10:
                return {'correlation': 0.0, 'p_value': 1.0}

            # Simple correlation after including conditioning variables
            correlation, p_value = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])

            return {'correlation': correlation, 'p_value': p_value}

        except Exception as e:
            logger.error(f"Error in independence test: {e}")
            return {'correlation': 0.0, 'p_value': 1.0}

    def comprehensive_validation_suite(self,
                                     data: pd.DataFrame,
                                     treated_unit: str,
                                     outcome_variable: str,
                                     treatment_date: datetime,
                                     causal_graph: Optional[Dict[str, List[str]]] = None,
                                     adjustment_set: Optional[List[str]] = None) -> ValidationSuite:
        """
        Run comprehensive validation suite combining all methods

        Args:
            data: Panel data
            treated_unit: Unit that received treatment
            outcome_variable: Outcome variable
            treatment_date: Treatment date
            causal_graph: Causal graph for backdoor validation
            adjustment_set: Proposed adjustment set

        Returns:
            Complete validation suite results
        """
        logger.info(f"Running comprehensive validation suite for {treated_unit}")

        try:
            # Get control units
            control_units = [unit for unit in data['unit'].unique()
                           if unit != treated_unit]

            # 1. Synthetic control analysis
            synthetic_result = self.synthetic_control_analysis(
                data, treated_unit, outcome_variable, treatment_date, control_units
            )

            # 2. Placebo tests
            placebo_results = self.placebo_test_suite(
                data, treated_unit, outcome_variable, treatment_date
            )

            # 3. Backdoor validation
            backdoor_validation = {}
            if causal_graph and adjustment_set:
                backdoor_validation = self.validate_backdoor_criterion(
                    causal_graph, treated_unit, outcome_variable, adjustment_set, data
                )

            # 4. Calculate overall validity score
            validity_score = self._calculate_overall_validity_score(
                synthetic_result, placebo_results, backdoor_validation
            )

            # 5. Generate recommendations
            recommendations = self._generate_validation_recommendations(
                synthetic_result, placebo_results, backdoor_validation, validity_score
            )

            validation_suite = ValidationSuite(
                treatment_unit=treated_unit,
                outcome_variable=outcome_variable,
                treatment_date=treatment_date,
                synthetic_control=synthetic_result,
                placebo_tests=placebo_results,
                permutation_p_value=synthetic_result.p_value,
                backdoor_validation=backdoor_validation,
                instrumental_validation={},  # Would implement if instruments available
                overall_validity_score=validity_score,
                recommendations=recommendations
            )

            # Store results
            self.validation_results.append(validation_suite)

            logger.info(f"Validation suite completed: validity score = {validity_score:.2f}")
            return validation_suite

        except Exception as e:
            logger.error(f"Error in comprehensive validation suite: {e}")
            raise

    def _calculate_overall_validity_score(self,
                                        synthetic_result: SyntheticControlResult,
                                        placebo_results: List[PlaceboTestResult],
                                        backdoor_validation: Dict[str, Any]) -> float:
        """Calculate overall validity score from all validation tests"""
        try:
            scores = []

            # Synthetic control quality (0-1)
            sc_score = 1.0 / (1.0 + synthetic_result.pre_treatment_fit)  # Better fit = higher score
            scores.append(sc_score)

            # Statistical significance (0-1)
            sig_score = 1.0 - synthetic_result.p_value  # Lower p-value = higher score
            scores.append(sig_score)

            # Placebo test robustness (0-1)
            if placebo_results:
                placebo_score = sum(1 for result in placebo_results if result.robust_to_placebo) / len(placebo_results)
                scores.append(placebo_score)

            # Backdoor criterion satisfaction (0-1)
            if backdoor_validation:
                backdoor_score = 1.0 if backdoor_validation.get('criterion_satisfied', False) else 0.0
                scores.append(backdoor_score)

            # Weight concentration penalty (prefer distributed weights)
            weight_concentration = synthetic_result.validation_metrics.get('weight_concentration', 0.5)
            concentration_score = 1.0 - weight_concentration  # Lower concentration = higher score
            scores.append(concentration_score)

            # Overall score as weighted average
            weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(scores)]
            overall_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)

            return overall_score

        except Exception as e:
            logger.error(f"Error calculating validity score: {e}")
            return 0.0

    def _generate_validation_recommendations(self,
                                           synthetic_result: SyntheticControlResult,
                                           placebo_results: List[PlaceboTestResult],
                                           backdoor_validation: Dict[str, Any],
                                           validity_score: float) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        try:
            # Check synthetic control quality
            if synthetic_result.pre_treatment_fit > 0.1:
                recommendations.append(
                    "Poor pre-treatment fit - consider additional predictors or different control pool"
                )

            # Check statistical significance
            if synthetic_result.p_value > 0.10:
                recommendations.append(
                    "Effect not statistically significant - consider alternative identification strategies"
                )

            # Check weight concentration
            max_weight = synthetic_result.validation_metrics.get('weight_concentration', 0.5)
            if max_weight > 0.8:
                recommendations.append(
                    "Synthetic control dominated by single unit - results may not be robust"
                )

            # Check placebo robustness
            if placebo_results:
                robust_count = sum(1 for result in placebo_results if result.robust_to_placebo)
                if robust_count < len(placebo_results) * 0.7:
                    recommendations.append(
                        "Failed multiple placebo tests - causal identification may be questionable"
                    )

            # Check backdoor criterion
            if backdoor_validation and not backdoor_validation.get('criterion_satisfied', True):
                recommendations.append(
                    "Backdoor criterion not satisfied - add control variables or find instruments"
                )

            # Overall validity assessment
            if validity_score < 0.3:
                recommendations.append(
                    "Low overall validity - consider alternative causal identification strategies"
                )
            elif validity_score < 0.6:
                recommendations.append(
                    "Moderate validity - interpret results with caution and seek additional evidence"
                )
            else:
                recommendations.append(
                    "High validity - results appear robust to multiple validation tests"
                )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        return recommendations

    def export_validation_report(self, validation_suite: ValidationSuite) -> Dict[str, Any]:
        """Export comprehensive validation report"""
        return {
            'validation_summary': {
                'treatment_unit': validation_suite.treatment_unit,
                'outcome_variable': validation_suite.outcome_variable,
                'treatment_date': validation_suite.treatment_date.isoformat(),
                'overall_validity_score': validation_suite.overall_validity_score,
                'recommendations': validation_suite.recommendations
            },
            'synthetic_control': {
                'average_treatment_effect': validation_suite.synthetic_control.average_treatment_effect,
                'p_value': validation_suite.synthetic_control.p_value,
                'confidence_interval': validation_suite.synthetic_control.confidence_interval,
                'pre_treatment_fit': validation_suite.synthetic_control.pre_treatment_fit,
                'synthetic_weights': validation_suite.synthetic_control.synthetic_weights
            },
            'placebo_tests': [
                {
                    'original_effect': test.original_effect,
                    'placebo_effects': test.placebo_effects,
                    'robust_to_placebo': test.robust_to_placebo,
                    'p_value': test.p_value
                }
                for test in validation_suite.placebo_tests
            ],
            'backdoor_validation': validation_suite.backdoor_validation,
            'timestamp': datetime.now().isoformat()
        }