"""
Performance Optimization Recommendation Engine
Advanced optimization system for trading strategy parameter tuning and performance enhancement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle

warnings.filterwarnings('ignore')

@dataclass
class OptimizationParameter:
    """Optimization parameter definition"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    parameter_type: str = "continuous"  # continuous, discrete, categorical
    importance: float = 1.0
    description: str = ""

@dataclass
class OptimizationResult:
    """Single optimization result"""
    parameters: Dict[str, float]
    objective_value: float
    metrics: Dict[str, float]
    backtest_results: Dict
    confidence: float
    optimization_time: float

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    parameter_name: str
    current_value: float
    recommended_value: float
    expected_improvement: float
    confidence: float
    risk_impact: str
    implementation_priority: str  # HIGH, MEDIUM, LOW
    rationale: str

class PerformanceOptimizer:
    """Advanced performance optimization system"""

    def __init__(self, objective_function: Callable = None,
                 optimization_method: str = "bayesian",
                 n_trials: int = 100):
        self.objective_function = objective_function or self._default_objective
        self.optimization_method = optimization_method
        self.n_trials = n_trials

        # Optimization state
        self.optimization_history = []
        self.best_parameters = {}
        self.parameter_importance = {}

        # Models
        self.surrogate_model = None
        self.parameter_sensitivity_model = None

        # Gary×Taleb specific objectives
        self.gary_taleb_objectives = {
            'sharpe_ratio': {'weight': 0.25, 'target': 2.0},
            'calmar_ratio': {'weight': 0.20, 'target': 1.5},
            'antifragility_score': {'weight': 0.20, 'target': 0.7},
            'dpi_score': {'weight': 0.15, 'target': 0.8},
            'max_drawdown': {'weight': 0.10, 'target': -0.10},
            'profit_factor': {'weight': 0.10, 'target': 1.8}
        }

    def optimize_strategy_parameters(self,
                                   parameters: List[OptimizationParameter],
                                   strategy_function: Callable,
                                   data: pd.DataFrame,
                                   validation_split: float = 0.3) -> List[OptimizationResult]:
        """Comprehensive strategy parameter optimization"""

        print(f"Starting parameter optimization using {self.optimization_method} method...")
        print(f"Optimizing {len(parameters)} parameters with {self.n_trials} trials")

        # Split data for validation
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        validation_data = data.iloc[split_idx:]

        results = []

        if self.optimization_method == "bayesian":
            results = self._bayesian_optimization(parameters, strategy_function, train_data, validation_data)
        elif self.optimization_method == "optuna":
            results = self._optuna_optimization(parameters, strategy_function, train_data, validation_data)
        elif self.optimization_method == "differential_evolution":
            results = self._differential_evolution_optimization(parameters, strategy_function, train_data, validation_data)
        elif self.optimization_method == "grid_search":
            results = self._grid_search_optimization(parameters, strategy_function, train_data, validation_data)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        # Post-process results
        results = self._post_process_results(results, validation_data, strategy_function)

        # Update best parameters
        if results:
            self.best_parameters = results[0].parameters

        print(f"Optimization completed. Found {len(results)} valid parameter sets.")

        return results

    def _bayesian_optimization(self, parameters: List[OptimizationParameter],
                             strategy_function: Callable,
                             train_data: pd.DataFrame,
                             validation_data: pd.DataFrame) -> List[OptimizationResult]:
        """Bayesian optimization using Gaussian Process"""

        # Define parameter bounds
        bounds = [(p.min_value, p.max_value) for p in parameters]
        param_names = [p.name for p in parameters]

        # Initialize Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)

        # Initial random samples
        n_initial = min(20, self.n_trials // 4)
        X_samples = []
        y_samples = []

        print("Generating initial samples...")
        for i in range(n_initial):
            # Random parameter sample
            param_values = [np.random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
            param_dict = dict(zip(param_names, param_values))

            # Evaluate
            start_time = datetime.now()
            objective_value = self._evaluate_parameters(param_dict, strategy_function, train_data)
            eval_time = (datetime.now() - start_time).total_seconds()

            X_samples.append(param_values)
            y_samples.append(objective_value)

            print(f"Initial sample {i+1}/{n_initial}: Objective = {objective_value:.4f}")

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)

        # Bayesian optimization loop
        results = []

        for iteration in range(n_initial, self.n_trials):
            # Fit GP model
            gp.fit(X_samples, y_samples)

            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)

                # Expected Improvement
                best_f = np.max(y_samples)
                improvement = mu - best_f
                Z = improvement / sigma if sigma > 0 else 0
                ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)

                return -ei[0]  # Minimize negative EI

            # Optimize acquisition function
            acq_result = minimize(acquisition,
                                X_samples[np.argmax(y_samples)],
                                bounds=bounds,
                                method='L-BFGS-B')

            # Evaluate new point
            next_x = acq_result.x
            param_dict = dict(zip(param_names, next_x))

            start_time = datetime.now()
            objective_value = self._evaluate_parameters(param_dict, strategy_function, train_data)
            eval_time = (datetime.now() - start_time).total_seconds()

            # Update samples
            X_samples = np.vstack([X_samples, next_x])
            y_samples = np.append(y_samples, objective_value)

            # Store result
            backtest_result = strategy_function(train_data, **param_dict)
            metrics = self._calculate_optimization_metrics(backtest_result)

            result = OptimizationResult(
                parameters=param_dict,
                objective_value=objective_value,
                metrics=metrics,
                backtest_results=backtest_result,
                confidence=0.8,  # Simplified confidence
                optimization_time=eval_time
            )
            results.append(result)

            print(f"Iteration {iteration+1}/{self.n_trials}: Objective = {objective_value:.4f}")

        # Sort by objective value
        results.sort(key=lambda x: x.objective_value, reverse=True)

        return results

    def _optuna_optimization(self, parameters: List[OptimizationParameter],
                           strategy_function: Callable,
                           train_data: pd.DataFrame,
                           validation_data: pd.DataFrame) -> List[OptimizationResult]:
        """Optuna-based optimization"""

        def objective(trial):
            # Suggest parameter values
            param_dict = {}
            for param in parameters:
                if param.parameter_type == "continuous":
                    param_dict[param.name] = trial.suggest_float(
                        param.name, param.min_value, param.max_value
                    )
                elif param.parameter_type == "discrete":
                    if param.step_size:
                        param_dict[param.name] = trial.suggest_discrete_uniform(
                            param.name, param.min_value, param.max_value, param.step_size
                        )
                    else:
                        param_dict[param.name] = trial.suggest_int(
                            param.name, int(param.min_value), int(param.max_value)
                        )

            # Evaluate parameters
            return self._evaluate_parameters(param_dict, strategy_function, train_data)

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        # Convert results
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                backtest_result = strategy_function(train_data, **trial.params)
                metrics = self._calculate_optimization_metrics(backtest_result)

                result = OptimizationResult(
                    parameters=trial.params,
                    objective_value=trial.value,
                    metrics=metrics,
                    backtest_results=backtest_result,
                    confidence=0.8,
                    optimization_time=0  # Not tracked in basic Optuna
                )
                results.append(result)

        return sorted(results, key=lambda x: x.objective_value, reverse=True)

    def _differential_evolution_optimization(self, parameters: List[OptimizationParameter],
                                           strategy_function: Callable,
                                           train_data: pd.DataFrame,
                                           validation_data: pd.DataFrame) -> List[OptimizationResult]:
        """Differential Evolution optimization"""

        bounds = [(p.min_value, p.max_value) for p in parameters]
        param_names = [p.name for p in parameters]

        def objective(x):
            param_dict = dict(zip(param_names, x))
            return -self._evaluate_parameters(param_dict, strategy_function, train_data)  # Minimize negative

        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.n_trials // 10,
            popsize=10,
            seed=42
        )

        # Create result
        param_dict = dict(zip(param_names, result.x))
        backtest_result = strategy_function(train_data, **param_dict)
        metrics = self._calculate_optimization_metrics(backtest_result)

        opt_result = OptimizationResult(
            parameters=param_dict,
            objective_value=-result.fun,
            metrics=metrics,
            backtest_results=backtest_result,
            confidence=0.9,
            optimization_time=0
        )

        return [opt_result]

    def _grid_search_optimization(self, parameters: List[OptimizationParameter],
                                strategy_function: Callable,
                                train_data: pd.DataFrame,
                                validation_data: pd.DataFrame) -> List[OptimizationResult]:
        """Grid search optimization"""

        # Create parameter grids
        param_grids = []
        param_names = []

        for param in parameters:
            param_names.append(param.name)
            if param.step_size:
                grid = np.arange(param.min_value, param.max_value + param.step_size, param.step_size)
            else:
                grid = np.linspace(param.min_value, param.max_value, 10)
            param_grids.append(grid)

        # Generate all combinations
        from itertools import product
        param_combinations = list(product(*param_grids))

        # Limit combinations to n_trials
        if len(param_combinations) > self.n_trials:
            indices = np.random.choice(len(param_combinations), self.n_trials, replace=False)
            param_combinations = [param_combinations[i] for i in indices]

        # Evaluate all combinations
        results = []
        for i, combination in enumerate(param_combinations):
            param_dict = dict(zip(param_names, combination))

            start_time = datetime.now()
            objective_value = self._evaluate_parameters(param_dict, strategy_function, train_data)
            eval_time = (datetime.now() - start_time).total_seconds()

            backtest_result = strategy_function(train_data, **param_dict)
            metrics = self._calculate_optimization_metrics(backtest_result)

            result = OptimizationResult(
                parameters=param_dict,
                objective_value=objective_value,
                metrics=metrics,
                backtest_results=backtest_result,
                confidence=0.7,
                optimization_time=eval_time
            )
            results.append(result)

            print(f"Grid search {i+1}/{len(param_combinations)}: Objective = {objective_value:.4f}")

        return sorted(results, key=lambda x: x.objective_value, reverse=True)

    def _evaluate_parameters(self, param_dict: Dict[str, float],
                           strategy_function: Callable,
                           data: pd.DataFrame) -> float:
        """Evaluate parameter set using objective function"""

        try:
            # Run backtest with parameters
            backtest_result = strategy_function(data, **param_dict)

            # Calculate objective value
            objective_value = self.objective_function(backtest_result)

            return objective_value

        except Exception as e:
            print(f"Error evaluating parameters {param_dict}: {str(e)}")
            return -999999  # Large negative value for failed evaluations

    def _default_objective(self, backtest_result: Dict) -> float:
        """Default Gary×Taleb objective function"""

        if 'returns' not in backtest_result or len(backtest_result['returns']) == 0:
            return -999999

        returns = backtest_result['returns']
        equity_curve = backtest_result.get('equity_curve', (1 + returns).cumprod() * 200)

        # Calculate metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, equity_curve)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Gary×Taleb specific metrics
        antifragility_score = self._calculate_antifragility_score(returns)
        dpi_score = self._calculate_dpi_score(returns, backtest_result)
        profit_factor = self._calculate_profit_factor(returns)

        # Weighted objective
        objective = 0

        for metric_name, config in self.gary_taleb_objectives.items():
            if metric_name == 'sharpe_ratio':
                value = min(sharpe_ratio / config['target'], 1.0)
            elif metric_name == 'calmar_ratio':
                value = min(calmar_ratio / config['target'], 1.0)
            elif metric_name == 'antifragility_score':
                value = min(antifragility_score / config['target'], 1.0)
            elif metric_name == 'dpi_score':
                value = min(dpi_score / config['target'], 1.0)
            elif metric_name == 'max_drawdown':
                value = max(0, 1 + max_drawdown / config['target'])  # Less negative is better
            elif metric_name == 'profit_factor':
                value = min(profit_factor / config['target'], 1.0)
            else:
                value = 0

            objective += config['weight'] * value

        return objective

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_return = returns.mean() * 252 - 0.02
        volatility = returns.std() * np.sqrt(252)
        return excess_return / volatility if volatility > 0 else 0

    def _calculate_calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def _calculate_antifragility_score(self, returns: pd.Series) -> float:
        """Calculate Taleb-inspired antifragility score"""
        if len(returns) < 20:
            return 0.5

        vol_periods = returns.rolling(20).std()
        return_periods = returns.rolling(20).mean()

        correlation = vol_periods.corr(return_periods)
        return max(0, correlation) if not np.isnan(correlation) else 0

    def _calculate_dpi_score(self, returns: pd.Series, backtest_result: Dict) -> float:
        """Calculate Gary's DPI score"""
        if len(returns) < 20:
            return 0.5

        # Momentum component
        momentum = returns.rolling(20).mean().mean()

        # Stability component
        volatility_stability = 1 - (returns.rolling(10).std().std() / returns.rolling(20).std().mean()) if returns.rolling(20).std().mean() > 0 else 0

        # Combined score
        dpi_score = (np.tanh(momentum * 100) * 0.5 + max(0, min(1, volatility_stability)) * 0.5)

        return max(0, min(1, (dpi_score + 1) / 2))

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf')

        return positive_returns.sum() / abs(negative_returns.sum())

    def _calculate_optimization_metrics(self, backtest_result: Dict) -> Dict[str, float]:
        """Calculate comprehensive metrics for optimization result"""

        if 'returns' not in backtest_result:
            return {}

        returns = backtest_result['returns']
        equity_curve = backtest_result.get('equity_curve', (1 + returns).cumprod() * 200)

        metrics = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns, equity_curve),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'volatility': returns.std() * np.sqrt(252),
            'antifragility_score': self._calculate_antifragility_score(returns),
            'dpi_score': self._calculate_dpi_score(returns, backtest_result),
            'profit_factor': self._calculate_profit_factor(returns),
            'win_rate': (returns > 0).sum() / len(returns)
        }

        return metrics

    def _post_process_results(self, results: List[OptimizationResult],
                            validation_data: pd.DataFrame,
                            strategy_function: Callable) -> List[OptimizationResult]:
        """Post-process optimization results with validation"""

        print("Post-processing results with out-of-sample validation...")

        validated_results = []

        for result in results:
            try:
                # Validate on out-of-sample data
                validation_backtest = strategy_function(validation_data, **result.parameters)
                validation_metrics = self._calculate_optimization_metrics(validation_backtest)

                # Calculate validation score
                validation_objective = self.objective_function(validation_backtest)

                # Update result with validation data
                result.metrics.update({f'val_{k}': v for k, v in validation_metrics.items()})
                result.metrics['validation_objective'] = validation_objective
                result.metrics['validation_degradation'] = result.objective_value - validation_objective

                # Update confidence based on validation performance
                degradation = result.metrics['validation_degradation']
                if degradation < 0.1:
                    result.confidence = min(1.0, result.confidence * 1.2)
                elif degradation < 0.2:
                    result.confidence = result.confidence
                else:
                    result.confidence = result.confidence * 0.8

                validated_results.append(result)

            except Exception as e:
                print(f"Error in validation for parameters {result.parameters}: {str(e)}")
                continue

        return sorted(validated_results, key=lambda x: x.metrics.get('validation_objective', -999999), reverse=True)

    def generate_optimization_recommendations(self, optimization_results: List[OptimizationResult],
                                            current_parameters: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Generate actionable optimization recommendations"""

        if not optimization_results:
            return []

        best_result = optimization_results[0]
        recommendations = []

        for param_name, current_value in current_parameters.items():
            if param_name in best_result.parameters:
                recommended_value = best_result.parameters[param_name]

                # Calculate expected improvement
                current_objective = self._estimate_objective_for_params(current_parameters, optimization_results)
                expected_improvement = best_result.objective_value - current_objective

                # Determine confidence
                confidence = self._calculate_parameter_confidence(param_name, optimization_results)

                # Assess risk impact
                risk_impact = self._assess_parameter_risk_impact(param_name, current_value, recommended_value)

                # Determine priority
                priority = self._determine_implementation_priority(
                    abs(recommended_value - current_value) / abs(current_value) if current_value != 0 else 1,
                    expected_improvement,
                    confidence
                )

                # Generate rationale
                rationale = self._generate_parameter_rationale(
                    param_name, current_value, recommended_value, optimization_results
                )

                recommendation = OptimizationRecommendation(
                    parameter_name=param_name,
                    current_value=current_value,
                    recommended_value=recommended_value,
                    expected_improvement=expected_improvement,
                    confidence=confidence,
                    risk_impact=risk_impact,
                    implementation_priority=priority,
                    rationale=rationale
                )

                recommendations.append(recommendation)

        return sorted(recommendations, key=lambda x: x.expected_improvement, reverse=True)

    def _estimate_objective_for_params(self, params: Dict[str, float],
                                     optimization_results: List[OptimizationResult]) -> float:
        """Estimate objective value for given parameters"""

        # Find closest result
        min_distance = float('inf')
        closest_objective = 0

        for result in optimization_results:
            distance = sum((params.get(k, 0) - v) ** 2 for k, v in result.parameters.items())
            if distance < min_distance:
                min_distance = distance
                closest_objective = result.objective_value

        return closest_objective

    def _calculate_parameter_confidence(self, param_name: str,
                                      optimization_results: List[OptimizationResult]) -> float:
        """Calculate confidence in parameter recommendation"""

        # Analyze parameter consistency across top results
        top_results = optimization_results[:min(10, len(optimization_results))]
        param_values = [r.parameters.get(param_name, 0) for r in top_results]

        if len(param_values) < 2:
            return 0.5

        # Lower variance = higher confidence
        variance = np.var(param_values)
        mean_value = np.mean(param_values)
        cv = variance / abs(mean_value) if mean_value != 0 else 1

        confidence = max(0.1, min(1.0, 1.0 - cv))

        return confidence

    def _assess_parameter_risk_impact(self, param_name: str, current_value: float, recommended_value: float) -> str:
        """Assess risk impact of parameter change"""

        change_magnitude = abs(recommended_value - current_value) / abs(current_value) if current_value != 0 else 1

        if change_magnitude < 0.1:
            return "LOW: Minor parameter adjustment"
        elif change_magnitude < 0.3:
            return "MEDIUM: Moderate parameter change"
        elif change_magnitude < 0.5:
            return "HIGH: Significant parameter modification"
        else:
            return "CRITICAL: Major parameter overhaul required"

    def _determine_implementation_priority(self, change_magnitude: float,
                                         expected_improvement: float,
                                         confidence: float) -> str:
        """Determine implementation priority"""

        priority_score = expected_improvement * confidence / (change_magnitude + 0.1)

        if priority_score > 0.5:
            return "HIGH"
        elif priority_score > 0.2:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_parameter_rationale(self, param_name: str, current_value: float,
                                    recommended_value: float, optimization_results: List[OptimizationResult]) -> str:
        """Generate rationale for parameter recommendation"""

        change_direction = "increase" if recommended_value > current_value else "decrease"
        change_magnitude = abs(recommended_value - current_value) / abs(current_value) if current_value != 0 else 1

        # Analyze parameter impact
        top_results = optimization_results[:5]
        param_values = [r.parameters.get(param_name, current_value) for r in top_results]
        avg_top_value = np.mean(param_values)

        rationale = f"Optimization suggests to {change_direction} {param_name} by {change_magnitude:.1%}. "
        rationale += f"Top performing parameter sets average {avg_top_value:.3f} for this parameter. "

        # Add specific insights based on parameter name
        if 'threshold' in param_name.lower():
            rationale += "This affects signal sensitivity and trade frequency."
        elif 'stop' in param_name.lower():
            rationale += "This impacts risk management and drawdown control."
        elif 'position' in param_name.lower():
            rationale += "This influences position sizing and leverage."
        elif 'window' in param_name.lower() or 'period' in param_name.lower():
            rationale += "This changes the lookback period for calculations."

        return rationale

    def generate_optimization_report(self, optimization_results: List[OptimizationResult],
                                   recommendations: List[OptimizationRecommendation],
                                   current_parameters: Dict[str, float]) -> str:
        """Generate comprehensive optimization report"""

        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE OPTIMIZATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Optimization Method: {self.optimization_method}")
        report.append(f"Total Trials: {self.n_trials}")
        report.append(f"Valid Results: {len(optimization_results)}")
        report.append("")

        if optimization_results:
            best_result = optimization_results[0]

            # Current vs Optimized Performance
            report.append("PERFORMANCE COMPARISON")
            report.append("-" * 30)
            current_objective = self._estimate_objective_for_params(current_parameters, optimization_results)
            improvement = best_result.objective_value - current_objective

            report.append(f"Current Objective Score: {current_objective:.4f}")
            report.append(f"Optimized Objective Score: {best_result.objective_value:.4f}")
            report.append(f"Expected Improvement: {improvement:.4f} ({improvement/current_objective*100:.1f}%)")
            report.append("")

            # Key Metrics Comparison
            report.append("KEY METRICS IMPROVEMENT")
            report.append("-" * 30)
            metrics_to_show = ['sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'antifragility_score', 'dpi_score']

            for metric in metrics_to_show:
                if metric in best_result.metrics:
                    val_metric = best_result.metrics.get(f'val_{metric}', best_result.metrics[metric])
                    report.append(f"{metric.replace('_', ' ').title()}: {val_metric:.3f}")

            report.append("")

            # Parameter Recommendations
            report.append("PARAMETER OPTIMIZATION RECOMMENDATIONS")
            report.append("-" * 45)

            for i, rec in enumerate(recommendations[:10], 1):  # Top 10 recommendations
                report.append(f"\n{i}. {rec.parameter_name} [{rec.implementation_priority} Priority]")
                report.append(f"   Current: {rec.current_value:.4f}")
                report.append(f"   Recommended: {rec.recommended_value:.4f}")
                report.append(f"   Expected Improvement: {rec.expected_improvement:.4f}")
                report.append(f"   Confidence: {rec.confidence:.1%}")
                report.append(f"   Risk Impact: {rec.risk_impact}")
                report.append(f"   Rationale: {rec.rationale}")

            # Implementation Plan
            report.append("\n\nIMPLEMENTATION PLAN")
            report.append("-" * 25)

            high_priority = [r for r in recommendations if r.implementation_priority == "HIGH"]
            medium_priority = [r for r in recommendations if r.implementation_priority == "MEDIUM"]

            if high_priority:
                report.append("Phase 1 (High Priority):")
                for rec in high_priority:
                    report.append(f"  • Adjust {rec.parameter_name} to {rec.recommended_value:.4f}")

            if medium_priority:
                report.append("Phase 2 (Medium Priority):")
                for rec in medium_priority[:5]:  # Top 5 medium priority
                    report.append(f"  • Adjust {rec.parameter_name} to {rec.recommended_value:.4f}")

            # Risk Assessment
            report.append("\n\nRISK ASSESSMENT")
            report.append("-" * 20)

            critical_changes = [r for r in recommendations if "CRITICAL" in r.risk_impact]
            high_risk_changes = [r for r in recommendations if "HIGH" in r.risk_impact]

            if critical_changes:
                report.append("⚠️  CRITICAL RISK CHANGES:")
                for rec in critical_changes:
                    report.append(f"   • {rec.parameter_name}: {rec.risk_impact}")

            if high_risk_changes:
                report.append("⚠️  HIGH RISK CHANGES:")
                for rec in high_risk_changes:
                    report.append(f"   • {rec.parameter_name}: {rec.risk_impact}")

            if not critical_changes and not high_risk_changes:
                report.append("✅ No high-risk parameter changes identified")

            # Validation Results
            if 'validation_objective' in best_result.metrics:
                report.append("\n\nOUT-OF-SAMPLE VALIDATION")
                report.append("-" * 30)

                val_objective = best_result.metrics['validation_objective']
                degradation = best_result.metrics.get('validation_degradation', 0)

                report.append(f"In-Sample Objective: {best_result.objective_value:.4f}")
                report.append(f"Out-of-Sample Objective: {val_objective:.4f}")
                report.append(f"Performance Degradation: {degradation:.4f}")

                if degradation < 0.1:
                    report.append("✅ Excellent out-of-sample performance")
                elif degradation < 0.2:
                    report.append("✅ Good out-of-sample performance")
                else:
                    report.append("⚠️  Significant performance degradation detected")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Example strategy function
    def example_strategy(data: pd.DataFrame, **params) -> Dict:
        # Simulate a simple strategy with parameters
        returns = np.random.normal(0.0005, 0.015, len(data))

        # Apply parameter effects
        threshold = params.get('signal_threshold', 0.5)
        position_size = params.get('position_size', 1.0)

        # Modify returns based on parameters
        adjusted_returns = returns * position_size * (1 + threshold * 0.1)

        return {
            'returns': pd.Series(adjusted_returns, index=data.index),
            'equity_curve': pd.Series((1 + adjusted_returns).cumprod() * 200, index=data.index)
        }

    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sample_data = pd.DataFrame({'price': 100 + np.cumsum(np.random.normal(0, 1, len(dates)))}, index=dates)

    # Define parameters to optimize
    parameters = [
        OptimizationParameter(
            name='signal_threshold',
            current_value=0.5,
            min_value=0.1,
            max_value=0.9,
            description="Signal strength threshold"
        ),
        OptimizationParameter(
            name='position_size',
            current_value=1.0,
            min_value=0.1,
            max_value=1.5,
            description="Position sizing multiplier"
        )
    ]

    # Initialize optimizer
    optimizer = PerformanceOptimizer(optimization_method="bayesian", n_trials=50)

    # Run optimization
    results = optimizer.optimize_strategy_parameters(parameters, example_strategy, sample_data)

    # Generate recommendations
    current_params = {'signal_threshold': 0.5, 'position_size': 1.0}
    recommendations = optimizer.generate_optimization_recommendations(results, current_params)

    # Generate report
    report = optimizer.generate_optimization_report(results, recommendations, current_params)
    print(report)

    print(f"\nOptimization completed successfully!")
    if results:
        print(f"Best objective value: {results[0].objective_value:.4f}")
        print(f"Best parameters: {results[0].parameters}")