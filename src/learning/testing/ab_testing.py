"""
A/B Testing Framework for Gary×Taleb Trading System

Production-grade A/B testing system for comparing model versions,
strategies, and parameter configurations with statistical significance testing.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path
import threading
import time
import uuid
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu

class ExperimentStatus(Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class TrafficSplitType(Enum):
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    STRATIFIED = "stratified"
    TIME_BASED = "time_based"

@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    control_variant: str
    treatment_variants: List[str]
    traffic_split: Dict[str, float]  # variant -> traffic percentage
    split_type: TrafficSplitType
    min_sample_size: int
    max_duration_days: int
    significance_level: float = 0.05
    minimum_detectable_effect: float = 0.02
    power: float = 0.8
    primary_metric: str = "gary_dpi"
    secondary_metrics: List[str] = None
    stratification_keys: List[str] = None
    early_stopping_enabled: bool = True
    early_stopping_lookback_days: int = 7

@dataclass
class VariantConfig:
    """Configuration for experiment variant"""
    variant_id: str
    name: str
    description: str
    model_config: Dict[str, Any]
    strategy_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    risk_config: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Result data for experiment participant"""
    experiment_id: str
    variant_id: str
    user_id: str
    timestamp: datetime
    primary_metric_value: float
    secondary_metrics: Dict[str, float]
    trade_data: Dict[str, Any]
    context_data: Dict[str, Any]

@dataclass
class StatisticalResult:
    """Statistical test result"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    relative_change: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_statistic: float
    test_type: str
    sample_size_control: int
    sample_size_treatment: int

@dataclass
class ExperimentSummary:
    """Summary of A/B test experiment"""
    experiment_id: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    total_participants: int
    variant_distribution: Dict[str, int]
    statistical_results: Dict[str, StatisticalResult]
    recommendation: str
    confidence_score: float
    early_stopping_triggered: bool
    business_impact: Dict[str, float]

class ABTestingFramework:
    """
    A/B testing framework for comparing trading models and strategies
    with proper statistical significance testing and early stopping.
    """

    def __init__(self):
        self.logger = self._setup_logging()
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/ab_testing.db")
        self._init_database()

        # Active experiments
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_variants: Dict[str, Dict[str, VariantConfig]] = {}

        # Traffic assignment
        self.traffic_assignments: Dict[str, str] = {}  # user_id -> variant_id

        # Background monitoring
        self.is_monitoring = False
        self.monitor_thread = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for A/B testing"""
        logger = logging.getLogger('ABTesting')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/ab_testing.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for A/B testing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    hypothesis TEXT,
                    status TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS variants (
                    variant_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS traffic_assignments (
                    user_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, experiment_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                    FOREIGN KEY (variant_id) REFERENCES variants (variant_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    primary_metric_value REAL,
                    secondary_metrics_json TEXT,
                    trade_data_json TEXT,
                    context_data_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                    FOREIGN KEY (variant_id) REFERENCES variants (variant_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS statistical_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    control_variant TEXT NOT NULL,
                    treatment_variant TEXT NOT NULL,
                    control_mean REAL,
                    treatment_mean REAL,
                    effect_size REAL,
                    relative_change REAL,
                    p_value REAL,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    is_significant BOOLEAN,
                    test_statistic REAL,
                    test_type TEXT,
                    sample_size_control INTEGER,
                    sample_size_treatment INTEGER,
                    calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

    def create_experiment(self, config: ExperimentConfig,
                         variants: List[VariantConfig]) -> str:
        """
        Create a new A/B test experiment

        Args:
            config: Experiment configuration
            variants: List of variant configurations

        Returns:
            experiment_id: Unique experiment identifier
        """
        try:
            # Validate configuration
            self._validate_experiment_config(config, variants)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Insert experiment
                conn.execute('''
                    INSERT INTO experiments (experiment_id, name, description, hypothesis, status, config_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    config.experiment_id,
                    config.name,
                    config.description,
                    config.hypothesis,
                    ExperimentStatus.PLANNED.value,
                    json.dumps(asdict(config))
                ))

                # Insert variants
                for variant in variants:
                    conn.execute('''
                        INSERT INTO variants (variant_id, experiment_id, name, description, config_json)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        variant.variant_id,
                        config.experiment_id,
                        variant.name,
                        variant.description,
                        json.dumps(asdict(variant))
                    ))

            # Store in memory
            self.active_experiments[config.experiment_id] = config
            self.experiment_variants[config.experiment_id] = {v.variant_id: v for v in variants}

            self.logger.info(f"Created experiment: {config.experiment_id}")
            return config.experiment_id

        except Exception as e:
            self.logger.error(f"Error creating experiment: {e}")
            raise

    def _validate_experiment_config(self, config: ExperimentConfig, variants: List[VariantConfig]):
        """Validate experiment configuration"""
        # Check traffic split sums to 1.0
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")

        # Check all variants have traffic allocation
        variant_ids = {v.variant_id for v in variants}
        traffic_variants = set(config.traffic_split.keys())

        if variant_ids != traffic_variants:
            raise ValueError("Mismatch between variants and traffic split")

        # Check control variant exists
        if config.control_variant not in variant_ids:
            raise ValueError(f"Control variant {config.control_variant} not found in variants")

        # Check treatment variants exist
        for treatment in config.treatment_variants:
            if treatment not in variant_ids:
                raise ValueError(f"Treatment variant {treatment} not found in variants")

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Update status in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE experiments
                    SET status = ?, start_date = ?
                    WHERE experiment_id = ?
                ''', (ExperimentStatus.RUNNING.value, datetime.now().isoformat(), experiment_id))

            # Start monitoring if not already running
            if not self.is_monitoring:
                self.start_monitoring()

            self.logger.info(f"Started experiment: {experiment_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting experiment: {e}")
            return False

    def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an A/B test experiment"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Update status in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE experiments
                    SET status = ?, end_date = ?
                    WHERE experiment_id = ?
                ''', (ExperimentStatus.COMPLETED.value, datetime.now().isoformat(), experiment_id))

            self.logger.info(f"Stopped experiment {experiment_id}: {reason}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping experiment: {e}")
            return False

    def assign_variant(self, experiment_id: str, user_id: str,
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Assign user to experiment variant

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            context: Additional context for stratified assignment

        Returns:
            variant_id: Assigned variant ID
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            config = self.active_experiments[experiment_id]

            # Check if user already assigned
            assignment_key = f"{user_id}_{experiment_id}"
            if assignment_key in self.traffic_assignments:
                return self.traffic_assignments[assignment_key]

            # Assign variant based on split type
            if config.split_type == TrafficSplitType.RANDOM:
                variant_id = self._random_assignment(config, user_id)
            elif config.split_type == TrafficSplitType.DETERMINISTIC:
                variant_id = self._deterministic_assignment(config, user_id)
            elif config.split_type == TrafficSplitType.STRATIFIED:
                variant_id = self._stratified_assignment(config, user_id, context)
            elif config.split_type == TrafficSplitType.TIME_BASED:
                variant_id = self._time_based_assignment(config, user_id)
            else:
                variant_id = self._random_assignment(config, user_id)

            # Store assignment
            self.traffic_assignments[assignment_key] = variant_id

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO traffic_assignments (user_id, experiment_id, variant_id)
                    VALUES (?, ?, ?)
                ''', (user_id, experiment_id, variant_id))

            return variant_id

        except Exception as e:
            self.logger.error(f"Error assigning variant: {e}")
            # Return control variant as fallback
            return self.active_experiments[experiment_id].control_variant

    def _random_assignment(self, config: ExperimentConfig, user_id: str) -> str:
        """Random variant assignment"""
        # Use user_id for deterministic randomness
        np.random.seed(hash(user_id) % (2**32))
        random_val = np.random.random()

        cumulative_prob = 0
        for variant_id, traffic_share in config.traffic_split.items():
            cumulative_prob += traffic_share
            if random_val <= cumulative_prob:
                return variant_id

        # Fallback to control
        return config.control_variant

    def _deterministic_assignment(self, config: ExperimentConfig, user_id: str) -> str:
        """Deterministic assignment based on user ID hash"""
        user_hash = hash(user_id) % 100  # 0-99

        cumulative_threshold = 0
        for variant_id, traffic_share in config.traffic_split.items():
            cumulative_threshold += int(traffic_share * 100)
            if user_hash < cumulative_threshold:
                return variant_id

        # Fallback to control
        return config.control_variant

    def _stratified_assignment(self, config: ExperimentConfig, user_id: str,
                             context: Optional[Dict[str, Any]]) -> str:
        """Stratified assignment based on context"""
        # For now, fall back to random assignment
        # In production, implement stratification logic based on context
        return self._random_assignment(config, user_id)

    def _time_based_assignment(self, config: ExperimentConfig, user_id: str) -> str:
        """Time-based assignment (e.g., alternate by day)"""
        day_of_year = datetime.now().timetuple().tm_yday
        variant_index = day_of_year % len(config.traffic_split)
        variant_id = list(config.traffic_split.keys())[variant_index]
        return variant_id

    def record_result(self, experiment_id: str, user_id: str,
                     primary_metric_value: float,
                     secondary_metrics: Optional[Dict[str, float]] = None,
                     trade_data: Optional[Dict[str, Any]] = None,
                     context_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record experiment result for a user

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            primary_metric_value: Value of primary metric (e.g., Gary DPI)
            secondary_metrics: Secondary metric values
            trade_data: Trade-specific data
            context_data: Additional context

        Returns:
            success: Whether recording was successful
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Get user's variant assignment
            assignment_key = f"{user_id}_{experiment_id}"
            if assignment_key not in self.traffic_assignments:
                # Assign variant if not already assigned
                variant_id = self.assign_variant(experiment_id, user_id, context_data)
            else:
                variant_id = self.traffic_assignments[assignment_key]

            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                timestamp=datetime.now(),
                primary_metric_value=primary_metric_value,
                secondary_metrics=secondary_metrics or {},
                trade_data=trade_data or {},
                context_data=context_data or {}
            )

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO experiment_results (
                        experiment_id, variant_id, user_id, timestamp,
                        primary_metric_value, secondary_metrics_json,
                        trade_data_json, context_data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.experiment_id,
                    result.variant_id,
                    result.user_id,
                    result.timestamp.isoformat(),
                    result.primary_metric_value,
                    json.dumps(result.secondary_metrics),
                    json.dumps(result.trade_data),
                    json.dumps(result.context_data)
                ))

            return True

        except Exception as e:
            self.logger.error(f"Error recording result: {e}")
            return False

    def analyze_experiment(self, experiment_id: str,
                          force_analysis: bool = False) -> Optional[ExperimentSummary]:
        """
        Analyze experiment results with statistical testing

        Args:
            experiment_id: Experiment ID
            force_analysis: Force analysis even if minimum sample size not met

        Returns:
            ExperimentSummary with results and recommendations
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            config = self.active_experiments[experiment_id]

            # Load results from database
            results_data = self._load_experiment_results(experiment_id)

            if not results_data:
                self.logger.warning(f"No results found for experiment {experiment_id}")
                return None

            # Check minimum sample size
            if not force_analysis and len(results_data) < config.min_sample_size:
                self.logger.info(f"Insufficient sample size for experiment {experiment_id}")
                return None

            # Group results by variant
            variant_results = {}
            for result in results_data:
                variant_id = result['variant_id']
                if variant_id not in variant_results:
                    variant_results[variant_id] = []
                variant_results[variant_id].append(result)

            # Perform statistical analysis
            statistical_results = {}

            # Analyze primary metric
            primary_metric_results = self._analyze_metric(
                variant_results, config.control_variant, config.treatment_variants,
                'primary_metric_value', config.primary_metric, config.significance_level
            )
            statistical_results.update(primary_metric_results)

            # Analyze secondary metrics
            if config.secondary_metrics:
                for metric_name in config.secondary_metrics:
                    secondary_results = self._analyze_secondary_metric(
                        variant_results, config.control_variant, config.treatment_variants,
                        metric_name, config.significance_level
                    )
                    statistical_results.update(secondary_results)

            # Generate recommendations
            recommendation, confidence_score = self._generate_recommendation(
                statistical_results, config
            )

            # Calculate business impact
            business_impact = self._calculate_business_impact(statistical_results, variant_results)

            # Check for early stopping
            early_stopping_triggered = self._check_early_stopping(config, statistical_results)

            # Create summary
            start_date, end_date = self._get_experiment_dates(experiment_id)
            duration = (end_date or datetime.now()) - start_date

            summary = ExperimentSummary(
                experiment_id=experiment_id,
                status=self._get_experiment_status(experiment_id),
                start_date=start_date,
                end_date=end_date,
                duration_days=duration.days,
                total_participants=len(results_data),
                variant_distribution={k: len(v) for k, v in variant_results.items()},
                statistical_results=statistical_results,
                recommendation=recommendation,
                confidence_score=confidence_score,
                early_stopping_triggered=early_stopping_triggered,
                business_impact=business_impact
            )

            # Save statistical results to database
            self._save_statistical_results(experiment_id, statistical_results)

            return summary

        except Exception as e:
            self.logger.error(f"Error analyzing experiment: {e}")
            return None

    def _load_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load experiment results from database"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT experiment_id, variant_id, user_id, timestamp,
                       primary_metric_value, secondary_metrics_json,
                       trade_data_json, context_data_json
                FROM experiment_results
                WHERE experiment_id = ?
                ORDER BY timestamp
            '''
            cursor = conn.execute(query, (experiment_id,))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                result = {
                    'experiment_id': row[0],
                    'variant_id': row[1],
                    'user_id': row[2],
                    'timestamp': datetime.fromisoformat(row[3]),
                    'primary_metric_value': row[4],
                    'secondary_metrics': json.loads(row[5]) if row[5] else {},
                    'trade_data': json.loads(row[6]) if row[6] else {},
                    'context_data': json.loads(row[7]) if row[7] else {}
                }
                results.append(result)

            return results

    def _analyze_metric(self, variant_results: Dict[str, List], control_variant: str,
                       treatment_variants: List[str], metric_key: str, metric_name: str,
                       significance_level: float) -> Dict[str, StatisticalResult]:
        """Analyze a specific metric across variants"""
        results = {}

        if control_variant not in variant_results:
            return results

        control_values = [r[metric_key] for r in variant_results[control_variant]
                         if r[metric_key] is not None]

        for treatment_variant in treatment_variants:
            if treatment_variant not in variant_results:
                continue

            treatment_values = [r[metric_key] for r in variant_results[treatment_variant]
                              if r[metric_key] is not None]

            if len(control_values) < 2 or len(treatment_values) < 2:
                continue

            # Perform statistical test
            stat_result = self._perform_statistical_test(
                control_values, treatment_values, significance_level
            )

            # Create result object
            comparison_key = f"{metric_name}_{control_variant}_vs_{treatment_variant}"
            results[comparison_key] = StatisticalResult(
                metric_name=metric_name,
                control_mean=stat_result['control_mean'],
                treatment_mean=stat_result['treatment_mean'],
                effect_size=stat_result['effect_size'],
                relative_change=stat_result['relative_change'],
                p_value=stat_result['p_value'],
                confidence_interval=stat_result['confidence_interval'],
                is_significant=stat_result['is_significant'],
                test_statistic=stat_result['test_statistic'],
                test_type=stat_result['test_type'],
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values)
            )

        return results

    def _analyze_secondary_metric(self, variant_results: Dict[str, List], control_variant: str,
                                 treatment_variants: List[str], metric_name: str,
                                 significance_level: float) -> Dict[str, StatisticalResult]:
        """Analyze secondary metric"""
        results = {}

        if control_variant not in variant_results:
            return results

        # Extract secondary metric values
        control_values = []
        for r in variant_results[control_variant]:
            if metric_name in r['secondary_metrics']:
                control_values.append(r['secondary_metrics'][metric_name])

        for treatment_variant in treatment_variants:
            if treatment_variant not in variant_results:
                continue

            treatment_values = []
            for r in variant_results[treatment_variant]:
                if metric_name in r['secondary_metrics']:
                    treatment_values.append(r['secondary_metrics'][metric_name])

            if len(control_values) < 2 or len(treatment_values) < 2:
                continue

            # Perform statistical test
            stat_result = self._perform_statistical_test(
                control_values, treatment_values, significance_level
            )

            # Create result object
            comparison_key = f"{metric_name}_{control_variant}_vs_{treatment_variant}"
            results[comparison_key] = StatisticalResult(
                metric_name=metric_name,
                control_mean=stat_result['control_mean'],
                treatment_mean=stat_result['treatment_mean'],
                effect_size=stat_result['effect_size'],
                relative_change=stat_result['relative_change'],
                p_value=stat_result['p_value'],
                confidence_interval=stat_result['confidence_interval'],
                is_significant=stat_result['is_significant'],
                test_statistic=stat_result['test_statistic'],
                test_type=stat_result['test_type'],
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values)
            )

        return results

    def _perform_statistical_test(self, control_values: List[float],
                                 treatment_values: List[float],
                                 significance_level: float) -> Dict[str, Any]:
        """Perform appropriate statistical test"""
        control_array = np.array(control_values)
        treatment_array = np.array(treatment_values)

        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)

        # Check normality (simple approach)
        control_normal = len(control_values) >= 30  # CLT assumption
        treatment_normal = len(treatment_values) >= 30

        if control_normal and treatment_normal:
            # Use t-test
            test_stat, p_value = ttest_ind(control_array, treatment_array, equal_var=False)
            test_type = "Welch's t-test"

            # Calculate confidence interval for difference in means
            pooled_se = np.sqrt(
                np.var(control_array, ddof=1) / len(control_array) +
                np.var(treatment_array, ddof=1) / len(treatment_array)
            )

            df = len(control_array) + len(treatment_array) - 2
            t_critical = stats.t.ppf(1 - significance_level / 2, df)

            mean_diff = treatment_mean - control_mean
            margin_error = t_critical * pooled_se
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error

        else:
            # Use Mann-Whitney U test (non-parametric)
            test_stat, p_value = mannwhitneyu(control_array, treatment_array, alternative='two-sided')
            test_type = "Mann-Whitney U test"

            # Bootstrap confidence interval
            n_bootstrap = 1000
            differences = []
            for _ in range(n_bootstrap):
                control_sample = np.random.choice(control_array, len(control_array), replace=True)
                treatment_sample = np.random.choice(treatment_array, len(treatment_array), replace=True)
                differences.append(np.mean(treatment_sample) - np.mean(control_sample))

            ci_lower = np.percentile(differences, significance_level / 2 * 100)
            ci_upper = np.percentile(differences, (1 - significance_level / 2) * 100)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_array) - 1) * np.var(control_array, ddof=1) +
             (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)) /
            (len(control_array) + len(treatment_array) - 2)
        )

        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        # Relative change
        relative_change = (treatment_mean - control_mean) / abs(control_mean) if control_mean != 0 else 0

        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'effect_size': effect_size,
            'relative_change': relative_change,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < significance_level,
            'test_statistic': test_stat,
            'test_type': test_type
        }

    def _generate_recommendation(self, statistical_results: Dict[str, StatisticalResult],
                               config: ExperimentConfig) -> Tuple[str, float]:
        """Generate experiment recommendation"""
        primary_metric_results = [
            r for r in statistical_results.values()
            if r.metric_name == config.primary_metric
        ]

        if not primary_metric_results:
            return "Insufficient data for recommendation", 0.0

        # Find best performing treatment
        best_result = max(primary_metric_results, key=lambda r: r.treatment_mean)

        if best_result.is_significant and best_result.relative_change > config.minimum_detectable_effect:
            confidence = min(1.0, (1 - best_result.p_value) * abs(best_result.effect_size))
            return f"Recommend {best_result.metric_name} treatment variant", confidence
        elif best_result.is_significant and best_result.relative_change < -config.minimum_detectable_effect:
            confidence = min(1.0, (1 - best_result.p_value) * abs(best_result.effect_size))
            return "Recommend control variant", confidence
        else:
            return "No significant difference detected", 0.5

    def _calculate_business_impact(self, statistical_results: Dict[str, StatisticalResult],
                                 variant_results: Dict[str, List]) -> Dict[str, float]:
        """Calculate estimated business impact"""
        # Simplified business impact calculation
        # In practice, this would be more sophisticated

        impact = {}

        for key, result in statistical_results.items():
            if result.is_significant:
                # Estimate impact based on effect size and sample size
                estimated_impact = result.effect_size * result.sample_size_treatment * 100  # Placeholder
                impact[key] = estimated_impact

        return impact

    def _check_early_stopping(self, config: ExperimentConfig,
                            statistical_results: Dict[str, StatisticalResult]) -> bool:
        """Check if early stopping criteria are met"""
        if not config.early_stopping_enabled:
            return False

        # Check if primary metric shows strong significance
        primary_results = [
            r for r in statistical_results.values()
            if r.metric_name == config.primary_metric
        ]

        for result in primary_results:
            if result.is_significant and result.p_value < 0.01:  # Strong significance
                return True

        return False

    def _get_experiment_dates(self, experiment_id: str) -> Tuple[datetime, Optional[datetime]]:
        """Get experiment start and end dates"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT start_date, end_date FROM experiments WHERE experiment_id = ?"
            cursor = conn.execute(query, (experiment_id,))
            row = cursor.fetchone()

            if row:
                start_date = datetime.fromisoformat(row[0]) if row[0] else datetime.now()
                end_date = datetime.fromisoformat(row[1]) if row[1] else None
                return start_date, end_date
            else:
                return datetime.now(), None

    def _get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """Get current experiment status"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT status FROM experiments WHERE experiment_id = ?"
            cursor = conn.execute(query, (experiment_id,))
            row = cursor.fetchone()

            if row:
                return ExperimentStatus(row[0])
            else:
                return ExperimentStatus.PLANNED

    def _save_statistical_results(self, experiment_id: str,
                                statistical_results: Dict[str, StatisticalResult]):
        """Save statistical results to database"""
        with sqlite3.connect(self.db_path) as conn:
            for key, result in statistical_results.items():
                # Parse control and treatment variants from key
                parts = key.split('_vs_')
                if len(parts) == 2:
                    control_variant = parts[0].split('_')[-1]
                    treatment_variant = parts[1]
                else:
                    control_variant = "unknown"
                    treatment_variant = "unknown"

                conn.execute('''
                    INSERT INTO statistical_results (
                        experiment_id, metric_name, control_variant, treatment_variant,
                        control_mean, treatment_mean, effect_size, relative_change,
                        p_value, confidence_interval_lower, confidence_interval_upper,
                        is_significant, test_statistic, test_type,
                        sample_size_control, sample_size_treatment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experiment_id,
                    result.metric_name,
                    control_variant,
                    treatment_variant,
                    result.control_mean,
                    result.treatment_mean,
                    result.effect_size,
                    result.relative_change,
                    result.p_value,
                    result.confidence_interval[0],
                    result.confidence_interval[1],
                    result.is_significant,
                    result.test_statistic,
                    result.test_type,
                    result.sample_size_control,
                    result.sample_size_treatment
                ))

    def start_monitoring(self):
        """Start background monitoring for experiments"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Started A/B test monitoring")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        self.logger.info("Stopped A/B test monitoring")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Check all active experiments
                for experiment_id in list(self.active_experiments.keys()):
                    config = self.active_experiments[experiment_id]

                    # Check if experiment should be stopped due to duration
                    start_date, _ = self._get_experiment_dates(experiment_id)
                    duration = datetime.now() - start_date

                    if duration.days >= config.max_duration_days:
                        self.stop_experiment(experiment_id, "Maximum duration reached")
                        continue

                    # Check for early stopping
                    summary = self.analyze_experiment(experiment_id, force_analysis=True)
                    if summary and summary.early_stopping_triggered:
                        self.stop_experiment(experiment_id, "Early stopping criteria met")

                time.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment summary for reporting"""
        summary = self.analyze_experiment(experiment_id, force_analysis=True)

        if summary:
            return {
                'experiment_id': summary.experiment_id,
                'status': summary.status.value,
                'duration_days': summary.duration_days,
                'total_participants': summary.total_participants,
                'variant_distribution': summary.variant_distribution,
                'recommendation': summary.recommendation,
                'confidence_score': summary.confidence_score,
                'significant_results': [
                    {
                        'metric': result.metric_name,
                        'p_value': result.p_value,
                        'effect_size': result.effect_size,
                        'relative_change': result.relative_change
                    }
                    for result in summary.statistical_results.values()
                    if result.is_significant
                ]
            }

        return None

if __name__ == "__main__":
    # Example usage
    ab_testing = ABTestingFramework()

    # Create experiment configuration
    config = ExperimentConfig(
        experiment_id=str(uuid.uuid4()),
        name="Gary DPI Model Comparison",
        description="Compare new Gary DPI calculation vs baseline",
        hypothesis="New Gary DPI calculation will improve trading performance",
        control_variant="baseline_gary_dpi",
        treatment_variants=["enhanced_gary_dpi"],
        traffic_split={"baseline_gary_dpi": 0.5, "enhanced_gary_dpi": 0.5},
        split_type=TrafficSplitType.RANDOM,
        min_sample_size=100,
        max_duration_days=30,
        primary_metric="gary_dpi"
    )

    # Create variants
    control_variant = VariantConfig(
        variant_id="baseline_gary_dpi",
        name="Baseline Gary DPI",
        description="Current Gary DPI calculation",
        model_config={"version": "1.0"},
        strategy_config={},
        feature_config={},
        risk_config={}
    )

    treatment_variant = VariantConfig(
        variant_id="enhanced_gary_dpi",
        name="Enhanced Gary DPI",
        description="Enhanced Gary DPI with volatility adjustment",
        model_config={"version": "2.0"},
        strategy_config={},
        feature_config={},
        risk_config={}
    )

    # Create and start experiment
    experiment_id = ab_testing.create_experiment(config, [control_variant, treatment_variant])
    ab_testing.start_experiment(experiment_id)

    print(f"Created and started experiment: {experiment_id}")
    print("Use ab_testing.assign_variant() to assign users to variants")
    print("Use ab_testing.record_result() to record trading results")
    print("Use ab_testing.analyze_experiment() to get statistical analysis")