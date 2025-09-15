"""
Enhanced Kelly Criterion with Survival-First Constraints for Super-Gary Trading Framework

This module implements survival-first position sizing using enhanced Kelly Criterion
with risk-of-ruin constraints, correlation clustering awareness, and multi-asset
optimization for the Gary×Taleb trading framework.

Mathematical Foundation:
- Fractional Kelly: f* = min(kelly_raw * fraction, CVaR_limit / expected_loss)
- Survival Constraint: P(ruin) < ε, where ε = maximum acceptable ruin probability
- Correlation Clustering: Cross-asset netting by macro drivers
- CVaR Guardrails: Conditional Value at Risk limits for tail protection

Key Enhancements:
- Risk-of-ruin bounded Kelly sizing
- Multi-asset Kelly frontiers with correlation awareness
- Duration, inflation, equity beta decomposition
- Hygiene gates for vol/liquidity/crowding detection
- Integration with calibration and convexity systems
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, t
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SurvivalMode(Enum):
    """Survival-first sizing modes"""
    CONSERVATIVE = "conservative"    # P(ruin) < 0.01
    MODERATE = "moderate"           # P(ruin) < 0.05
    AGGRESSIVE = "aggressive"       # P(ruin) < 0.10
    CRITICAL = "critical"           # Emergency mode, P(ruin) < 0.001

@dataclass
class AssetRiskProfile:
    """Risk profile for individual asset"""
    asset: str
    expected_return: float
    volatility: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    skewness: float
    kurtosis: float
    liquidity_score: float  # [0,1] higher is more liquid
    beta_equity: float
    beta_duration: float
    beta_inflation: float
    correlation_cluster: int
    crowding_score: float  # [0,1] higher is more crowded

@dataclass
class KellySurvivalConstraints:
    """Survival constraints for Kelly optimization"""
    max_ruin_probability: float
    max_single_asset_weight: float
    max_cluster_weight: float
    min_liquidity_score: float
    max_crowding_score: float
    max_leverage: float
    max_daily_var: float
    max_portfolio_cvar: float
    correlation_threshold: float

@dataclass
class MultiAssetKellyResult:
    """Result from multi-asset Kelly optimization"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    survival_probability: float
    portfolio_var: float
    portfolio_cvar: float
    cluster_exposures: Dict[int, float]
    constraint_violations: List[str]
    optimization_status: str

class EnhancedKellyCriterion:
    """
    Enhanced Kelly Criterion with survival-first constraints

    Implements sophisticated position sizing that prioritizes survival while
    optimizing for risk-adjusted returns across multiple assets with
    correlation awareness and macro factor decomposition.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Risk profiles and constraints
        self.asset_profiles: Dict[str, AssetRiskProfile] = {}
        self.survival_constraints = self._initialize_survival_constraints()
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.factor_loadings: Optional[pd.DataFrame] = None

        # Performance tracking
        self.sizing_history: List[Dict] = []
        self.survival_metrics: List[Dict] = []

        # Integration with other systems
        self.calibration_system = None  # Will be injected
        self.convexity_system = None   # Will be injected

        # Data persistence
        self.data_path = Path(self.config.get('data_path', './data/kelly_enhanced'))
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._initialize_system()

    def _default_config(self) -> Dict:
        """Default configuration for enhanced Kelly system"""
        return {
            'base_kelly_fraction': 0.25,  # Conservative Kelly multiplier
            'survival_mode': SurvivalMode.MODERATE,
            'max_ruin_probability': 0.05,  # 5% maximum ruin probability
            'correlation_lookback': 252,   # Days for correlation calculation
            'volatility_lookback': 63,     # Days for volatility calculation
            'factor_update_frequency': 21,  # Days between factor updates
            'liquidity_threshold': 0.3,    # Minimum liquidity score
            'crowding_threshold': 0.7,     # Maximum crowding score
            'max_single_asset': 0.15,      # 15% maximum single asset weight
            'max_cluster_weight': 0.35,    # 35% maximum cluster weight
            'emergency_cash_ratio': 0.1,   # 10% emergency cash buffer
            'rebalance_threshold': 0.05,   # 5% drift before rebalancing
            'cvar_confidence': 0.95,       # 95% confidence for CVaR
            'kelly_cap': 1.0,              # Maximum Kelly value before rejection
            'min_expected_return': 0.02,   # 2% minimum expected return
            'max_leverage': 2.0,           # 2x maximum leverage
            'optimization_method': 'SLSQP' # Scipy optimization method
        }

    def _initialize_system(self):
        """Initialize enhanced Kelly system"""
        try:
            # Load existing data if available
            self._load_system_data()
            self.logger.info("Enhanced Kelly system initialized")
        except Exception as e:
            self.logger.warning(f"Could not load historical data: {e}")
            self.logger.info("Starting fresh Kelly optimization system")

    def _initialize_survival_constraints(self) -> KellySurvivalConstraints:
        """Initialize survival constraints based on config"""
        mode = self.config.get('survival_mode', SurvivalMode.MODERATE)

        # Adjust constraints based on survival mode
        ruin_prob_map = {
            SurvivalMode.CONSERVATIVE: 0.01,
            SurvivalMode.MODERATE: 0.05,
            SurvivalMode.AGGRESSIVE: 0.10,
            SurvivalMode.CRITICAL: 0.001
        }

        return KellySurvivalConstraints(
            max_ruin_probability=ruin_prob_map[mode],
            max_single_asset_weight=self.config['max_single_asset'],
            max_cluster_weight=self.config['max_cluster_weight'],
            min_liquidity_score=self.config['liquidity_threshold'],
            max_crowding_score=self.config['crowding_threshold'],
            max_leverage=self.config['max_leverage'],
            max_daily_var=0.03,  # 3% daily VaR limit
            max_portfolio_cvar=0.05,  # 5% portfolio CVaR limit
            correlation_threshold=0.7  # High correlation threshold
        )

    def add_asset_profile(self,
                         asset: str,
                         returns_data: pd.Series,
                         market_data: Optional[Dict] = None) -> AssetRiskProfile:
        """
        Add or update asset risk profile

        Args:
            asset: Asset identifier
            returns_data: Historical returns data
            market_data: Additional market data (beta, liquidity, etc.)

        Returns:
            Computed asset risk profile
        """
        try:
            # Calculate basic risk metrics
            expected_return = returns_data.mean() * 252  # Annualized
            volatility = returns_data.std() * np.sqrt(252)  # Annualized
            skewness = returns_data.skew()
            kurtosis = returns_data.kurtosis()

            # Calculate VaR and CVaR
            var_95 = returns_data.quantile(0.05)  # 5th percentile
            cvar_95 = returns_data[returns_data <= var_95].mean()  # Expected shortfall

            # Calculate maximum drawdown
            cumulative_returns = (1 + returns_data).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()

            # Extract or estimate market data
            market_data = market_data or {}
            liquidity_score = market_data.get('liquidity_score', 0.5)
            beta_equity = market_data.get('beta_equity', 1.0)
            beta_duration = market_data.get('beta_duration', 0.0)
            beta_inflation = market_data.get('beta_inflation', 0.0)
            crowding_score = market_data.get('crowding_score', 0.5)

            # Assign to correlation cluster (will be updated when clustering is run)
            correlation_cluster = market_data.get('correlation_cluster', 0)

            profile = AssetRiskProfile(
                asset=asset,
                expected_return=expected_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                skewness=skewness,
                kurtosis=kurtosis,
                liquidity_score=liquidity_score,
                beta_equity=beta_equity,
                beta_duration=beta_duration,
                beta_inflation=beta_inflation,
                correlation_cluster=correlation_cluster,
                crowding_score=crowding_score
            )

            self.asset_profiles[asset] = profile
            self.logger.info(f"Added risk profile for {asset}")
            return profile

        except Exception as e:
            self.logger.error(f"Error creating asset profile for {asset}: {e}")
            raise

    def update_correlation_matrix(self, returns_data: pd.DataFrame):
        """
        Update correlation matrix and perform clustering

        Args:
            returns_data: DataFrame with returns for all assets
        """
        try:
            # Calculate correlation matrix
            self.correlation_matrix = returns_data.corr()

            # Perform correlation clustering
            self._update_correlation_clusters(returns_data)

            self.logger.info("Updated correlation matrix and clusters")

        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")

    def calculate_survival_kelly(self,
                               asset: str,
                               base_kelly: float,
                               confidence_adjustment: float = 1.0) -> float:
        """
        Calculate survival-first Kelly fraction for single asset

        Args:
            asset: Asset identifier
            base_kelly: Base Kelly calculation
            confidence_adjustment: Calibration-based adjustment

        Returns:
            Survival-adjusted Kelly fraction
        """
        if asset not in self.asset_profiles:
            self.logger.warning(f"No risk profile for {asset}, using conservative sizing")
            return self.config['base_kelly_fraction'] * 0.5

        profile = self.asset_profiles[asset]

        try:
            # Apply base Kelly fraction
            kelly_raw = base_kelly * self.config['base_kelly_fraction']

            # Adjust for calibration quality
            kelly_calibrated = kelly_raw * confidence_adjustment

            # Apply survival constraints
            kelly_survival = self._apply_survival_constraints(kelly_calibrated, profile)

            # Apply hygiene gates
            kelly_final = self._apply_hygiene_gates(kelly_survival, profile)

            self.logger.debug(f"Kelly calculation for {asset}: "
                            f"raw={kelly_raw:.3f}, calibrated={kelly_calibrated:.3f}, "
                            f"survival={kelly_survival:.3f}, final={kelly_final:.3f}")

            return kelly_final

        except Exception as e:
            self.logger.error(f"Error calculating survival Kelly for {asset}: {e}")
            return self.config['base_kelly_fraction'] * 0.1  # Very conservative fallback

    def optimize_multi_asset_portfolio(self,
                                     expected_returns: Dict[str, float],
                                     confidence_scores: Optional[Dict[str, float]] = None) -> MultiAssetKellyResult:
        """
        Optimize multi-asset portfolio using enhanced Kelly with constraints

        Args:
            expected_returns: Expected returns by asset
            confidence_scores: Calibration confidence scores by asset

        Returns:
            Optimal portfolio allocation
        """
        try:
            # Prepare data
            assets = list(expected_returns.keys())
            if not all(asset in self.asset_profiles for asset in assets):
                raise ValueError("Missing risk profiles for some assets")

            # Get confidence scores
            confidence_scores = confidence_scores or {asset: 1.0 for asset in assets}

            # Optimize portfolio
            result = self._solve_multi_asset_kelly(assets, expected_returns, confidence_scores)

            # Record result
            self.sizing_history.append({
                'timestamp': datetime.now(),
                'assets': assets,
                'weights': result.optimal_weights,
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'survival_probability': result.survival_probability
            })

            return result

        except Exception as e:
            self.logger.error(f"Error optimizing multi-asset portfolio: {e}")
            # Return conservative equal-weight fallback
            return self._create_fallback_allocation(assets)

    def calculate_kelly_frontiers(self,
                                assets: List[str],
                                return_scenarios: np.ndarray,
                                confidence_levels: List[float]) -> Dict[str, Any]:
        """
        Calculate Kelly efficiency frontiers for different confidence levels

        Args:
            assets: List of assets to optimize
            return_scenarios: Array of return scenarios for Monte Carlo
            confidence_levels: List of confidence levels to analyze

        Returns:
            Kelly frontier analysis results
        """
        frontiers = {}

        for confidence in confidence_levels:
            try:
                # Adjust expected returns by confidence
                adjusted_returns = {}
                for asset in assets:
                    base_return = np.mean(return_scenarios, axis=0)[assets.index(asset)]
                    adjusted_returns[asset] = base_return * confidence

                # Optimize for this confidence level
                result = self.optimize_multi_asset_portfolio(
                    adjusted_returns,
                    {asset: confidence for asset in assets}
                )

                frontiers[f"confidence_{confidence:.1f}"] = {
                    'weights': result.optimal_weights,
                    'expected_return': result.expected_return,
                    'volatility': result.expected_volatility,
                    'sharpe': result.sharpe_ratio,
                    'survival_prob': result.survival_probability
                }

            except Exception as e:
                self.logger.error(f"Error calculating frontier for confidence {confidence}: {e}")

        return frontiers

    def get_factor_decomposition(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Decompose portfolio exposure by macro factors

        Args:
            weights: Portfolio weights by asset

        Returns:
            Factor exposures (equity, duration, inflation)
        """
        total_equity_beta = 0
        total_duration_beta = 0
        total_inflation_beta = 0

        for asset, weight in weights.items():
            if asset in self.asset_profiles:
                profile = self.asset_profiles[asset]
                total_equity_beta += weight * profile.beta_equity
                total_duration_beta += weight * profile.beta_duration
                total_inflation_beta += weight * profile.beta_inflation

        return {
            'equity_beta': total_equity_beta,
            'duration_beta': total_duration_beta,
            'inflation_beta': total_inflation_beta,
            'total_factor_exposure': abs(total_equity_beta) + abs(total_duration_beta) + abs(total_inflation_beta)
        }

    def assess_crowding_risk(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess portfolio crowding risk

        Args:
            weights: Portfolio weights by asset

        Returns:
            Crowding risk assessment
        """
        total_crowding_exposure = 0
        high_crowding_assets = []
        cluster_crowding = {}

        for asset, weight in weights.items():
            if asset in self.asset_profiles:
                profile = self.asset_profiles[asset]
                asset_crowding = weight * profile.crowding_score
                total_crowding_exposure += asset_crowding

                if profile.crowding_score > self.config['crowding_threshold']:
                    high_crowding_assets.append({
                        'asset': asset,
                        'weight': weight,
                        'crowding_score': profile.crowding_score,
                        'crowding_exposure': asset_crowding
                    })

                # Cluster crowding
                cluster = profile.correlation_cluster
                if cluster not in cluster_crowding:
                    cluster_crowding[cluster] = 0
                cluster_crowding[cluster] += asset_crowding

        return {
            'total_crowding_exposure': total_crowding_exposure,
            'high_crowding_assets': high_crowding_assets,
            'cluster_crowding': cluster_crowding,
            'crowding_risk_level': 'high' if total_crowding_exposure > 0.5 else 'medium' if total_crowding_exposure > 0.3 else 'low'
        }

    def calculate_portfolio_cvar(self,
                               weights: Dict[str, float],
                               confidence: float = 0.95) -> float:
        """
        Calculate portfolio Conditional Value at Risk

        Args:
            weights: Portfolio weights
            confidence: Confidence level for CVaR

        Returns:
            Portfolio CVaR
        """
        if self.correlation_matrix is None:
            self.logger.warning("No correlation matrix available, using conservative CVaR estimate")
            # Use sum of individual CVaRs (conservative)
            total_cvar = 0
            for asset, weight in weights.items():
                if asset in self.asset_profiles:
                    total_cvar += abs(weight) * abs(self.asset_profiles[asset].cvar_95)
            return total_cvar

        try:
            # Get asset CVaRs
            assets = list(weights.keys())
            asset_cvars = np.array([self.asset_profiles[asset].cvar_95 for asset in assets])
            weights_array = np.array([weights[asset] for asset in assets])

            # Get correlation submatrix
            corr_sub = self.correlation_matrix.loc[assets, assets].values

            # Calculate portfolio CVaR using correlation
            portfolio_var = np.sqrt(np.dot(weights_array, np.dot(corr_sub, weights_array)))

            # Approximate CVaR using normal distribution (conservative approximation)
            z_score = norm.ppf(1 - confidence)
            portfolio_cvar = portfolio_var * z_score

            return abs(portfolio_cvar)

        except Exception as e:
            self.logger.error(f"Error calculating portfolio CVaR: {e}")
            # Fallback to sum of individual CVaRs
            return sum(abs(weights.get(asset, 0)) * abs(profile.cvar_95)
                      for asset, profile in self.asset_profiles.items())

    def _apply_survival_constraints(self, kelly_fraction: float, profile: AssetRiskProfile) -> float:
        """Apply survival constraints to Kelly fraction"""
        # Risk-of-ruin constraint using approximate formula
        # P(ruin) ≈ exp(-2 * edge * capital / variance)
        # Rearranging: max_fraction = sqrt(-ln(max_ruin_prob) * variance / (2 * edge * capital))

        if profile.expected_return <= 0:
            return 0  # No positive expected return

        try:
            # Calculate implied risk-of-ruin for current Kelly fraction
            edge = profile.expected_return
            variance = profile.volatility ** 2

            # Approximate risk of ruin using Kelly formula
            if kelly_fraction > 0:
                implied_ruin_risk = np.exp(-2 * edge * kelly_fraction / variance)
            else:
                implied_ruin_risk = 0

            # If risk too high, reduce Kelly fraction
            if implied_ruin_risk > self.survival_constraints.max_ruin_probability:
                # Calculate maximum Kelly that satisfies survival constraint
                max_safe_kelly = -np.log(self.survival_constraints.max_ruin_probability) * variance / (2 * edge)
                kelly_fraction = min(kelly_fraction, max_safe_kelly)

            # Additional CVaR constraint
            estimated_cvar_impact = kelly_fraction * abs(profile.cvar_95)
            if estimated_cvar_impact > self.survival_constraints.max_portfolio_cvar:
                kelly_fraction = min(kelly_fraction,
                                   self.survival_constraints.max_portfolio_cvar / abs(profile.cvar_95))

            return max(0, kelly_fraction)

        except Exception as e:
            self.logger.error(f"Error applying survival constraints: {e}")
            return kelly_fraction * 0.5  # Conservative fallback

    def _apply_hygiene_gates(self, kelly_fraction: float, profile: AssetRiskProfile) -> float:
        """Apply hygiene gates for vol/liquidity/crowding"""
        # Liquidity gate
        if profile.liquidity_score < self.survival_constraints.min_liquidity_score:
            liquidity_penalty = profile.liquidity_score / self.survival_constraints.min_liquidity_score
            kelly_fraction *= liquidity_penalty

        # Crowding gate
        if profile.crowding_score > self.survival_constraints.max_crowding_score:
            crowding_penalty = (1 - profile.crowding_score) / (1 - self.survival_constraints.max_crowding_score)
            kelly_fraction *= max(0.1, crowding_penalty)  # Minimum 10% of original

        # Volatility gate (reduce sizing for extreme volatility)
        if profile.volatility > 0.5:  # 50% annualized volatility threshold
            vol_penalty = 0.5 / profile.volatility
            kelly_fraction *= vol_penalty

        # Skewness gate (reduce sizing for negative skewness)
        if profile.skewness < -1:
            skew_penalty = 1 + (profile.skewness / 10)  # Reduce for negative skew
            kelly_fraction *= max(0.5, skew_penalty)

        return max(0, kelly_fraction)

    def _solve_multi_asset_kelly(self,
                               assets: List[str],
                               expected_returns: Dict[str, float],
                               confidence_scores: Dict[str, float]) -> MultiAssetKellyResult:
        """Solve multi-asset Kelly optimization problem"""
        n_assets = len(assets)

        # Objective function: negative expected log return (we minimize)
        def objective(weights):
            portfolio_return = sum(weights[i] * expected_returns[assets[i]] * confidence_scores[assets[i]]
                                 for i in range(n_assets))

            # Portfolio variance calculation
            portfolio_var = 0
            if self.correlation_matrix is not None:
                for i in range(n_assets):
                    for j in range(n_assets):
                        asset_i, asset_j = assets[i], assets[j]
                        vol_i = self.asset_profiles[asset_i].volatility
                        vol_j = self.asset_profiles[asset_j].volatility
                        corr = self.correlation_matrix.loc[asset_i, asset_j] if asset_i in self.correlation_matrix.index and asset_j in self.correlation_matrix.columns else 0
                        portfolio_var += weights[i] * weights[j] * vol_i * vol_j * corr
            else:
                # Fallback: assume zero correlation
                portfolio_var = sum(weights[i]**2 * self.asset_profiles[assets[i]].volatility**2
                                  for i in range(n_assets))

            # Kelly objective: maximize log(1 + f*R) ≈ f*E[R] - f²*Var[R]/2
            if portfolio_var <= 0:
                return -portfolio_return  # Just maximize return if no variance

            kelly_objective = portfolio_return - 0.5 * portfolio_var
            return -kelly_objective  # Minimize negative

        # Constraints
        constraints = []

        # Budget constraint: sum of weights = 1 (allow cash)
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Individual asset limits
        for i in range(n_assets):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=i: self.survival_constraints.max_single_asset_weight - abs(w[idx])
            })

        # Cluster limits (if clustering available)
        cluster_constraints = self._create_cluster_constraints(assets, n_assets)
        constraints.extend(cluster_constraints)

        # Portfolio CVaR constraint
        def cvar_constraint(weights):
            weights_dict = {assets[i]: weights[i] for i in range(n_assets)}
            portfolio_cvar = self.calculate_portfolio_cvar(weights_dict)
            return self.survival_constraints.max_portfolio_cvar - portfolio_cvar

        constraints.append({'type': 'ineq', 'fun': cvar_constraint})

        # Bounds: allow short positions but limit leverage
        bounds = [(-self.survival_constraints.max_leverage/n_assets,
                  self.survival_constraints.max_leverage/n_assets) for _ in range(n_assets)]

        # Initial guess: equal weight
        x0 = np.array([1.0/n_assets] * n_assets)

        try:
            # Solve optimization
            result = minimize(
                objective, x0, method=self.config['optimization_method'],
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                optimal_weights = {assets[i]: result.x[i] for i in range(n_assets)}
                return self._create_kelly_result(optimal_weights, expected_returns, confidence_scores, "success")
            else:
                self.logger.warning(f"Optimization failed: {result.message}")
                return self._create_fallback_allocation(assets, expected_returns)

        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self._create_fallback_allocation(assets, expected_returns)

    def _create_cluster_constraints(self, assets: List[str], n_assets: int) -> List[Dict]:
        """Create cluster weight constraints"""
        constraints = []

        # Group assets by cluster
        clusters = {}
        for i, asset in enumerate(assets):
            if asset in self.asset_profiles:
                cluster = self.asset_profiles[asset].correlation_cluster
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(i)

        # Add cluster weight constraints
        for cluster, asset_indices in clusters.items():
            def cluster_constraint(weights, indices=asset_indices):
                cluster_weight = sum(abs(weights[i]) for i in indices)
                return self.survival_constraints.max_cluster_weight - cluster_weight

            constraints.append({'type': 'ineq', 'fun': cluster_constraint})

        return constraints

    def _create_kelly_result(self,
                           weights: Dict[str, float],
                           expected_returns: Dict[str, float],
                           confidence_scores: Dict[str, float],
                           status: str) -> MultiAssetKellyResult:
        """Create Kelly optimization result"""
        # Calculate portfolio metrics
        portfolio_return = sum(weights[asset] * expected_returns[asset] * confidence_scores[asset]
                             for asset in weights.keys())

        # Portfolio volatility
        portfolio_variance = 0
        if self.correlation_matrix is not None:
            for asset1 in weights.keys():
                for asset2 in weights.keys():
                    if asset1 in self.asset_profiles and asset2 in self.asset_profiles:
                        vol1 = self.asset_profiles[asset1].volatility
                        vol2 = self.asset_profiles[asset2].volatility
                        corr = self.correlation_matrix.loc[asset1, asset2] if asset1 in self.correlation_matrix.index and asset2 in self.correlation_matrix.columns else 0
                        portfolio_variance += weights[asset1] * weights[asset2] * vol1 * vol2 * corr

        portfolio_volatility = np.sqrt(max(0, portfolio_variance))

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        # Survival probability (approximate)
        portfolio_cvar = self.calculate_portfolio_cvar(weights)
        survival_probability = 1 - self.survival_constraints.max_ruin_probability

        # Cluster exposures
        cluster_exposures = {}
        for asset, weight in weights.items():
            if asset in self.asset_profiles:
                cluster = self.asset_profiles[asset].correlation_cluster
                cluster_exposures[cluster] = cluster_exposures.get(cluster, 0) + abs(weight)

        # Check constraint violations
        violations = []
        for asset, weight in weights.items():
            if abs(weight) > self.survival_constraints.max_single_asset_weight:
                violations.append(f"Asset {asset} exceeds individual weight limit")

        if portfolio_cvar > self.survival_constraints.max_portfolio_cvar:
            violations.append("Portfolio CVaR exceeds limit")

        return MultiAssetKellyResult(
            optimal_weights=weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_estimate=portfolio_cvar * 2,  # Rough estimate
            survival_probability=survival_probability,
            portfolio_var=portfolio_variance,
            portfolio_cvar=portfolio_cvar,
            cluster_exposures=cluster_exposures,
            constraint_violations=violations,
            optimization_status=status
        )

    def _create_fallback_allocation(self, assets: List[str], expected_returns: Optional[Dict[str, float]] = None) -> MultiAssetKellyResult:
        """Create conservative fallback allocation"""
        # Equal weight with cash buffer
        cash_weight = self.config['emergency_cash_ratio']
        asset_weight = (1 - cash_weight) / len(assets)

        weights = {asset: asset_weight for asset in assets}
        weights['CASH'] = cash_weight

        if expected_returns is None:
            expected_returns = {asset: 0.05 for asset in assets}  # Assume 5% return

        confidence_scores = {asset: 0.5 for asset in assets}  # Conservative confidence

        return self._create_kelly_result(weights, expected_returns, confidence_scores, "fallback")

    def _update_correlation_clusters(self, returns_data: pd.DataFrame):
        """Update correlation-based clustering of assets"""
        try:
            if len(returns_data.columns) < 2:
                return

            # Use correlation distance for clustering
            corr_matrix = returns_data.corr()
            distance_matrix = 1 - corr_matrix.abs()

            # Perform clustering
            n_clusters = min(4, len(returns_data.columns))  # Maximum 4 clusters

            # Use PCA for dimensionality reduction if needed
            if len(returns_data.columns) > 10:
                pca = PCA(n_components=min(5, len(returns_data.columns)))
                features = pca.fit_transform(returns_data.T)
            else:
                features = distance_matrix.values

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)

            # Update asset profiles with cluster assignments
            for i, asset in enumerate(returns_data.columns):
                if asset in self.asset_profiles:
                    self.asset_profiles[asset].correlation_cluster = cluster_labels[i]

            self.logger.info(f"Updated correlation clusters: {n_clusters} clusters")

        except Exception as e:
            self.logger.error(f"Error updating correlation clusters: {e}")

    def _save_system_data(self):
        """Save system data to disk"""
        try:
            # Save asset profiles
            profiles_data = {}
            for asset, profile in self.asset_profiles.items():
                profiles_data[asset] = {
                    'asset': profile.asset,
                    'expected_return': profile.expected_return,
                    'volatility': profile.volatility,
                    'max_drawdown': profile.max_drawdown,
                    'var_95': profile.var_95,
                    'cvar_95': profile.cvar_95,
                    'skewness': profile.skewness,
                    'kurtosis': profile.kurtosis,
                    'liquidity_score': profile.liquidity_score,
                    'beta_equity': profile.beta_equity,
                    'beta_duration': profile.beta_duration,
                    'beta_inflation': profile.beta_inflation,
                    'correlation_cluster': profile.correlation_cluster,
                    'crowding_score': profile.crowding_score
                }

            with open(self.data_path / 'asset_profiles.json', 'w') as f:
                json.dump(profiles_data, f, indent=2)

            # Save correlation matrix
            if self.correlation_matrix is not None:
                self.correlation_matrix.to_csv(self.data_path / 'correlation_matrix.csv')

            self.logger.info("System data saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving system data: {e}")

    def _load_system_data(self):
        """Load system data from disk"""
        # Load asset profiles
        profiles_file = self.data_path / 'asset_profiles.json'
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                profiles_data = json.load(f)

            for asset, data in profiles_data.items():
                profile = AssetRiskProfile(**data)
                self.asset_profiles[asset] = profile

        # Load correlation matrix
        corr_file = self.data_path / 'correlation_matrix.csv'
        if corr_file.exists():
            self.correlation_matrix = pd.read_csv(corr_file, index_col=0)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create enhanced Kelly system
    kelly_system = EnhancedKellyCriterion()

    # Simulate asset data
    np.random.seed(42)
    assets = ['SPY', 'TLT', 'GLD', 'VIX']

    # Generate sample returns data
    returns_data = {}
    for asset in assets:
        returns = np.random.normal(0.0003, 0.015, 252)  # Daily returns
        returns_data[asset] = pd.Series(returns)

        # Add asset profile
        market_data = {
            'liquidity_score': np.random.uniform(0.5, 1.0),
            'beta_equity': np.random.uniform(0.5, 1.5),
            'beta_duration': np.random.uniform(-0.5, 0.5),
            'beta_inflation': np.random.uniform(-0.3, 0.3),
            'crowding_score': np.random.uniform(0.2, 0.8)
        }
        kelly_system.add_asset_profile(asset, returns_data[asset], market_data)

    # Update correlation matrix
    returns_df = pd.DataFrame(returns_data)
    kelly_system.update_correlation_matrix(returns_df)

    # Test single asset sizing
    single_kelly = kelly_system.calculate_survival_kelly('SPY', 0.15, 0.8)
    print(f"Survival Kelly for SPY: {single_kelly:.3f}")

    # Test multi-asset optimization
    expected_returns = {asset: np.random.uniform(0.05, 0.12) for asset in assets}
    confidence_scores = {asset: np.random.uniform(0.6, 0.9) for asset in assets}

    result = kelly_system.optimize_multi_asset_portfolio(expected_returns, confidence_scores)
    print(f"Optimization Status: {result.optimization_status}")
    print(f"Optimal Weights: {result.optimal_weights}")
    print(f"Expected Return: {result.expected_return:.3f}")
    print(f"Expected Volatility: {result.expected_volatility:.3f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Survival Probability: {result.survival_probability:.3f}")

    # Test factor decomposition
    factor_exposure = kelly_system.get_factor_decomposition(result.optimal_weights)
    print(f"Factor Exposures: {factor_exposure}")

    # Test crowding risk
    crowding_risk = kelly_system.assess_crowding_risk(result.optimal_weights)
    print(f"Crowding Risk Level: {crowding_risk['crowding_risk_level']}")
    print(f"Total Crowding Exposure: {crowding_risk['total_crowding_exposure']:.3f}")

    # Test Kelly frontiers
    return_scenarios = np.random.normal(0.08, 0.15, (1000, len(assets)))
    frontiers = kelly_system.calculate_kelly_frontiers(assets, return_scenarios, [0.5, 0.7, 0.9])

    print("\nKelly Frontiers:")
    for confidence, frontier in frontiers.items():
        print(f"{confidence}: Return={frontier['expected_return']:.3f}, "
              f"Vol={frontier['volatility']:.3f}, Sharpe={frontier['sharpe']:.3f}")