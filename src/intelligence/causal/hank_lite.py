"""
HANK-lite: Heterogeneous Agent New Keynesian Model

Implements Gary's vision of a lightweight HANK model with heterogeneous agents
and sticky finance frictions for trading and policy analysis. This model captures
distributional effects of monetary and fiscal policy with Bayesian daily calibration.

Core Philosophy: "Multiple income cohorts with different MPCs, sticky finance frictions"

Mathematical Foundation:
- Heterogeneous agents across income/wealth distribution
- Different marginal propensities to consume (MPCs) by cohort
- Sticky financial frictions (credit constraints, housing market)
- New Keynesian price/wage rigidities
- Bayesian updating of parameters from daily market data
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
from scipy.stats import multivariate_normal, gamma, beta
import json

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of heterogeneous agents"""
    HAND_TO_MOUTH = "hand_to_mouth"  # High MPC, liquidity constrained
    MIDDLE_CLASS = "middle_class"    # Moderate MPC, some assets
    WEALTHY = "wealthy"              # Low MPC, significant assets
    ENTREPRENEURS = "entrepreneurs"  # Variable MPC, business owners


class FinancialFriction(Enum):
    """Types of financial frictions"""
    CREDIT_CONSTRAINT = "credit_constraint"
    HOUSING_STICKINESS = "housing_stickiness"
    PORTFOLIO_ADJUSTMENT = "portfolio_adjustment"
    LIQUIDITY_CONSTRAINT = "liquidity_constraint"


@dataclass
class AgentCohort:
    """Cohort of heterogeneous agents"""
    agent_type: AgentType
    population_share: float
    income_mean: float
    income_std: float
    wealth_mean: float
    wealth_std: float
    mpc: float  # Marginal propensity to consume
    mpc_std: float  # MPC uncertainty
    discount_factor: float
    risk_aversion: float
    housing_share: float  # Share of wealth in housing
    financial_assets: float  # Share in financial assets
    credit_limit_ratio: float  # Credit limit as ratio of income
    liquidity_buffer: float  # Desired liquid assets ratio


@dataclass
class PolicyShock:
    """Monetary or fiscal policy shock"""
    shock_type: str  # 'monetary', 'fiscal', 'transfer'
    magnitude: float
    persistence: float  # AR(1) persistence parameter
    announcement_effect: float  # Immediate announcement impact
    implementation_lag: int  # Periods until full implementation
    targeting: Dict[AgentType, float]  # Differential impact by agent type


@dataclass
class MarketState:
    """Current state of the economy"""
    aggregate_consumption: float
    aggregate_investment: float
    aggregate_output: float
    inflation: float
    interest_rate: float
    unemployment: float
    housing_prices: float
    wage_level: float
    credit_spread: float
    agent_states: Dict[AgentType, Dict[str, float]]


@dataclass
class HANKParameters:
    """HANK model parameters with Bayesian posteriors"""
    # Production parameters
    technology_level: float = 1.0
    capital_share: float = 0.33
    depreciation_rate: float = 0.025

    # Preferences
    discount_factor_mean: float = 0.99
    risk_aversion_mean: float = 2.0

    # Nominal rigidities
    price_stickiness: float = 0.75  # Calvo parameter
    wage_stickiness: float = 0.75

    # Financial frictions
    credit_elasticity: float = 0.5
    housing_adjustment_cost: float = 0.1
    portfolio_adjustment_cost: float = 0.05

    # Monetary policy
    taylor_rule_inflation: float = 1.5
    taylor_rule_output: float = 0.5
    interest_rate_smoothing: float = 0.8

    # Bayesian uncertainty (standard deviations)
    parameter_uncertainty: Dict[str, float] = field(default_factory=lambda: {
        'price_stickiness': 0.1,
        'wage_stickiness': 0.1,
        'credit_elasticity': 0.2,
        'mpc_heterogeneity': 0.15
    })


class HANKLiteModel:
    """
    HANK-lite model for heterogeneous agent analysis

    This model implements a simplified version of Heterogeneous Agent New Keynesian
    models focused on distributional effects of policy and market shocks.
    """

    def __init__(self, parameters: Optional[HANKParameters] = None):
        """
        Initialize HANK-lite model

        Args:
            parameters: Model parameters (uses defaults if None)
        """
        self.parameters = parameters or HANKParameters()
        self.agent_cohorts: Dict[AgentType, AgentCohort] = {}
        self.current_state: Optional[MarketState] = None
        self.simulation_history: List[MarketState] = []
        self.calibration_data: Dict[str, pd.Series] = {}

        # Initialize agent cohorts with realistic distributions
        self._initialize_agent_cohorts()

        # Bayesian posterior tracking
        self.posterior_means: Dict[str, float] = {}
        self.posterior_vars: Dict[str, float] = {}

        logger.info("HANK-lite model initialized")

    def _initialize_agent_cohorts(self):
        """Initialize heterogeneous agent cohorts with US-calibrated parameters"""

        # Hand-to-mouth agents (bottom 40% of wealth distribution)
        self.agent_cohorts[AgentType.HAND_TO_MOUTH] = AgentCohort(
            agent_type=AgentType.HAND_TO_MOUTH,
            population_share=0.40,
            income_mean=35000,
            income_std=8000,
            wealth_mean=5000,  # Very low wealth
            wealth_std=3000,
            mpc=0.95,  # Very high MPC
            mpc_std=0.10,
            discount_factor=0.96,  # More impatient
            risk_aversion=3.0,  # High risk aversion
            housing_share=0.20,  # Low homeownership
            financial_assets=0.10,  # Very few financial assets
            credit_limit_ratio=0.5,  # Limited credit access
            liquidity_buffer=0.05  # Minimal liquidity buffer
        )

        # Middle class agents (next 50% of wealth distribution)
        self.agent_cohorts[AgentType.MIDDLE_CLASS] = AgentCohort(
            agent_type=AgentType.MIDDLE_CLASS,
            population_share=0.50,
            income_mean=70000,
            income_std=20000,
            wealth_mean=150000,  # Moderate wealth, mostly housing
            wealth_std=75000,
            mpc=0.60,  # Moderate MPC
            mpc_std=0.15,
            discount_factor=0.98,
            risk_aversion=2.0,
            housing_share=0.70,  # High homeownership
            financial_assets=0.25,  # Some financial assets
            credit_limit_ratio=1.5,  # Better credit access
            liquidity_buffer=0.15  # Moderate liquidity buffer
        )

        # Wealthy agents (top 10% of wealth distribution)
        self.agent_cohorts[AgentType.WEALTHY] = AgentCohort(
            agent_type=AgentType.WEALTHY,
            population_share=0.09,
            income_mean=250000,
            income_std=100000,
            wealth_mean=1500000,  # High wealth, diversified
            wealth_std=800000,
            mpc=0.15,  # Low MPC
            mpc_std=0.08,
            discount_factor=0.99,
            risk_aversion=1.5,  # Less risk averse
            housing_share=0.40,  # Diversified wealth
            financial_assets=0.55,  # Heavy in financial assets
            credit_limit_ratio=5.0,  # Excellent credit access
            liquidity_buffer=0.30  # High liquidity buffer
        )

        # Entrepreneurs (top 1%, business owners)
        self.agent_cohorts[AgentType.ENTREPRENEURS] = AgentCohort(
            agent_type=AgentType.ENTREPRENEURS,
            population_share=0.01,
            income_mean=500000,
            income_std=300000,
            wealth_mean=5000000,  # Very high wealth
            wealth_std=3000000,
            mpc=0.25,  # Variable MPC (depends on business cycle)
            mpc_std=0.20,
            discount_factor=0.98,  # Business investment focus
            risk_aversion=1.2,  # Risk tolerant
            housing_share=0.20,  # Low housing share
            financial_assets=0.30,  # Business assets dominant
            credit_limit_ratio=10.0,  # Excellent credit access
            liquidity_buffer=0.40  # High liquidity for opportunities
        )

    def simulate_policy_shock(self, policy_shock: PolicyShock, periods: int = 24) -> List[MarketState]:
        """
        Simulate the effects of a policy shock on heterogeneous agents

        Args:
            policy_shock: Policy shock specification
            periods: Number of periods to simulate

        Returns:
            Time series of market states
        """
        logger.info(f"Simulating {policy_shock.shock_type} policy shock of magnitude {policy_shock.magnitude}")

        simulation_results = []

        # Initialize state if needed
        if self.current_state is None:
            self.current_state = self._initialize_steady_state()

        # Store initial state
        initial_state = self._copy_state(self.current_state)

        try:
            for period in range(periods):
                # Calculate shock magnitude for this period
                shock_magnitude = self._calculate_shock_magnitude(policy_shock, period)

                # Update agent states based on shock
                new_agent_states = self._update_agent_states(
                    self.current_state, policy_shock, shock_magnitude, period
                )

                # Calculate aggregate variables
                new_state = self._calculate_aggregate_state(new_agent_states, period)

                # Apply financial frictions
                new_state = self._apply_financial_frictions(new_state, period)

                # Apply nominal rigidities
                new_state = self._apply_nominal_rigidities(new_state, self.current_state)

                # Update current state
                self.current_state = new_state
                simulation_results.append(self._copy_state(new_state))

            # Store simulation in history
            self.simulation_history.extend(simulation_results)

            logger.info(f"Policy shock simulation completed: {len(simulation_results)} periods")
            return simulation_results

        except Exception as e:
            logger.error(f"Error in policy shock simulation: {e}")
            # Reset to initial state
            self.current_state = initial_state
            raise

    def _initialize_steady_state(self) -> MarketState:
        """Initialize the model in steady state"""
        # Calculate steady-state values
        steady_state_consumption = 0.0
        steady_state_investment = 0.0

        agent_states = {}

        for agent_type, cohort in self.agent_cohorts.items():
            # Steady-state consumption for this cohort
            cohort_consumption = cohort.income_mean * cohort.mpc * cohort.population_share
            steady_state_consumption += cohort_consumption

            # Steady-state investment (simplified)
            cohort_investment = cohort.wealth_mean * 0.05 * cohort.population_share  # 5% of wealth
            steady_state_investment += cohort_investment

            agent_states[agent_type] = {
                'consumption': cohort_consumption,
                'income': cohort.income_mean * cohort.population_share,
                'wealth': cohort.wealth_mean * cohort.population_share,
                'liquidity': cohort.wealth_mean * cohort.liquidity_buffer * cohort.population_share,
                'credit_utilization': 0.3,  # 30% of credit limit used
                'housing_value': cohort.wealth_mean * cohort.housing_share * cohort.population_share
            }

        # Aggregate output (simplified production function)
        steady_state_output = steady_state_consumption + steady_state_investment

        return MarketState(
            aggregate_consumption=steady_state_consumption,
            aggregate_investment=steady_state_investment,
            aggregate_output=steady_state_output,
            inflation=0.02,  # 2% target inflation
            interest_rate=0.03,  # 3% neutral rate
            unemployment=0.05,  # 5% natural rate
            housing_prices=100.0,  # Normalized index
            wage_level=100.0,  # Normalized index
            credit_spread=0.01,  # 1% credit spread
            agent_states=agent_states
        )

    def _calculate_shock_magnitude(self, policy_shock: PolicyShock, period: int) -> float:
        """Calculate shock magnitude for a given period with persistence and lags"""
        # Implementation lag
        if period < policy_shock.implementation_lag:
            return policy_shock.announcement_effect

        # AR(1) persistence after implementation
        periods_since_implementation = period - policy_shock.implementation_lag
        persistence_factor = policy_shock.persistence ** periods_since_implementation

        return policy_shock.magnitude * persistence_factor

    def _update_agent_states(self, current_state: MarketState, policy_shock: PolicyShock,
                           shock_magnitude: float, period: int) -> Dict[AgentType, Dict[str, float]]:
        """Update individual agent states based on policy shock"""
        new_agent_states = {}

        for agent_type, cohort in self.agent_cohorts.items():
            current_agent_state = current_state.agent_states[agent_type]

            # Get agent-specific shock magnitude
            agent_shock = shock_magnitude * policy_shock.targeting.get(agent_type, 1.0)

            # Update based on shock type
            if policy_shock.shock_type == 'monetary':
                new_state = self._monetary_shock_effects(
                    current_agent_state, cohort, agent_shock, current_state
                )
            elif policy_shock.shock_type == 'fiscal':
                new_state = self._fiscal_shock_effects(
                    current_agent_state, cohort, agent_shock, current_state
                )
            elif policy_shock.shock_type == 'transfer':
                new_state = self._transfer_shock_effects(
                    current_agent_state, cohort, agent_shock, current_state
                )
            else:
                new_state = current_agent_state.copy()

            new_agent_states[agent_type] = new_state

        return new_agent_states

    def _monetary_shock_effects(self, agent_state: Dict[str, float], cohort: AgentCohort,
                              shock_magnitude: float, market_state: MarketState) -> Dict[str, float]:
        """Calculate effects of monetary policy shock on agent"""
        new_state = agent_state.copy()

        # Interest rate change
        new_interest_rate = market_state.interest_rate + shock_magnitude

        # Wealth effect through asset prices (more for wealthy agents with financial assets)
        asset_price_effect = -shock_magnitude * 10 * cohort.financial_assets  # Duration effect
        wealth_change = new_state['wealth'] * asset_price_effect

        # Housing wealth effect
        housing_price_effect = -shock_magnitude * 5  # Housing duration lower than bonds
        housing_wealth_change = new_state['housing_value'] * housing_price_effect

        # Update wealth
        new_state['wealth'] += wealth_change + housing_wealth_change

        # Consumption response based on MPC and wealth effect
        consumption_change = (wealth_change + housing_wealth_change) * cohort.mpc

        # Credit constraint effects (more for hand-to-mouth)
        if cohort.agent_type == AgentType.HAND_TO_MOUTH:
            # Higher rates tighten credit constraints
            credit_tightening = shock_magnitude * 0.1  # 10% reduction per 100bp
            new_state['credit_utilization'] *= (1 + credit_tightening)
            new_state['credit_utilization'] = min(1.0, new_state['credit_utilization'])

            # Reduce consumption due to credit constraints
            consumption_change -= new_state['consumption'] * credit_tightening * 0.5

        # Update consumption
        new_state['consumption'] += consumption_change

        return new_state

    def _fiscal_shock_effects(self, agent_state: Dict[str, float], cohort: AgentCohort,
                            shock_magnitude: float, market_state: MarketState) -> Dict[str, float]:
        """Calculate effects of fiscal policy shock on agent"""
        new_state = agent_state.copy()

        # Direct income effect (government spending or tax change)
        income_change = shock_magnitude * cohort.income_mean * 0.1  # 10% transmission

        # Update income
        new_state['income'] += income_change

        # Consumption response based on MPC
        consumption_change = income_change * cohort.mpc

        # Ricardian equivalence effects (more for wealthy, forward-looking agents)
        if cohort.agent_type in [AgentType.WEALTHY, AgentType.ENTREPRENEURS]:
            # Anticipate future tax increases, reduce consumption
            ricardian_offset = consumption_change * 0.3  # 30% offset
            consumption_change -= ricardian_offset

        # Update consumption
        new_state['consumption'] += consumption_change

        return new_state

    def _transfer_shock_effects(self, agent_state: Dict[str, float], cohort: AgentCohort,
                              shock_magnitude: float, market_state: MarketState) -> Dict[str, float]:
        """Calculate effects of transfer shock (stimulus, unemployment benefits, etc.)"""
        new_state = agent_state.copy()

        # Direct transfer (typically targeted to lower-income groups)
        transfer_amount = shock_magnitude * 1000  # Dollar amount

        # Liquidity effect
        new_state['liquidity'] += transfer_amount

        # Consumption response based on MPC and liquidity constraints
        if cohort.agent_type == AgentType.HAND_TO_MOUTH:
            # High MPC, spend immediately
            consumption_change = transfer_amount * cohort.mpc
        else:
            # Lower MPC, save more of the transfer
            consumption_change = transfer_amount * cohort.mpc * 0.5

        # Update consumption
        new_state['consumption'] += consumption_change

        return new_state

    def _calculate_aggregate_state(self, agent_states: Dict[AgentType, Dict[str, float]],
                                 period: int) -> MarketState:
        """Calculate aggregate variables from individual agent states"""
        # Sum across all agents
        aggregate_consumption = sum(state['consumption'] for state in agent_states.values())
        aggregate_investment = self._calculate_aggregate_investment(agent_states)

        # Simple production function: Y = C + I (closed economy, no government for simplicity)
        aggregate_output = aggregate_consumption + aggregate_investment

        # Phillips curve for inflation
        output_gap = (aggregate_output / 100.0) - 1.0  # Deviation from normalized steady state
        inflation = 0.02 + 0.5 * output_gap  # Simple Phillips curve

        # Taylor rule for interest rate
        inflation_gap = inflation - 0.02  # Deviation from 2% target
        interest_rate = 0.03 + self.parameters.taylor_rule_inflation * inflation_gap + \
                      self.parameters.taylor_rule_output * output_gap

        # Unemployment (Okun's law)
        unemployment = 0.05 - 0.5 * output_gap  # Inverse relationship with output

        # Housing prices (wealth effect and credit conditions)
        housing_price_change = 0.1 * output_gap - 0.2 * (interest_rate - 0.03)
        housing_prices = 100.0 * (1 + housing_price_change)

        # Wage level (sticky wages)
        wage_change = 0.5 * inflation + 0.3 * output_gap
        wage_level = 100.0 * (1 + wage_change)

        # Credit spread (financial conditions)
        credit_spread = 0.01 + 0.5 * max(0, unemployment - 0.05)  # Widens with unemployment

        return MarketState(
            aggregate_consumption=aggregate_consumption,
            aggregate_investment=aggregate_investment,
            aggregate_output=aggregate_output,
            inflation=inflation,
            interest_rate=interest_rate,
            unemployment=unemployment,
            housing_prices=housing_prices,
            wage_level=wage_level,
            credit_spread=credit_spread,
            agent_states=agent_states
        )

    def _calculate_aggregate_investment(self, agent_states: Dict[AgentType, Dict[str, float]]) -> float:
        """Calculate aggregate investment based on agent wealth and business conditions"""
        total_investment = 0.0

        for agent_type, state in agent_states.items():
            cohort = self.agent_cohorts[agent_type]

            if agent_type == AgentType.ENTREPRENEURS:
                # Business investment based on expected returns and credit conditions
                investment_rate = 0.10  # 10% of wealth in business investment
                total_investment += state['wealth'] * investment_rate
            else:
                # Household investment (housing, financial assets)
                investment_rate = 0.02  # 2% of wealth
                total_investment += state['wealth'] * investment_rate

        return total_investment

    def _apply_financial_frictions(self, state: MarketState, period: int) -> MarketState:
        """Apply financial frictions to the economy"""
        new_state = self._copy_state(state)

        # Credit constraint effects
        if state.credit_spread > 0.02:  # High credit spread
            # Reduce consumption for credit-constrained agents
            for agent_type, agent_state in new_state.agent_states.items():
                cohort = self.agent_cohorts[agent_type]

                if cohort.credit_limit_ratio < 2.0:  # Credit constrained
                    friction_factor = 1 - (state.credit_spread - 0.01) * 2  # 2x effect
                    agent_state['consumption'] *= max(0.8, friction_factor)

        # Housing market stickiness
        housing_adjustment_speed = 0.1  # 10% adjustment per period
        housing_target = 100.0 * (1 + 0.1 * (state.aggregate_output / 100.0 - 1))
        new_state.housing_prices = (
            state.housing_prices * (1 - housing_adjustment_speed) +
            housing_target * housing_adjustment_speed
        )

        # Portfolio adjustment costs
        for agent_type, agent_state in new_state.agent_states.items():
            cohort = self.agent_cohorts[agent_type]

            # Reduce wealth changes due to adjustment costs
            wealth_change_rate = 0.95  # 5% adjustment cost
            agent_state['wealth'] *= wealth_change_rate

        return new_state

    def _apply_nominal_rigidities(self, new_state: MarketState, old_state: MarketState) -> MarketState:
        """Apply price and wage stickiness"""
        result_state = self._copy_state(new_state)

        # Price stickiness (Calvo pricing)
        price_adjustment_prob = 1 - self.parameters.price_stickiness
        inflation_target = new_state.inflation
        inflation_adjusted = (
            old_state.inflation * (1 - price_adjustment_prob) +
            inflation_target * price_adjustment_prob
        )
        result_state.inflation = inflation_adjusted

        # Wage stickiness
        wage_adjustment_prob = 1 - self.parameters.wage_stickiness
        wage_target = new_state.wage_level
        wage_adjusted = (
            old_state.wage_level * (1 - wage_adjustment_prob) +
            wage_target * wage_adjustment_prob
        )
        result_state.wage_level = wage_adjusted

        return result_state

    def _copy_state(self, state: MarketState) -> MarketState:
        """Create a deep copy of market state"""
        return MarketState(
            aggregate_consumption=state.aggregate_consumption,
            aggregate_investment=state.aggregate_investment,
            aggregate_output=state.aggregate_output,
            inflation=state.inflation,
            interest_rate=state.interest_rate,
            unemployment=state.unemployment,
            housing_prices=state.housing_prices,
            wage_level=state.wage_level,
            credit_spread=state.credit_spread,
            agent_states={k: v.copy() for k, v in state.agent_states.items()}
        )

    def bayesian_calibration(self, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Perform Bayesian calibration of model parameters using market data

        Args:
            market_data: Dictionary of observed time series data

        Returns:
            Updated parameter estimates and uncertainty
        """
        logger.info("Performing Bayesian calibration")

        try:
            # Store calibration data
            self.calibration_data.update(market_data)

            # Define priors for key parameters
            priors = {
                'price_stickiness': {'mean': 0.75, 'std': 0.1, 'dist': 'beta'},
                'wage_stickiness': {'mean': 0.75, 'std': 0.1, 'dist': 'beta'},
                'credit_elasticity': {'mean': 0.5, 'std': 0.2, 'dist': 'normal'},
                'mpc_hand_to_mouth': {'mean': 0.95, 'std': 0.05, 'dist': 'beta'},
                'mpc_middle_class': {'mean': 0.60, 'std': 0.10, 'dist': 'beta'},
                'mpc_wealthy': {'mean': 0.15, 'std': 0.05, 'dist': 'beta'}
            }

            # Simplified Bayesian updating using moments matching
            calibration_results = {}

            for param_name, prior in priors.items():
                if param_name.startswith('mpc_'):
                    # Update MPC parameters based on consumption data
                    agent_type_str = param_name.split('_', 1)[1]
                    updated_estimate = self._update_mpc_parameter(
                        agent_type_str, prior, market_data
                    )
                else:
                    # Update structural parameters based on aggregate data
                    updated_estimate = self._update_structural_parameter(
                        param_name, prior, market_data
                    )

                calibration_results[param_name] = updated_estimate

                # Update model parameters
                if param_name == 'price_stickiness':
                    self.parameters.price_stickiness = updated_estimate['mean']
                elif param_name == 'wage_stickiness':
                    self.parameters.wage_stickiness = updated_estimate['mean']
                elif param_name == 'credit_elasticity':
                    self.parameters.credit_elasticity = updated_estimate['mean']

            # Update agent cohort MPCs
            if 'mpc_hand_to_mouth' in calibration_results:
                self.agent_cohorts[AgentType.HAND_TO_MOUTH].mpc = calibration_results['mpc_hand_to_mouth']['mean']
            if 'mpc_middle_class' in calibration_results:
                self.agent_cohorts[AgentType.MIDDLE_CLASS].mpc = calibration_results['mpc_middle_class']['mean']
            if 'mpc_wealthy' in calibration_results:
                self.agent_cohorts[AgentType.WEALTHY].mpc = calibration_results['mpc_wealthy']['mean']

            logger.info("Bayesian calibration completed")
            return {
                'timestamp': datetime.now(),
                'calibrated_parameters': calibration_results,
                'model_fit': self._calculate_model_fit(market_data),
                'parameter_uncertainty': {name: result['std'] for name, result in calibration_results.items()}
            }

        except Exception as e:
            logger.error(f"Error in Bayesian calibration: {e}")
            raise

    def _update_mpc_parameter(self, agent_type_str: str, prior: Dict, market_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Update MPC parameter using consumption data"""
        try:
            # Get consumption and income data
            consumption_data = market_data.get('consumption', pd.Series())
            income_data = market_data.get('income', pd.Series())

            if len(consumption_data) < 5 or len(income_data) < 5:
                # Insufficient data, return prior
                return {'mean': prior['mean'], 'std': prior['std']}

            # Calculate empirical MPC (simplified)
            consumption_changes = consumption_data.pct_change().dropna()
            income_changes = income_data.pct_change().dropna()

            if len(consumption_changes) > 0 and len(income_changes) > 0:
                # Align series
                common_index = consumption_changes.index.intersection(income_changes.index)
                if len(common_index) > 3:
                    aligned_consumption = consumption_changes.loc[common_index]
                    aligned_income = income_changes.loc[common_index]

                    # Simple regression to estimate MPC
                    correlation = aligned_consumption.corr(aligned_income)
                    empirical_mpc = correlation * (aligned_consumption.std() / aligned_income.std())

                    # Bayesian update (simplified)
                    prior_precision = 1 / (prior['std'] ** 2)
                    data_precision = len(common_index) / 0.1  # Assume data variance

                    posterior_precision = prior_precision + data_precision
                    posterior_mean = (
                        prior['mean'] * prior_precision + empirical_mpc * data_precision
                    ) / posterior_precision
                    posterior_std = 1 / np.sqrt(posterior_precision)

                    return {'mean': posterior_mean, 'std': posterior_std}

            # Return prior if no update possible
            return {'mean': prior['mean'], 'std': prior['std']}

        except Exception as e:
            logger.error(f"Error updating MPC parameter: {e}")
            return {'mean': prior['mean'], 'std': prior['std']}

    def _update_structural_parameter(self, param_name: str, prior: Dict, market_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Update structural parameter using aggregate data"""
        try:
            # Simplified parameter update based on data fit
            if param_name == 'price_stickiness':
                # Use inflation volatility to infer price stickiness
                inflation_data = market_data.get('inflation', pd.Series())
                if len(inflation_data) > 5:
                    inflation_volatility = inflation_data.std()
                    # Higher volatility suggests lower stickiness
                    empirical_stickiness = max(0.1, min(0.9, 1 - inflation_volatility * 5))

                    # Bayesian update
                    prior_precision = 1 / (prior['std'] ** 2)
                    data_precision = len(inflation_data) / 0.05

                    posterior_precision = prior_precision + data_precision
                    posterior_mean = (
                        prior['mean'] * prior_precision + empirical_stickiness * data_precision
                    ) / posterior_precision
                    posterior_std = 1 / np.sqrt(posterior_precision)

                    return {'mean': posterior_mean, 'std': posterior_std}

            elif param_name == 'credit_elasticity':
                # Use credit spread and consumption relationship
                credit_data = market_data.get('credit_spread', pd.Series())
                consumption_data = market_data.get('consumption', pd.Series())

                if len(credit_data) > 5 and len(consumption_data) > 5:
                    # Calculate correlation
                    common_index = credit_data.index.intersection(consumption_data.index)
                    if len(common_index) > 3:
                        credit_aligned = credit_data.loc[common_index]
                        consumption_aligned = consumption_data.pct_change().loc[common_index]

                        correlation = credit_aligned.corr(consumption_aligned)
                        empirical_elasticity = abs(correlation)  # Use absolute correlation

                        # Bayesian update
                        prior_precision = 1 / (prior['std'] ** 2)
                        data_precision = len(common_index) / 0.1

                        posterior_precision = prior_precision + data_precision
                        posterior_mean = (
                            prior['mean'] * prior_precision + empirical_elasticity * data_precision
                        ) / posterior_precision
                        posterior_std = 1 / np.sqrt(posterior_precision)

                        return {'mean': posterior_mean, 'std': posterior_std}

            # Return prior if no specific update rule
            return {'mean': prior['mean'], 'std': prior['std']}

        except Exception as e:
            logger.error(f"Error updating structural parameter {param_name}: {e}")
            return {'mean': prior['mean'], 'std': prior['std']}

    def _calculate_model_fit(self, market_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate model fit statistics"""
        try:
            fit_stats = {}

            # Compare model predictions to actual data
            if len(self.simulation_history) > 0:
                # Get recent simulation data
                recent_states = self.simulation_history[-min(len(self.simulation_history), 12):]

                # Calculate RMSE for key variables
                if 'inflation' in market_data and len(market_data['inflation']) >= len(recent_states):
                    actual_inflation = market_data['inflation'].iloc[-len(recent_states):].values
                    predicted_inflation = [state.inflation for state in recent_states]
                    fit_stats['inflation_rmse'] = np.sqrt(np.mean((actual_inflation - predicted_inflation)**2))

                if 'unemployment' in market_data and len(market_data['unemployment']) >= len(recent_states):
                    actual_unemployment = market_data['unemployment'].iloc[-len(recent_states):].values
                    predicted_unemployment = [state.unemployment for state in recent_states]
                    fit_stats['unemployment_rmse'] = np.sqrt(np.mean((actual_unemployment - predicted_unemployment)**2))

            return fit_stats

        except Exception as e:
            logger.error(f"Error calculating model fit: {e}")
            return {}

    def get_distributional_analysis(self) -> Dict[str, Any]:
        """Get distributional analysis from current model state"""
        if not self.current_state:
            return {}

        analysis = {
            'timestamp': datetime.now(),
            'agent_consumption_shares': {},
            'consumption_inequality': {},
            'policy_transmission': {},
            'financial_stability': {}
        }

        # Calculate consumption shares by agent type
        total_consumption = self.current_state.aggregate_consumption
        for agent_type, state in self.current_state.agent_states.items():
            share = state['consumption'] / total_consumption
            analysis['agent_consumption_shares'][agent_type.value] = share

        # Calculate consumption inequality (Gini approximation)
        consumption_by_type = [
            state['consumption'] for state in self.current_state.agent_states.values()
        ]
        analysis['consumption_inequality'] = {
            'consumption_concentration': max(consumption_by_type) / sum(consumption_by_type),
            'bottom_40_share': sum(consumption_by_type[:2]) / sum(consumption_by_type)  # Approximate
        }

        # Policy transmission effectiveness
        analysis['policy_transmission'] = {
            'hand_to_mouth_responsiveness': self.agent_cohorts[AgentType.HAND_TO_MOUTH].mpc,
            'wealthy_responsiveness': self.agent_cohorts[AgentType.WEALTHY].mpc,
            'transmission_heterogeneity': (
                self.agent_cohorts[AgentType.HAND_TO_MOUTH].mpc -
                self.agent_cohorts[AgentType.WEALTHY].mpc
            )
        }

        # Financial stability indicators
        credit_stress_indicators = []
        for agent_type, state in self.current_state.agent_states.items():
            if 'credit_utilization' in state:
                credit_stress_indicators.append(state['credit_utilization'])

        analysis['financial_stability'] = {
            'average_credit_utilization': np.mean(credit_stress_indicators),
            'max_credit_utilization': max(credit_stress_indicators),
            'credit_stress_score': sum(1 for x in credit_stress_indicators if x > 0.8) / len(credit_stress_indicators)
        }

        return analysis

    def export_model_summary(self) -> Dict[str, Any]:
        """Export comprehensive model summary"""
        return {
            'model_type': 'HANK-lite',
            'timestamp': datetime.now(),
            'agent_cohorts': {
                agent_type.value: {
                    'population_share': cohort.population_share,
                    'mpc': cohort.mpc,
                    'income_mean': cohort.income_mean,
                    'wealth_mean': cohort.wealth_mean
                }
                for agent_type, cohort in self.agent_cohorts.items()
            },
            'current_parameters': {
                'price_stickiness': self.parameters.price_stickiness,
                'wage_stickiness': self.parameters.wage_stickiness,
                'credit_elasticity': self.parameters.credit_elasticity
            },
            'current_state': self.current_state.__dict__ if self.current_state else None,
            'simulation_periods': len(self.simulation_history),
            'calibration_data_available': list(self.calibration_data.keys())
        }