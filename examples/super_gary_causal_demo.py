"""
Super-Gary Causal Intelligence Demo

Demonstrates the complete causal intelligence system implementation including:
- Information Mycelium (Distributional Flow Ledger)
- Causal DAG with do-operator simulations
- HANK-lite model with heterogeneous agents
- Synthetic controls for validation
- Natural experiments registry
- Complete integration with existing trading systems

This demo shows Gary's vision of sophisticated causal inference for trading decisions.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append('/c/Users/17175/Desktop/trader-ai/src')

# Import causal intelligence components
from intelligence.mycelium.distributional_flow_ledger import (
    DistributionalFlowLedger, FlowEvent, IncomeDecile, FlowCaptor
)
from intelligence.causal.causal_dag import CausalDAG
from intelligence.causal.hank_lite import HANKLiteModel, PolicyShock, AgentType
from intelligence.causal.synthetic_controls import SyntheticControlValidator
from intelligence.experiments.registry import NaturalExperimentsRegistry
from intelligence.causal_intelligence_factory import CausalIntelligenceFactory, CausalIntelligenceConfig
from intelligence.causal_integration import create_super_gary_system

# Mock Phase2SystemFactory for demo
class MockPhase2SystemFactory:
    """Mock Phase2 factory for demonstration"""

    def __init__(self):
        self.phase2_systems = {
            'kill_switch': MockKillSwitch(),
            'kelly_calculator': MockKellyCalculator(),
            'evt_engine': MockEVTEngine()
        }

    def get_integrated_system(self):
        return {
            'dpi_calculator': MockDPICalculator(),
            'kelly_calculator': self.phase2_systems['kelly_calculator'],
            'portfolio_manager': MockPortfolioManager(),
            'broker': MockBroker(),
            'market_data': MockMarketData()
        }

class MockDPICalculator:
    """Mock DPI calculator"""
    def calculate_dpi(self, symbol, lookback_days=None):
        # Return mock DPI score based on symbol
        if 'SPY' in symbol:
            return 0.3, MockDPIComponents()
        elif 'TLT' in symbol:
            return -0.2, MockDPIComponents()
        else:
            return 0.1, MockDPIComponents()

class MockDPIComponents:
    """Mock DPI components"""
    def __init__(self):
        self.order_flow_pressure = 0.2
        self.volume_weighted_skew = 0.1
        self.price_momentum_bias = 0.15
        self.volatility_clustering = 0.05

class MockKellyCalculator:
    """Mock Kelly calculator"""
    def calculate_kelly_size(self, symbol, dpi_score, available_cash):
        kelly_fraction = abs(dpi_score) * 0.25  # Max 25% of Kelly
        return {
            'position_size': kelly_fraction,
            'kelly_fraction': kelly_fraction,
            'confidence': 0.7
        }

class MockPortfolioManager:
    """Mock portfolio manager"""
    def get_current_positions(self):
        return {}

class MockBroker:
    """Mock broker"""
    def get_account_value(self):
        return 10000.0

class MockMarketData:
    """Mock market data"""
    def get_current_price(self, symbol):
        return 100.0

class MockKillSwitch:
    """Mock kill switch"""
    pass

class MockEVTEngine:
    """Mock EVT engine"""
    pass

def setup_logging():
    """Set up logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def demo_distributional_flow_ledger(logger):
    """Demonstrate the Distributional Flow Ledger"""
    logger.info("=== DISTRIBUTIONAL FLOW LEDGER DEMO ===")

    # Initialize DFL
    dfl = DistributionalFlowLedger()

    # Simulate some flow events
    logger.info("Recording flow events...")

    # Rent payment from low-income decile
    rent_flow = FlowEvent(
        timestamp=datetime.now(),
        amount=1200.0,
        source_decile=IncomeDecile.D2,
        captor_type=FlowCaptor.LANDLORDS,
        captor_id="megacorp_properties",
        flow_category="rent",
        urgency_score=0.95,
        elasticity=-0.1,
        metadata={"location": "urban", "housing_type": "apartment"}
    )
    dfl.record_flow_event(rent_flow)

    # Credit card payment from middle class
    credit_flow = FlowEvent(
        timestamp=datetime.now(),
        amount=500.0,
        source_decile=IncomeDecile.D5,
        captor_type=FlowCaptor.CREDITORS,
        captor_id="big_bank_credit",
        flow_category="credit_payment",
        urgency_score=0.8,
        elasticity=-0.3
    )
    dfl.record_flow_event(credit_flow)

    # Analyze marginal flows from policy
    logger.info("Analyzing marginal flow from $1B stimulus...")
    marginal_analysis = dfl.track_marginal_flow(1000000000, "stimulus")

    logger.info(f"Total captured by captors: ${marginal_analysis['total_captured']:,.0f}")
    logger.info(f"Capture efficiency: {marginal_analysis['capture_efficiency']:.1%}")

    # Housing affordability analysis
    logger.info("Analyzing housing affordability...")
    housing_analysis = dfl.analyze_housing_affordability()
    overall_metrics = housing_analysis['overall_metrics']

    logger.info(f"Cost-burdened deciles: {overall_metrics['cost_burdened_deciles']}/10")
    logger.info(f"Affordability crisis score: {overall_metrics['affordability_crisis_score']:.2f}")

    # DPI integration
    logger.info("Integrating with DPI analysis...")
    dpi_integration = dfl.integrate_with_dpi(0.4, "SPY")

    logger.info(f"Original DPI: {dpi_integration['original_dpi']:.3f}")
    logger.info(f"Adjusted DPI: {dpi_integration['adjusted_dpi']:.3f}")
    logger.info(f"Risk level: {dpi_integration['trading_recommendation']['risk_level']}")

    dfl.close()
    return dfl

def demo_causal_dag(logger):
    """Demonstrate the Causal DAG"""
    logger.info("\n=== CAUSAL DAG DEMO ===")

    # Initialize Causal DAG
    dag = CausalDAG()

    # Test causal identification
    logger.info("Testing causal identification...")
    identification = dag.identify_causal_effect("monetary_policy", "asset_prices")

    logger.info(f"Causal effect identifiable: {identification['identifiable']}")
    if identification['identifiable']:
        logger.info(f"Identification method: {identification['method']}")

    # Perform do-operator intervention
    logger.info("Performing do-operator intervention...")
    try:
        do_result = dag.do_operator(
            intervention_node="monetary_policy",
            intervention_value=0.005,  # 50bp rate cut
            target_node="asset_prices"
        )

        logger.info(f"Expected effect on asset prices: {do_result.expected_effect:.3f}")
        logger.info(f"Confidence interval: [{do_result.confidence_interval[0]:.3f}, {do_result.confidence_interval[1]:.3f}]")
        logger.info(f"Causal mechanism: {' -> '.join(do_result.causal_mechanism)}")

    except ValueError as e:
        logger.info(f"Do-operator failed: {e}")

    # Policy shock simulation
    logger.info("Simulating monetary policy shock...")
    simulation = dag.policy_shock_simulation(
        policy_type="monetary",
        shock_magnitude=0.5,
        target_variables=["asset_prices", "inflation"],
        time_horizon=6
    )

    logger.info(f"Policy shock simulation completed for {simulation['time_horizon']} periods")

    # Show asset price effects
    if "asset_prices" in simulation['time_series_effects']:
        effects = simulation['time_series_effects']['asset_prices']
        logger.info(f"Asset price effects over time: {[f'{e['effect']:.3f}' for e in effects]}")

    return dag

def demo_hank_model(logger):
    """Demonstrate the HANK-lite model"""
    logger.info("\n=== HANK-LITE MODEL DEMO ===")

    # Initialize HANK model
    hank = HANKLiteModel()

    # Create policy shock
    fiscal_shock = PolicyShock(
        shock_type="fiscal",
        magnitude=0.03,  # 3% of GDP stimulus
        persistence=0.8,
        announcement_effect=0.02,
        implementation_lag=2,
        targeting={
            AgentType.HAND_TO_MOUTH: 1.5,  # More targeted to low-income
            AgentType.MIDDLE_CLASS: 1.0,
            AgentType.WEALTHY: 0.5,
            AgentType.ENTREPRENEURS: 0.8
        }
    )

    # Simulate policy shock
    logger.info("Simulating fiscal policy shock...")
    simulation_results = hank.simulate_policy_shock(fiscal_shock, periods=8)

    logger.info(f"Simulation completed for {len(simulation_results)} periods")

    # Show key results
    initial_state = simulation_results[0]
    final_state = simulation_results[-1]

    consumption_change = (final_state.aggregate_consumption / initial_state.aggregate_consumption - 1) * 100
    inflation_change = (final_state.inflation - initial_state.inflation) * 100
    unemployment_change = (final_state.unemployment - initial_state.unemployment) * 100

    logger.info(f"Consumption change: {consumption_change:+.1f}%")
    logger.info(f"Inflation change: {inflation_change:+.1f} percentage points")
    logger.info(f"Unemployment change: {unemployment_change:+.1f} percentage points")

    # Distributional analysis
    logger.info("Analyzing distributional effects...")
    distributional_analysis = hank.get_distributional_analysis()

    logger.info(f"Policy transmission heterogeneity: {distributional_analysis['policy_transmission']['transmission_heterogeneity']:.2f}")
    logger.info(f"Credit stress score: {distributional_analysis['financial_stability']['credit_stress_score']:.2f}")

    return hank

def demo_synthetic_controls(logger):
    """Demonstrate Synthetic Controls"""
    logger.info("\n=== SYNTHETIC CONTROLS DEMO ===")

    # Initialize validator
    validator = SyntheticControlValidator()

    # Create synthetic panel data for demonstration
    np.random.seed(42)

    # Create mock panel data
    units = ['treated_unit', 'control_1', 'control_2', 'control_3', 'control_4']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')

    data = []
    for unit in units:
        for date in dates:
            # Treatment occurs in Jan 2022 for treated_unit
            is_treated = (unit == 'treated_unit' and date >= pd.Timestamp('2022-01-01'))

            # Base outcome with unit-specific trend
            base_outcome = 100 + np.random.normal(0, 5)
            if unit == 'treated_unit':
                base_outcome += np.random.normal(2, 1)  # Slightly higher baseline

            # Treatment effect
            if is_treated:
                treatment_effect = 10 + np.random.normal(0, 2)  # 10% average effect
                base_outcome += treatment_effect

            data.append({
                'unit': unit,
                'date': date,
                'outcome': base_outcome,
                'predictor_1': np.random.normal(50, 10),
                'predictor_2': np.random.normal(20, 5)
            })

    panel_data = pd.DataFrame(data)

    # Run synthetic control analysis
    logger.info("Running synthetic control analysis...")
    try:
        sc_result = validator.synthetic_control_analysis(
            data=panel_data,
            treated_unit='treated_unit',
            outcome_variable='outcome',
            treatment_date=pd.Timestamp('2022-01-01'),
            control_units=['control_1', 'control_2', 'control_3', 'control_4']
        )

        logger.info(f"Average treatment effect: {sc_result.average_treatment_effect:.2f}")
        logger.info(f"P-value: {sc_result.p_value:.3f}")
        logger.info(f"Pre-treatment fit (RMSE): {sc_result.pre_treatment_fit:.2f}")

        # Show synthetic weights
        logger.info("Synthetic control weights:")
        for unit, weight in sc_result.synthetic_weights.items():
            if weight > 0.01:  # Only show significant weights
                logger.info(f"  {unit}: {weight:.3f}")

    except Exception as e:
        logger.info(f"Synthetic control analysis failed: {e}")

    return validator

def demo_natural_experiments_registry(logger):
    """Demonstrate Natural Experiments Registry"""
    logger.info("\n=== NATURAL EXPERIMENTS REGISTRY DEMO ===")

    # Initialize registry
    registry = NaturalExperimentsRegistry()

    # Search for experiments
    logger.info("Searching for high-quality experiments...")
    high_quality_experiments = registry.search_experiments(
        min_quality_score=0.8,
        min_market_relevance=0.7
    )

    logger.info(f"Found {len(high_quality_experiments)} high-quality experiments")

    for exp in high_quality_experiments[:3]:  # Show top 3
        logger.info(f"  - {exp.title}")
        logger.info(f"    Quality: {exp.quality_score:.2f}, Market relevance: {exp.market_relevance:.2f}")
        logger.info(f"    Strategy: {exp.identification_strategy.value}")

    # Policy shock analysis
    logger.info("\nAnalyzing policy shocks...")
    policy_analysis = registry.get_policy_shock_analysis(
        date_range=(datetime.now() - timedelta(days=365), datetime.now())
    )

    if 'total_shocks' in policy_analysis:
        logger.info(f"Total policy shocks in last year: {policy_analysis['total_shocks']}")
        logger.info(f"Average shock magnitude: {policy_analysis.get('average_magnitude', 0):.2f}")

    # Suggest experiments for research question
    logger.info("\nSuggesting experiments for interest rate effects...")
    suggestions = registry.suggest_natural_experiments(
        research_question="How do interest rate changes affect stock market returns?",
        treatment_variable="interest_rate",
        outcome_variable="stock_returns",
        available_data=["fed_data", "stock_prices", "economic_indicators"]
    )

    logger.info(f"Found {len(suggestions)} experiment suggestions")
    for suggestion in suggestions[:2]:  # Show top 2
        logger.info(f"  - {suggestion['title']}")
        logger.info(f"    Suitability score: {suggestion['suitability_score']:.2f}")
        logger.info(f"    Data feasibility: {suggestion['data_feasibility']:.2f}")

    registry.close()
    return registry

def demo_integrated_causal_intelligence(logger):
    """Demonstrate the integrated causal intelligence system"""
    logger.info("\n=== INTEGRATED CAUSAL INTELLIGENCE DEMO ===")

    # Create mock Phase2 factory
    phase2_factory = MockPhase2SystemFactory()

    # Create causal intelligence configuration
    config = CausalIntelligenceConfig(
        enable_distributional_flows=True,
        enable_causal_dag=True,
        enable_hank_model=True,
        enable_synthetic_controls=True,
        enable_experiments_registry=True,
        dfl_db_path=":memory:",
        registry_db_path=":memory:"
    )

    # Create Super-Gary system
    logger.info("Creating Super-Gary system...")
    super_gary = create_super_gary_system(phase2_factory, config)

    logger.info("Super-Gary system created successfully!")
    logger.info(f"Causal integration active: {super_gary['causal_integration_active']}")

    # Generate enhanced trading signal
    logger.info("\nGenerating enhanced trading signal for SPY...")
    enhanced_signal = super_gary['generate_causally_enhanced_signal'](
        symbol='SPY',
        available_cash=10000.0,
        market_context={'sector': 'broad_market'}
    )

    logger.info(f"Symbol: {enhanced_signal.symbol}")
    logger.info(f"Base DPI score: {enhanced_signal.base_dpi_score:.3f}")
    logger.info(f"Distributionally adjusted DPI: {enhanced_signal.distributional_adjusted_dpi:.3f}")
    logger.info(f"Final position size: {enhanced_signal.final_position_size:.1%}")
    logger.info(f"Causal confidence: {enhanced_signal.causal_confidence:.2f}")

    if enhanced_signal.causal_warnings:
        logger.info("Causal warnings:")
        for warning in enhanced_signal.causal_warnings:
            logger.info(f"  - {warning}")

    if enhanced_signal.sector_rotation_signal:
        logger.info(f"Sector rotation signals: {enhanced_signal.sector_rotation_signal}")

    # Test policy impact analysis
    logger.info("\nAnalyzing policy impact...")
    policy_impact = super_gary['causal_factory'].perform_policy_impact_analysis(
        policy_type="monetary",
        policy_magnitude=0.25,  # 25bp rate change
        affected_symbols=["SPY", "TLT"]
    )

    logger.info(f"Policy type: {policy_impact['policy_type']}")
    logger.info(f"Policy magnitude: {policy_impact['policy_magnitude']}")

    for symbol in ["SPY", "TLT"]:
        if symbol in policy_impact['trading_implications']:
            implications = policy_impact['trading_implications'][symbol]
            if 'expected_direction' in implications:
                logger.info(f"{symbol} expected direction: {implications['expected_direction']}")
                logger.info(f"{symbol} magnitude estimate: {implications.get('magnitude_estimate', 0):.3f}")

    # Clean up
    super_gary['causal_factory'].close()

    return super_gary

def main():
    """Main demonstration function"""
    logger = setup_logging()

    logger.info("🚀 Starting Super-Gary Causal Intelligence Demo")
    logger.info("Implementing Gary's vision of sophisticated causal inference for trading")

    try:
        # Demo individual components
        demo_distributional_flow_ledger(logger)
        demo_causal_dag(logger)
        demo_hank_model(logger)
        demo_synthetic_controls(logger)
        demo_natural_experiments_registry(logger)

        # Demo integrated system
        demo_integrated_causal_intelligence(logger)

        logger.info("\n🎉 Super-Gary Causal Intelligence Demo completed successfully!")
        logger.info("\nKey capabilities demonstrated:")
        logger.info("✅ Information Mycelium - tracking wealth flows by income decile")
        logger.info("✅ Causal DAG - do-operator simulations and counterfactuals")
        logger.info("✅ HANK-lite - heterogeneous agent modeling with sticky frictions")
        logger.info("✅ Synthetic Controls - counterfactual validation")
        logger.info("✅ Natural Experiments Registry - causal identification strategies")
        logger.info("✅ Complete Integration - enhanced DPI and risk management")

        logger.info("\nGary's vision achieved: 'Follow the Flow, Prove Causality'")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()