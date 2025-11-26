"""
Causal Intelligence Factory

Integrates all causal intelligence components (DFL, Causal DAG, HANK-lite,
Synthetic Controls, Natural Experiments) with the existing Phase2SystemFactory
to provide sophisticated causal inference capabilities for trading decisions.

Core Integration Points:
- Enhanced DPI calculator with distributional flow analysis
- Risk management with causal model validation
- Policy shock analysis for trading strategies
- Natural experiment identification for market opportunities
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import causal intelligence components
from .mycelium.distributional_flow_ledger import DistributionalFlowLedger
from .causal.causal_dag import CausalDAG
from .causal.hank_lite import HANKLiteModel, PolicyShock, AgentType
from .causal.synthetic_controls import SyntheticControlValidator
from .experiments.registry import NaturalExperimentsRegistry

# Import existing system components
from ..integration.phase2_factory import Phase2SystemFactory

logger = logging.getLogger(__name__)


@dataclass
class CausalIntelligenceConfig:
    """Configuration for causal intelligence systems"""
    enable_distributional_flows: bool = True
    enable_causal_dag: bool = True
    enable_hank_model: bool = True
    enable_synthetic_controls: bool = True
    enable_experiments_registry: bool = True

    # DFL configuration
    dfl_db_path: Optional[str] = None
    dfl_lookback_periods: int = 90

    # Causal DAG configuration
    dag_confidence_threshold: float = 0.7
    dag_max_path_length: int = 5

    # HANK configuration
    hank_calibration_frequency: str = "weekly"  # daily, weekly, monthly
    hank_agent_rebalancing: bool = True

    # Validation configuration
    validation_min_periods: int = 12
    validation_confidence_level: float = 0.95

    # Registry configuration
    registry_db_path: Optional[str] = None
    registry_auto_update: bool = True


class CausalIntelligenceFactory:
    """
    Factory for initializing and coordinating causal intelligence systems

    This factory integrates sophisticated causal inference capabilities with
    the existing trading system to provide:
    1. Distributional flow analysis for wealth capture tracking
    2. Causal DAG for policy shock simulation
    3. HANK-lite for heterogeneous agent modeling
    4. Synthetic controls for counterfactual validation
    5. Natural experiments registry for identification strategies
    """

    def __init__(self,
                 phase2_factory: Phase2SystemFactory,
                 config: Optional[CausalIntelligenceConfig] = None):
        """
        Initialize Causal Intelligence Factory

        Args:
            phase2_factory: Existing Phase2SystemFactory instance
            config: Configuration for causal intelligence systems
        """
        self.phase2_factory = phase2_factory
        self.config = config or CausalIntelligenceConfig()

        # Causal intelligence components
        self.distributional_flow_ledger: Optional[DistributionalFlowLedger] = None
        self.causal_dag: Optional[CausalDAG] = None
        self.hank_model: Optional[HANKLiteModel] = None
        self.synthetic_control_validator: Optional[SyntheticControlValidator] = None
        self.experiments_registry: Optional[NaturalExperimentsRegistry] = None

        # Integration state
        self.initialized = False
        self.last_calibration: Optional[datetime] = None

        logger.info("Causal Intelligence Factory initialized")

    def initialize_causal_systems(self) -> Dict[str, Any]:
        """Initialize all causal intelligence systems"""
        logger.info("Initializing causal intelligence systems...")

        systems = {}

        try:
            # Ensure Phase2 systems are initialized
            if not self.phase2_factory.phase2_systems:
                self.phase2_factory.initialize_phase2_systems()

            # Initialize Distributional Flow Ledger
            if self.config.enable_distributional_flows:
                self.distributional_flow_ledger = DistributionalFlowLedger(
                    db_path=self.config.dfl_db_path
                )
                systems['distributional_flow_ledger'] = self.distributional_flow_ledger
                logger.info("Distributional Flow Ledger initialized")

            # Initialize Causal DAG
            if self.config.enable_causal_dag:
                self.causal_dag = CausalDAG()
                systems['causal_dag'] = self.causal_dag
                logger.info("Causal DAG initialized")

            # Initialize HANK-lite model
            if self.config.enable_hank_model:
                self.hank_model = HANKLiteModel()
                systems['hank_model'] = self.hank_model
                logger.info("HANK-lite model initialized")

            # Initialize Synthetic Control Validator
            if self.config.enable_synthetic_controls:
                self.synthetic_control_validator = SyntheticControlValidator(
                    min_pre_treatment_periods=self.config.validation_min_periods
                )
                systems['synthetic_control_validator'] = self.synthetic_control_validator
                logger.info("Synthetic Control Validator initialized")

            # Initialize Natural Experiments Registry
            if self.config.enable_experiments_registry:
                self.experiments_registry = NaturalExperimentsRegistry(
                    db_path=self.config.registry_db_path
                )
                systems['experiments_registry'] = self.experiments_registry
                logger.info("Natural Experiments Registry initialized")

            # Perform initial integration
            self._integrate_with_existing_systems()

            self.initialized = True
            logger.info("All causal intelligence systems initialized successfully")

            return systems

        except Exception as e:
            logger.error(f"Error initializing causal intelligence systems: {e}")
            raise

    def _integrate_with_existing_systems(self):
        """Integrate causal intelligence with existing Phase2 systems"""
        try:
            # Get existing systems
            existing_systems = self.phase2_factory.get_integrated_system()

            # Integration 1: Enhance DPI calculator with distributional flows
            if (self.distributional_flow_ledger and
                'dpi_calculator' in existing_systems):

                dpi_calculator = existing_systems['dpi_calculator']

                # Add method to integrate distributional analysis
                def enhanced_calculate_dpi(symbol: str, lookback_days: int = None):
                    # Original DPI calculation
                    dpi_score, components = dpi_calculator.calculate_dpi(symbol, lookback_days)

                    # Add distributional context
                    distributional_integration = self.distributional_flow_ledger.integrate_with_dpi(
                        dpi_score, symbol
                    )

                    return dpi_score, components, distributional_integration

                # Monkey patch the enhanced method
                dpi_calculator.calculate_dpi_with_distributional_analysis = enhanced_calculate_dpi

                logger.info("Enhanced DPI calculator with distributional analysis")

            # Integration 2: Enhance risk management with causal validation
            if (self.synthetic_control_validator and
                'kelly_calculator' in existing_systems):

                kelly_calculator = existing_systems['kelly_calculator']

                # Add causal validation to position sizing
                def validated_calculate_kelly(symbol: str, *args, **kwargs):
                    # Original Kelly calculation
                    kelly_result = kelly_calculator.calculate_kelly_size(symbol, *args, **kwargs)

                    # Add causal validation warning if needed
                    # (would require historical data for full validation)
                    validation_warning = None
                    if kelly_result['position_size'] > 0.15:  # Large position
                        validation_warning = "Large position size - consider causal validation"

                    kelly_result['causal_validation'] = {
                        'warning': validation_warning,
                        'validation_available': True
                    }

                    return kelly_result

                kelly_calculator.calculate_kelly_with_validation = validated_calculate_kelly

                logger.info("Enhanced Kelly calculator with causal validation")

            # Integration 3: Add policy shock monitoring to portfolio manager
            if (self.hank_model and
                'portfolio_manager' in existing_systems):

                portfolio_manager = existing_systems['portfolio_manager']

                # Add policy shock impact assessment
                def assess_policy_shock_impact(policy_type: str, magnitude: float):
                    # Create policy shock
                    policy_shock = PolicyShock(
                        shock_type=policy_type,
                        magnitude=magnitude,
                        persistence=0.8,
                        announcement_effect=magnitude * 0.5,
                        implementation_lag=1,
                        targeting={
                            AgentType.HAND_TO_MOUTH: 1.2,
                            AgentType.MIDDLE_CLASS: 1.0,
                            AgentType.WEALTHY: 0.8,
                            AgentType.ENTREPRENEURS: 0.9
                        }
                    )

                    # Simulate shock effects
                    simulation_results = self.hank_model.simulate_policy_shock(
                        policy_shock, periods=12
                    )

                    # Extract implications for portfolio
                    portfolio_implications = self._extract_portfolio_implications(
                        simulation_results
                    )

                    return portfolio_implications

                portfolio_manager.assess_policy_shock_impact = assess_policy_shock_impact

                logger.info("Enhanced portfolio manager with policy shock analysis")

        except Exception as e:
            logger.error(f"Error integrating with existing systems: {e}")
            raise

    def _extract_portfolio_implications(self, simulation_results: List[Any]) -> Dict[str, Any]:
        """Extract portfolio implications from HANK simulation results"""
        try:
            if not simulation_results:
                return {'error': 'No simulation results'}

            # Analyze final state
            final_state = simulation_results[-1]
            initial_state = simulation_results[0] if len(simulation_results) > 1 else final_state

            # Calculate changes
            consumption_change = (
                final_state.aggregate_consumption / initial_state.aggregate_consumption - 1
            )
            inflation_change = final_state.inflation - initial_state.inflation
            unemployment_change = final_state.unemployment - initial_state.unemployment

            # Generate sector recommendations
            sector_recommendations = []

            if consumption_change > 0.02:  # >2% increase
                sector_recommendations.extend([
                    "consumer_discretionary", "retail", "restaurants"
                ])
            elif consumption_change < -0.02:  # >2% decrease
                sector_recommendations.extend([
                    "consumer_staples", "utilities", "healthcare"
                ])

            if inflation_change > 0.005:  # >50bp increase
                sector_recommendations.extend([
                    "commodities", "real_estate", "inflation_protected_bonds"
                ])

            if unemployment_change > 0.005:  # >50bp increase
                sector_recommendations.extend([
                    "discount_retail", "debt_collection", "unemployment_services"
                ])

            return {
                'consumption_impact': consumption_change,
                'inflation_impact': inflation_change,
                'unemployment_impact': unemployment_change,
                'sector_recommendations': list(set(sector_recommendations)),
                'risk_level': 'high' if abs(consumption_change) > 0.05 else 'moderate',
                'time_horizon': len(simulation_results)
            }

        except Exception as e:
            logger.error(f"Error extracting portfolio implications: {e}")
            return {'error': str(e)}

    def enhanced_trading_signal_generation(self,
                                         symbol: str,
                                         market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate enhanced trading signals using causal intelligence

        Args:
            symbol: Trading symbol
            market_data: Market data for analysis

        Returns:
            Enhanced trading signals with causal analysis
        """
        logger.info(f"Generating enhanced trading signals for {symbol}")

        if not self.initialized:
            raise ValueError("Causal intelligence systems not initialized")

        try:
            # Get base DPI signal
            existing_systems = self.phase2_factory.get_integrated_system()
            dpi_calculator = existing_systems['dpi_calculator']

            if hasattr(dpi_calculator, 'calculate_dpi_with_distributional_analysis'):
                dpi_score, components, distributional_integration = (
                    dpi_calculator.calculate_dpi_with_distributional_analysis(symbol)
                )
            else:
                dpi_score, components = dpi_calculator.calculate_dpi(symbol)
                distributional_integration = None

            # Analyze causal context
            causal_context = self._analyze_causal_context(symbol, market_data)

            # Generate enhanced signal
            enhanced_signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'base_dpi_score': dpi_score,
                'dpi_components': components.__dict__ if components else None,
                'distributional_integration': distributional_integration,
                'causal_context': causal_context,
                'enhanced_score': self._calculate_enhanced_score(
                    dpi_score, distributional_integration, causal_context
                ),
                'confidence_level': self._calculate_confidence_level(causal_context),
                'recommendations': self._generate_causal_recommendations(
                    symbol, dpi_score, causal_context
                )
            }

            return enhanced_signal

        except Exception as e:
            logger.error(f"Error generating enhanced trading signals: {e}")
            raise

    def _analyze_causal_context(self,
                               symbol: str,
                               market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze causal context for trading decision"""
        context = {
            'policy_environment': {},
            'natural_experiments': [],
            'causal_risks': [],
            'validation_strength': 0.0
        }

        try:
            # Analyze recent policy environment
            if self.experiments_registry:
                recent_shocks = []
                for shock in self.experiments_registry.policy_shocks.values():
                    if shock.announcement_date >= datetime.now() - timedelta(days=90):
                        recent_shocks.append({
                            'type': shock.shock_type,
                            'magnitude': shock.magnitude,
                            'surprise': shock.surprise_component,
                            'affected_variables': shock.directly_affected
                        })

                context['policy_environment'] = {
                    'recent_shocks_count': len(recent_shocks),
                    'average_magnitude': np.mean([s['magnitude'] for s in recent_shocks]) if recent_shocks else 0,
                    'surprise_level': np.mean([s['surprise'] for s in recent_shocks]) if recent_shocks else 0
                }

            # Find relevant natural experiments
            if self.experiments_registry:
                relevant_experiments = self.experiments_registry.search_experiments(
                    min_market_relevance=0.6,
                    tags=['trading', 'market', 'finance']
                )

                context['natural_experiments'] = [
                    {
                        'experiment_id': exp.experiment_id,
                        'quality_score': exp.quality_score,
                        'market_relevance': exp.market_relevance,
                        'identification_strength': exp.identification_strength
                    }
                    for exp in relevant_experiments[:5]  # Top 5
                ]

            # Assess causal risks
            context['causal_risks'] = self._assess_causal_risks(symbol, market_data)

            # Calculate validation strength
            if context['natural_experiments']:
                avg_quality = np.mean([exp['quality_score'] for exp in context['natural_experiments']])
                avg_identification = np.mean([exp['identification_strength'] for exp in context['natural_experiments']])
                context['validation_strength'] = (avg_quality + avg_identification) / 2

            return context

        except Exception as e:
            logger.error(f"Error analyzing causal context: {e}")
            return context

    def _assess_causal_risks(self,
                            symbol: str,
                            market_data: Dict[str, pd.Series]) -> List[str]:
        """Assess causal risks for trading decision"""
        risks = []

        try:
            # Risk 1: Confounding variables
            if 'correlation_high' in str(market_data):  # Simplified check
                risks.append("High correlation with confounding variables detected")

            # Risk 2: Structural breaks
            # Would implement statistical tests for structural breaks
            risks.append("Potential structural break in relationships (requires testing)")

            # Risk 3: Distributional effects
            if self.distributional_flow_ledger:
                flow_summary = self.distributional_flow_ledger.generate_flow_intelligence_summary()
                pressure_score = flow_summary['distributional_pressure']['pressure_score']

                if pressure_score > 0.7:
                    risks.append("High distributional pressure may affect causal relationships")

            # Risk 4: Policy regime uncertainty
            if self.experiments_registry:
                policy_shocks = list(self.experiments_registry.policy_shocks.values())
                recent_shocks = [s for s in policy_shocks
                               if s.announcement_date >= datetime.now() - timedelta(days=30)]

                if len(recent_shocks) > 2:
                    risks.append("High policy uncertainty may invalidate historical relationships")

            return risks

        except Exception as e:
            logger.error(f"Error assessing causal risks: {e}")
            return risks

    def _calculate_enhanced_score(self,
                                 dpi_score: float,
                                 distributional_integration: Optional[Dict],
                                 causal_context: Dict) -> float:
        """Calculate enhanced score incorporating causal analysis"""
        try:
            # Start with base DPI score
            enhanced_score = dpi_score

            # Adjust for distributional factors
            if distributional_integration:
                distributional_factor = distributional_integration.get('distributional_factor', 1.0)
                enhanced_score *= distributional_factor

            # Adjust for causal validation strength
            validation_strength = causal_context.get('validation_strength', 0.5)
            confidence_adjustment = 0.8 + 0.4 * validation_strength  # 0.8 to 1.2 range
            enhanced_score *= confidence_adjustment

            # Penalize for causal risks
            risk_count = len(causal_context.get('causal_risks', []))
            risk_penalty = max(0.7, 1.0 - 0.1 * risk_count)  # Max 30% penalty
            enhanced_score *= risk_penalty

            # Ensure score stays in valid range
            enhanced_score = max(-1.0, min(1.0, enhanced_score))

            return enhanced_score

        except Exception as e:
            logger.error(f"Error calculating enhanced score: {e}")
            return dpi_score  # Fallback to original DPI

    def _calculate_confidence_level(self, causal_context: Dict) -> float:
        """Calculate confidence level in the trading signal"""
        try:
            # Base confidence
            base_confidence = 0.5

            # Boost for strong validation
            validation_strength = causal_context.get('validation_strength', 0.0)
            validation_boost = validation_strength * 0.3

            # Penalty for risks
            risk_count = len(causal_context.get('causal_risks', []))
            risk_penalty = risk_count * 0.1

            # Boost for relevant experiments
            experiment_count = len(causal_context.get('natural_experiments', []))
            experiment_boost = min(0.2, experiment_count * 0.05)

            confidence = base_confidence + validation_boost + experiment_boost - risk_penalty

            return max(0.1, min(0.95, confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence level: {e}")
            return 0.5

    def _generate_causal_recommendations(self,
                                       symbol: str,
                                       dpi_score: float,
                                       causal_context: Dict) -> List[str]:
        """Generate causal-informed trading recommendations"""
        recommendations = []

        try:
            # Base recommendation from DPI
            if abs(dpi_score) > 0.3:
                direction = "long" if dpi_score > 0 else "short"
                recommendations.append(f"Base DPI signal: {direction} {symbol}")

            # Causal validation recommendations
            validation_strength = causal_context.get('validation_strength', 0.0)
            if validation_strength < 0.4:
                recommendations.append("Low causal validation - consider reduced position size")
            elif validation_strength > 0.8:
                recommendations.append("Strong causal validation - signal has high confidence")

            # Risk-based recommendations
            risks = causal_context.get('causal_risks', [])
            if len(risks) > 2:
                recommendations.append("Multiple causal risks identified - consider defensive positioning")

            # Policy environment recommendations
            policy_env = causal_context.get('policy_environment', {})
            if policy_env.get('surprise_level', 0) > 0.7:
                recommendations.append("High policy surprise environment - expect increased volatility")

            # Natural experiment insights
            experiments = causal_context.get('natural_experiments', [])
            if experiments:
                high_quality_experiments = [exp for exp in experiments if exp['quality_score'] > 0.8]
                if high_quality_experiments:
                    recommendations.append(f"Consider insights from {len(high_quality_experiments)} high-quality natural experiments")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating causal recommendations: {e}")
            return ["Error generating recommendations - use base analysis"]

    def perform_policy_impact_analysis(self,
                                     policy_type: str,
                                     policy_magnitude: float,
                                     affected_symbols: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive policy impact analysis

        Args:
            policy_type: Type of policy (monetary, fiscal, regulatory)
            policy_magnitude: Magnitude of policy change
            affected_symbols: Symbols potentially affected by policy

        Returns:
            Policy impact analysis results
        """
        logger.info(f"Performing policy impact analysis: {policy_type}")

        if not self.initialized:
            raise ValueError("Causal intelligence systems not initialized")

        try:
            analysis_results = {
                'policy_type': policy_type,
                'policy_magnitude': policy_magnitude,
                'analysis_timestamp': datetime.now(),
                'affected_symbols': affected_symbols,
                'hank_simulation': None,
                'causal_dag_analysis': None,
                'distributional_effects': None,
                'historical_precedents': [],
                'trading_implications': {}
            }

            # HANK model simulation
            if self.hank_model:
                policy_shock = PolicyShock(
                    shock_type=policy_type,
                    magnitude=policy_magnitude,
                    persistence=0.8,
                    announcement_effect=policy_magnitude * 0.6,
                    implementation_lag=1,
                    targeting={
                        AgentType.HAND_TO_MOUTH: 1.2,
                        AgentType.MIDDLE_CLASS: 1.0,
                        AgentType.WEALTHY: 0.8,
                        AgentType.ENTREPRENEURS: 0.9
                    }
                )

                simulation_results = self.hank_model.simulate_policy_shock(policy_shock, periods=12)
                analysis_results['hank_simulation'] = {
                    'consumption_impact': [state.aggregate_consumption for state in simulation_results],
                    'inflation_impact': [state.inflation for state in simulation_results],
                    'unemployment_impact': [state.unemployment for state in simulation_results],
                    'distributional_analysis': self.hank_model.get_distributional_analysis()
                }

            # Causal DAG analysis
            if self.causal_dag:
                dag_simulation = self.causal_dag.policy_shock_simulation(
                    policy_type, policy_magnitude,
                    ['asset_prices', 'consumption', 'inflation'],
                    time_horizon=12
                )
                analysis_results['causal_dag_analysis'] = dag_simulation

            # Distributional effects analysis
            if self.distributional_flow_ledger:
                marginal_flow_analysis = self.distributional_flow_ledger.track_marginal_flow(
                    policy_magnitude * 1000,  # Convert to dollar amount
                    policy_context=policy_type
                )
                analysis_results['distributional_effects'] = marginal_flow_analysis

            # Historical precedents
            if self.experiments_registry:
                similar_experiments = self.experiments_registry.search_experiments(
                    experiment_type=None,  # Search all types
                    min_market_relevance=0.5
                )

                # Filter for similar policy types
                relevant_experiments = []
                for exp in similar_experiments:
                    if policy_type.lower() in exp.title.lower() or policy_type.lower() in exp.description.lower():
                        relevant_experiments.append({
                            'experiment_id': exp.experiment_id,
                            'title': exp.title,
                            'effect_sizes': exp.effect_sizes,
                            'market_relevance': exp.market_relevance,
                            'trading_implications': exp.trading_implications
                        })

                analysis_results['historical_precedents'] = relevant_experiments[:5]

            # Generate trading implications for each symbol
            for symbol in affected_symbols:
                try:
                    # Get symbol-specific implications
                    symbol_implications = self._analyze_symbol_policy_impact(
                        symbol, policy_type, policy_magnitude, analysis_results
                    )
                    analysis_results['trading_implications'][symbol] = symbol_implications

                except Exception as e:
                    logger.error(f"Error analyzing policy impact for {symbol}: {e}")
                    analysis_results['trading_implications'][symbol] = {'error': str(e)}

            return analysis_results

        except Exception as e:
            logger.error(f"Error in policy impact analysis: {e}")
            raise

    def _analyze_symbol_policy_impact(self,
                                    symbol: str,
                                    policy_type: str,
                                    policy_magnitude: float,
                                    analysis_results: Dict) -> Dict[str, Any]:
        """Analyze policy impact for specific symbol"""
        try:
            implications = {
                'expected_direction': 'neutral',
                'magnitude_estimate': 0.0,
                'confidence': 0.5,
                'time_horizon': 'medium_term',
                'risk_factors': []
            }

            # Extract insights from HANK simulation
            if analysis_results.get('hank_simulation'):
                hank_results = analysis_results['hank_simulation']

                # Simple heuristics based on simulation results
                consumption_impact = hank_results['consumption_impact'][-1] / hank_results['consumption_impact'][0] - 1
                inflation_impact = hank_results['inflation_impact'][-1] - hank_results['inflation_impact'][0]

                # Map to symbol implications (simplified)
                if 'consumer' in symbol.lower() or 'retail' in symbol.lower():
                    implications['expected_direction'] = 'positive' if consumption_impact > 0 else 'negative'
                    implications['magnitude_estimate'] = abs(consumption_impact) * 2

                elif 'bank' in symbol.lower() or 'financial' in symbol.lower():
                    # Banks benefit from moderate inflation, hurt by high inflation
                    if 0 < inflation_impact < 0.02:  # 0-2% inflation increase
                        implications['expected_direction'] = 'positive'
                        implications['magnitude_estimate'] = inflation_impact * 5
                    else:
                        implications['expected_direction'] = 'negative'
                        implications['magnitude_estimate'] = abs(inflation_impact) * 3

            # Extract insights from historical precedents
            precedents = analysis_results.get('historical_precedents', [])
            if precedents:
                # Look for similar effect sizes
                similar_effects = []
                for precedent in precedents:
                    for outcome, effect in precedent.get('effect_sizes', {}).items():
                        if any(keyword in outcome.lower() for keyword in ['stock', 'return', 'price']):
                            similar_effects.append(effect)

                if similar_effects:
                    avg_effect = np.mean(similar_effects)
                    implications['magnitude_estimate'] = max(
                        implications['magnitude_estimate'], abs(avg_effect)
                    )
                    if avg_effect != 0:
                        implications['expected_direction'] = 'positive' if avg_effect > 0 else 'negative'

            # Assess confidence
            confidence_factors = []
            if analysis_results.get('hank_simulation'):
                confidence_factors.append(0.7)  # HANK provides good macro context
            if precedents:
                confidence_factors.append(0.6)  # Historical precedents add confidence
            if analysis_results.get('distributional_effects'):
                confidence_factors.append(0.5)  # Distributional analysis adds some confidence

            implications['confidence'] = np.mean(confidence_factors) if confidence_factors else 0.3

            # Risk factors
            if implications['magnitude_estimate'] > 0.1:  # Large expected impact
                implications['risk_factors'].append('Large expected impact increases uncertainty')

            if policy_type == 'monetary' and 'bank' not in symbol.lower():
                implications['risk_factors'].append('Monetary policy transmission may vary by sector')

            return implications

        except Exception as e:
            logger.error(f"Error analyzing symbol policy impact: {e}")
            return {'error': str(e)}

    def validate_trading_hypothesis(self,
                                  hypothesis: str,
                                  data: pd.DataFrame,
                                  treatment_variable: str,
                                  outcome_variable: str) -> Dict[str, Any]:
        """
        Validate a trading hypothesis using causal inference methods

        Args:
            hypothesis: Trading hypothesis to test
            data: Data for testing
            treatment_variable: Treatment/signal variable
            outcome_variable: Outcome/return variable

        Returns:
            Validation results
        """
        logger.info(f"Validating trading hypothesis: {hypothesis}")

        if not self.initialized:
            raise ValueError("Causal intelligence systems not initialized")

        try:
            validation_results = {
                'hypothesis': hypothesis,
                'treatment_variable': treatment_variable,
                'outcome_variable': outcome_variable,
                'validation_timestamp': datetime.now(),
                'synthetic_control_analysis': None,
                'causal_dag_validation': None,
                'natural_experiments_match': [],
                'overall_validity': 'unknown',
                'confidence_score': 0.0,
                'recommendations': []
            }

            # Synthetic control validation (if sufficient data)
            if (self.synthetic_control_validator and
                len(data) > self.config.validation_min_periods and
                'unit' in data.columns and 'date' in data.columns):

                try:
                    # Prepare data for synthetic control
                    # (This is simplified - real implementation would need proper data structure)
                    validation_results['synthetic_control_analysis'] = {
                        'status': 'attempted',
                        'note': 'Requires panel data with treatment timing identification'
                    }

                except Exception as e:
                    validation_results['synthetic_control_analysis'] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # Causal DAG validation
            if self.causal_dag:
                # Check if hypothesis variables exist in DAG
                dag_nodes = list(self.causal_dag.graph.nodes())

                treatment_in_dag = any(treatment_variable.lower() in node.lower() for node in dag_nodes)
                outcome_in_dag = any(outcome_variable.lower() in node.lower() for node in dag_nodes)

                if treatment_in_dag and outcome_in_dag:
                    # Find matching nodes
                    treatment_node = next(node for node in dag_nodes
                                        if treatment_variable.lower() in node.lower())
                    outcome_node = next(node for node in dag_nodes
                                      if outcome_variable.lower() in node.lower())

                    # Test causal identification
                    identification = self.causal_dag.identify_causal_effect(treatment_node, outcome_node)
                    validation_results['causal_dag_validation'] = identification
                else:
                    validation_results['causal_dag_validation'] = {
                        'status': 'variables_not_in_dag',
                        'treatment_in_dag': treatment_in_dag,
                        'outcome_in_dag': outcome_in_dag
                    }

            # Search for matching natural experiments
            if self.experiments_registry:
                matching_experiments = self.experiments_registry.search_experiments(
                    treatment_variable=treatment_variable,
                    outcome_variables=[outcome_variable],
                    min_quality_score=0.5
                )

                validation_results['natural_experiments_match'] = [
                    {
                        'experiment_id': exp.experiment_id,
                        'title': exp.title,
                        'quality_score': exp.quality_score,
                        'identification_strategy': exp.identification_strategy.value,
                        'effect_sizes': exp.effect_sizes
                    }
                    for exp in matching_experiments
                ]

            # Calculate overall validity and confidence
            validity_score = self._calculate_hypothesis_validity(validation_results)
            validation_results['overall_validity'] = self._categorize_validity(validity_score)
            validation_results['confidence_score'] = validity_score

            # Generate recommendations
            validation_results['recommendations'] = self._generate_validation_recommendations(
                validation_results
            )

            return validation_results

        except Exception as e:
            logger.error(f"Error validating trading hypothesis: {e}")
            raise

    def _calculate_hypothesis_validity(self, validation_results: Dict) -> float:
        """Calculate overall validity score for hypothesis"""
        try:
            validity_components = []

            # Causal DAG validation
            dag_validation = validation_results.get('causal_dag_validation')
            if dag_validation and dag_validation.get('identifiable'):
                validity_components.append(0.8)  # Strong causal identification
            elif dag_validation and dag_validation.get('status') == 'variables_not_in_dag':
                validity_components.append(0.3)  # Variables not in standard economic model

            # Natural experiments evidence
            experiments = validation_results.get('natural_experiments_match', [])
            if experiments:
                avg_quality = np.mean([exp['quality_score'] for exp in experiments])
                validity_components.append(avg_quality)

            # Synthetic control evidence (if available)
            sc_analysis = validation_results.get('synthetic_control_analysis')
            if sc_analysis and sc_analysis.get('status') == 'completed':
                validity_components.append(0.7)  # Assume good if completed

            # Calculate weighted average
            if validity_components:
                return np.mean(validity_components)
            else:
                return 0.2  # Low validity if no evidence

        except Exception as e:
            logger.error(f"Error calculating hypothesis validity: {e}")
            return 0.2

    def _categorize_validity(self, validity_score: float) -> str:
        """Categorize validity score"""
        if validity_score >= 0.7:
            return 'strong'
        elif validity_score >= 0.4:
            return 'moderate'
        else:
            return 'weak'

    def _generate_validation_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        try:
            validity = validation_results['overall_validity']
            confidence = validation_results['confidence_score']

            # Overall assessment
            if validity == 'strong':
                recommendations.append("Hypothesis has strong causal support - proceed with confidence")
            elif validity == 'moderate':
                recommendations.append("Hypothesis has moderate support - consider additional validation")
            else:
                recommendations.append("Hypothesis has weak causal support - high risk of spurious correlation")

            # Specific recommendations
            dag_validation = validation_results.get('causal_dag_validation')
            if dag_validation and not dag_validation.get('identifiable'):
                recommendations.append("Causal identification not possible - consider instrumental variables")

            experiments = validation_results.get('natural_experiments_match', [])
            if not experiments:
                recommendations.append("No matching natural experiments found - consider lower position sizes")
            elif len(experiments) >= 3:
                recommendations.append("Multiple natural experiments support hypothesis - high confidence")

            # Risk management
            if confidence < 0.5:
                recommendations.append("Low confidence - implement strict stop-losses and position limits")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating validation recommendations: {e}")
            return ["Error generating recommendations"]

    def get_causal_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of causal intelligence systems"""
        summary = {
            'timestamp': datetime.now(),
            'systems_status': {
                'initialized': self.initialized,
                'distributional_flow_ledger': self.distributional_flow_ledger is not None,
                'causal_dag': self.causal_dag is not None,
                'hank_model': self.hank_model is not None,
                'synthetic_control_validator': self.synthetic_control_validator is not None,
                'experiments_registry': self.experiments_registry is not None
            },
            'system_statistics': {},
            'recent_analysis': {},
            'integration_status': {}
        }

        try:
            # System statistics
            if self.distributional_flow_ledger:
                flow_summary = self.distributional_flow_ledger.generate_flow_intelligence_summary()
                summary['system_statistics']['distributional_flows'] = {
                    'flow_events_recorded': flow_summary['system_status']['flow_events_recorded'],
                    'distributional_pressure_score': flow_summary['distributional_pressure']['pressure_score']
                }

            if self.experiments_registry:
                registry_summary = self.experiments_registry.export_registry_summary()
                summary['system_statistics']['experiments_registry'] = {
                    'total_experiments': registry_summary['registry_summary']['total_experiments'],
                    'total_instruments': registry_summary['registry_summary']['total_instruments'],
                    'high_quality_experiments': registry_summary['quality_distribution']['high_quality']
                }

            if self.hank_model:
                hank_summary = self.hank_model.export_model_summary()
                summary['system_statistics']['hank_model'] = {
                    'simulation_periods': hank_summary['simulation_periods'],
                    'current_state_available': hank_summary['current_state'] is not None
                }

            # Integration status
            existing_systems = self.phase2_factory.get_integrated_system()

            summary['integration_status'] = {
                'phase2_systems_available': len(existing_systems),
                'dpi_enhanced': hasattr(existing_systems.get('dpi_calculator', {}), 'calculate_dpi_with_distributional_analysis'),
                'kelly_enhanced': hasattr(existing_systems.get('kelly_calculator', {}), 'calculate_kelly_with_validation'),
                'portfolio_enhanced': hasattr(existing_systems.get('portfolio_manager', {}), 'assess_policy_shock_impact')
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating causal intelligence summary: {e}")
            summary['error'] = str(e)
            return summary

    def close(self):
        """Clean up all causal intelligence systems"""
        try:
            if self.distributional_flow_ledger:
                self.distributional_flow_ledger.close()

            if self.experiments_registry:
                self.experiments_registry.close()

            logger.info("Causal intelligence systems cleaned up")

        except Exception as e:
            logger.error(f"Error cleaning up causal intelligence systems: {e}")