"""
Causal Integration Module

Wires causal intelligence systems into existing DPI calculator and risk management
components to provide enhanced trading decisions with causal validation.

This module provides the final integration layer that connects:
- Distributional Flow Ledger with DPI calculations
- Causal DAG with risk assessment
- HANK model with position sizing
- Natural experiments with strategy validation
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import causal intelligence components
from .causal_intelligence_factory import CausalIntelligenceFactory, CausalIntelligenceConfig

logger = logging.getLogger(__name__)


@dataclass
class CausallyEnhancedSignal:
    """Enhanced trading signal with causal analysis"""
    symbol: str
    timestamp: datetime

    # Original signals
    base_dpi_score: float
    kelly_size: float
    risk_score: float

    # Causal enhancements
    distributional_adjusted_dpi: float
    causal_confidence: float
    policy_shock_factor: float
    natural_experiment_support: float

    # Integrated recommendation
    final_position_size: float
    confidence_level: float
    causal_warnings: List[str]
    sector_rotation_signal: List[str]

    # Validation metrics
    causal_validity_score: float
    counterfactual_robustness: float


class CausallyEnhancedDPICalculator:
    """
    Enhanced DPI Calculator with causal intelligence integration

    Extends the base DPI calculator with:
    - Distributional flow analysis
    - Causal DAG validation
    - Policy shock considerations
    - Natural experiment insights
    """

    def __init__(self, base_dpi_calculator, causal_factory: CausalIntelligenceFactory):
        """
        Initialize enhanced DPI calculator

        Args:
            base_dpi_calculator: Original DPI calculator instance
            causal_factory: Initialized causal intelligence factory
        """
        self.base_calculator = base_dpi_calculator
        self.causal_factory = causal_factory

        # Ensure causal systems are initialized
        if not self.causal_factory.initialized:
            self.causal_factory.initialize_causal_systems()

        logger.info("Causally Enhanced DPI Calculator initialized")

    def calculate_enhanced_dpi(self,
                              symbol: str,
                              lookback_days: int = None,
                              market_context: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate DPI with causal enhancements

        Args:
            symbol: Trading symbol
            lookback_days: Historical data window
            market_context: Additional market context

        Returns:
            Tuple of (enhanced_dpi_score, detailed_analysis)
        """
        logger.info(f"Calculating enhanced DPI for {symbol}")

        try:
            # Get base DPI calculation
            base_dpi, components = self.base_calculator.calculate_dpi(symbol, lookback_days)

            # Get distributional flow analysis
            distributional_integration = None
            if self.causal_factory.distributional_flow_ledger:
                distributional_integration = self.causal_factory.distributional_flow_ledger.integrate_with_dpi(
                    base_dpi, symbol
                )

            # Get causal DAG analysis
            causal_dag_analysis = self._get_causal_dag_analysis(symbol, base_dpi)

            # Get policy environment analysis
            policy_analysis = self._analyze_policy_environment(symbol)

            # Get natural experiment validation
            experiment_validation = self._validate_with_natural_experiments(symbol)

            # Calculate enhanced DPI score
            enhanced_dpi = self._calculate_enhanced_dpi_score(
                base_dpi,
                distributional_integration,
                causal_dag_analysis,
                policy_analysis,
                experiment_validation
            )

            # Compile detailed analysis
            detailed_analysis = {
                'base_dpi_score': base_dpi,
                'enhanced_dpi_score': enhanced_dpi,
                'base_components': components.__dict__ if components else None,
                'distributional_integration': distributional_integration,
                'causal_dag_analysis': causal_dag_analysis,
                'policy_analysis': policy_analysis,
                'experiment_validation': experiment_validation,
                'enhancement_factors': self._calculate_enhancement_factors(
                    base_dpi, enhanced_dpi, distributional_integration, policy_analysis
                ),
                'causal_warnings': self._generate_causal_warnings(
                    causal_dag_analysis, policy_analysis, experiment_validation
                ),
                'timestamp': datetime.now()
            }

            return enhanced_dpi, detailed_analysis

        except Exception as e:
            logger.error(f"Error calculating enhanced DPI for {symbol}: {e}")
            # Fallback to base DPI
            base_dpi, components = self.base_calculator.calculate_dpi(symbol, lookback_days)
            return base_dpi, {
                'base_dpi_score': base_dpi,
                'enhanced_dpi_score': base_dpi,
                'error': str(e),
                'fallback_used': True
            }

    def _get_causal_dag_analysis(self, symbol: str, dpi_score: float) -> Dict[str, Any]:
        """Get causal DAG analysis for the symbol"""
        try:
            if not self.causal_factory.causal_dag:
                return {'status': 'not_available'}

            # Map symbol to economic variables (simplified)
            if 'SPY' in symbol or 'stock' in symbol.lower():
                outcome_variable = 'asset_prices'
            elif 'TLT' in symbol or 'bond' in symbol.lower():
                outcome_variable = 'bond_yields'
            else:
                outcome_variable = 'asset_prices'  # Default

            # Test causal identification for monetary policy
            monetary_identification = self.causal_factory.causal_dag.identify_causal_effect(
                'monetary_policy', outcome_variable
            )

            # Test causal identification for fiscal policy
            fiscal_identification = self.causal_factory.causal_dag.identify_causal_effect(
                'fiscal_policy', outcome_variable
            )

            return {
                'outcome_variable': outcome_variable,
                'monetary_policy_identification': monetary_identification,
                'fiscal_policy_identification': fiscal_identification,
                'causal_paths_available': monetary_identification['identifiable'] or fiscal_identification['identifiable']
            }

        except Exception as e:
            logger.error(f"Error in causal DAG analysis: {e}")
            return {'status': 'error', 'error': str(e)}

    def _analyze_policy_environment(self, symbol: str) -> Dict[str, Any]:
        """Analyze current policy environment"""
        try:
            if not self.causal_factory.experiments_registry:
                return {'status': 'not_available'}

            # Get recent policy shocks
            recent_shocks = []
            cutoff_date = datetime.now() - timedelta(days=90)

            for shock in self.causal_factory.experiments_registry.policy_shocks.values():
                if shock.announcement_date >= cutoff_date:
                    recent_shocks.append({
                        'type': shock.shock_type,
                        'magnitude': shock.magnitude,
                        'surprise': shock.surprise_component,
                        'date': shock.announcement_date
                    })

            # Assess policy uncertainty
            policy_uncertainty = len(recent_shocks) / 10.0  # Scale by typical number

            # Assess policy impact on symbol
            symbol_relevance = self._assess_policy_symbol_relevance(symbol, recent_shocks)

            return {
                'recent_shocks': recent_shocks,
                'policy_uncertainty': min(1.0, policy_uncertainty),
                'symbol_relevance': symbol_relevance,
                'environment_assessment': self._categorize_policy_environment(policy_uncertainty)
            }

        except Exception as e:
            logger.error(f"Error analyzing policy environment: {e}")
            return {'status': 'error', 'error': str(e)}

    def _validate_with_natural_experiments(self, symbol: str) -> Dict[str, Any]:
        """Validate signal with natural experiments"""
        try:
            if not self.causal_factory.experiments_registry:
                return {'status': 'not_available'}

            # Search for relevant experiments
            relevant_experiments = self.causal_factory.experiments_registry.search_experiments(
                min_market_relevance=0.5,
                tags=['trading', 'financial', 'market']
            )

            # Calculate validation strength
            if relevant_experiments:
                avg_quality = np.mean([exp.quality_score for exp in relevant_experiments])
                avg_identification = np.mean([exp.identification_strength for exp in relevant_experiments])
                validation_strength = (avg_quality + avg_identification) / 2
            else:
                validation_strength = 0.0

            return {
                'relevant_experiments_count': len(relevant_experiments),
                'validation_strength': validation_strength,
                'top_experiments': [
                    {
                        'id': exp.experiment_id,
                        'quality': exp.quality_score,
                        'market_relevance': exp.market_relevance
                    }
                    for exp in relevant_experiments[:3]
                ]
            }

        except Exception as e:
            logger.error(f"Error validating with natural experiments: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_enhanced_dpi_score(self,
                                    base_dpi: float,
                                    distributional_integration: Optional[Dict],
                                    causal_dag_analysis: Dict,
                                    policy_analysis: Dict,
                                    experiment_validation: Dict) -> float:
        """Calculate enhanced DPI score"""
        try:
            enhanced_score = base_dpi

            # Apply distributional adjustment
            if distributional_integration:
                distributional_factor = distributional_integration.get('distributional_factor', 1.0)
                enhanced_score *= distributional_factor

            # Apply causal validation adjustment
            if causal_dag_analysis.get('causal_paths_available'):
                causal_boost = 1.1  # 10% boost for strong causal identification
                enhanced_score *= causal_boost

            # Apply policy environment adjustment
            policy_uncertainty = policy_analysis.get('policy_uncertainty', 0.5)
            if policy_uncertainty > 0.7:  # High uncertainty
                uncertainty_penalty = 0.9  # 10% penalty
                enhanced_score *= uncertainty_penalty

            # Apply natural experiment validation
            validation_strength = experiment_validation.get('validation_strength', 0.0)
            validation_adjustment = 1.0 + (validation_strength - 0.5) * 0.2  # ±10% max
            enhanced_score *= validation_adjustment

            # Ensure score stays in valid range
            enhanced_score = max(-1.0, min(1.0, enhanced_score))

            return enhanced_score

        except Exception as e:
            logger.error(f"Error calculating enhanced DPI score: {e}")
            return base_dpi

    def _calculate_enhancement_factors(self,
                                     base_dpi: float,
                                     enhanced_dpi: float,
                                     distributional_integration: Optional[Dict],
                                     policy_analysis: Dict) -> Dict[str, float]:
        """Calculate individual enhancement factors"""
        factors = {
            'total_enhancement': enhanced_dpi / base_dpi if base_dpi != 0 else 1.0,
            'distributional_factor': 1.0,
            'policy_factor': 1.0,
            'causal_validation_factor': 1.0
        }

        try:
            if distributional_integration:
                factors['distributional_factor'] = distributional_integration.get('distributional_factor', 1.0)

            policy_uncertainty = policy_analysis.get('policy_uncertainty', 0.5)
            if policy_uncertainty > 0.7:
                factors['policy_factor'] = 0.9
            elif policy_uncertainty < 0.3:
                factors['policy_factor'] = 1.1

            return factors

        except Exception as e:
            logger.error(f"Error calculating enhancement factors: {e}")
            return factors

    def _generate_causal_warnings(self,
                                 causal_dag_analysis: Dict,
                                 policy_analysis: Dict,
                                 experiment_validation: Dict) -> List[str]:
        """Generate causal warnings for the signal"""
        warnings = []

        try:
            # Check causal identification
            if not causal_dag_analysis.get('causal_paths_available'):
                warnings.append("Weak causal identification - signal may be spurious")

            # Check policy environment
            policy_uncertainty = policy_analysis.get('policy_uncertainty', 0.5)
            if policy_uncertainty > 0.8:
                warnings.append("High policy uncertainty - increased volatility expected")

            # Check experimental validation
            validation_strength = experiment_validation.get('validation_strength', 0.0)
            if validation_strength < 0.3:
                warnings.append("Limited experimental validation - proceed with caution")

            # Check for recent policy shocks
            recent_shocks = policy_analysis.get('recent_shocks', [])
            if len(recent_shocks) > 3:
                warnings.append("Multiple recent policy shocks - relationships may be unstable")

            return warnings

        except Exception as e:
            logger.error(f"Error generating causal warnings: {e}")
            return ["Error generating warnings - manual review recommended"]

    def _assess_policy_symbol_relevance(self, symbol: str, recent_shocks: List[Dict]) -> float:
        """Assess how relevant recent policy shocks are to the symbol"""
        if not recent_shocks:
            return 0.0

        relevance_score = 0.0

        for shock in recent_shocks:
            shock_type = shock['type'].lower()
            symbol_lower = symbol.lower()

            # Simple heuristic mapping
            if 'monetary' in shock_type:
                if any(keyword in symbol_lower for keyword in ['spy', 'stock', 'etf']):
                    relevance_score += 0.8
                elif any(keyword in symbol_lower for keyword in ['tlt', 'bond']):
                    relevance_score += 0.9
            elif 'fiscal' in shock_type:
                if any(keyword in symbol_lower for keyword in ['spy', 'consumer', 'retail']):
                    relevance_score += 0.7

        return min(1.0, relevance_score / len(recent_shocks))

    def _categorize_policy_environment(self, uncertainty: float) -> str:
        """Categorize the policy environment"""
        if uncertainty > 0.8:
            return "high_uncertainty"
        elif uncertainty > 0.5:
            return "moderate_uncertainty"
        elif uncertainty > 0.2:
            return "low_uncertainty"
        else:
            return "stable"


class CausallyEnhancedRiskManager:
    """
    Enhanced Risk Manager with causal intelligence integration

    Extends risk management with:
    - Causal validation of risk models
    - Policy shock impact assessment
    - Distributional risk analysis
    - Natural experiment insights
    """

    def __init__(self, base_kelly_calculator, causal_factory: CausalIntelligenceFactory):
        """
        Initialize enhanced risk manager

        Args:
            base_kelly_calculator: Original Kelly calculator
            causal_factory: Initialized causal intelligence factory
        """
        self.base_kelly_calculator = base_kelly_calculator
        self.causal_factory = causal_factory

        logger.info("Causally Enhanced Risk Manager initialized")

    def calculate_enhanced_position_size(self,
                                       symbol: str,
                                       enhanced_dpi: float,
                                       base_kelly_size: float,
                                       causal_analysis: Dict[str, Any],
                                       available_cash: float) -> Dict[str, Any]:
        """
        Calculate position size with causal risk adjustments

        Args:
            symbol: Trading symbol
            enhanced_dpi: Enhanced DPI score
            base_kelly_size: Base Kelly position size
            causal_analysis: Causal analysis from enhanced DPI
            available_cash: Available cash for position

        Returns:
            Enhanced position sizing recommendation
        """
        logger.info(f"Calculating enhanced position size for {symbol}")

        try:
            # Start with base Kelly size
            adjusted_size = base_kelly_size

            # Apply causal risk adjustments
            causal_confidence = self._calculate_causal_confidence(causal_analysis)
            confidence_adjustment = 0.5 + 0.5 * causal_confidence  # 50% to 100% of base size
            adjusted_size *= confidence_adjustment

            # Apply distributional risk adjustments
            distributional_risk = self._assess_distributional_risk(causal_analysis)
            distributional_adjustment = 1.0 - distributional_risk * 0.3  # Up to 30% reduction
            adjusted_size *= distributional_adjustment

            # Apply policy shock risk adjustments
            policy_risk = self._assess_policy_shock_risk(causal_analysis)
            policy_adjustment = 1.0 - policy_risk * 0.2  # Up to 20% reduction
            adjusted_size *= policy_adjustment

            # Calculate final position size in dollar terms
            dollar_size = adjusted_size * available_cash

            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(
                causal_confidence, distributional_risk, policy_risk
            )

            # Calculate stop-loss and take-profit levels
            stop_loss, take_profit = self._calculate_risk_levels(
                enhanced_dpi, causal_confidence
            )

            return {
                'symbol': symbol,
                'base_kelly_size': base_kelly_size,
                'causal_adjusted_size': adjusted_size,
                'dollar_position_size': dollar_size,
                'position_size_percentage': adjusted_size,
                'causal_confidence': causal_confidence,
                'distributional_risk': distributional_risk,
                'policy_risk': policy_risk,
                'adjustments': {
                    'confidence_adjustment': confidence_adjustment,
                    'distributional_adjustment': distributional_adjustment,
                    'policy_adjustment': policy_adjustment
                },
                'risk_warnings': risk_warnings,
                'stop_loss_level': stop_loss,
                'take_profit_level': take_profit,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error calculating enhanced position size: {e}")
            return {
                'symbol': symbol,
                'base_kelly_size': base_kelly_size,
                'causal_adjusted_size': base_kelly_size,
                'dollar_position_size': base_kelly_size * available_cash,
                'error': str(e),
                'fallback_used': True
            }

    def _calculate_causal_confidence(self, causal_analysis: Dict[str, Any]) -> float:
        """Calculate overall causal confidence"""
        try:
            confidence_factors = []

            # Causal DAG confidence
            dag_analysis = causal_analysis.get('causal_dag_analysis', {})
            if dag_analysis.get('causal_paths_available'):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)

            # Experimental validation confidence
            experiment_validation = causal_analysis.get('experiment_validation', {})
            validation_strength = experiment_validation.get('validation_strength', 0.0)
            confidence_factors.append(validation_strength)

            # Policy analysis confidence
            policy_analysis = causal_analysis.get('policy_analysis', {})
            policy_uncertainty = policy_analysis.get('policy_uncertainty', 0.5)
            policy_confidence = 1.0 - policy_uncertainty
            confidence_factors.append(policy_confidence)

            return np.mean(confidence_factors) if confidence_factors else 0.5

        except Exception as e:
            logger.error(f"Error calculating causal confidence: {e}")
            return 0.5

    def _assess_distributional_risk(self, causal_analysis: Dict[str, Any]) -> float:
        """Assess distributional risk from flow analysis"""
        try:
            distributional_integration = causal_analysis.get('distributional_integration')
            if not distributional_integration:
                return 0.5  # Moderate risk if no data

            flow_context = distributional_integration.get('flow_context', {})
            pressure_score = flow_context.get('pressure_score', 0.5)

            # Higher pressure = higher risk
            return min(1.0, pressure_score)

        except Exception as e:
            logger.error(f"Error assessing distributional risk: {e}")
            return 0.5

    def _assess_policy_shock_risk(self, causal_analysis: Dict[str, Any]) -> float:
        """Assess policy shock risk"""
        try:
            policy_analysis = causal_analysis.get('policy_analysis', {})
            policy_uncertainty = policy_analysis.get('policy_uncertainty', 0.5)
            symbol_relevance = policy_analysis.get('symbol_relevance', 0.5)

            # Risk increases with uncertainty and relevance
            policy_risk = policy_uncertainty * symbol_relevance

            return min(1.0, policy_risk)

        except Exception as e:
            logger.error(f"Error assessing policy shock risk: {e}")
            return 0.5

    def _generate_risk_warnings(self,
                               causal_confidence: float,
                               distributional_risk: float,
                               policy_risk: float) -> List[str]:
        """Generate risk warnings"""
        warnings = []

        if causal_confidence < 0.4:
            warnings.append("Low causal confidence - high risk of spurious signals")

        if distributional_risk > 0.7:
            warnings.append("High distributional pressure - consumer spending risk")

        if policy_risk > 0.7:
            warnings.append("High policy shock risk - expect increased volatility")

        return warnings

    def _calculate_risk_levels(self,
                              enhanced_dpi: float,
                              causal_confidence: float) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels"""
        try:
            # Base levels on signal strength and confidence
            signal_strength = abs(enhanced_dpi)

            # Stop-loss: tighter for lower confidence
            stop_loss_base = 0.02  # 2% base stop-loss
            confidence_multiplier = 1.0 + (1.0 - causal_confidence)  # 1.0 to 2.0
            stop_loss = stop_loss_base * confidence_multiplier

            # Take-profit: based on signal strength
            take_profit_base = 0.04  # 4% base take-profit
            signal_multiplier = 1.0 + signal_strength  # 1.0 to 2.0
            take_profit = take_profit_base * signal_multiplier

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            return 0.03, 0.05  # Default 3% stop, 5% profit


def integrate_causal_intelligence_with_phase2(phase2_factory, config: Optional[CausalIntelligenceConfig] = None):
    """
    Integrate causal intelligence with existing Phase2 systems

    Args:
        phase2_factory: Existing Phase2SystemFactory
        config: Causal intelligence configuration

    Returns:
        Enhanced systems dictionary
    """
    logger.info("Integrating causal intelligence with Phase2 systems")

    try:
        # Initialize causal intelligence factory
        causal_factory = CausalIntelligenceFactory(phase2_factory, config)
        causal_factory.initialize_causal_systems()

        # Get existing systems
        existing_systems = phase2_factory.get_integrated_system()

        # Create enhanced DPI calculator
        enhanced_dpi_calculator = CausallyEnhancedDPICalculator(
            existing_systems['dpi_calculator'],
            causal_factory
        )

        # Create enhanced risk manager
        enhanced_risk_manager = CausallyEnhancedRiskManager(
            existing_systems['kelly_calculator'],
            causal_factory
        )

        # Create unified signal generator
        def generate_causally_enhanced_signal(symbol: str,
                                            available_cash: float,
                                            market_context: Optional[Dict] = None) -> CausallyEnhancedSignal:
            """Generate unified causally enhanced trading signal"""

            # Get enhanced DPI
            enhanced_dpi, dpi_analysis = enhanced_dpi_calculator.calculate_enhanced_dpi(
                symbol, market_context=market_context
            )

            # Get base Kelly size
            base_kelly_result = existing_systems['kelly_calculator'].calculate_kelly_size(
                symbol, enhanced_dpi, available_cash
            )
            base_kelly_size = base_kelly_result.get('position_size', 0.0)

            # Get enhanced position sizing
            position_sizing = enhanced_risk_manager.calculate_enhanced_position_size(
                symbol, enhanced_dpi, base_kelly_size, dpi_analysis, available_cash
            )

            # Create unified signal
            signal = CausallyEnhancedSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                base_dpi_score=dpi_analysis.get('base_dpi_score', 0.0),
                kelly_size=base_kelly_size,
                risk_score=position_sizing.get('policy_risk', 0.5),
                distributional_adjusted_dpi=enhanced_dpi,
                causal_confidence=position_sizing.get('causal_confidence', 0.5),
                policy_shock_factor=position_sizing.get('policy_risk', 0.5),
                natural_experiment_support=dpi_analysis.get('experiment_validation', {}).get('validation_strength', 0.0),
                final_position_size=position_sizing.get('causal_adjusted_size', 0.0),
                confidence_level=position_sizing.get('causal_confidence', 0.5),
                causal_warnings=dpi_analysis.get('causal_warnings', []) + position_sizing.get('risk_warnings', []),
                sector_rotation_signal=causal_factory.distributional_flow_ledger.generate_flow_intelligence_summary().get(
                    'trading_implications', {}
                ).get('recommended_sectors', []) if causal_factory.distributional_flow_ledger else [],
                causal_validity_score=dpi_analysis.get('experiment_validation', {}).get('validation_strength', 0.0),
                counterfactual_robustness=position_sizing.get('causal_confidence', 0.5)
            )

            return signal

        # Enhanced systems dictionary
        enhanced_systems = existing_systems.copy()
        enhanced_systems.update({
            'causal_factory': causal_factory,
            'enhanced_dpi_calculator': enhanced_dpi_calculator,
            'enhanced_risk_manager': enhanced_risk_manager,
            'generate_causally_enhanced_signal': generate_causally_enhanced_signal,
            'causal_integration_active': True
        })

        logger.info("Causal intelligence integration completed successfully")
        return enhanced_systems

    except Exception as e:
        logger.error(f"Error integrating causal intelligence: {e}")
        raise


# Convenience function for quick integration
def create_super_gary_system(phase2_factory, causal_config: Optional[CausalIntelligenceConfig] = None):
    """
    Create the complete 'Super-Gary' system with causal intelligence

    This is the main entry point for creating the enhanced trading system
    with all causal intelligence components integrated.

    Args:
        phase2_factory: Existing Phase2SystemFactory
        causal_config: Optional causal intelligence configuration

    Returns:
        Complete enhanced trading system
    """
    logger.info("Creating Super-Gary system with causal intelligence")

    # Use default config if none provided
    if causal_config is None:
        causal_config = CausalIntelligenceConfig()

    # Integrate causal intelligence
    enhanced_systems = integrate_causal_intelligence_with_phase2(phase2_factory, causal_config)

    logger.info("Super-Gary system created successfully")
    return enhanced_systems