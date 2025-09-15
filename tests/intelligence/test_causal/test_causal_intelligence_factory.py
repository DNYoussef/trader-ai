"""
Unit tests for Causal Intelligence Factory

Tests the integration of all causal intelligence components with the existing
Phase2SystemFactory and enhanced trading signal generation.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the modules to test
import sys
sys.path.append('/c/Users/17175/Desktop/trader-ai/src')

from intelligence.causal_intelligence_factory import (
    CausalIntelligenceFactory,
    CausalIntelligenceConfig
)


class TestCausalIntelligenceFactory(unittest.TestCase):
    """Test cases for Causal Intelligence Factory"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Phase2SystemFactory
        self.mock_phase2_factory = Mock()
        self.mock_phase2_factory.phase2_systems = {
            'kill_switch': Mock(),
            'kelly_calculator': Mock(),
            'evt_engine': Mock()
        }
        self.mock_phase2_factory.get_integrated_system.return_value = {
            'dpi_calculator': Mock(),
            'kelly_calculator': Mock(),
            'portfolio_manager': Mock(),
            'broker': Mock(),
            'market_data': Mock()
        }

        # Create factory with test configuration
        self.config = CausalIntelligenceConfig(
            enable_distributional_flows=True,
            enable_causal_dag=True,
            enable_hank_model=True,
            enable_synthetic_controls=True,
            enable_experiments_registry=True,
            dfl_db_path=":memory:",
            registry_db_path=":memory:"
        )

        self.factory = CausalIntelligenceFactory(
            phase2_factory=self.mock_phase2_factory,
            config=self.config
        )

    def test_initialization(self):
        """Test that factory initializes correctly"""
        self.assertIsInstance(self.factory, CausalIntelligenceFactory)
        self.assertEqual(self.factory.phase2_factory, self.mock_phase2_factory)
        self.assertEqual(self.factory.config, self.config)
        self.assertFalse(self.factory.initialized)

    def test_causal_systems_initialization(self):
        """Test initialization of all causal intelligence systems"""
        systems = self.factory.initialize_causal_systems()

        # Verify all systems were initialized
        expected_systems = [
            'distributional_flow_ledger',
            'causal_dag',
            'hank_model',
            'synthetic_control_validator',
            'experiments_registry'
        ]

        for system_name in expected_systems:
            self.assertIn(system_name, systems)
            self.assertIsNotNone(systems[system_name])

        # Verify factory state
        self.assertTrue(self.factory.initialized)
        self.assertIsNotNone(self.factory.distributional_flow_ledger)
        self.assertIsNotNone(self.factory.causal_dag)
        self.assertIsNotNone(self.factory.hank_model)
        self.assertIsNotNone(self.factory.synthetic_control_validator)
        self.assertIsNotNone(self.factory.experiments_registry)

    def test_selective_system_initialization(self):
        """Test initialization with some systems disabled"""
        # Create config with some systems disabled
        selective_config = CausalIntelligenceConfig(
            enable_distributional_flows=True,
            enable_causal_dag=False,
            enable_hank_model=True,
            enable_synthetic_controls=False,
            enable_experiments_registry=True
        )

        selective_factory = CausalIntelligenceFactory(
            phase2_factory=self.mock_phase2_factory,
            config=selective_config
        )

        systems = selective_factory.initialize_causal_systems()

        # Verify only enabled systems were initialized
        self.assertIn('distributional_flow_ledger', systems)
        self.assertNotIn('causal_dag', systems)
        self.assertIn('hank_model', systems)
        self.assertNotIn('synthetic_control_validator', systems)
        self.assertIn('experiments_registry', systems)

    @patch('intelligence.causal_intelligence_factory.CausalIntelligenceFactory._integrate_with_existing_systems')
    def test_integration_with_existing_systems(self, mock_integrate):
        """Test integration with existing Phase2 systems"""
        # Initialize systems (which should call integration)
        self.factory.initialize_causal_systems()

        # Verify integration was called
        mock_integrate.assert_called_once()

    def test_enhanced_trading_signal_generation(self):
        """Test enhanced trading signal generation"""
        # Initialize systems first
        self.factory.initialize_causal_systems()

        # Mock DPI calculator response
        mock_dpi_calculator = self.factory.phase2_factory.get_integrated_system()['dpi_calculator']
        mock_dpi_calculator.calculate_dpi.return_value = (0.5, Mock())

        # Create test market data
        test_market_data = {
            'price': pd.Series([100, 101, 102, 103, 104]),
            'volume': pd.Series([1000, 1100, 900, 1200, 1050])
        }

        # Generate enhanced signals
        signals = self.factory.enhanced_trading_signal_generation("TEST", test_market_data)

        # Verify signal structure
        expected_keys = [
            'symbol', 'timestamp', 'base_dpi_score', 'causal_context',
            'enhanced_score', 'confidence_level', 'recommendations'
        ]

        for key in expected_keys:
            self.assertIn(key, signals)

        # Verify signal values
        self.assertEqual(signals['symbol'], "TEST")
        self.assertIsInstance(signals['timestamp'], datetime)
        self.assertIsInstance(signals['base_dpi_score'], float)
        self.assertIsInstance(signals['enhanced_score'], float)
        self.assertIsInstance(signals['confidence_level'], float)
        self.assertIsInstance(signals['recommendations'], list)

        # Enhanced score should be within valid range
        self.assertGreaterEqual(signals['enhanced_score'], -1.0)
        self.assertLessEqual(signals['enhanced_score'], 1.0)

        # Confidence level should be between 0 and 1
        self.assertGreaterEqual(signals['confidence_level'], 0.0)
        self.assertLessEqual(signals['confidence_level'], 1.0)

    def test_enhanced_signals_without_initialization(self):
        """Test that enhanced signals fail without initialization"""
        test_market_data = {'price': pd.Series([100, 101, 102])}

        with self.assertRaises(ValueError):
            self.factory.enhanced_trading_signal_generation("TEST", test_market_data)

    def test_policy_impact_analysis(self):
        """Test comprehensive policy impact analysis"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Test policy impact analysis
        analysis = self.factory.perform_policy_impact_analysis(
            policy_type="monetary",
            policy_magnitude=0.5,
            affected_symbols=["SPY", "TLT", "GLD"]
        )

        # Verify analysis structure
        expected_keys = [
            'policy_type', 'policy_magnitude', 'analysis_timestamp',
            'affected_symbols', 'hank_simulation', 'causal_dag_analysis',
            'distributional_effects', 'trading_implications'
        ]

        for key in expected_keys:
            self.assertIn(key, analysis)

        # Verify analysis values
        self.assertEqual(analysis['policy_type'], "monetary")
        self.assertEqual(analysis['policy_magnitude'], 0.5)
        self.assertEqual(analysis['affected_symbols'], ["SPY", "TLT", "GLD"])

        # Verify trading implications for each symbol
        for symbol in ["SPY", "TLT", "GLD"]:
            self.assertIn(symbol, analysis['trading_implications'])

    def test_trading_hypothesis_validation(self):
        """Test trading hypothesis validation"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Create test data
        test_data = pd.DataFrame({
            'unit': ['A', 'A', 'A', 'B', 'B', 'B'],
            'date': pd.date_range('2023-01-01', periods=6),
            'treatment': [0, 0, 1, 0, 0, 1],
            'outcome': [1, 2, 3, 1.5, 2.5, 3.5]
        })

        # Validate hypothesis
        validation = self.factory.validate_trading_hypothesis(
            hypothesis="Interest rate changes affect stock returns",
            data=test_data,
            treatment_variable="interest_rate",
            outcome_variable="stock_return"
        )

        # Verify validation structure
        expected_keys = [
            'hypothesis', 'treatment_variable', 'outcome_variable',
            'validation_timestamp', 'overall_validity', 'confidence_score',
            'recommendations'
        ]

        for key in expected_keys:
            self.assertIn(key, validation)

        # Verify validation values
        self.assertEqual(validation['hypothesis'], "Interest rate changes affect stock returns")
        self.assertEqual(validation['treatment_variable'], "interest_rate")
        self.assertEqual(validation['outcome_variable'], "stock_return")
        self.assertIn(validation['overall_validity'], ['strong', 'moderate', 'weak', 'unknown'])
        self.assertIsInstance(validation['confidence_score'], float)
        self.assertIsInstance(validation['recommendations'], list)

    def test_causal_intelligence_summary(self):
        """Test comprehensive causal intelligence summary"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        summary = self.factory.get_causal_intelligence_summary()

        # Verify summary structure
        expected_sections = [
            'timestamp', 'systems_status', 'system_statistics',
            'recent_analysis', 'integration_status'
        ]

        for section in expected_sections:
            self.assertIn(section, summary)

        # Verify systems status
        systems_status = summary['systems_status']
        self.assertTrue(systems_status['initialized'])
        self.assertTrue(systems_status['distributional_flow_ledger'])
        self.assertTrue(systems_status['causal_dag'])
        self.assertTrue(systems_status['hank_model'])
        self.assertTrue(systems_status['synthetic_control_validator'])
        self.assertTrue(systems_status['experiments_registry'])

    def test_portfolio_implications_extraction(self):
        """Test extraction of portfolio implications from HANK simulation"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Create mock simulation results
        mock_states = []
        for i in range(12):
            mock_state = Mock()
            mock_state.aggregate_consumption = 100 + i * 2  # Increasing consumption
            mock_state.inflation = 0.02 + i * 0.001       # Slightly increasing inflation
            mock_state.unemployment = 0.05 - i * 0.001    # Slightly decreasing unemployment
            mock_states.append(mock_state)

        implications = self.factory._extract_portfolio_implications(mock_states)

        # Verify implications structure
        expected_keys = [
            'consumption_impact', 'inflation_impact', 'unemployment_impact',
            'sector_recommendations', 'risk_level', 'time_horizon'
        ]

        for key in expected_keys:
            self.assertIn(key, implications)

        # Verify values are reasonable
        self.assertIsInstance(implications['consumption_impact'], float)
        self.assertIsInstance(implications['inflation_impact'], float)
        self.assertIsInstance(implications['unemployment_impact'], float)
        self.assertIsInstance(implications['sector_recommendations'], list)
        self.assertIn(implications['risk_level'], ['high', 'moderate', 'low'])
        self.assertEqual(implications['time_horizon'], len(mock_states))

    def test_causal_context_analysis(self):
        """Test causal context analysis for trading decisions"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        test_market_data = {
            'price': pd.Series([100, 101, 102]),
            'volume': pd.Series([1000, 1100, 900])
        }

        context = self.factory._analyze_causal_context("TEST", test_market_data)

        # Verify context structure
        expected_keys = [
            'policy_environment', 'natural_experiments',
            'causal_risks', 'validation_strength'
        ]

        for key in expected_keys:
            self.assertIn(key, context)

        # Verify context values
        self.assertIsInstance(context['policy_environment'], dict)
        self.assertIsInstance(context['natural_experiments'], list)
        self.assertIsInstance(context['causal_risks'], list)
        self.assertIsInstance(context['validation_strength'], float)

        # Validation strength should be between 0 and 1
        self.assertGreaterEqual(context['validation_strength'], 0.0)
        self.assertLessEqual(context['validation_strength'], 1.0)

    def test_enhanced_score_calculation(self):
        """Test enhanced score calculation with causal factors"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        base_dpi = 0.5
        distributional_integration = {'distributional_factor': 1.2}
        causal_context = {
            'validation_strength': 0.8,
            'causal_risks': ['risk1', 'risk2']
        }

        enhanced_score = self.factory._calculate_enhanced_score(
            base_dpi, distributional_integration, causal_context
        )

        # Score should be numeric and in valid range
        self.assertIsInstance(enhanced_score, float)
        self.assertGreaterEqual(enhanced_score, -1.0)
        self.assertLessEqual(enhanced_score, 1.0)

        # Score should be different from base DPI due to adjustments
        self.assertNotEqual(enhanced_score, base_dpi)

    def test_confidence_level_calculation(self):
        """Test confidence level calculation"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Test high confidence scenario
        high_confidence_context = {
            'validation_strength': 0.9,
            'causal_risks': [],
            'natural_experiments': [{'quality': 0.8}, {'quality': 0.9}]
        }

        confidence = self.factory._calculate_confidence_level(high_confidence_context)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.1)
        self.assertLessEqual(confidence, 0.95)

        # Test low confidence scenario
        low_confidence_context = {
            'validation_strength': 0.2,
            'causal_risks': ['risk1', 'risk2', 'risk3'],
            'natural_experiments': []
        }

        low_confidence = self.factory._calculate_confidence_level(low_confidence_context)
        self.assertLess(low_confidence, confidence)  # Should be lower than high confidence

    def test_causal_recommendations_generation(self):
        """Test generation of causal-informed recommendations"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Test strong signal scenario
        strong_context = {
            'validation_strength': 0.9,
            'causal_risks': [],
            'policy_environment': {'surprise_level': 0.2},
            'natural_experiments': [{'quality_score': 0.9}]
        }

        recommendations = self.factory._generate_causal_recommendations(
            "TEST", 0.7, strong_context
        )

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Should contain positive recommendations for strong signal
        recommendation_text = ' '.join(recommendations).lower()
        self.assertTrue(any(keyword in recommendation_text
                          for keyword in ['long', 'confidence', 'strong']))

    def test_symbol_policy_impact_analysis(self):
        """Test symbol-specific policy impact analysis"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Mock analysis results
        mock_analysis = {
            'hank_simulation': {
                'consumption_impact': [100, 102, 104],
                'inflation_impact': [0.02, 0.025, 0.03]
            },
            'historical_precedents': [
                {'effect_sizes': {'stock_returns': 0.05}}
            ]
        }

        # Test consumer stock
        consumer_implications = self.factory._analyze_symbol_policy_impact(
            "consumer_discretionary_etf", "fiscal", 0.5, mock_analysis
        )

        expected_keys = [
            'expected_direction', 'magnitude_estimate', 'confidence',
            'time_horizon', 'risk_factors'
        ]

        for key in expected_keys:
            self.assertIn(key, consumer_implications)

        self.assertIn(consumer_implications['expected_direction'],
                      ['positive', 'negative', 'neutral'])
        self.assertIsInstance(consumer_implications['magnitude_estimate'], float)
        self.assertIsInstance(consumer_implications['risk_factors'], list)

    def test_causal_risk_assessment(self):
        """Test assessment of causal risks"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        test_market_data = {'correlation_high': 'test'}  # Mock high correlation signal

        risks = self.factory._assess_causal_risks("TEST", test_market_data)

        self.assertIsInstance(risks, list)
        # Should identify some risks based on the system state

    def test_hypothesis_validity_calculation(self):
        """Test hypothesis validity score calculation"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Test strong validity case
        strong_validation = {
            'causal_dag_validation': {'identifiable': True},
            'natural_experiments_match': [
                {'quality_score': 0.9},
                {'quality_score': 0.8}
            ],
            'synthetic_control_analysis': {'status': 'completed'}
        }

        strong_score = self.factory._calculate_hypothesis_validity(strong_validation)

        # Test weak validity case
        weak_validation = {
            'causal_dag_validation': {'identifiable': False},
            'natural_experiments_match': [],
            'synthetic_control_analysis': None
        }

        weak_score = self.factory._calculate_hypothesis_validity(weak_validation)

        # Strong should score higher than weak
        self.assertGreater(strong_score, weak_score)
        self.assertGreaterEqual(strong_score, 0.0)
        self.assertLessEqual(strong_score, 1.0)

    def test_validation_recommendations_generation(self):
        """Test generation of validation recommendations"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        validation_results = {
            'overall_validity': 'strong',
            'confidence_score': 0.8,
            'causal_dag_validation': {'identifiable': True},
            'natural_experiments_match': [
                {'quality_score': 0.9}
            ]
        }

        recommendations = self.factory._generate_validation_recommendations(validation_results)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Should have positive recommendations for strong validity
        recommendation_text = ' '.join(recommendations).lower()
        self.assertTrue(any(keyword in recommendation_text
                          for keyword in ['strong', 'confidence', 'proceed']))

    def test_cleanup(self):
        """Test proper cleanup of causal intelligence systems"""
        # Initialize systems
        self.factory.initialize_causal_systems()

        # Cleanup should not raise exceptions
        try:
            self.factory.close()
        except Exception as e:
            self.fail(f"Cleanup should not raise exceptions: {e}")

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test enhanced signals before initialization
        with self.assertRaises(ValueError):
            self.factory.enhanced_trading_signal_generation("TEST", {})

        # Test policy analysis before initialization
        with self.assertRaises(ValueError):
            self.factory.perform_policy_impact_analysis("monetary", 0.5, ["TEST"])

        # Test hypothesis validation before initialization
        with self.assertRaises(ValueError):
            self.factory.validate_trading_hypothesis(
                "test", pd.DataFrame(), "treatment", "outcome"
            )


if __name__ == '__main__':
    unittest.main()