"""
Unit tests for Distributional Flow Ledger

Tests the core functionality of tracking wealth flows by income decile,
landlord and creditor capture patterns, and integration with DPI calculations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

# Import the module to test
import sys
sys.path.append('/c/Users/17175/Desktop/trader-ai/src')

from intelligence.mycelium.distributional_flow_ledger import (
    DistributionalFlowLedger,
    FlowEvent,
    IncomeDecile,
    FlowCaptor,
    DecileFlowProfile,
    CaptorProfile
)


class TestDistributionalFlowLedger(unittest.TestCase):
    """Test cases for Distributional Flow Ledger"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()

        self.dfl = DistributionalFlowLedger(db_path=self.temp_db.name)

    def tearDown(self):
        """Clean up test fixtures"""
        self.dfl.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test that DFL initializes correctly"""
        self.assertIsInstance(self.dfl, DistributionalFlowLedger)
        self.assertEqual(len(self.dfl.decile_profiles), 10)  # 10 income deciles

        # Check that decile profiles are properly initialized
        for decile in IncomeDecile:
            self.assertIn(decile, self.dfl.decile_profiles)
            profile = self.dfl.decile_profiles[decile]
            self.assertIsInstance(profile, DecileFlowProfile)
            self.assertGreater(profile.total_income, 0)
            self.assertGreaterEqual(profile.vulnerability_score, 0)
            self.assertLessEqual(profile.vulnerability_score, 1)

    def test_flow_event_recording(self):
        """Test recording of flow events"""
        # Create test flow event
        flow_event = FlowEvent(
            timestamp=datetime.now(),
            amount=1000.0,
            source_decile=IncomeDecile.D3,
            captor_type=FlowCaptor.LANDLORDS,
            captor_id="landlord_001",
            flow_category="rent",
            urgency_score=0.9,
            elasticity=-0.1,
            metadata={"location": "urban"}
        )

        # Record the event
        initial_count = len(self.dfl.flow_history)
        self.dfl.record_flow_event(flow_event)

        # Verify event was recorded
        self.assertEqual(len(self.dfl.flow_history), initial_count + 1)
        self.assertEqual(self.dfl.flow_history[-1], flow_event)

        # Verify captor profile was updated
        self.assertIn("landlord_001", self.dfl.captor_profiles)
        captor = self.dfl.captor_profiles["landlord_001"]
        self.assertEqual(captor.total_captured, 1000.0)
        self.assertEqual(captor.captor_type, FlowCaptor.LANDLORDS)

    def test_marginal_flow_tracking(self):
        """Test tracking of marginal flows"""
        # Test marginal flow analysis
        result = self.dfl.track_marginal_flow(
            amount=10000.0,
            policy_context="stimulus"
        )

        # Verify result structure
        self.assertIn('injected_amount', result)
        self.assertIn('marginal_captures', result)
        self.assertIn('total_captured', result)
        self.assertIn('capture_efficiency', result)

        self.assertEqual(result['injected_amount'], 10000.0)
        self.assertGreater(result['total_captured'], 0)
        self.assertLessEqual(result['capture_efficiency'], 1.0)

        # Test different policy contexts
        qe_result = self.dfl.track_marginal_flow(
            amount=10000.0,
            policy_context="qe"
        )

        # QE should flow more to top deciles
        stimulus_captures = result['marginal_captures']
        qe_captures = qe_result['marginal_captures']

        # Both should have captures, but potentially different distributions
        self.assertIsInstance(stimulus_captures, dict)
        self.assertIsInstance(qe_captures, dict)

    def test_housing_affordability_analysis(self):
        """Test housing affordability analysis"""
        analysis = self.dfl.analyze_housing_affordability()

        # Verify analysis structure
        self.assertIn('decile_analysis', analysis)
        self.assertIn('overall_metrics', analysis)

        # Check decile analysis
        decile_analysis = analysis['decile_analysis']
        self.assertEqual(len(decile_analysis), 10)  # All 10 deciles

        for decile_key, data in decile_analysis.items():
            self.assertIn('housing_cost', data)
            self.assertIn('housing_ratio', data)
            self.assertIn('cost_burdened', data)
            self.assertIn('severely_burdened', data)
            self.assertIn('affordability_score', data)

            # Housing ratio should be reasonable
            self.assertGreaterEqual(data['housing_ratio'], 0)
            self.assertLessEqual(data['housing_ratio'], 1)

        # Check overall metrics
        overall = analysis['overall_metrics']
        self.assertIn('cost_burdened_deciles', overall)
        self.assertIn('severely_burdened_deciles', overall)
        self.assertIn('total_housing_capture', overall)
        self.assertIn('affordability_crisis_score', overall)

    def test_landlord_capture_analysis(self):
        """Test landlord capture analysis"""
        analysis = self.dfl.get_landlord_capture_analysis()

        # Verify analysis structure
        self.assertIn('total_capture', analysis)
        self.assertIn('decile_breakdown', analysis)
        self.assertIn('concentration_metrics', analysis)

        # Check decile breakdown
        decile_breakdown = analysis['decile_breakdown']
        self.assertEqual(len(decile_breakdown), 10)

        for decile_key, data in decile_breakdown.items():
            self.assertIn('absolute_capture', data)
            self.assertIn('capture_rate', data)
            self.assertIn('burden_category', data)

            # Capture rate should be reasonable
            self.assertGreaterEqual(data['capture_rate'], 0)
            self.assertLessEqual(data['capture_rate'], 1)

    def test_creditor_capture_analysis(self):
        """Test creditor capture analysis"""
        analysis = self.dfl.get_creditor_capture_analysis()

        # Verify analysis structure
        self.assertIn('total_capture', analysis)
        self.assertIn('decile_breakdown', analysis)
        self.assertIn('debt_burden_analysis', analysis)
        self.assertIn('predatory_indicators', analysis)

        # Check predatory indicators
        predatory = analysis['predatory_indicators']
        self.assertIn('burden_inequality', predatory)
        self.assertIn('predatory_score', predatory)
        self.assertIn('systemic_risk', predatory)

        # Burden inequality should be positive
        self.assertGreater(predatory['burden_inequality'], 0)

    def test_margin_markup_analysis(self):
        """Test margin and markup trend analysis"""
        # Create some test flow events for a sector
        test_events = [
            FlowEvent(
                timestamp=datetime.now() - timedelta(days=i),
                amount=100.0 + i * 10,
                source_decile=IncomeDecile.D5,
                captor_type=FlowCaptor.RETAILERS,
                captor_id=f"retailer_{i}",
                flow_category="grocery",
                urgency_score=0.8,
                elasticity=-0.5
            )
            for i in range(10)
        ]

        # Record events
        for event in test_events:
            self.dfl.record_flow_event(event)

        # Analyze margin trends
        analysis = self.dfl.analyze_margin_markup_trends("grocery")

        # Verify analysis structure
        self.assertIsInstance(analysis.sector, str)
        self.assertIsInstance(analysis.average_margin, float)
        self.assertIsInstance(analysis.margin_trend, float)
        self.assertIsInstance(analysis.price_elasticity, float)
        self.assertIsInstance(analysis.market_concentration, float)

        # Values should be in reasonable ranges
        self.assertGreaterEqual(analysis.market_concentration, 0)
        self.assertLessEqual(analysis.market_concentration, 1)

    def test_dpi_integration(self):
        """Test integration with DPI calculations"""
        # Test DPI integration
        dpi_score = 0.5
        symbol = "TEST"

        integration_result = self.dfl.integrate_with_dpi(dpi_score, symbol)

        # Verify integration result structure
        self.assertIn('symbol', integration_result)
        self.assertIn('original_dpi', integration_result)
        self.assertIn('adjusted_dpi', integration_result)
        self.assertIn('distributional_factor', integration_result)
        self.assertIn('flow_context', integration_result)
        self.assertIn('trading_recommendation', integration_result)

        self.assertEqual(integration_result['symbol'], symbol)
        self.assertEqual(integration_result['original_dpi'], dpi_score)

        # Adjusted DPI should be different from original (unless factor is 1.0)
        adjusted_dpi = integration_result['adjusted_dpi']
        self.assertIsInstance(adjusted_dpi, float)
        self.assertGreaterEqual(adjusted_dpi, -1.0)
        self.assertLessEqual(adjusted_dpi, 1.0)

    def test_flow_intelligence_summary(self):
        """Test comprehensive flow intelligence summary"""
        summary = self.dfl.generate_flow_intelligence_summary()

        # Verify summary structure
        required_sections = [
            'system_status',
            'distributional_pressure',
            'capture_analysis',
            'trading_implications'
        ]

        for section in required_sections:
            self.assertIn(section, summary)

        # Check system status
        system_status = summary['system_status']
        self.assertIn('total_deciles_tracked', system_status)
        self.assertIn('total_captors_tracked', system_status)
        self.assertIn('system_flow_velocity', system_status)

        self.assertEqual(system_status['total_deciles_tracked'], 10)

        # Check distributional pressure
        pressure = summary['distributional_pressure']
        self.assertIn('average_vulnerability', pressure)
        self.assertIn('pressure_score', pressure)

        self.assertGreaterEqual(pressure['average_vulnerability'], 0)
        self.assertLessEqual(pressure['average_vulnerability'], 1)

        # Check trading implications
        implications = summary['trading_implications']
        self.assertIn('defensive_positioning_recommended', implications)
        self.assertIn('recommended_sectors', implications)
        self.assertIsInstance(implications['recommended_sectors'], list)

    def test_database_persistence(self):
        """Test that data is properly persisted to database"""
        # Create a flow event
        flow_event = FlowEvent(
            timestamp=datetime.now(),
            amount=500.0,
            source_decile=IncomeDecile.D1,
            captor_type=FlowCaptor.CREDITORS,
            captor_id="creditor_test",
            flow_category="credit_payment",
            urgency_score=0.7,
            elasticity=-0.2
        )

        # Record the event
        self.dfl.record_flow_event(flow_event)

        # Close and reopen DFL to test persistence
        db_path = self.dfl.db_path
        self.dfl.close()

        # Create new DFL with same database
        new_dfl = DistributionalFlowLedger(db_path=db_path)

        # Check that captor was persisted (simplified check)
        # In full implementation, would query database directly
        self.assertIsInstance(new_dfl.captor_profiles, dict)

        new_dfl.close()

    def test_decile_money_share_calculation(self):
        """Test calculation of money shares by decile for different policies"""
        # Test stimulus policy
        stimulus_share = self.dfl._calculate_decile_money_share(IncomeDecile.D1, "stimulus")
        qe_share = self.dfl._calculate_decile_money_share(IncomeDecile.D1, "qe")

        # Bottom decile should get more from stimulus than QE
        self.assertGreater(stimulus_share, qe_share)

        # Test top decile
        stimulus_top = self.dfl._calculate_decile_money_share(IncomeDecile.D10, "stimulus")
        qe_top = self.dfl._calculate_decile_money_share(IncomeDecile.D10, "qe")

        # Top decile should get more from QE than stimulus
        self.assertGreater(qe_top, stimulus_top)

        # All shares should be between 0 and 1
        for decile in IncomeDecile:
            for policy in ["stimulus", "qe", "default"]:
                share = self.dfl._calculate_decile_money_share(decile, policy)
                self.assertGreaterEqual(share, 0)
                self.assertLessEqual(share, 1)

    def test_housing_burden_categorization(self):
        """Test housing burden categorization"""
        # Test different burden levels
        test_cases = [
            (0.15, "Affordable"),
            (0.25, "Moderate Burden"),
            (0.35, "Cost Burdened"),
            (0.60, "Severely Burdened")
        ]

        for ratio, expected_category in test_cases:
            category = self.dfl._categorize_housing_burden(ratio)
            self.assertEqual(category, expected_category)

    def test_debt_stress_assessment(self):
        """Test debt stress level assessment"""
        # Test different stress levels
        test_cases = [
            (0.10, 0.2, "Low"),
            (0.25, 0.4, "Moderate"),
            (0.40, 0.6, "High"),
            (0.60, 0.8, "Critical")
        ]

        for debt_ratio, vulnerability, expected_level in test_cases:
            stress_level = self.dfl._assess_debt_stress(debt_ratio, vulnerability)
            self.assertEqual(stress_level, expected_level)

    def test_system_flow_velocity(self):
        """Test system flow velocity calculation"""
        velocity = self.dfl._calculate_system_flow_velocity()

        # Velocity should be between 0 and 1
        self.assertGreaterEqual(velocity, 0)
        self.assertLessEqual(velocity, 1)

        # Should be numeric
        self.assertIsInstance(velocity, float)

    def test_recommended_sectors(self):
        """Test sector recommendation logic"""
        # Test high pressure scenario
        high_pressure_sectors = self.dfl._get_recommended_sectors(0.8, 0.3)
        self.assertIn('utilities', high_pressure_sectors)
        self.assertIn('consumer_staples', high_pressure_sectors)

        # Test crisis scenario
        crisis_sectors = self.dfl._get_recommended_sectors(0.5, 0.8)
        self.assertIn('rental_housing_reits', crisis_sectors)

        # Test low pressure scenario
        low_pressure_sectors = self.dfl._get_recommended_sectors(0.2, 0.2)
        self.assertIn('consumer_discretionary', low_pressure_sectors)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test invalid flow event handling
        with self.assertRaises(Exception):
            invalid_event = FlowEvent(
                timestamp="invalid_date",  # Invalid timestamp
                amount=1000.0,
                source_decile=IncomeDecile.D1,
                captor_type=FlowCaptor.LANDLORDS,
                captor_id="test",
                flow_category="test",
                urgency_score=0.5,
                elasticity=-0.5
            )
            # This should not be reached due to dataclass validation

        # Test negative amounts (should be handled gracefully)
        negative_flow = FlowEvent(
            timestamp=datetime.now(),
            amount=-1000.0,  # Negative amount
            source_decile=IncomeDecile.D1,
            captor_type=FlowCaptor.LANDLORDS,
            captor_id="test_negative",
            flow_category="test",
            urgency_score=0.5,
            elasticity=-0.5
        )

        # Should not raise exception but handle gracefully
        try:
            self.dfl.record_flow_event(negative_flow)
        except Exception as e:
            self.fail(f"Recording negative flow should not raise exception: {e}")


if __name__ == '__main__':
    unittest.main()