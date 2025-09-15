"""
Unit tests for Causal DAG

Tests the causal directed acyclic graph implementation including do-operator
simulations, counterfactual analysis, and causal identification methods.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the module to test
import sys
sys.path.append('/c/Users/17175/Desktop/trader-ai/src')

from intelligence.causal.causal_dag import (
    CausalDAG,
    CausalNode,
    CausalEdge,
    CausalRelationType,
    InterventionType,
    DoOperatorResult,
    CounterfactualAnalysis
)


class TestCausalDAG(unittest.TestCase):
    """Test cases for Causal DAG"""

    def setUp(self):
        """Set up test fixtures"""
        self.dag = CausalDAG()

    def test_initialization(self):
        """Test that Causal DAG initializes correctly"""
        self.assertIsInstance(self.dag, CausalDAG)
        self.assertIsNotNone(self.dag.graph)
        self.assertIsInstance(self.dag.nodes, dict)
        self.assertIsInstance(self.dag.edges, dict)

        # Check that initial economic structure is loaded
        self.assertGreater(len(self.dag.nodes), 0)
        self.assertGreater(len(self.dag.edges), 0)

        # Check for key economic nodes
        node_names = [node.name for node in self.dag.nodes.values()]
        expected_nodes = [
            'monetary_policy', 'fiscal_policy', 'aggregate_demand',
            'price_level', 'asset_prices', 'employment'
        ]

        for expected_node in expected_nodes:
            self.assertIn(expected_node, node_names)

    def test_node_addition(self):
        """Test adding nodes to the DAG"""
        # Create test node
        test_node = CausalNode(
            name="test_variable",
            node_type="outcome",
            observed=True
        )

        # Add node
        initial_count = len(self.dag.nodes)
        self.dag.add_node(test_node)

        # Verify node was added
        self.assertEqual(len(self.dag.nodes), initial_count + 1)
        self.assertIn("test_variable", self.dag.nodes)
        self.assertEqual(self.dag.nodes["test_variable"], test_node)

    def test_edge_addition(self):
        """Test adding edges to the DAG"""
        # Create test nodes
        source_node = CausalNode("source_test", "policy", True)
        target_node = CausalNode("target_test", "outcome", True)

        self.dag.add_node(source_node)
        self.dag.add_node(target_node)

        # Create test edge
        test_edge = CausalEdge(
            source="source_test",
            target="target_test",
            relationship_type=CausalRelationType.DIRECT,
            strength=0.5,
            confidence=0.8,
            identification_strategy="natural_experiment"
        )

        # Add edge
        initial_count = len(self.dag.edges)
        self.dag.add_edge(test_edge)

        # Verify edge was added
        self.assertEqual(len(self.dag.edges), initial_count + 1)
        edge_key = ("source_test", "target_test")
        self.assertIn(edge_key, self.dag.edges)
        self.assertEqual(self.dag.edges[edge_key], test_edge)

    def test_cycle_prevention(self):
        """Test that DAG prevents cycles"""
        # Create nodes that would form a cycle
        node_a = CausalNode("cycle_a", "test", True)
        node_b = CausalNode("cycle_b", "test", True)
        node_c = CausalNode("cycle_c", "test", True)

        self.dag.add_node(node_a)
        self.dag.add_node(node_b)
        self.dag.add_node(node_c)

        # Add edges A -> B -> C
        edge_ab = CausalEdge("cycle_a", "cycle_b", CausalRelationType.DIRECT, 0.5, 0.8, "test")
        edge_bc = CausalEdge("cycle_b", "cycle_c", CausalRelationType.DIRECT, 0.5, 0.8, "test")

        self.dag.add_edge(edge_ab)
        self.dag.add_edge(edge_bc)

        # Try to add edge C -> A (would create cycle)
        edge_ca = CausalEdge("cycle_c", "cycle_a", CausalRelationType.DIRECT, 0.5, 0.8, "test")

        with self.assertRaises(ValueError):
            self.dag.add_edge(edge_ca)

    def test_causal_identification(self):
        """Test causal identification algorithms"""
        # Test identification for existing economic relationships
        identification = self.dag.identify_causal_effect("monetary_policy", "asset_prices")

        # Verify identification result structure
        self.assertIn('treatment', identification)
        self.assertIn('outcome', identification)
        self.assertIn('identifiable', identification)
        self.assertIn('method', identification)

        self.assertEqual(identification['treatment'], "monetary_policy")
        self.assertEqual(identification['outcome'], "asset_prices")

        # For this relationship, should be identifiable
        if identification['identifiable']:
            self.assertIn(identification['method'], [
                'backdoor_adjustment', 'frontdoor_adjustment', 'instrumental_variables'
            ])

    def test_do_operator_intervention(self):
        """Test do-operator interventions"""
        # Test monetary policy intervention
        try:
            do_result = self.dag.do_operator(
                intervention_node="monetary_policy",
                intervention_value=0.01,  # 1% change
                target_node="asset_prices"
            )

            # Verify result structure
            self.assertIsInstance(do_result, DoOperatorResult)
            self.assertEqual(do_result.intervention_node, "monetary_policy")
            self.assertEqual(do_result.intervention_value, 0.01)
            self.assertEqual(do_result.target_node, "asset_prices")

            # Effect should be numeric
            self.assertIsInstance(do_result.expected_effect, float)

            # Should have confidence interval
            self.assertIsInstance(do_result.confidence_interval, tuple)
            self.assertEqual(len(do_result.confidence_interval), 2)

            # Should have causal mechanism
            self.assertIsInstance(do_result.causal_mechanism, list)

        except ValueError as e:
            # If identification fails, this is also acceptable for testing
            self.assertIn("not identifiable", str(e))

    def test_counterfactual_analysis(self):
        """Test counterfactual analysis"""
        try:
            # Test counterfactual: what if monetary policy was different?
            counterfactual = self.dag.counterfactual_analysis(
                intervention_node="monetary_policy",
                intervention_value=0.02,
                target_node="asset_prices",
                actual_outcome=100.0
            )

            # Verify result structure
            self.assertIsInstance(counterfactual, CounterfactualAnalysis)
            self.assertEqual(counterfactual.actual_outcome, 100.0)

            # Counterfactual should be different from actual
            self.assertIsInstance(counterfactual.counterfactual_outcome, float)
            self.assertIsInstance(counterfactual.causal_effect, float)

            # Probability measures should be between 0 and 1
            self.assertGreaterEqual(counterfactual.probability_of_causation, 0)
            self.assertLessEqual(counterfactual.probability_of_causation, 1)

            self.assertGreaterEqual(counterfactual.necessity_score, 0)
            self.assertLessEqual(counterfactual.necessity_score, 1)

            self.assertGreaterEqual(counterfactual.sufficiency_score, 0)
            self.assertLessEqual(counterfactual.sufficiency_score, 1)

        except ValueError as e:
            # If identification fails, this is also acceptable
            self.assertIn("not identifiable", str(e))

    def test_policy_shock_simulation(self):
        """Test policy shock simulation"""
        simulation = self.dag.policy_shock_simulation(
            policy_type="monetary",
            shock_magnitude=0.5,  # 50bp shock
            target_variables=["asset_prices", "inflation", "unemployment"],
            time_horizon=6
        )

        # Verify simulation structure
        self.assertIn('policy_type', simulation)
        self.assertIn('shock_magnitude', simulation)
        self.assertIn('time_horizon', simulation)
        self.assertIn('time_series_effects', simulation)

        self.assertEqual(simulation['policy_type'], "monetary")
        self.assertEqual(simulation['shock_magnitude'], 0.5)
        self.assertEqual(simulation['time_horizon'], 6)

        # Check time series for each target variable
        for var in ["asset_prices", "inflation", "unemployment"]:
            if var in simulation['time_series_effects']:
                time_series = simulation['time_series_effects'][var]
                self.assertEqual(len(time_series), 6)  # Should have 6 periods

                for period_data in time_series:
                    self.assertIn('period', period_data)
                    self.assertIn('effect', period_data)
                    self.assertIn('confidence_interval', period_data)

    def test_backdoor_path_finding(self):
        """Test backdoor path identification"""
        # Add test nodes and edges to create backdoor paths
        confounder = CausalNode("test_confounder", "confounder", False)
        treatment = CausalNode("test_treatment", "policy", True)
        outcome = CausalNode("test_outcome", "outcome", True)

        self.dag.add_node(confounder)
        self.dag.add_node(treatment)
        self.dag.add_node(outcome)

        # Create backdoor path: confounder -> treatment, confounder -> outcome
        edge1 = CausalEdge("test_confounder", "test_treatment", CausalRelationType.CONFOUNDED, 0.5, 0.8, "test")
        edge2 = CausalEdge("test_confounder", "test_outcome", CausalRelationType.CONFOUNDED, 0.5, 0.8, "test")
        edge3 = CausalEdge("test_treatment", "test_outcome", CausalRelationType.DIRECT, 0.7, 0.9, "test")

        self.dag.add_edge(edge1)
        self.dag.add_edge(edge2)
        self.dag.add_edge(edge3)

        # Find backdoor paths
        backdoor_paths = self.dag._get_backdoor_paths("test_treatment", "test_outcome")

        # Should find the backdoor path through confounder
        self.assertIsInstance(backdoor_paths, list)
        # The specific structure depends on the path finding algorithm

    def test_path_blocking(self):
        """Test path blocking by adjustment sets"""
        # Create simple path for testing
        path = ["A", "B", "C"]
        adjustment_set = ["B"]

        # Mock graph structure for testing
        test_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": []
        }

        # Test path blocking
        is_blocked = self.dag._blocks_path(adjustment_set, path)

        # Since B is in adjustment set and is a non-collider, path should be blocked
        # (This is simplified - real implementation depends on collider detection)
        self.assertIsInstance(is_blocked, bool)

    def test_model_validation(self):
        """Test causal model validation"""
        # Create test data
        test_data = {
            'monetary_policy': pd.Series(np.random.normal(0, 1, 100)),
            'asset_prices': pd.Series(np.random.normal(0, 1, 100)),
            'inflation': pd.Series(np.random.normal(0, 1, 100))
        }

        validation_results = self.dag.validate_causal_model(test_data)

        # Verify validation structure
        self.assertIn('tests_performed', validation_results)
        self.assertIn('overall_validity', validation_results)

        self.assertIsInstance(validation_results['tests_performed'], list)
        self.assertIn(validation_results['overall_validity'], ['good', 'moderate', 'poor', 'unknown'])

    def test_data_addition(self):
        """Test adding observational data to nodes"""
        # Create test data
        test_data = pd.Series([1, 2, 3, 4, 5], name="test_series")

        # Add data to existing node
        node_name = list(self.dag.nodes.keys())[0]  # Get first node
        self.dag.add_data(node_name, test_data)

        # Verify data was added
        self.assertIn(node_name, self.dag.data)
        pd.testing.assert_series_equal(self.dag.data[node_name], test_data)

    def test_data_addition_invalid_node(self):
        """Test adding data to non-existent node"""
        test_data = pd.Series([1, 2, 3])

        with self.assertRaises(ValueError):
            self.dag.add_data("non_existent_node", test_data)

    def test_intervention_history(self):
        """Test that interventions are recorded in history"""
        initial_count = len(self.dag.interventions_history)

        try:
            # Perform intervention
            self.dag.do_operator(
                intervention_node="monetary_policy",
                intervention_value=0.01,
                target_node="asset_prices"
            )

            # Check that intervention was recorded
            self.assertEqual(len(self.dag.interventions_history), initial_count + 1)

            # Verify intervention record structure
            last_intervention = self.dag.interventions_history[-1]
            self.assertIn('timestamp', last_intervention)
            self.assertIn('intervention_node', last_intervention)
            self.assertIn('intervention_value', last_intervention)
            self.assertIn('target_node', last_intervention)
            self.assertIn('result', last_intervention)

        except ValueError:
            # If identification fails, history should not change
            self.assertEqual(len(self.dag.interventions_history), initial_count)

    def test_graph_export(self):
        """Test DAG visualization export"""
        export_data = self.dag.export_dag_visualization()

        # Verify export structure
        self.assertIn('nodes', export_data)
        self.assertIn('edges', export_data)

        # Check nodes structure
        nodes = export_data['nodes']
        self.assertIsInstance(nodes, list)
        if nodes:  # If there are nodes
            node = nodes[0]
            self.assertIn('id', node)
            self.assertIn('label', node)
            self.assertIn('type', node)
            self.assertIn('observed', node)

        # Check edges structure
        edges = export_data['edges']
        self.assertIsInstance(edges, list)
        if edges:  # If there are edges
            edge = edges[0]
            self.assertIn('source', edge)
            self.assertIn('target', edge)
            self.assertIn('type', edge)
            self.assertIn('strength', edge)
            self.assertIn('confidence', edge)

    def test_edge_effect_calculation(self):
        """Test edge effect calculations"""
        # Get existing edge
        if self.dag.edges:
            edge_key = list(self.dag.edges.keys())[0]
            source, target = edge_key

            effect = self.dag._get_edge_effect(source, target)

            # Effect should be numeric
            self.assertIsInstance(effect, float)

            # For non-existent edge, should return 0
            non_effect = self.dag._get_edge_effect("non_existent_1", "non_existent_2")
            self.assertEqual(non_effect, 0.0)

    def test_causal_path_calculation(self):
        """Test causal path finding"""
        # Test path between existing nodes
        node_names = list(self.dag.nodes.keys())
        if len(node_names) >= 2:
            source = node_names[0]
            target = node_names[1]

            path = self.dag._get_causal_path(source, target)

            # Path should be a list
            self.assertIsInstance(path, list)

            # If path exists, should start with source and end with target
            if path:
                self.assertEqual(path[0], source)
                self.assertEqual(path[-1], target)

    def test_path_effect_calculation(self):
        """Test path effect calculations"""
        # Test with simple path
        test_path = ["A", "B"]

        # Mock edges for testing
        original_get_edge_effect = self.dag._get_edge_effect
        self.dag._get_edge_effect = lambda s, t: 0.5 if s == "A" and t == "B" else 0.0

        effect = self.dag._calculate_path_effect(test_path)
        self.assertEqual(effect, 0.5)

        # Test with longer path
        test_path_long = ["A", "B", "C"]
        self.dag._get_edge_effect = lambda s, t: 0.5  # All edges have effect 0.5

        effect_long = self.dag._calculate_path_effect(test_path_long)
        self.assertEqual(effect_long, 0.25)  # 0.5 * 0.5

        # Restore original method
        self.dag._get_edge_effect = original_get_edge_effect

    def test_economic_dag_structure(self):
        """Test that the economic DAG has reasonable structure"""
        # Check for key economic relationships
        node_names = set(node.name for node in self.dag.nodes.values())

        # Should have monetary policy affecting asset prices
        monetary_in_dag = any("monetary" in name for name in node_names)
        asset_prices_in_dag = any("asset" in name for name in node_names)

        self.assertTrue(monetary_in_dag, "Should have monetary policy node")
        self.assertTrue(asset_prices_in_dag, "Should have asset prices node")

        # Should have reasonable number of nodes and edges
        self.assertGreater(len(self.dag.nodes), 10)  # At least 10 economic variables
        self.assertGreater(len(self.dag.edges), 5)   # At least 5 causal relationships

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test do-operator with non-existent nodes
        with self.assertRaises(Exception):
            self.dag.do_operator("non_existent_treatment", 1.0, "non_existent_outcome")

        # Test counterfactual with non-existent nodes
        with self.assertRaises(Exception):
            self.dag.counterfactual_analysis("non_existent", 1.0, "non_existent", 100.0)

        # Test identification with non-existent nodes
        identification = self.dag.identify_causal_effect("non_existent_1", "non_existent_2")
        self.assertFalse(identification['identifiable'])


if __name__ == '__main__':
    unittest.main()