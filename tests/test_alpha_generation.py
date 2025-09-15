"""
Comprehensive Test Suite for Alpha Generation Systems

This test suite provides comprehensive testing for all Phase 5 alpha generation
components including unit tests, integration tests, and performance tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from intelligence.narrative.narrative_gap import (
    NarrativeGapEngine, SellSideNotesExtractor, MediaSentimentExtractor,
    DFLPricingEngine, NGSignal, MarketConsensus, DFLDistribution
)
from learning.shadow_book import (
    ShadowBookEngine, Trade, TradeType, ActionType, CounterfactualScenario
)
from learning.policy_twin import (
    PolicyTwin, EthicalTrade, AlphaType, SocialImpactCategory,
    StakeholderWelfareFramework, MarketEfficiencyFramework
)
from intelligence.alpha.alpha_integration import (
    AlphaIntegrationEngine, AlphaSignal, PortfolioState
)
from intelligence.alpha.backtesting_framework import (
    AlphaBacktester, BacktestConfig, MarketDataSimulator
)
from intelligence.alpha.realtime_pipeline import (
    RealTimePipeline, SimulatedDataFeed, MarketData
)

class TestNarrativeGapEngine:
    """Test suite for Narrative Gap Engine"""

    @pytest.fixture
    def ng_engine(self):
        return NarrativeGapEngine()

    @pytest.fixture
    def sample_consensus(self):
        return MarketConsensus(
            symbol="AAPL",
            price_target=160.0,
            time_horizon=30,
            confidence=0.8,
            consensus_strength=0.7,
            narrative_coherence=0.6
        )

    def test_ng_engine_initialization(self, ng_engine):
        """Test NG engine initialization"""
        assert ng_engine is not None
        assert len(ng_engine.consensus_extractors) == 2
        assert ng_engine.dfl_engine is not None

    @pytest.mark.asyncio
    async def test_calculate_narrative_gap(self, ng_engine):
        """Test narrative gap calculation"""
        symbol = "AAPL"
        current_price = 150.0

        signal = await ng_engine.calculate_narrative_gap(symbol, current_price)

        assert isinstance(signal, NGSignal)
        assert signal.symbol == symbol
        assert 0.0 <= signal.ng_score <= 1.0
        assert signal.gap_magnitude >= 0.0
        assert 0.0 <= signal.time_to_diffusion <= 1.0
        assert 0.0 <= signal.catalyst_proximity <= 1.0

    def test_sellside_extractor(self):
        """Test sell-side notes extractor"""
        extractor = SellSideNotesExtractor()

        # Test price target extraction
        text = "We are raising AAPL price target to $170 based on strong fundamentals"
        targets = extractor._extract_price_targets(text)

        assert len(targets) > 0
        assert 170.0 in targets

    def test_dfl_pricing_engine(self):
        """Test DFL pricing engine"""
        engine = DFLPricingEngine()

        distribution = engine.generate_dfl_distribution("AAPL", 150.0, 30)

        assert isinstance(distribution, DFLDistribution)
        assert distribution.symbol == "AAPL"
        assert len(distribution.expected_path) == 31  # 30 days + initial
        assert len(distribution.confidence_bands[0]) == 31
        assert len(distribution.confidence_bands[1]) == 31

    def test_gap_magnitude_calculation(self, ng_engine):
        """Test gap magnitude calculation"""
        market_path = np.array([100, 105, 110, 115, 120])
        dfl_path = np.array([100, 102, 108, 112, 118])

        gap = ng_engine._calculate_gap_magnitude(market_path, dfl_path)

        assert isinstance(gap, float)
        assert 0.0 <= gap <= 1.0

class TestShadowBookSystem:
    """Test suite for Shadow Book System"""

    @pytest.fixture
    def shadow_book(self):
        return ShadowBookEngine(db_path=":memory:")

    @pytest.fixture
    def sample_trade(self):
        return Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            trade_type=TradeType.ACTUAL,
            action_type=ActionType.ENTRY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            confidence=0.8
        )

    def test_shadow_book_initialization(self, shadow_book):
        """Test shadow book initialization"""
        assert shadow_book is not None
        assert len(shadow_book.positions) == 3  # actual, shadow, counterfactual
        assert shadow_book.trades == []

    def test_record_actual_trade(self, shadow_book, sample_trade):
        """Test recording actual trade"""
        result = shadow_book.record_actual_trade(sample_trade)

        assert result is True
        assert len(shadow_book.trades) == 1
        assert shadow_book.trades[0].trade_id == "TEST_001"

    def test_shadow_scenario_generation(self, shadow_book, sample_trade):
        """Test shadow scenario generation"""
        shadow_book.record_actual_trade(sample_trade)

        # Check that scenarios were created
        scenario_count = len(shadow_book.scenarios)
        assert scenario_count > 0

        # Check for expected scenario types
        scenario_names = [s.name for s in shadow_book.scenarios.values()]
        assert any("Larger Size" in name for name in scenario_names)
        assert any("Smaller Size" in name for name in scenario_names)

    def test_counterfactual_analysis(self, shadow_book, sample_trade):
        """Test counterfactual analysis"""
        shadow_book.record_actual_trade(sample_trade)

        # Get first scenario
        scenario_id = list(shadow_book.scenarios.keys())[0]

        analysis = shadow_book.analyze_counterfactual(scenario_id, sample_trade.trade_id)

        assert "scenario_id" in analysis
        assert "actual_performance" in analysis
        assert "counterfactual_performance" in analysis
        assert "performance_difference" in analysis

    def test_optimization_insights(self, shadow_book, sample_trade):
        """Test optimization insights generation"""
        shadow_book.record_actual_trade(sample_trade)

        insights = shadow_book.generate_optimization_insights()

        assert "insights" in insights
        assert "recommendations" in insights
        assert "performance_comparison" in insights

    def test_performance_metrics(self, shadow_book, sample_trade):
        """Test performance metrics calculation"""
        shadow_book.record_actual_trade(sample_trade)

        performance = shadow_book.get_shadow_book_performance("actual")

        assert "total_pnl" in performance
        assert "unrealized_pnl" in performance
        assert "realized_pnl" in performance

class TestPolicyTwin:
    """Test suite for Policy Twin System"""

    @pytest.fixture
    def policy_twin(self):
        return PolicyTwin()

    @pytest.fixture
    def sample_trade_data(self):
        return {
            "trade_id": "POLICY_TEST_001",
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 1000,
            "price": 150.0,
            "strategy_id": "momentum_strategy"
        }

    def test_policy_twin_initialization(self, policy_twin):
        """Test policy twin initialization"""
        assert policy_twin is not None
        assert len(policy_twin.ethical_frameworks) == 2

    def test_ethical_frameworks(self):
        """Test ethical frameworks"""
        stakeholder_framework = StakeholderWelfareFramework()
        efficiency_framework = MarketEfficiencyFramework()

        test_data = {
            "symbol": "AAPL",
            "strategy_id": "arbitrage_strategy",
            "quantity": 1000
        }

        # Test stakeholder framework
        alpha_type, impact = stakeholder_framework.evaluate_trade_ethics(test_data)
        assert isinstance(alpha_type, AlphaType)
        assert isinstance(impact, float)

        # Test efficiency framework
        alpha_type2, impact2 = efficiency_framework.evaluate_trade_ethics(test_data)
        assert isinstance(alpha_type2, AlphaType)
        assert isinstance(impact2, float)

    def test_trade_ethics_analysis(self, policy_twin, sample_trade_data):
        """Test trade ethics analysis"""
        ethical_trade = policy_twin.analyze_trade_ethics(sample_trade_data)

        assert isinstance(ethical_trade, EthicalTrade)
        assert ethical_trade.symbol == "AAPL"
        assert isinstance(ethical_trade.alpha_type, AlphaType)
        assert -1.0 <= ethical_trade.social_impact_score <= 1.0
        assert 0.0 <= ethical_trade.transparency_level <= 1.0

    def test_social_impact_metrics(self, policy_twin, sample_trade_data):
        """Test social impact metrics calculation"""
        # Analyze some trades first
        for i in range(5):
            trade_data = sample_trade_data.copy()
            trade_data["trade_id"] = f"TEST_{i}"
            policy_twin.analyze_trade_ethics(trade_data)

        metrics = policy_twin.calculate_social_impact_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) > 0

        for metric in metrics:
            assert hasattr(metric, 'category')
            assert hasattr(metric, 'metric_name')
            assert hasattr(metric, 'value')

    def test_policy_recommendations(self, policy_twin, sample_trade_data):
        """Test policy recommendations generation"""
        # Generate some trade history
        for i in range(10):
            trade_data = sample_trade_data.copy()
            trade_data["trade_id"] = f"POLICY_TEST_{i}"
            trade_data["quantity"] = np.random.uniform(500, 5000)
            policy_twin.analyze_trade_ethics(trade_data)

        recommendations = policy_twin.generate_policy_recommendations()

        assert isinstance(recommendations, list)
        # Should generate at least monitoring recommendation
        assert len(recommendations) >= 1

    def test_transparency_report(self, policy_twin, sample_trade_data):
        """Test transparency report generation"""
        # Generate some trade history
        for i in range(5):
            trade_data = sample_trade_data.copy()
            trade_data["trade_id"] = f"TRANSPARENCY_TEST_{i}"
            policy_twin.analyze_trade_ethics(trade_data)

        report = policy_twin.create_transparency_report()

        assert "report_period" in report
        assert "trade_summary" in report
        assert "impact_metrics" in report
        assert "transparency_initiatives" in report

class TestAlphaIntegration:
    """Test suite for Alpha Integration Engine"""

    @pytest.fixture
    def integration_engine(self):
        return AlphaIntegrationEngine()

    @pytest.fixture
    def sample_portfolio_state(self):
        return PortfolioState(
            total_capital=10_000_000,
            available_capital=2_000_000,
            current_positions={"AAPL": 500_000},
            risk_utilization=0.5,
            var_usage=0.4
        )

    def test_integration_engine_initialization(self, integration_engine):
        """Test integration engine initialization"""
        assert integration_engine is not None
        assert integration_engine.ng_engine is not None
        assert integration_engine.shadow_book is not None
        assert integration_engine.policy_twin is not None

    @pytest.mark.asyncio
    async def test_generate_integrated_signal(self, integration_engine, sample_portfolio_state):
        """Test integrated signal generation"""
        symbol = "AAPL"
        current_price = 150.0

        signal = await integration_engine.generate_integrated_signal(
            symbol, current_price, sample_portfolio_state
        )

        assert isinstance(signal, AlphaSignal)
        assert signal.symbol == symbol
        assert 0.0 <= signal.final_score <= 1.0
        assert signal.action in ["buy", "sell", "hold"]
        assert 0.0 <= signal.urgency <= 1.0

    def test_position_sizing_calculation(self, integration_engine, sample_portfolio_state):
        """Test position sizing calculation"""
        # Create mock NG signal
        class MockNGSignal:
            ng_score = 0.8
            confidence = 0.7
            gap_magnitude = 0.5

        ng_signal = MockNGSignal()

        sizing = integration_engine._calculate_position_sizing(ng_signal, sample_portfolio_state)

        assert "recommended_size" in sizing
        assert "max_size" in sizing
        assert "confidence_adjusted_size" in sizing
        assert sizing["recommended_size"] >= 0

    def test_risk_adjustments(self, integration_engine, sample_portfolio_state):
        """Test risk adjustments calculation"""
        # Create mock NG signal
        class MockNGSignal:
            ng_score = 0.8
            confidence = 0.7

        ng_signal = MockNGSignal()

        risk_adj = integration_engine._calculate_risk_adjustments(
            ng_signal, sample_portfolio_state, "AAPL"
        )

        assert "var_impact" in risk_adj
        assert "correlation_impact" in risk_adj
        assert "overall_risk_score" in risk_adj
        assert 0.0 <= risk_adj["overall_risk_score"] <= 1.0

    def test_portfolio_signals_generation(self, integration_engine, sample_portfolio_state):
        """Test portfolio-level signal generation"""
        symbols = ["AAPL", "MSFT"]
        current_prices = {"AAPL": 150.0, "MSFT": 300.0}

        signals = integration_engine.generate_portfolio_signals(
            symbols, current_prices, sample_portfolio_state
        )

        assert isinstance(signals, list)
        assert len(signals) <= len(symbols)

class TestBacktestingFramework:
    """Test suite for Backtesting Framework"""

    @pytest.fixture
    def backtest_config(self):
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),  # Short period for testing
            initial_capital=1_000_000,
            symbols=["AAPL", "MSFT"],
            transaction_costs=0.001
        )

    def test_market_data_simulator(self, backtest_config):
        """Test market data simulator"""
        simulator = MarketDataSimulator(
            backtest_config.symbols,
            backtest_config.start_date,
            backtest_config.end_date
        )

        # Test data generation
        assert len(simulator.data_cache) == len(backtest_config.symbols)

        # Test price data retrieval
        data = simulator.get_price_data("AAPL")
        assert len(data) > 0
        assert "close" in data.columns
        assert "returns" in data.columns

    def test_backtest_config_validation(self, backtest_config):
        """Test backtest configuration"""
        assert backtest_config.start_date < backtest_config.end_date
        assert backtest_config.initial_capital > 0
        assert len(backtest_config.symbols) > 0
        assert 0 <= backtest_config.transaction_costs <= 1

    @pytest.mark.asyncio
    async def test_backtest_execution(self, backtest_config):
        """Test backtest execution (abbreviated)"""
        backtester = AlphaBacktester(backtest_config)

        # Test initialization
        assert backtester.config == backtest_config
        assert backtester.market_data is not None
        assert backtester.alpha_engine is not None

class TestRealTimePipeline:
    """Test suite for Real-time Pipeline"""

    @pytest.fixture
    def pipeline_config(self):
        return {
            'symbols': ['AAPL', 'MSFT'],
            'initial_capital': 1_000_000,
            'data_frequency': 1.0,
            'signal_frequency': 5.0
        }

    def test_simulated_data_feed(self, pipeline_config):
        """Test simulated data feed"""
        feed = SimulatedDataFeed(pipeline_config['symbols'], 1.0)

        assert feed.symbols == pipeline_config['symbols']
        assert not feed.connected
        assert len(feed.subscribed_symbols) == 0

    @pytest.mark.asyncio
    async def test_data_feed_connection(self, pipeline_config):
        """Test data feed connection"""
        feed = SimulatedDataFeed(pipeline_config['symbols'], 1.0)

        # Test connection
        connected = await feed.connect()
        assert connected
        assert feed.connected

        # Test subscription
        subscribed = await feed.subscribe(pipeline_config['symbols'])
        assert subscribed

        # Test data stream (get one data point)
        async for data in feed.get_data_stream():
            assert isinstance(data, MarketData)
            assert data.symbol in pipeline_config['symbols']
            assert data.price > 0
            break  # Only test one data point

        # Test disconnection
        await feed.disconnect()
        assert not feed.connected

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization"""
        pipeline = RealTimePipeline(pipeline_config)

        assert pipeline.symbols == pipeline_config['symbols']
        assert not pipeline.running
        assert pipeline.alpha_engine is not None
        assert pipeline.data_feed is not None

class TestPerformanceAndStress:
    """Performance and stress tests for alpha generation systems"""

    @pytest.mark.asyncio
    async def test_ng_engine_performance(self):
        """Test NG engine performance with multiple symbols"""
        engine = NarrativeGapEngine()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        start_time = datetime.now()

        tasks = [
            engine.calculate_narrative_gap(symbol, np.random.uniform(100, 500))
            for symbol in symbols
        ]

        signals = await asyncio.gather(*tasks)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        assert len(signals) == len(symbols)
        assert execution_time < 10.0  # Should complete within 10 seconds

        for signal in signals:
            assert isinstance(signal, NGSignal)
            assert 0.0 <= signal.ng_score <= 1.0

    def test_shadow_book_scale(self):
        """Test shadow book with many trades"""
        shadow_book = ShadowBookEngine(db_path=":memory:")

        # Generate many trades
        for i in range(100):
            trade = Trade(
                trade_id=f"SCALE_TEST_{i}",
                symbol=f"SYM_{i % 10}",
                trade_type=TradeType.ACTUAL,
                action_type=ActionType.ENTRY,
                quantity=np.random.uniform(100, 1000),
                price=np.random.uniform(50, 500),
                timestamp=datetime.now(),
                strategy_id="scale_test",
                confidence=np.random.uniform(0.5, 1.0)
            )

            shadow_book.record_actual_trade(trade)

        # Test performance metrics calculation
        performance = shadow_book.get_shadow_book_performance("actual")
        assert "total_pnl" in performance

        # Test optimization insights
        insights = shadow_book.generate_optimization_insights()
        assert "insights" in insights

    def test_policy_twin_scale(self):
        """Test policy twin with many ethical evaluations"""
        policy_twin = PolicyTwin()

        # Generate many trade evaluations
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        strategies = ["momentum", "arbitrage", "mean_reversion", "news_based"]

        for i in range(50):
            trade_data = {
                "trade_id": f"SCALE_ETHICS_{i}",
                "symbol": np.random.choice(symbols),
                "action": np.random.choice(["buy", "sell"]),
                "quantity": np.random.uniform(100, 5000),
                "price": np.random.uniform(50, 500),
                "strategy_id": np.random.choice(strategies)
            }

            ethical_trade = policy_twin.analyze_trade_ethics(trade_data)
            assert isinstance(ethical_trade, EthicalTrade)

        # Test metrics calculation
        metrics = policy_twin.calculate_social_impact_metrics()
        assert len(metrics) > 0

        # Test recommendations
        recommendations = policy_twin.generate_policy_recommendations()
        assert isinstance(recommendations, list)

# Integration tests
class TestEndToEndIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_alpha_generation_workflow(self):
        """Test complete alpha generation workflow"""
        # Initialize all components
        integration_engine = AlphaIntegrationEngine()

        portfolio_state = PortfolioState(
            total_capital=10_000_000,
            available_capital=5_000_000,
            current_positions={"AAPL": 1_000_000},
            risk_utilization=0.3,
            var_usage=0.25
        )

        # Generate signal
        signal = await integration_engine.generate_integrated_signal(
            "AAPL", 150.0, portfolio_state
        )

        # Verify complete signal
        assert isinstance(signal, AlphaSignal)
        assert signal.symbol == "AAPL"
        assert hasattr(signal, 'ng_score')
        assert hasattr(signal, 'ethical_score')
        assert hasattr(signal, 'risk_adjusted_score')
        assert hasattr(signal, 'final_score')

        # Verify metadata completeness
        assert 'ng_signal_data' in signal.metadata
        assert 'position_sizing_breakdown' in signal.metadata
        assert 'ethical_considerations' in signal.metadata

    def test_component_compatibility(self):
        """Test that all components work together"""
        # Test data flow between components
        ng_engine = NarrativeGapEngine()
        shadow_book = ShadowBookEngine(db_path=":memory:")
        policy_twin = PolicyTwin()

        # Verify all components can be instantiated together
        assert ng_engine is not None
        assert shadow_book is not None
        assert policy_twin is not None

        # Test basic functionality
        trade_data = {
            "trade_id": "COMPAT_TEST",
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 1000,
            "price": 150.0,
            "strategy_id": "test"
        }

        ethical_trade = policy_twin.analyze_trade_ethics(trade_data)
        assert isinstance(ethical_trade, EthicalTrade)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])