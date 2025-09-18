"""
PRODUCTION AUDIT: Enhanced Strategy System
End-to-end functionality verification and reality testing
"""

import sys
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from strategies.black_swan_strategies import BlackSwanStrategyToolbox
from strategies.enhanced_market_state import create_enhanced_market_state
from strategies.convex_reward_function import ConvexRewardFunction, TradeOutcome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAuditor:
    """Comprehensive production audit of enhanced system"""

    def __init__(self):
        self.db_path = Path("data/historical_market.db")
        self.toolbox = BlackSwanStrategyToolbox()
        self.reward_calc = ConvexRewardFunction()
        self.audit_results = {}

    def audit_database_connection(self) -> bool:
        """Audit 1: Database connectivity and data quality"""

        print("=== AUDIT 1: DATABASE CONNECTION ===")

        if not self.db_path.exists():
            print(f"FAIL: Database not found at {self.db_path}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"OK Tables found: {tables}")

                # Check market_data
                cursor.execute("SELECT COUNT(*) FROM market_data")
                count = cursor.fetchone()[0]
                print(f"OK Market data records: {count}")

                # Check data quality
                cursor.execute("SELECT MIN(date), MAX(date) FROM market_data")
                date_range = cursor.fetchone()
                print(f"OK Date range: {date_range[0]} to {date_range[1]}")

                # Check symbols
                cursor.execute("SELECT DISTINCT symbol FROM market_data LIMIT 10")
                symbols = [row[0] for row in cursor.fetchall()]
                print(f"OK Symbols available: {symbols}")

                self.audit_results['database'] = {
                    'status': 'PASS',
                    'record_count': count,
                    'date_range': date_range,
                    'symbols': symbols
                }

                return True

        except Exception as e:
            print(f"FAIL: Database error: {e}")
            self.audit_results['database'] = {'status': 'FAIL', 'error': str(e)}
            return False

    def audit_enhanced_market_state(self) -> bool:
        """Audit 2: Enhanced market state functionality"""

        print("\\n=== AUDIT 2: ENHANCED MARKET STATE ===")

        try:
            # Create test market state
            market_state = create_enhanced_market_state(
                timestamp=datetime.now(),
                vix_level=25.0,
                spy_returns_5d=-0.03,
                spy_returns_20d=-0.01,
                put_call_ratio=1.5,
                market_breadth=0.35,
                volume_ratio=1.8,
                regime='volatile'
            )

            print(f"OK Market state created: {market_state.regime}")

            # Test feature extraction
            features = market_state.get_enhanced_market_features()
            print(f"OK Features extracted: {len(features)} features")

            # Test critical features
            critical_features = ['vix_level', 'signal_quality_score', 'gini_coefficient', 'sector_dispersion']
            missing_features = []
            for feature in critical_features:
                if feature not in features:
                    missing_features.append(feature)
                else:
                    print(f"  {feature}: {features[feature]}")

            if missing_features:
                print(f"FAIL: Missing features: {missing_features}")
                return False

            # Test threshold adjustment
            base_thresholds = {'vix_threshold': 20, 'momentum_threshold': 0.05}
            adjusted = market_state.get_regime_adjusted_thresholds(base_thresholds)
            print(f"OK Threshold adjustment: {adjusted}")

            self.audit_results['market_state'] = {
                'status': 'PASS',
                'features_count': len(features),
                'signal_quality': features.get('signal_quality_score', 0),
                'regime': market_state.regime
            }

            return True

        except Exception as e:
            print(f"FAIL: Market state error: {e}")
            import traceback
            traceback.print_exc()
            self.audit_results['market_state'] = {'status': 'FAIL', 'error': str(e)}
            return False

    def audit_strategy_signal_generation(self) -> bool:
        """Audit 3: Strategy signal generation with real data"""

        print("\\n=== AUDIT 3: STRATEGY SIGNAL GENERATION ===")

        try:
            # Load real historical data
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT date, symbol, open, high, low, close, volume, returns
                FROM market_data
                WHERE date >= '2024-01-01'
                ORDER BY date DESC
                LIMIT 500
                """
                df = pd.read_sql_query(query, conn)

            print(f"OK Loaded {len(df)} historical records")

            # Test multiple market scenarios
            scenarios = [
                {'name': 'Crisis', 'vix': 35, 'spy_5d': -0.08, 'put_call': 2.0, 'breadth': 0.2, 'volume': 2.5},
                {'name': 'Volatile', 'vix': 25, 'spy_5d': -0.03, 'put_call': 1.4, 'breadth': 0.35, 'volume': 1.8},
                {'name': 'Normal', 'vix': 18, 'spy_5d': 0.01, 'put_call': 1.0, 'breadth': 0.6, 'volume': 1.2},
                {'name': 'Momentum', 'vix': 15, 'spy_5d': 0.04, 'put_call': 0.8, 'breadth': 0.75, 'volume': 1.5}
            ]

            total_signals = 0
            strategy_signal_counts = {}

            for scenario in scenarios:
                print(f"\\n--- Testing {scenario['name']} Scenario ---")

                market_state = create_enhanced_market_state(
                    timestamp=datetime.now(),
                    vix_level=scenario['vix'],
                    spy_returns_5d=scenario['spy_5d'],
                    spy_returns_20d=scenario['spy_5d'] * 0.6,
                    put_call_ratio=scenario['put_call'],
                    market_breadth=scenario['breadth'],
                    volume_ratio=scenario['volume'],
                    regime='crisis' if scenario['vix'] > 30 else 'volatile' if scenario['vix'] > 20 else 'normal'
                )

                signals = self.toolbox.analyze_all_strategies(market_state, df)
                print(f"Generated {len(signals)} signals")

                for signal in signals:
                    strategy_name = signal.strategy_name
                    if strategy_name not in strategy_signal_counts:
                        strategy_signal_counts[strategy_name] = 0
                    strategy_signal_counts[strategy_name] += 1
                    total_signals += 1

                    print(f"  {strategy_name}: {signal.action} {signal.symbol} (conf={signal.confidence:.2f})")

            print(f"\\n=== SIGNAL GENERATION RESULTS ===")
            print(f"Total signals: {total_signals}")
            print(f"Active strategies: {len(strategy_signal_counts)}")

            for strategy, count in strategy_signal_counts.items():
                print(f"  {strategy}: {count} signals")

            # Audit criteria
            if total_signals < 4:
                print(f"FAIL: Insufficient signals generated ({total_signals})")
                return False

            if len(strategy_signal_counts) < 2:
                print(f"FAIL: Too few strategies active ({len(strategy_signal_counts)})")
                return False

            print("OK Signal generation audit PASSED")

            self.audit_results['signal_generation'] = {
                'status': 'PASS',
                'total_signals': total_signals,
                'active_strategies': len(strategy_signal_counts),
                'strategy_counts': strategy_signal_counts
            }

            return True

        except Exception as e:
            print(f"FAIL: Signal generation error: {e}")
            import traceback
            traceback.print_exc()
            self.audit_results['signal_generation'] = {'status': 'FAIL', 'error': str(e)}
            return False

    def audit_convex_reward_calculation(self) -> bool:
        """Audit 4: Convex reward function with real scenarios"""

        print("\\n=== AUDIT 4: CONVEX REWARD CALCULATION ===")

        try:
            # Test different return scenarios
            test_scenarios = [
                {'returns': 0.15, 'desc': 'Black Swan Gain'},
                {'returns': 0.05, 'desc': 'Normal Gain'},
                {'returns': -0.02, 'desc': 'Small Loss'},
                {'returns': -0.15, 'desc': 'Large Loss'},
                {'returns': 0.35, 'desc': 'Extreme Gain'}
            ]

            results = []

            for scenario in test_scenarios:
                trade_outcome = TradeOutcome(
                    strategy_name='test_strategy',
                    entry_date=datetime.now() - timedelta(days=10),
                    exit_date=datetime.now(),
                    symbol='SPY',
                    returns=scenario['returns'],
                    max_drawdown=min(0, scenario['returns']),
                    holding_period_days=10,
                    volatility_during_trade=0.02,
                    is_black_swan_period=abs(scenario['returns']) > 0.10,
                    black_swan_captured=scenario['returns'] > 0.10,
                    convexity_achieved=max(0, scenario['returns'] / 0.05)
                )

                reward_metrics = self.reward_calc.calculate_reward(trade_outcome)
                final_reward = reward_metrics.final_reward

                results.append({
                    'scenario': scenario['desc'],
                    'returns': scenario['returns'],
                    'final_reward': final_reward,
                    'convexity_bonus': reward_metrics.convexity_bonus,
                    'black_swan_multiplier': reward_metrics.black_swan_multiplier
                })

                print(f"{scenario['desc']}: {scenario['returns']:.1%} return -> {final_reward:.4f} reward")

            # Verify convexity properties
            positive_returns = [r for r in results if r['returns'] > 0]
            negative_returns = [r for r in results if r['returns'] < 0]

            # Check that large gains have disproportionately high rewards
            if len(positive_returns) >= 2:
                large_gain = max(positive_returns, key=lambda x: x['returns'])
                small_gain = min(positive_returns, key=lambda x: x['returns'])

                reward_ratio = large_gain['final_reward'] / small_gain['final_reward']
                return_ratio = large_gain['returns'] / small_gain['returns']

                print(f"\\nConvexity check:")
                print(f"  Return ratio: {return_ratio:.2f}x")
                print(f"  Reward ratio: {reward_ratio:.2f}x")

                if reward_ratio <= return_ratio:
                    print("FAIL: Rewards not sufficiently convex")
                    return False

                print("OK Convex reward structure verified")

            self.audit_results['reward_calculation'] = {
                'status': 'PASS',
                'test_scenarios': len(results),
                'convexity_verified': True
            }

            return True

        except Exception as e:
            print(f"FAIL: Reward calculation error: {e}")
            import traceback
            traceback.print_exc()
            self.audit_results['reward_calculation'] = {'status': 'FAIL', 'error': str(e)}
            return False

    def audit_end_to_end_workflow(self) -> bool:
        """Audit 5: Complete end-to-end workflow"""

        print("\\n=== AUDIT 5: END-TO-END WORKFLOW ===")

        try:
            # Load real data
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT date, symbol, open, high, low, close, volume, returns
                FROM market_data
                WHERE date >= '2023-01-01'
                ORDER BY date DESC
                LIMIT 200
                """
                df = pd.read_sql_query(query, conn)

            # Create realistic market state
            market_state = create_enhanced_market_state(
                timestamp=datetime.now(),
                vix_level=22.5,
                spy_returns_5d=-0.025,
                spy_returns_20d=0.01,
                put_call_ratio=1.3,
                market_breadth=0.4,
                volume_ratio=1.6,
                regime='volatile'
            )

            print(f"Market state: {market_state.regime}, Quality: {market_state.signal_quality_score:.2f}")

            # Generate signals
            signals = self.toolbox.analyze_all_strategies(market_state, df)
            print(f"Generated {len(signals)} signals")

            if not signals:
                print("FAIL: No signals generated in end-to-end test")
                return False

            # Process each signal through reward function
            total_processed = 0
            successful_rewards = 0

            for signal in signals:
                try:
                    # Simulate forward return
                    simulated_return = np.random.normal(0.02, 0.05)  # 2% mean, 5% std

                    trade_outcome = TradeOutcome(
                        strategy_name=signal.strategy_name,
                        entry_date=market_state.timestamp,
                        exit_date=market_state.timestamp + timedelta(days=10),
                        symbol=signal.symbol,
                        returns=simulated_return,
                        max_drawdown=min(0, simulated_return),
                        holding_period_days=10,
                        volatility_during_trade=0.025,
                        is_black_swan_period=abs(simulated_return) > 0.10,
                        black_swan_captured=simulated_return > 0.10,
                        convexity_achieved=max(0, simulated_return / 0.05)
                    )

                    reward_metrics = self.reward_calc.calculate_reward(trade_outcome)
                    final_reward = reward_metrics.final_reward

                    print(f"  {signal.strategy_name}: {simulated_return:.2%} → {final_reward:.4f}")
                    total_processed += 1

                    if final_reward is not None and not np.isnan(final_reward):
                        successful_rewards += 1

                except Exception as e:
                    print(f"  ERROR processing {signal.strategy_name}: {e}")

            print(f"\\nProcessed {total_processed}/{len(signals)} signals successfully")
            print(f"Successful rewards: {successful_rewards}/{total_processed}")

            if successful_rewards < len(signals) * 0.8:  # 80% success rate
                print("FAIL: Too many reward calculation failures")
                return False

            print("OK End-to-end workflow PASSED")

            self.audit_results['end_to_end'] = {
                'status': 'PASS',
                'signals_generated': len(signals),
                'signals_processed': total_processed,
                'successful_rewards': successful_rewards,
                'success_rate': successful_rewards / total_processed if total_processed > 0 else 0
            }

            return True

        except Exception as e:
            print(f"FAIL: End-to-end workflow error: {e}")
            import traceback
            traceback.print_exc()
            self.audit_results['end_to_end'] = {'status': 'FAIL', 'error': str(e)}
            return False

    def generate_audit_report(self):
        """Generate final audit report"""

        print("\\n" + "="*60)
        print("PRODUCTION AUDIT REPORT")
        print("="*60)

        total_audits = len(self.audit_results)
        passed_audits = sum(1 for result in self.audit_results.values() if result.get('status') == 'PASS')

        print(f"Audits completed: {total_audits}")
        print(f"Audits passed: {passed_audits}")
        print(f"Success rate: {passed_audits/total_audits:.1%}")

        print("\\nDetailed Results:")
        for audit_name, result in self.audit_results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"  {audit_name}: {status}")
            if status == 'FAIL' and 'error' in result:
                print(f"    Error: {result['error']}")

        if passed_audits == total_audits:
            print("\\n=== FINAL VERDICT: PRODUCTION READY ===")
            print("All audits passed. Enhanced system is fully functional.")
        else:
            print("\\n=== FINAL VERDICT: NEEDS ATTENTION ===")
            print("Some audits failed. System requires fixes before production.")

        return passed_audits == total_audits

def main():
    """Run complete production audit"""

    print("STARTING COMPREHENSIVE PRODUCTION AUDIT")
    print("Enhanced Black Swan Strategy System")
    print("="*60)

    auditor = ProductionAuditor()

    # Run all audits
    audits = [
        auditor.audit_database_connection,
        auditor.audit_enhanced_market_state,
        auditor.audit_strategy_signal_generation,
        auditor.audit_convex_reward_calculation,
        auditor.audit_end_to_end_workflow
    ]

    all_passed = True
    for audit in audits:
        try:
            if not audit():
                all_passed = False
        except Exception as e:
            print(f"CRITICAL ERROR in audit: {e}")
            all_passed = False

    # Generate final report
    production_ready = auditor.generate_audit_report()

    if production_ready:
        print("\\nPRODUCTION AUDIT COMPLETE - SYSTEM READY")
    else:
        print("\\nPRODUCTION AUDIT FAILED - SYSTEM NOT READY")

    return production_ready

if __name__ == "__main__":
    main()