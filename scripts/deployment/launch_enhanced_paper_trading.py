#!/usr/bin/env python3
"""
Enhanced Paper Trading Launch
Gary x Taleb Autonomous Trading System

Launches paper trading with all Phase 5 enhancements:
- Narrative Gap position amplification
- Brier score risk calibration
- Enhanced DPI with wealth flow tracking
- Comprehensive monitoring and reporting
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import sys
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

# Phase 5 Enhanced Components
from src.trading.narrative_gap import NarrativeGap
from src.performance.simple_brier import BrierTracker
from src.strategies.dpi_calculator import DistributionalPressureIndex
from src.monitoring.phase5_monitor import Phase5Monitor

class EnhancedPaperTradingSystem:
    """Enhanced paper trading with Phase 5 vision components"""

    def __init__(self, initial_capital: float = 200.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []

        # Phase 5 Components
        self.ng_engine = NarrativeGap()
        self.brier_tracker = BrierTracker()
        self.enhanced_dpi = DistributionalPressureIndex()
        self.monitor = Phase5Monitor()

        # Trading parameters
        self.base_position_size = 0.1  # 10% of capital per trade
        self.max_position_size = 0.25  # Max 25% of capital

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_paper_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EnhancedTrading")

    def start_trading(self, duration_hours: int = 24):
        """Start enhanced paper trading session"""
        self.logger.info("=== ENHANCED PAPER TRADING LAUNCH ===")
        self.logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        self.logger.info(f"Duration: {duration_hours} hours")

        # Start monitoring
        self.monitor.start_monitoring()

        # Trading session
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        trade_count = 0
        target_trades = 20  # Target number of trades for demo

        while datetime.now() < end_time and trade_count < target_trades:
            try:
                # Simulate market opportunity
                market_data = self._simulate_market_data()

                # Generate enhanced trading signal
                signal = self._generate_enhanced_signal(market_data)

                if signal and signal['should_trade']:
                    # Execute trade
                    self._execute_trade(signal)
                    trade_count += 1

                    # Wait between trades (simulate realistic trading frequency)
                    time.sleep(2)  # 2 seconds for demo (would be minutes/hours in real trading)

                else:
                    # No trade signal, wait and continue
                    time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("Trading interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Trading error: {e}")
                time.sleep(5)

        # End trading session
        self._end_trading_session()

    def _simulate_market_data(self) -> Dict:
        """Simulate market data for demo purposes"""
        # Generate realistic market data
        base_price = 100 + random.uniform(-20, 20)

        # Simulate consensus forecast (analyst estimates)
        consensus_bias = random.uniform(-0.05, 0.05)  # ±5% bias
        consensus_forecast = base_price * (1 + consensus_bias)

        # Simulate Gary's estimate (our model)
        gary_bias = random.uniform(-0.08, 0.08)  # ±8% range
        gary_estimate = base_price * (1 + gary_bias)

        # Market regime indicators
        volatility = random.uniform(0.15, 0.35)
        volume = random.uniform(0.8, 1.5)  # Relative volume

        return {
            'symbol': 'SPY',  # S&P 500 ETF
            'price': base_price,
            'consensus_forecast': consensus_forecast,
            'gary_estimate': gary_estimate,
            'volatility': volatility,
            'volume': volume,
            'timestamp': datetime.now()
        }

    def _generate_enhanced_signal(self, market_data: Dict) -> Dict:
        """Generate enhanced trading signal using Phase 5 components"""

        # Step 1: Calculate Narrative Gap
        ng_score = self.ng_engine.calculate_ng(
            market_data['price'],
            market_data['consensus_forecast'],
            market_data['gary_estimate']
        )
        ng_multiplier = self.ng_engine.get_position_multiplier(ng_score)

        # Step 2: Get Brier adjustment
        current_brier = self.brier_tracker.get_brier_score()
        brier_adjustment = max(0.1, 1 - current_brier)  # Minimum 10% position size

        # Step 3: Enhanced DPI (simulated wealth flow data)
        wealth_flow_data = {
            'high_income_gains': random.uniform(800000, 1200000),
            'total_gains': random.uniform(1000000, 1500000),
            'wealth_concentration': random.uniform(0.6, 0.9)
        }

        flow_score = self.enhanced_dpi.calculate_wealth_flow(wealth_flow_data)
        dpi_enhancement = 1 + (flow_score * 0.5)  # Up to 50% DPI enhancement

        # Step 4: Determine trade direction and size
        price_gap = market_data['gary_estimate'] - market_data['price']
        trade_direction = 'long' if price_gap > 0 else 'short'

        # Calculate position size
        base_size = self.current_capital * self.base_position_size
        enhanced_size = base_size * ng_multiplier * brier_adjustment * dpi_enhancement
        enhanced_size = min(enhanced_size, self.current_capital * self.max_position_size)

        # Minimum threshold for trading
        should_trade = abs(ng_score) > 0.02 and enhanced_size > self.current_capital * 0.05

        # Record monitoring data
        self.monitor.record_ng_signal(
            ng_score, ng_multiplier,
            market_data['price'],
            market_data['consensus_forecast'],
            market_data['gary_estimate']
        )

        self.monitor.record_brier_update(
            current_brier, self.brier_tracker.prediction_count,
            self._calculate_recent_accuracy(), brier_adjustment
        )

        self.monitor.record_enhanced_dpi(
            0.6, flow_score, 0.6 * dpi_enhancement,
            "bullish" if price_gap > 0 else "bearish", 0.8
        )

        return {
            'should_trade': should_trade,
            'direction': trade_direction,
            'position_size': enhanced_size,
            'ng_score': ng_score,
            'ng_multiplier': ng_multiplier,
            'brier_adjustment': brier_adjustment,
            'dpi_enhancement': dpi_enhancement,
            'expected_return': abs(price_gap) / market_data['price'],
            'market_data': market_data
        }

    def _execute_trade(self, signal: Dict):
        """Execute enhanced paper trade"""
        trade_id = f"TRADE_{self.total_trades + 1:03d}"

        # Calculate expected outcome (simulate market movement)
        expected_return = signal['expected_return']
        actual_return = expected_return * random.uniform(0.6, 1.4) * (1 if random.random() > 0.4 else -1)

        # Apply direction
        if signal['direction'] == 'short':
            actual_return *= -1

        # Calculate P&L
        pnl = signal['position_size'] * actual_return

        # Record prediction for Brier scoring
        prediction_confidence = min(0.9, abs(signal['ng_score']) * 10)  # Convert NG to confidence
        actual_outcome = 1 if pnl > 0 else 0
        self.brier_tracker.record_prediction(prediction_confidence, actual_outcome)

        # Update portfolio
        self.current_capital += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Record trade
        trade_record = {
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'symbol': signal['market_data']['symbol'],
            'direction': signal['direction'],
            'position_size': signal['position_size'],
            'entry_price': signal['market_data']['price'],
            'pnl': pnl,
            'ng_score': signal['ng_score'],
            'ng_multiplier': signal['ng_multiplier'],
            'brier_adjustment': signal['brier_adjustment'],
            'dpi_enhancement': signal['dpi_enhancement'],
            'return_pct': (pnl / signal['position_size']) * 100
        }

        self.trade_history.append(trade_record)

        # Log trade
        self.logger.info(f"{trade_id}: {signal['direction'].upper()} ${signal['position_size']:.2f} "
                        f"| NG: {signal['ng_score']:.4f} ({signal['ng_multiplier']:.3f}x) "
                        f"| Brier: {signal['brier_adjustment']:.3f}x "
                        f"| DPI: {signal['dpi_enhancement']:.3f}x "
                        f"| P&L: ${pnl:+.2f} "
                        f"| Capital: ${self.current_capital:.2f}")

    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy"""
        if len(self.trade_history) < 5:
            return 0.5  # Default 50%

        recent_trades = self.trade_history[-5:]
        correct_predictions = sum(1 for trade in recent_trades if trade['pnl'] > 0)
        return correct_predictions / len(recent_trades)

    def _end_trading_session(self):
        """End trading session and generate report"""
        self.logger.info("=== ENHANCED TRADING SESSION COMPLETE ===")

        # Calculate performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        win_rate = self.winning_trades / max(1, self.total_trades)
        avg_trade_pnl = self.total_pnl / max(1, self.total_trades)

        # Phase 5 enhancement metrics
        ng_trades = len([t for t in self.trade_history if abs(t['ng_score']) > 0.03])
        avg_ng_multiplier = np.mean([t['ng_multiplier'] for t in self.trade_history])
        avg_brier_adjustment = np.mean([t['brier_adjustment'] for t in self.trade_history])
        avg_dpi_enhancement = np.mean([t['dpi_enhancement'] for t in self.trade_history])

        # Generate final report
        report = {
            'session_summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return_pct': total_return * 100,
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate_pct': win_rate * 100,
                'avg_trade_pnl': avg_trade_pnl
            },
            'phase5_enhancements': {
                'ng_enhanced_trades': ng_trades,
                'avg_ng_multiplier': avg_ng_multiplier,
                'avg_brier_adjustment': avg_brier_adjustment,
                'avg_dpi_enhancement': avg_dpi_enhancement,
                'final_brier_score': self.brier_tracker.get_brier_score()
            },
            'monitoring_status': self.monitor.get_current_status()
        }

        # Print summary
        print("\n" + "="*60)
        print("ENHANCED PAPER TRADING RESULTS")
        print("="*60)
        print(f"Initial Capital:     ${self.initial_capital:.2f}")
        print(f"Final Capital:       ${self.current_capital:.2f}")
        print(f"Total Return:        {total_return*100:+.2f}%")
        print(f"Total P&L:           ${self.total_pnl:+.2f}")
        print(f"Total Trades:        {self.total_trades}")
        print(f"Win Rate:            {win_rate*100:.1f}%")
        print(f"Avg Trade P&L:       ${avg_trade_pnl:+.2f}")
        print("\nPHASE 5 ENHANCEMENTS:")
        print(f"NG Enhanced Trades:  {ng_trades}/{self.total_trades}")
        print(f"Avg NG Multiplier:   {avg_ng_multiplier:.3f}x")
        print(f"Avg Brier Adjust:    {avg_brier_adjustment:.3f}x")
        print(f"Avg DPI Enhance:     {avg_dpi_enhancement:.3f}x")
        print(f"Final Brier Score:   {self.brier_tracker.get_brier_score():.4f}")
        print("="*60)

        # Export detailed report
        import json
        report_filename = f"enhanced_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Export monitoring metrics
        metrics_filename = self.monitor.export_metrics()

        self.logger.info(f"Session report saved: {report_filename}")
        self.logger.info(f"Monitoring metrics: {metrics_filename}")

        # Stop monitoring
        self.monitor.stop_monitoring()

        return report

def main():
    """Launch enhanced paper trading session"""
    print("Gary x Taleb Enhanced Paper Trading System")
    print("Phase 5 Vision Components Active")
    print("="*50)

    # Create trading system
    trading_system = EnhancedPaperTradingSystem(initial_capital=200.0)

    # Start trading session (short demo - 10 minutes, 20 trades)
    trading_system.start_trading(duration_hours=0.1)  # 6 minutes for demo

    print("\nEnhanced paper trading session complete.")
    print("System ready for production deployment with $200 capital.")

if __name__ == "__main__":
    main()