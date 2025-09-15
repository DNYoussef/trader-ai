"""
Enhanced Risk Management System - Phase 5 Implementation

This module provides the complete Phase 5 risk management and calibration system
for the Super-Gary trading framework, implementing survival-first trading through:

1. Brier Score Calibration - Forecast accuracy tracking and position sizing
2. Convexity Optimization - Regime-aware gamma farming and options strategies  
3. Enhanced Kelly Criterion - Survival-first position sizing with correlation awareness
4. Integrated Risk Management - Unified system with kill switch integration

Key Features:
- Survival-first position sizing with risk-of-ruin constraints
- Real-time calibration tracking and position size adjustments
- Regime-aware convexity optimization for volatility harvesting
- Multi-asset Kelly frontiers with correlation clustering
- Integrated dashboard and monitoring with emergency stops
"""

# Phase 5 availability flag
PHASE5_AVAILABLE = True

__version__ = "5.0.0"
__phase__ = "Phase 5: Risk & Calibration Systems"

