"""
Gate Management System

This package provides a comprehensive gate-based trading system that manages
trading privileges and constraints based on capital levels, enforcing risk
management rules and tracking performance metrics.
"""

from .gate_manager import (
    GateLevel,
    ViolationType,
    GateConfig,
    TradeValidationResult,
    ViolationRecord,
    GraduationMetrics,
    GateManager
)

__all__ = [
    'GateLevel',
    'ViolationType', 
    'GateConfig',
    'TradeValidationResult',
    'ViolationRecord',
    'GraduationMetrics',
    'GateManager'
]