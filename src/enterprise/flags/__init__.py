"""
Feature Flag System

Provides enterprise-grade feature flag management with:
- Decorator-based feature flags
- Runtime configuration
- A/B testing support
- Performance impact monitoring
- Zero-impact non-breaking integration
"""

from .feature_flags import (
    FeatureFlag,
    conditional_execution,
    enterprise_feature,
    enterprise_gate,
    feature_flag,
    flag_manager,
)

__all__ = [
    "FeatureFlag",
    "enterprise_feature", 
    "flag_manager",
    "feature_flag",
    "conditional_execution",
    "enterprise_gate",
]
