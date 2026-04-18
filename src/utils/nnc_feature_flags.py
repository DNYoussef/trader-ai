"""
NNC Feature Flag Loader

Simple feature flag loader for NNC features.
Reads from config/feature_flags.json.

Can be integrated with enterprise flag system later.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "feature_flags.json"

# Cached flags
_cached_flags: Optional[dict] = None
_cache_path: Optional[Path] = None


def _load_flags(config_path: Optional[Path] = None) -> dict:
    """Load feature flags from JSON config."""
    global _cached_flags, _cache_path

    path = config_path or DEFAULT_CONFIG_PATH

    # Return cached if same path
    if _cached_flags is not None and _cache_path == path:
        return _cached_flags

    try:
        if path.exists():
            with open(path, 'r') as f:
                flags = json.load(f)
                _cached_flags = flags
                _cache_path = path
                return flags
        else:
            logger.warning(f"NNC feature flags config not found at {path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading NNC feature flags: {e}")
        return {}


def get_flag(name: str, default: Any = False) -> Any:
    """
    Get a feature flag value.

    Args:
        name: Flag name (e.g., 'use_nnc_returns')
        default: Default value if flag not found

    Returns:
        Flag value or default
    """
    flags = _load_flags()
    return flags.get(name, default)


def is_nnc_enabled() -> bool:
    """Check if NNC is globally enabled."""
    return get_flag('nnc_enabled', False)


def use_nnc_returns() -> bool:
    """Check if geometric mean returns should be used."""
    return is_nnc_enabled() and get_flag('use_nnc_returns', False)


def use_nnc_risk() -> bool:
    """Check if multiplicative risk compounding should be used."""
    return is_nnc_enabled() and get_flag('use_nnc_risk', False)


def use_nnc_kelly() -> bool:
    """Check if beta-arithmetic Kelly criterion should be used."""
    return is_nnc_enabled() and get_flag('use_nnc_kelly', False)


def use_nnc_nav() -> bool:
    """Check if multiplicative NAV tracking should be used."""
    return is_nnc_enabled() and get_flag('use_nnc_nav', False)


def use_prelec_weighting() -> bool:
    """Check if Prelec probability weighting should be used."""
    return is_nnc_enabled() and get_flag('use_prelec_weighting', False)


def use_k_evolution() -> bool:
    """Check if gate-dependent k parameter should be used."""
    return is_nnc_enabled() and get_flag('use_k_evolution', False)


def use_star_euler_projection() -> bool:
    """Check if Star-Euler projection should be used."""
    return is_nnc_enabled() and get_flag('use_star_euler_projection', False)


def parallel_classical_nnc() -> bool:
    """Check if parallel classical + NNC calculations should be run."""
    return get_flag('parallel_classical_nnc', True)


def divergence_alert_threshold() -> float:
    """Get threshold for classical vs NNC divergence alerts."""
    return get_flag('divergence_alert_threshold', 0.10)


def reload_flags() -> dict:
    """Force reload flags from config file."""
    global _cached_flags, _cache_path
    _cached_flags = None
    _cache_path = None
    return _load_flags()


def set_flag(name: str, value: Any, persist: bool = False) -> None:
    """
    Set a feature flag value (runtime only unless persist=True).

    Args:
        name: Flag name
        value: Flag value
        persist: If True, save to config file
    """
    flags = _load_flags()
    flags[name] = value

    if persist:
        try:
            with open(_cache_path or DEFAULT_CONFIG_PATH, 'w') as f:
                json.dump(flags, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting flag {name}: {e}")
