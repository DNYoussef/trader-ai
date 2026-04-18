"""
Monte Carlo Simulations for Strategy Robustness

Provides statistical confidence bounds through:
- Shuffle Monte Carlo (destroys all temporal structure)
- Block Bootstrap (preserves volatility clustering)
- Ruin probability estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    n_paths: int
    metric_name: str

    # Distribution statistics
    mean: float
    std: float
    p5: float   # 5th percentile (worst case)
    p25: float
    p50: float  # Median
    p75: float
    p95: float  # 95th percentile (best case)

    # Risk metrics
    ruin_probability: float  # P(max_drawdown > threshold)
    expected_shortfall: float  # Average of worst 5%

    # Path data (for visualization)
    sample_paths: Optional[np.ndarray] = None


def shuffle_bar_returns_mc(
    bar_returns: np.ndarray,
    n_paths: int = 1000,
    seed: Optional[int] = None,
    compute_paths: bool = False,
) -> Dict[str, MonteCarloResult]:
    """
    Monte Carlo simulation with fully shuffled returns.

    This destroys ALL temporal structure (autocorrelation, volatility clustering).
    Use as a "null hypothesis" - if real performance isn't better than shuffled,
    the strategy likely has no edge.

    Args:
        bar_returns: Array of per-bar returns
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        compute_paths: Whether to store sample paths (memory intensive)

    Returns:
        Dict with MonteCarloResult for: cagr, max_drawdown, sharpe, time_to_recover
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(bar_returns)

    # Metrics to track
    cagrs = []
    max_dds = []
    sharpes = []
    recovery_times = []

    sample_paths = [] if compute_paths else None

    for _ in range(n_paths):
        # Shuffle returns
        shuffled = np.random.permutation(bar_returns)

        # Compute equity curve
        equity = np.exp(np.cumsum(shuffled))
        equity = np.concatenate([[1.0], equity])

        # CAGR (annualized)
        total_return = equity[-1] / equity[0]
        n_years = n / 252
        cagr = (total_return ** (1 / n_years) - 1) if n_years > 0 else 0
        cagrs.append(cagr)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = abs(np.min(drawdowns))
        max_dds.append(max_dd)

        # Sharpe
        sharpe = np.sqrt(252) * np.mean(shuffled) / (np.std(shuffled) + 1e-8)
        sharpes.append(sharpe)

        # Time to recover from max drawdown
        max_dd_idx = np.argmin(drawdowns)
        recovery_idx = np.where(equity[max_dd_idx:] >= running_max[max_dd_idx])[0]
        recovery_time = recovery_idx[0] if len(recovery_idx) > 0 else n - max_dd_idx
        recovery_times.append(recovery_time)

        if compute_paths and len(sample_paths) < 100:
            sample_paths.append(equity)

    results = {}

    # CAGR distribution
    results['cagr'] = _build_mc_result(
        'cagr', cagrs, n_paths,
        ruin_threshold=-0.20,  # 20% annual loss
        sample_paths=np.array(sample_paths) if sample_paths else None,
    )

    # Max Drawdown distribution
    results['max_drawdown'] = _build_mc_result(
        'max_drawdown', max_dds, n_paths,
        ruin_threshold=0.30,  # 30% drawdown
        invert_ruin=True,  # Higher is worse
    )

    # Sharpe distribution
    results['sharpe'] = _build_mc_result(
        'sharpe', sharpes, n_paths,
        ruin_threshold=0.0,  # Negative Sharpe
    )

    # Recovery time distribution
    results['recovery_time'] = _build_mc_result(
        'recovery_time', recovery_times, n_paths,
        ruin_threshold=126,  # 6 months to recover
        invert_ruin=True,
    )

    return results


def block_bootstrap_mc(
    bar_returns: np.ndarray,
    block_len: int = 20,
    n_paths: int = 1000,
    seed: Optional[int] = None,
    compute_paths: bool = False,
) -> Dict[str, MonteCarloResult]:
    """
    Block bootstrap Monte Carlo simulation.

    Preserves some volatility clustering by sampling contiguous blocks.
    More realistic than full shuffle for strategies that depend on
    volatility regimes.

    Args:
        bar_returns: Array of per-bar returns
        block_len: Length of each bootstrap block
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        compute_paths: Whether to store sample paths

    Returns:
        Dict with MonteCarloResult for: cagr, max_drawdown, sharpe
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(bar_returns)
    n_blocks = (n + block_len - 1) // block_len

    cagrs = []
    max_dds = []
    sharpes = []

    sample_paths = [] if compute_paths else None

    for _ in range(n_paths):
        # Sample blocks with replacement
        # Pre-allocate array to avoid O(n^2) memory operations
        bootstrapped = np.empty(n_blocks * block_len, dtype=np.float64)
        current_idx = 0

        for _ in range(n_blocks):
            start_idx = np.random.randint(0, max(1, n - block_len + 1))
            end_idx = min(start_idx + block_len, n)
            block = bar_returns[start_idx:end_idx]
            block_size = len(block)
            bootstrapped[current_idx:current_idx + block_size] = block
            current_idx += block_size

        bootstrapped = bootstrapped[:n]

        # Compute metrics
        equity = np.exp(np.cumsum(bootstrapped))
        equity = np.concatenate([[1.0], equity])

        # CAGR
        total_return = equity[-1] / equity[0]
        n_years = n / 252
        cagr = (total_return ** (1 / n_years) - 1) if n_years > 0 else 0
        cagrs.append(cagr)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = abs(np.min(drawdowns))
        max_dds.append(max_dd)

        # Sharpe
        sharpe = np.sqrt(252) * np.mean(bootstrapped) / (np.std(bootstrapped) + 1e-8)
        sharpes.append(sharpe)

        if compute_paths and len(sample_paths) < 100:
            sample_paths.append(equity)

    results = {}

    results['cagr'] = _build_mc_result(
        'cagr', cagrs, n_paths,
        ruin_threshold=-0.20,
        sample_paths=np.array(sample_paths) if sample_paths else None,
    )

    results['max_drawdown'] = _build_mc_result(
        'max_drawdown', max_dds, n_paths,
        ruin_threshold=0.30,
        invert_ruin=True,
    )

    results['sharpe'] = _build_mc_result(
        'sharpe', sharpes, n_paths,
        ruin_threshold=0.0,
    )

    return results


def compute_ruin_probability(
    bar_returns: np.ndarray,
    ruin_threshold: float = 0.30,
    n_paths: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute probability of ruin (exceeding drawdown threshold).

    Args:
        bar_returns: Array of per-bar returns
        ruin_threshold: Drawdown threshold for "ruin"
        n_paths: Number of Monte Carlo paths
        seed: Random seed

    Returns:
        Dict with ruin probabilities from shuffle and block bootstrap
    """
    shuffle_results = shuffle_bar_returns_mc(bar_returns, n_paths, seed)
    block_results = block_bootstrap_mc(bar_returns, n_paths=n_paths, seed=seed)

    return {
        'shuffle_ruin_prob': shuffle_results['max_drawdown'].ruin_probability,
        'block_ruin_prob': block_results['max_drawdown'].ruin_probability,
        'shuffle_p95_dd': shuffle_results['max_drawdown'].p95,
        'block_p95_dd': block_results['max_drawdown'].p95,
    }


def _build_mc_result(
    metric_name: str,
    values: List[float],
    n_paths: int,
    ruin_threshold: float,
    invert_ruin: bool = False,
    sample_paths: Optional[np.ndarray] = None,
) -> MonteCarloResult:
    """Build MonteCarloResult from simulation values."""
    arr = np.array(values)

    # Distribution statistics
    mean = np.mean(arr)
    std = np.std(arr)
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])

    # Ruin probability
    if invert_ruin:
        ruin_prob = np.mean(arr > ruin_threshold)
    else:
        ruin_prob = np.mean(arr < ruin_threshold)

    # Expected shortfall (CVaR) - average of worst 5%
    if invert_ruin:
        worst_5_pct = arr[arr >= np.percentile(arr, 95)]
    else:
        worst_5_pct = arr[arr <= np.percentile(arr, 5)]

    expected_shortfall = np.mean(worst_5_pct) if len(worst_5_pct) > 0 else p5

    return MonteCarloResult(
        n_paths=n_paths,
        metric_name=metric_name,
        mean=float(mean),
        std=float(std),
        p5=float(p5),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
        ruin_probability=float(ruin_prob),
        expected_shortfall=float(expected_shortfall),
        sample_paths=sample_paths,
    )


def export_mc_paths_artifact(
    mc_result: MonteCarloResult,
    out_path: str,
    title: str = "Monte Carlo Paths",
):
    """
    Export Monte Carlo paths visualization.

    Args:
        mc_result: MonteCarloResult with sample_paths
        out_path: Output file path
        title: Plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping MC paths export")
        return

    if mc_result.sample_paths is None:
        logger.warning("No sample paths available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot sample paths
    for path in mc_result.sample_paths[:50]:
        ax.plot(path, alpha=0.3, linewidth=0.5, color='blue')

    # Plot median path
    median_path = np.median(mc_result.sample_paths, axis=0)
    ax.plot(median_path, color='black', linewidth=2, label='Median')

    # Plot percentile bounds
    p5_path = np.percentile(mc_result.sample_paths, 5, axis=0)
    p95_path = np.percentile(mc_result.sample_paths, 95, axis=0)
    ax.fill_between(range(len(median_path)), p5_path, p95_path,
                    alpha=0.2, color='blue', label='5-95% CI')

    ax.set_title(f"{title}\nRuin Prob: {mc_result.ruin_probability:.1%}")
    ax.set_xlabel('Bar')
    ax.set_ylabel('Equity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"MC paths saved to {out_path}")


if __name__ == "__main__":
    # Test Monte Carlo
    print("=== Monte Carlo Test ===")

    np.random.seed(42)

    # Generate synthetic returns (with some edge)
    n = 252 * 2  # 2 years
    bar_returns = np.random.randn(n) * 0.01 + 0.0003  # Small positive drift

    print(f"True total return: {np.sum(bar_returns):.4f}")
    print(f"True Sharpe: {np.sqrt(252) * np.mean(bar_returns) / np.std(bar_returns):.3f}")

    # Shuffle MC
    print("\nShuffle Monte Carlo:")
    shuffle_results = shuffle_bar_returns_mc(bar_returns, n_paths=500, seed=42)
    for metric, result in shuffle_results.items():
        print(f"  {metric}:")
        print(f"    Mean: {result.mean:.4f}, Std: {result.std:.4f}")
        print(f"    5-95%: [{result.p5:.4f}, {result.p95:.4f}]")
        print(f"    Ruin prob: {result.ruin_probability:.1%}")

    # Block bootstrap MC
    print("\nBlock Bootstrap Monte Carlo:")
    block_results = block_bootstrap_mc(bar_returns, block_len=20, n_paths=500, seed=42)
    for metric, result in block_results.items():
        print(f"  {metric}:")
        print(f"    Mean: {result.mean:.4f}, Std: {result.std:.4f}")
        print(f"    5-95%: [{result.p5:.4f}, {result.p95:.4f}]")

    # Ruin probability
    print("\nRuin Probability Analysis:")
    ruin = compute_ruin_probability(bar_returns, ruin_threshold=0.20, n_paths=500)
    print(f"  {ruin}")

    print("\n=== Test Complete ===")
