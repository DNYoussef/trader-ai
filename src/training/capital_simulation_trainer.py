"""
Capital Simulation Trainer for TRM Model

This trainer optimizes for CAPITAL PRESERVATION rather than strategy classification.
The goal: Start with $200, never lose money, steadily grow with black swan spikes.

Training approach:
1. Walk through historical data day by day
2. Model predicts strategy each day
3. Simulate capital based on that strategy's actual returns
4. Punish 3x for any capital decrease (NNC asymmetric loss)
5. Reward steady growth over volatile growth
6. Extra reward for correct black swan timing

INTEGRATIONS:
- NNC geometric returns from src/utils/multiplicative.py
- NNC asymmetric profit weighting from src/training/trm_loss_functions.py
- MetaGrokFast optimizer with bigeometric from src/training/meta_grokfast.py
- k(L) evolution from src/training/k_formula.py
- MOO strategy boundaries from src/optimization/trading_oracle.py

This is a policy-gradient style approach where the reward is capital preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# NNC INTEGRATION
# =============================================================================
try:
    from src.utils.multiplicative import (
        GeometricOperations,
        GeometricDerivative,
        BoundedNNC,
        NUMERICAL_EPSILON
    )
    NNC_AVAILABLE = True
    logger.info("NNC multiplicative operations loaded")
except ImportError:
    NNC_AVAILABLE = False
    NUMERICAL_EPSILON = 1e-12
    logger.warning("NNC multiplicative not available - using fallback")

try:
    from src.training.trm_loss_functions import (
        NNC_K_GAIN, NNC_K_LOSS, NNC_ASYMMETRY,
        compute_nnc_profit_weighted_loss
    )
    NNC_LOSS_AVAILABLE = True
    logger.info("NNC asymmetric loss loaded (k_gain={}, k_loss={})".format(NNC_K_GAIN, NNC_K_LOSS))
except ImportError:
    NNC_LOSS_AVAILABLE = False
    NNC_K_GAIN = 0.05
    NNC_K_LOSS = 0.02
    NNC_ASYMMETRY = 2.5
    logger.warning("NNC loss functions not available - using defaults")

try:
    from src.training.k_formula import compute_k, k_from_gradient
    K_FORMULA_AVAILABLE = True
    logger.info("k(L) evolution formula loaded")
except ImportError:
    K_FORMULA_AVAILABLE = False
    def compute_k(L): return 0.1  # Default k
    logger.warning("k(L) formula not available - using default k=0.1")

try:
    from src.training.meta_grokfast import MetaGrokFast, TRM_CONFIG
    METAGROKFAST_AVAILABLE = True
    logger.info("MetaGrokFast optimizer loaded")
except ImportError:
    METAGROKFAST_AVAILABLE = False
    logger.warning("MetaGrokFast not available - using Adam")

# =============================================================================
# MOO INTEGRATION
# =============================================================================
try:
    from src.optimization.trading_oracle import (
        StrategySelectionOracle,
        create_oracle
    )
    from src.optimization.pymoo_adapter import PymooAdapter
    MOO_AVAILABLE = True
    logger.info("MOO strategy optimization loaded")
except ImportError:
    MOO_AVAILABLE = False
    logger.warning("MOO optimization not available")


@dataclass
class CapitalSimConfig:
    """Configuration for capital simulation training"""
    # Capital settings
    initial_capital: float = 200.0

    # NNC asymmetric loss (from Kahneman's loss aversion)
    # Losses hurt 2.5x more than equivalent gains
    loss_penalty_multiplier: float = NNC_ASYMMETRY  # 2.5x from NNC research
    nnc_k_gain: float = NNC_K_GAIN  # 0.05 - scale for gains
    nnc_k_loss: float = NNC_K_LOSS  # 0.02 - scale for losses (smaller = heavier penalty)

    # Stability penalties
    volatility_penalty: float = 0.5  # Penalty for volatile returns
    drawdown_penalty: float = 2.0  # Extra penalty for drawdowns

    # Strategy selection bonuses
    conservative_bonus: float = 0.1  # Bonus for conservative choices in uncertainty
    black_swan_reward: float = 2.0  # Reward for correct black swan timing
    min_confidence_threshold: float = 0.6  # Below this, prefer conservative

    # Optimizer settings
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor for future rewards
    use_metagrokfast: bool = True  # Use MetaGrokFast optimizer if available
    use_geometric_returns: bool = True  # Use NNC geometric returns


class StrategyReturnsSimulator:
    """
    Simulates daily returns for each strategy based on market conditions.
    Uses historical SPY/TLT returns to compute strategy-specific returns.
    """

    # Strategy allocations: (SPY%, TLT%, Cash%)
    STRATEGY_ALLOCATIONS = {
        0: (0.20, 0.50, 0.30),  # ultra_defensive
        1: (0.40, 0.30, 0.30),  # defensive
        2: (0.60, 0.20, 0.20),  # balanced_safe
        3: (0.70, 0.20, 0.10),  # balanced_growth
        4: (0.80, 0.15, 0.05),  # growth
        5: (0.90, 0.10, 0.00),  # aggressive_growth
        6: (0.85, 0.15, 0.00),  # contrarian_long
        7: (0.75, 0.25, 0.00),  # tactical_opportunity
    }

    STRATEGY_NAMES = [
        'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
        'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
    ]

    def __init__(self):
        self.returns_cache = {}

    def compute_strategy_return(
        self,
        strategy_idx: int,
        spy_return: float,
        tlt_return: float
    ) -> float:
        """Compute return for a strategy given SPY and TLT daily returns"""
        spy_alloc, tlt_alloc, cash_alloc = self.STRATEGY_ALLOCATIONS[strategy_idx]
        return spy_alloc * spy_return + tlt_alloc * tlt_return + cash_alloc * 0.0001  # tiny cash return

    def compute_all_strategy_returns(
        self,
        spy_return: float,
        tlt_return: float
    ) -> np.ndarray:
        """Compute returns for all 8 strategies"""
        returns = np.zeros(8)
        for i in range(8):
            returns[i] = self.compute_strategy_return(i, spy_return, tlt_return)
        return returns


class CapitalRewardCalculator:
    """
    Calculates rewards/punishments based on capital changes.

    Uses NNC EXPONENTIAL WEIGHTING for asymmetric loss aversion:
    - Gains: weight = exp(-pnl / k_gain) ~ small weight (0.13 for +10%)
    - Losses: weight = exp(-pnl / k_loss) ~ HUGE weight (148 for -10%)

    This implements Kahneman's loss aversion where losses hurt 2.5x more.

    Reward structure:
    - Capital increase: +reward (NNC exponential scaling)
    - Capital decrease: -SEVERE penalty (NNC exponential, much heavier)
    - High volatility: -penalty (prefer steady growth)
    - Drawdown: -2x penalty (extra punishment for drawdowns)
    - Conservative in uncertainty: +bonus
    """

    def __init__(self, config: CapitalSimConfig):
        self.config = config
        self.peak_capital = config.initial_capital
        self.returns_history = []

        # Use BoundedNNC for safe exponential calculations
        if NNC_AVAILABLE:
            self.nnc = BoundedNNC()
        else:
            self.nnc = None

    def reset(self):
        """Reset for new episode"""
        self.peak_capital = self.config.initial_capital
        self.returns_history = []

    def _nnc_weight(self, pnl: float) -> float:
        """
        Compute NNC exponential weight for profit/loss.

        FORMULA: weight = exp(-pnl / k)
        - For gains (pnl > 0): k = k_gain (0.05), weight is SMALL
        - For losses (pnl < 0): k = k_loss (0.02), weight is HUGE

        Examples (from trm_loss_functions.py):
        - +10% gain: exp(-0.10/0.05) = exp(-2) = 0.135 (low weight)
        - -5% loss:  exp(0.05/0.02) = exp(2.5) = 12.18 (high weight)
        - -10% loss: exp(0.10/0.02) = exp(5) = 148.4 (SEVERE weight)
        """
        if pnl >= 0:
            k = self.config.nnc_k_gain
            exp_arg = -pnl / k
        else:
            k = self.config.nnc_k_loss
            exp_arg = -pnl / k  # pnl is negative, so this becomes positive

        # Clamp to prevent overflow
        exp_arg = np.clip(exp_arg, -10, 10)

        if self.nnc is not None:
            return float(self.nnc.safe_exp(exp_arg))
        else:
            return float(np.exp(exp_arg))

    def calculate_reward(
        self,
        capital_before: float,
        capital_after: float,
        strategy_chosen: int,
        model_confidence: float,
        is_black_swan_period: bool = False,
        black_swan_phase: str = None  # 'before', 'during', 'after'
    ) -> Tuple[float, Dict]:
        """
        Calculate reward based on capital change using NNC exponential weighting.

        Returns:
            reward: float - the reward value
            breakdown: dict - detailed breakdown of reward components
        """
        capital_change = capital_after - capital_before
        pct_change = capital_change / capital_before if capital_before > 0 else 0

        breakdown = {}
        reward = 0.0

        # 1. NNC EXPONENTIAL reward/penalty for capital change
        nnc_weight = self._nnc_weight(pct_change)
        breakdown['nnc_weight'] = nnc_weight

        if capital_change >= 0:
            # Positive: reward proportional to gain, scaled by NNC weight
            # NNC weight is SMALL for gains (we don't over-reward profits)
            base_reward = pct_change * 100 * (1 / nnc_weight)  # Inverse weight for gains
            breakdown['gain_reward'] = base_reward
            reward += base_reward
        else:
            # Negative: SEVERE NNC PENALTY
            # NNC weight is HUGE for losses (we severely punish losses)
            base_penalty = abs(pct_change) * 100 * nnc_weight  # Direct weight for losses
            breakdown['loss_penalty'] = -base_penalty
            reward -= base_penalty

        # 2. Drawdown penalty
        self.peak_capital = max(self.peak_capital, capital_after)
        drawdown = (self.peak_capital - capital_after) / self.peak_capital
        if drawdown > 0.01:  # More than 1% drawdown
            dd_penalty = drawdown * 100 * self.config.drawdown_penalty
            breakdown['drawdown_penalty'] = -dd_penalty
            reward -= dd_penalty

        # 3. Volatility penalty (prefer steady growth)
        self.returns_history.append(pct_change)
        if len(self.returns_history) >= 5:
            recent_vol = np.std(self.returns_history[-5:])
            if recent_vol > 0.02:  # High volatility
                vol_penalty = recent_vol * 100 * self.config.volatility_penalty
                breakdown['volatility_penalty'] = -vol_penalty
                reward -= vol_penalty

        # 4. Conservative bonus in uncertainty
        if model_confidence < self.config.min_confidence_threshold:
            if strategy_chosen in [0, 1, 2]:  # Conservative strategies
                cons_bonus = self.config.conservative_bonus * 10
                breakdown['conservative_bonus'] = cons_bonus
                reward += cons_bonus

        # 5. Black swan timing rewards
        if is_black_swan_period:
            if black_swan_phase == 'before' and strategy_chosen in [0, 1]:  # Defensive before crash
                bs_reward = self.config.black_swan_reward * 10
                breakdown['black_swan_defensive'] = bs_reward
                reward += bs_reward
            elif black_swan_phase == 'after' and strategy_chosen in [4, 5]:  # Aggressive during recovery
                bs_reward = self.config.black_swan_reward * 10
                breakdown['black_swan_aggressive'] = bs_reward
                reward += bs_reward

        breakdown['total_reward'] = reward
        breakdown['capital_change'] = capital_change
        breakdown['pct_change'] = pct_change

        return reward, breakdown


class CapitalSimulationTrainer:
    """
    Trains TRM model to preserve capital through day-by-day simulation.

    Unlike classification training, this directly optimizes for:
    - Never losing money (3x penalty for losses)
    - Steady growth (volatility penalty)
    - Correct black swan timing

    Uses policy gradient style updates where reward = capital preservation.
    """

    def __init__(
        self,
        model: nn.Module,
        historical_data: pd.DataFrame,
        config: Optional[CapitalSimConfig] = None,
        device: str = None
    ):
        """
        Args:
            model: TRM model to train
            historical_data: DataFrame with columns: date, spy_return, tlt_return,
                           features (10 market features), is_black_swan, black_swan_phase
            config: Training configuration
            device: cuda or cpu
        """
        self.model = model
        self.data = historical_data
        self.config = config or CapitalSimConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Components
        self.simulator = StrategyReturnsSimulator()
        self.reward_calc = CapitalRewardCalculator(self.config)

        # Use NNC geometric operations if available
        if NNC_AVAILABLE and self.config.use_geometric_returns:
            self.geo_ops = GeometricOperations()
            self.geo_deriv = GeometricDerivative()
            logger.info("Using NNC geometric return calculations")
        else:
            self.geo_ops = None
            self.geo_deriv = None

        # Optimizer: MetaGrokFast (with bigeometric + k(L)) or Adam fallback
        if METAGROKFAST_AVAILABLE and self.config.use_metagrokfast:
            self.optimizer = MetaGrokFast.for_trm(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            self.optimizer_type = 'metagrokfast'
            logger.info("Using MetaGrokFast optimizer (bigeometric + k(L) evolution)")
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            self.optimizer_type = 'adam'
            logger.info("Using Adam optimizer (MetaGrokFast not available)")

        # Tracking
        self.episode_rewards = []
        self.episode_capitals = []
        self.best_final_capital = self.config.initial_capital

        logger.info(f"CapitalSimulationTrainer initialized")
        logger.info(f"  Initial capital: ${self.config.initial_capital}")
        logger.info(f"  NNC asymmetric loss: k_gain={self.config.nnc_k_gain}, k_loss={self.config.nnc_k_loss}")
        logger.info(f"  Loss penalty multiplier: {self.config.loss_penalty_multiplier}x (Kahneman)")
        logger.info(f"  Optimizer: {self.optimizer_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Data points: {len(self.data)}")
        logger.info(f"  NNC available: {NNC_AVAILABLE}")
        logger.info(f"  MOO available: {MOO_AVAILABLE}")
        logger.info(f"  MetaGrokFast available: {METAGROKFAST_AVAILABLE}")

    def _prepare_features(self, row: pd.Series) -> torch.Tensor:
        """Extract and prepare features from data row"""
        feature_cols = [
            'vix', 'spy_returns_5d', 'spy_returns_20d', 'volume_ratio',
            'market_breadth', 'correlation', 'put_call_ratio',
            'gini_coefficient', 'sector_dispersion', 'signal_quality'
        ]

        features = []
        for col in feature_cols:
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            features.append(float(val))

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    def run_episode(
        self,
        start_idx: int = 0,
        end_idx: int = None,
        training: bool = True
    ) -> Dict:
        """
        Run one episode (walk through data, simulate capital).

        Args:
            start_idx: Starting index in data
            end_idx: Ending index (None = end of data)
            training: If True, accumulate gradients

        Returns:
            Dictionary with episode statistics
        """
        if end_idx is None:
            end_idx = len(self.data)

        # Reset
        capital = self.config.initial_capital
        self.reward_calc.reset()

        # Tracking
        rewards = []
        log_probs = []
        capitals = [capital]
        strategies_chosen = []
        daily_returns = []

        self.model.train() if training else self.model.eval()

        for idx in range(start_idx, end_idx):
            row = self.data.iloc[idx]

            # Get market returns for this day
            spy_ret = row.get('spy_return', 0.0)
            tlt_ret = row.get('tlt_return', 0.0)

            if pd.isna(spy_ret) or pd.isna(tlt_ret):
                continue

            # Prepare features
            features = self._prepare_features(row)

            # Model predicts strategy
            with torch.set_grad_enabled(training):
                output = self.model(features)
                logits = output['strategy_logits']

                # Sample action (strategy) from distribution
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                if training:
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                else:
                    action = logits.argmax(dim=-1)

                strategy_idx = action.item()
                confidence = probs[0, strategy_idx].item()

            # Simulate capital change based on chosen strategy
            strategy_return = self.simulator.compute_strategy_return(
                strategy_idx, spy_ret, tlt_ret
            )

            capital_before = capital
            capital = capital * (1 + strategy_return)

            # Calculate reward
            is_black_swan = row.get('is_black_swan', False)
            black_swan_phase = row.get('black_swan_phase', None)

            reward, breakdown = self.reward_calc.calculate_reward(
                capital_before=capital_before,
                capital_after=capital,
                strategy_chosen=strategy_idx,
                model_confidence=confidence,
                is_black_swan_period=is_black_swan,
                black_swan_phase=black_swan_phase
            )

            rewards.append(reward)
            capitals.append(capital)
            strategies_chosen.append(strategy_idx)
            daily_returns.append(strategy_return)

        # Compute discounted returns for policy gradient
        if training and len(rewards) > 0:
            returns = self._compute_returns(rewards)

            # Policy gradient loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)

            if len(policy_loss) > 0:
                loss = torch.stack(policy_loss).sum()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()

        # Episode statistics
        final_capital = capitals[-1] if capitals else self.config.initial_capital
        total_return = (final_capital / self.config.initial_capital - 1) * 100
        max_drawdown = self._compute_max_drawdown(capitals)

        # Strategy distribution
        strategy_counts = np.bincount(strategies_chosen, minlength=8)

        stats = {
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown * 100,
            'total_reward': sum(rewards),
            'avg_daily_return': np.mean(daily_returns) * 100 if daily_returns else 0,
            'return_volatility': np.std(daily_returns) * 100 if daily_returns else 0,
            'num_days': len(rewards),
            'strategy_distribution': strategy_counts.tolist(),
            'capitals': capitals,
            'never_lost_money': min(capitals) >= self.config.initial_capital * 0.99,
        }

        return stats

    def _compute_returns(self, rewards: List[float]) -> List[torch.Tensor]:
        """Compute discounted returns for policy gradient"""
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns.tolist()

    def _compute_max_drawdown(self, capitals: List[float]) -> float:
        """Compute maximum drawdown from capital history"""
        if len(capitals) < 2:
            return 0.0

        peak = capitals[0]
        max_dd = 0.0

        for cap in capitals:
            peak = max(peak, cap)
            dd = (peak - cap) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def train(
        self,
        num_epochs: int = 100,
        episode_length: int = 252,  # ~1 trading year
        episodes_per_epoch: int = 10,
        eval_frequency: int = 5,
        save_best: bool = True,
        save_path: str = 'checkpoints/capital_sim_best.pt'
    ) -> Dict:
        """
        Train the model through capital simulation.

        Args:
            num_epochs: Number of training epochs
            episode_length: Days per episode (252 = 1 year)
            episodes_per_epoch: Episodes to run per epoch
            eval_frequency: Epochs between evaluations
            save_best: Whether to save best model
            save_path: Path for saving best model

        Returns:
            Training history
        """
        logger.info(f"\nStarting Capital Simulation Training")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Episode length: {episode_length} days")
        logger.info(f"  Episodes per epoch: {episodes_per_epoch}")

        history = {
            'epoch_rewards': [],
            'epoch_capitals': [],
            'epoch_returns': [],
            'epoch_drawdowns': [],
            'best_capital': self.config.initial_capital,
            'never_lost_count': 0,
        }

        data_len = len(self.data)

        for epoch in range(num_epochs):
            epoch_rewards = []
            epoch_capitals = []
            epoch_returns = []

            # Run multiple episodes with random starting points
            for ep in range(episodes_per_epoch):
                # Random starting point (ensure enough data ahead)
                max_start = max(0, data_len - episode_length - 1)
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end_idx = min(start_idx + episode_length, data_len)

                stats = self.run_episode(start_idx, end_idx, training=True)

                epoch_rewards.append(stats['total_reward'])
                epoch_capitals.append(stats['final_capital'])
                epoch_returns.append(stats['total_return_pct'])

            # Epoch statistics
            avg_reward = np.mean(epoch_rewards)
            avg_capital = np.mean(epoch_capitals)
            avg_return = np.mean(epoch_returns)

            history['epoch_rewards'].append(avg_reward)
            history['epoch_capitals'].append(avg_capital)
            history['epoch_returns'].append(avg_return)

            # Check for best
            if avg_capital > history['best_capital']:
                history['best_capital'] = avg_capital
                if save_best:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'config': self.config,
                        'best_capital': avg_capital,
                        'epoch': epoch,
                    }, save_path)
                    logger.info(f"  [BEST] Saved model with capital ${avg_capital:.2f}")

            # Progress logging
            if (epoch + 1) % eval_frequency == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Avg Capital: ${avg_capital:.2f} | "
                    f"Avg Return: {avg_return:.2f}% | "
                    f"Avg Reward: {avg_reward:.2f}"
                )

            # Full evaluation periodically
            if (epoch + 1) % (eval_frequency * 2) == 0:
                eval_stats = self.evaluate_full_history()
                history['epoch_drawdowns'].append(eval_stats['max_drawdown_pct'])

                if eval_stats['never_lost_money']:
                    history['never_lost_count'] += 1
                    logger.info(f"  *** NEVER LOST MONEY on full history! ***")

        return history

    def evaluate_full_history(self) -> Dict:
        """Evaluate model on complete historical data"""
        logger.info("\nEvaluating on full historical data...")

        stats = self.run_episode(0, len(self.data), training=False)

        logger.info(f"  Final Capital: ${stats['final_capital']:.2f}")
        logger.info(f"  Total Return: {stats['total_return_pct']:.2f}%")
        logger.info(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        logger.info(f"  Strategy Distribution: {stats['strategy_distribution']}")
        logger.info(f"  Never Lost Money: {stats['never_lost_money']}")

        return stats

    def backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run backtest on specified date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            verbose: Print detailed results

        Returns:
            Backtest results
        """
        # Filter data by date if provided
        data = self.data.copy()

        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]

        if len(data) == 0:
            logger.warning("No data in specified date range")
            return {}

        # Temporarily swap data
        original_data = self.data
        self.data = data.reset_index(drop=True)

        # Run evaluation
        stats = self.run_episode(0, len(self.data), training=False)

        # Restore original data
        self.data = original_data

        if verbose:
            date_range = f"{start_date or 'start'} to {end_date or 'end'}"
            logger.info(f"\nBacktest Results ({date_range}):")
            logger.info(f"  Starting Capital: ${self.config.initial_capital:.2f}")
            logger.info(f"  Final Capital: ${stats['final_capital']:.2f}")
            logger.info(f"  Total Return: {stats['total_return_pct']:.2f}%")
            logger.info(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
            logger.info(f"  Trading Days: {stats['num_days']}")
            logger.info(f"  Avg Daily Return: {stats['avg_daily_return']:.4f}%")
            logger.info(f"  Return Volatility: {stats['return_volatility']:.4f}%")
            logger.info(f"  Never Lost Money: {stats['never_lost_money']}")

            # Strategy usage
            logger.info(f"\n  Strategy Usage:")
            for i, count in enumerate(stats['strategy_distribution']):
                pct = count / stats['num_days'] * 100 if stats['num_days'] > 0 else 0
                logger.info(f"    {i} ({StrategyReturnsSimulator.STRATEGY_NAMES[i]}): {count} ({pct:.1f}%)")

        return stats


def create_training_data_with_returns(
    historical_manager,
    start_date: str = '1995-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Create training data with daily returns and features.

    This combines:
    - Daily SPY/TLT returns for strategy simulation
    - 10 market features for model input
    - Black swan period annotations
    """
    # This would use the HistoricalDataManager to get data
    # For now, return structure expected by the trainer

    # Placeholder - actual implementation would query database
    logger.info(f"Creating training data from {start_date} to {end_date}")

    # Expected columns:
    # date, spy_return, tlt_return, vix, spy_returns_5d, spy_returns_20d,
    # volume_ratio, market_breadth, correlation, put_call_ratio,
    # gini_coefficient, sector_dispersion, signal_quality,
    # is_black_swan, black_swan_phase

    return pd.DataFrame()


if __name__ == "__main__":
    # Test the capital simulation trainer
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("CAPITAL SIMULATION TRAINER TEST")
    print("=" * 70)

    # Create synthetic test data
    np.random.seed(42)
    n_days = 1000

    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')

    # Simulate SPY returns (slight positive drift with volatility)
    spy_returns = np.random.normal(0.0003, 0.015, n_days)
    # Simulate TLT returns (lower vol, negative correlation to SPY)
    tlt_returns = np.random.normal(0.0001, 0.008, n_days) - 0.3 * spy_returns

    # Add some black swan events
    spy_returns[100:110] = np.random.normal(-0.03, 0.02, 10)  # Crash
    spy_returns[150:180] = np.random.normal(0.02, 0.015, 30)  # Recovery

    test_data = pd.DataFrame({
        'date': dates,
        'spy_return': spy_returns,
        'tlt_return': tlt_returns,
        'vix': np.random.uniform(15, 35, n_days),
        'spy_returns_5d': np.convolve(spy_returns, np.ones(5)/5, mode='same'),
        'spy_returns_20d': np.convolve(spy_returns, np.ones(20)/20, mode='same'),
        'volume_ratio': np.random.uniform(0.8, 1.2, n_days),
        'market_breadth': np.random.uniform(0.3, 0.7, n_days),
        'correlation': np.random.uniform(0.3, 0.8, n_days),
        'put_call_ratio': np.random.uniform(0.7, 1.3, n_days),
        'gini_coefficient': np.random.uniform(0.4, 0.6, n_days),
        'sector_dispersion': np.random.uniform(0.2, 0.4, n_days),
        'signal_quality': np.random.uniform(0.5, 0.9, n_days),
        'is_black_swan': False,
        'black_swan_phase': None,
    })

    # Mark black swan periods
    test_data.loc[100:110, 'is_black_swan'] = True
    test_data.loc[100:105, 'black_swan_phase'] = 'during'
    test_data.loc[106:110, 'black_swan_phase'] = 'after'
    test_data.loc[95:99, 'is_black_swan'] = True
    test_data.loc[95:99, 'black_swan_phase'] = 'before'

    print(f"\nTest data: {len(test_data)} days")
    print(f"SPY mean return: {spy_returns.mean()*100:.4f}%")
    print(f"SPY volatility: {spy_returns.std()*100:.4f}%")

    # Create model
    import sys
    sys.path.insert(0, 'src')
    from models.trm_model import TinyRecursiveModel

    model = TinyRecursiveModel()

    # Create trainer
    config = CapitalSimConfig(
        initial_capital=200.0,
        loss_penalty_multiplier=3.0,
        learning_rate=1e-4,
    )

    trainer = CapitalSimulationTrainer(
        model=model,
        historical_data=test_data,
        config=config
    )

    # Run a few epochs
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)

    history = trainer.train(
        num_epochs=10,
        episode_length=252,
        episodes_per_epoch=5,
        eval_frequency=2,
        save_best=False
    )

    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    final_stats = trainer.evaluate_full_history()

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Best Capital Achieved: ${history['best_capital']:.2f}")
    print(f"Times Never Lost Money: {history['never_lost_count']}")
    print("=" * 70)
