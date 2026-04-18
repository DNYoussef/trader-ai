"""
Strategy Specification Module

Defines the StrategySpec dataclass that specifies a trading strategy's
parameters, validation status, and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime
import json


@dataclass
class StrategySpec:
    """
    Specification for a trading strategy (existing or generated).

    Used for:
    1. Defining existing strategies from black_swan_strategies.py
    2. Specifying generated strategies with parameters
    3. Tracking MCPT validation results
    4. Storing performance metrics

    Attributes:
        name: Unique strategy identifier
        strategy_id: Integer ID (0-7 for existing strategies)
        template_id: Base template for generated strategies
        params: Strategy-specific parameters
        market: Target market (e.g., 'SPY', 'QQQ')
        timeframe: Bar timeframe (e.g., '1H', '1D')
        fees_bps: Transaction costs in basis points

        # MCPT Validation Results
        insample_pvalue: p-value from in-sample MCPT (< 0.01 to pass)
        walkforward_pvalue: p-value from walk-forward MCPT (< 0.05 to pass)
        is_validated: True if strategy passed all validation gates

        # Performance Metrics
        profit_factor: Sum of gains / sum of losses
        sharpe_ratio: Risk-adjusted return
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return / max drawdown
        max_drawdown: Maximum peak-to-trough decline
        total_trades: Number of trades
        win_rate: Percentage of winning trades
    """

    # Identity
    name: str
    strategy_id: int = -1  # -1 for generated, 0-7 for existing
    template_id: str = ""
    params: Dict = field(default_factory=dict)

    # Target
    market: str = "SPY"
    timeframe: str = "1H"
    fees_bps: float = 10.0

    # MCPT Validation
    insample_pvalue: float = 1.0
    walkforward_pvalue: float = 1.0
    is_validated: bool = False

    # Performance Metrics
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

    # Metadata
    created_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'strategy_id': self.strategy_id,
            'template_id': self.template_id,
            'params': self.params,
            'market': self.market,
            'timeframe': self.timeframe,
            'fees_bps': self.fees_bps,
            'insample_pvalue': self.insample_pvalue,
            'walkforward_pvalue': self.walkforward_pvalue,
            'is_validated': self.is_validated,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'validated_at': self.validated_at.isoformat() if self.validated_at else None,
            'description': self.description,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategySpec':
        """Create from dictionary."""
        # Handle datetime fields
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])

        validated_at = None
        if data.get('validated_at'):
            validated_at = datetime.fromisoformat(data['validated_at'])

        return cls(
            name=data['name'],
            strategy_id=data.get('strategy_id', -1),
            template_id=data.get('template_id', ''),
            params=data.get('params', {}),
            market=data.get('market', 'SPY'),
            timeframe=data.get('timeframe', '1H'),
            fees_bps=data.get('fees_bps', 10.0),
            insample_pvalue=data.get('insample_pvalue', 1.0),
            walkforward_pvalue=data.get('walkforward_pvalue', 1.0),
            is_validated=data.get('is_validated', False),
            profit_factor=data.get('profit_factor', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            sortino_ratio=data.get('sortino_ratio', 0.0),
            calmar_ratio=data.get('calmar_ratio', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            total_trades=data.get('total_trades', 0),
            win_rate=data.get('win_rate', 0.0),
            created_at=created_at,
            validated_at=validated_at,
            description=data.get('description', ''),
            tags=data.get('tags', []),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'StrategySpec':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def passes_gate(self, gate_level: int) -> bool:
        """
        Check if strategy passes the specified gate level.

        Gate thresholds:
        - G0-G2: insample < 0.05, walkforward < 0.10 (learning)
        - G3-G5: insample < 0.02, walkforward < 0.05 (real money)
        - G6-G8: insample < 0.01, walkforward < 0.02 (significant capital)
        - G9-G12: insample < 0.005, walkforward < 0.01 (institutional)
        """
        if gate_level <= 2:
            return self.insample_pvalue < 0.05 and self.walkforward_pvalue < 0.10
        elif gate_level <= 5:
            return self.insample_pvalue < 0.02 and self.walkforward_pvalue < 0.05
        elif gate_level <= 8:
            return self.insample_pvalue < 0.01 and self.walkforward_pvalue < 0.02
        else:
            return self.insample_pvalue < 0.005 and self.walkforward_pvalue < 0.01


# Pre-defined specs for the 8 existing strategies
EXISTING_STRATEGIES = [
    StrategySpec(
        name="tail_hedge",
        strategy_id=0,
        description="Buys deep OTM puts when VIX is elevated but not extreme",
        tags=["protection", "tail_risk", "options"],
    ),
    StrategySpec(
        name="volatility_harvest",
        strategy_id=1,
        description="Goes long volatility when term structure inverts (backwardation)",
        tags=["volatility", "vix", "contango"],
    ),
    StrategySpec(
        name="crisis_alpha",
        strategy_id=2,
        description="Safe haven playbook during high VIX crisis periods",
        tags=["crisis", "safe_haven", "gld", "tlt"],
    ),
    StrategySpec(
        name="momentum_explosion",
        strategy_id=3,
        description="Captures strong trend breakouts in SPY/QQQ",
        tags=["momentum", "trend", "breakout"],
    ),
    StrategySpec(
        name="mean_reversion",
        strategy_id=4,
        description="Mean reversion on extreme moves in oversold assets",
        tags=["mean_reversion", "oversold", "bounce"],
    ),
    StrategySpec(
        name="correlation_breakdown",
        strategy_id=5,
        description="Trades decorrelated assets during regime changes",
        tags=["correlation", "regime_change", "decorrelation"],
    ),
    StrategySpec(
        name="inequality_arbitrage",
        strategy_id=6,
        description="Gary plays - exploiting inequality blindness in markets",
        tags=["inequality", "gary", "social"],
    ),
    StrategySpec(
        name="event_catalyst",
        strategy_id=7,
        description="Event-driven strategy for pre/post event positioning",
        tags=["event", "catalyst", "earnings"],
    ),
]


def get_strategy_spec(strategy_id: int) -> Optional[StrategySpec]:
    """Get strategy spec by ID (0-7)."""
    if 0 <= strategy_id < len(EXISTING_STRATEGIES):
        return EXISTING_STRATEGIES[strategy_id]
    return None


def get_strategy_by_name(name: str) -> Optional[StrategySpec]:
    """Get strategy spec by name."""
    for spec in EXISTING_STRATEGIES:
        if spec.name == name:
            return spec
    return None
