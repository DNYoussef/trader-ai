"""
Gate Management System for Capital-Based Trading Progression

This module implements a comprehensive gate system that manages trading
privileges and constraints based on capital levels, enforcing risk management
rules and tracking performance metrics for graduation/downgrade decisions.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GateLevel(Enum):
    """Trading gate levels based on capital ranges."""
    G0 = "G0"  # $200-499
    G1 = "G1"  # $500-999
    G2 = "G2"  # $1k-2.5k
    G3 = "G3"  # $2.5k-5k

class ViolationType(Enum):
    """Types of gate violations."""
    ASSET_NOT_ALLOWED = "asset_not_allowed"
    CASH_FLOOR_VIOLATION = "cash_floor_violation"
    OPTIONS_NOT_ALLOWED = "options_not_allowed"
    THETA_LIMIT_EXCEEDED = "theta_limit_exceeded"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    CONCENTRATION_EXCEEDED = "concentration_exceeded"

@dataclass
class GateConfig:
    """Configuration for a specific gate level."""
    level: GateLevel
    capital_min: float
    capital_max: float
    allowed_assets: Set[str]
    cash_floor_pct: float
    options_enabled: bool
    max_theta_pct: Optional[float] = None
    max_position_pct: float = 0.20  # 20% max position size
    max_concentration_pct: float = 0.30  # 30% max sector concentration
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.capital_min >= self.capital_max:
            raise ValueError(f"Invalid capital range: {self.capital_min} >= {self.capital_max}")
        if not 0 <= self.cash_floor_pct <= 1:
            raise ValueError(f"Invalid cash floor percentage: {self.cash_floor_pct}")
        if self.max_theta_pct is not None and not 0 <= self.max_theta_pct <= 1:
            raise ValueError(f"Invalid theta percentage: {self.max_theta_pct}")

@dataclass
class TradeValidationResult:
    """Result of trade validation against gate constraints."""
    is_valid: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_violation(self, violation_type: ViolationType, message: str, details: Dict[str, Any] = None):
        """Add a violation to the result."""
        self.is_valid = False
        self.violations.append({
            'type': violation_type.value,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, message: str, details: Dict[str, Any] = None):
        """Add a warning to the result."""
        self.warnings.append({
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })

@dataclass
class ViolationRecord:
    """Record of a gate violation."""
    timestamp: datetime
    gate_level: GateLevel
    violation_type: ViolationType
    message: str
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class GraduationMetrics:
    """Metrics tracked for gate graduation/downgrade decisions."""
    consecutive_compliant_days: int = 0
    total_violations_30d: int = 0
    avg_cash_utilization_30d: float = 0.0
    max_drawdown_30d: float = 0.0
    sharpe_ratio_30d: Optional[float] = None
    last_violation_date: Optional[datetime] = None
    performance_score: float = 0.0

class GateManager:
    """Manages trading gates, validation, and progression."""
    
    def __init__(self, data_dir: str = "./data/gates"):
        """Initialize the gate manager with configuration."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize gate configurations
        self.gate_configs = self._initialize_gate_configs()
        
        # Current state
        self.current_gate = GateLevel.G0
        self.current_capital = 0.0
        self.violation_history: List[ViolationRecord] = []
        self.graduation_metrics = GraduationMetrics()
        
        # Load persisted state
        self._load_state()
    
    def _initialize_gate_configs(self) -> Dict[GateLevel, GateConfig]:
        """Initialize all gate configurations."""
        configs = {}
        
        # G0: $200-499, ULTY/AMDY only, 50% cash floor, no options
        configs[GateLevel.G0] = GateConfig(
            level=GateLevel.G0,
            capital_min=200.0,
            capital_max=499.99,
            allowed_assets={'ULTY', 'AMDY'},
            cash_floor_pct=0.50,
            options_enabled=False,
            max_position_pct=0.25,  # More conservative for beginners
            max_concentration_pct=0.40
        )
        
        # G1: $500-999, adds IAU/GLDM/VTIP, 60% cash floor
        configs[GateLevel.G1] = GateConfig(
            level=GateLevel.G1,
            capital_min=500.0,
            capital_max=999.99,
            allowed_assets={'ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP'},
            cash_floor_pct=0.60,
            options_enabled=False,
            max_position_pct=0.22,
            max_concentration_pct=0.35
        )
        
        # G2: $1k-2.5k, adds factor ETFs, 65% cash floor
        configs[GateLevel.G2] = GateConfig(
            level=GateLevel.G2,
            capital_min=1000.0,
            capital_max=2499.99,
            allowed_assets={
                'ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP',
                'VTI', 'VTV', 'VUG', 'VEA', 'VWO',  # Factor ETFs
                'SCHD', 'DGRO', 'NOBL', 'VYM'  # Dividend ETFs
            },
            cash_floor_pct=0.65,
            options_enabled=False,
            max_position_pct=0.20,
            max_concentration_pct=0.30
        )
        
        # G3: $2.5k-5k, enables long options, 70% cash floor, 0.5% theta
        configs[GateLevel.G3] = GateConfig(
            level=GateLevel.G3,
            capital_min=2500.0,
            capital_max=4999.99,
            allowed_assets={
                'ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP',
                'VTI', 'VTV', 'VUG', 'VEA', 'VWO',
                'SCHD', 'DGRO', 'NOBL', 'VYM',
                'SPY', 'QQQ', 'IWM', 'DIA'  # Options-eligible ETFs
            },
            cash_floor_pct=0.70,
            options_enabled=True,
            max_theta_pct=0.005,  # 0.5% theta limit
            max_position_pct=0.20,
            max_concentration_pct=0.30
        )
        
        return configs
    
    def get_current_config(self) -> GateConfig:
        """Get the configuration for the current gate level."""
        return self.gate_configs[self.current_gate]
    
    def update_capital(self, new_capital: float) -> bool:
        """Update current capital and check if gate change is needed."""
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        # Check if we need to change gates based on capital
        new_gate = self._determine_gate_by_capital(new_capital)
        
        if new_gate != self.current_gate:
            logger.info(f"Capital change from ${old_capital:.2f} to ${new_capital:.2f} "
                       f"triggers gate change from {self.current_gate.value} to {new_gate.value}")
            self.current_gate = new_gate
            self._save_state()
            return True
        
        return False
    
    def _determine_gate_by_capital(self, capital: float) -> GateLevel:
        """Determine appropriate gate level based on capital amount."""
        for gate_level, config in self.gate_configs.items():
            if config.capital_min <= capital <= config.capital_max:
                return gate_level
        
        # If capital exceeds all gates, stay at highest gate
        if capital > max(config.capital_max for config in self.gate_configs.values()):
            return GateLevel.G3
        
        # If capital is below minimum, use G0
        return GateLevel.G0
    
    def validate_trade(self, trade_details: Dict[str, Any],
                      current_portfolio: Dict[str, Any]) -> TradeValidationResult:
        """
        Validate a trade against current gate constraints.

        Args:
            trade_details: Dict containing trade information
                - symbol: str, asset symbol
                - side: str, 'BUY' or 'SELL'
                - quantity: float, number of shares/contracts
                - price: float, expected execution price
                - trade_type: str, 'STOCK' or 'OPTION'
                - option_type: str, 'CALL' or 'PUT' (if applicable)
                - theta: float, theta exposure (if options)
                - kelly_percentage: float, optional Kelly % for enhanced validation
            current_portfolio: Dict containing current portfolio state
                - cash: float, available cash
                - positions: Dict[str, Dict], current positions
                - total_value: float, total portfolio value

        Returns:
            TradeValidationResult with validation outcome
        """
        result = TradeValidationResult(is_valid=True)
        config = self.get_current_config()
        
        symbol = trade_details.get('symbol', '').upper()
        trade_type = trade_details.get('trade_type', 'STOCK').upper()
        side = trade_details.get('side', '').upper()
        quantity = trade_details.get('quantity', 0)
        price = trade_details.get('price', 0)
        
        # 1. Check if asset is allowed
        if symbol not in config.allowed_assets:
            result.add_violation(
                ViolationType.ASSET_NOT_ALLOWED,
                f"Asset {symbol} not allowed in {config.level.value}",
                {'symbol': symbol, 'allowed_assets': list(config.allowed_assets)}
            )
        
        # 2. Check options permissions
        if trade_type == 'OPTION' and not config.options_enabled:
            result.add_violation(
                ViolationType.OPTIONS_NOT_ALLOWED,
                f"Options trading not enabled for {config.level.value}",
                {'trade_type': trade_type}
            )
        
        # 3. Check cash floor after trade
        trade_value = quantity * price
        if side == 'BUY':
            post_trade_cash = current_portfolio.get('cash', 0) - trade_value
            required_cash = current_portfolio.get('total_value', 0) * config.cash_floor_pct
            
            if post_trade_cash < required_cash:
                result.add_violation(
                    ViolationType.CASH_FLOOR_VIOLATION,
                    f"Trade would violate {config.cash_floor_pct*100:.0f}% cash floor",
                    {
                        'post_trade_cash': post_trade_cash,
                        'required_cash': required_cash,
                        'cash_floor_pct': config.cash_floor_pct
                    }
                )
        
        # 4. Check theta limits for options
        if (trade_type == 'OPTION' and config.max_theta_pct is not None and 
            side == 'BUY'):  # Only check theta for long options
            
            current_theta = current_portfolio.get('total_theta', 0)
            trade_theta = trade_details.get('theta', 0) * quantity
            new_total_theta = abs(current_theta + trade_theta)
            max_theta = current_portfolio.get('total_value', 0) * config.max_theta_pct
            
            if new_total_theta > max_theta:
                result.add_violation(
                    ViolationType.THETA_LIMIT_EXCEEDED,
                    f"Trade would exceed {config.max_theta_pct*100:.1f}% theta limit",
                    {
                        'current_theta': current_theta,
                        'trade_theta': trade_theta,
                        'new_total_theta': new_total_theta,
                        'max_theta': max_theta
                    }
                )
        
        # 5. Check position size limits
        if side == 'BUY':
            current_position_value = 0
            positions = current_portfolio.get('positions', {})
            if symbol in positions:
                pos = positions[symbol]
                current_position_value = pos.get('quantity', 0) * pos.get('current_price', price)
            
            new_position_value = current_position_value + trade_value
            max_position_value = current_portfolio.get('total_value', 0) * config.max_position_pct
            
            if new_position_value > max_position_value:
                result.add_violation(
                    ViolationType.POSITION_SIZE_EXCEEDED,
                    f"Trade would exceed {config.max_position_pct*100:.0f}% position size limit",
                    {
                        'symbol': symbol,
                        'current_position_value': current_position_value,
                        'new_position_value': new_position_value,
                        'max_position_value': max_position_value
                    }
                )
        
        # 6. Kelly percentage validation (if provided)
        kelly_pct = trade_details.get('kelly_percentage')
        if kelly_pct is not None:
            if kelly_pct > 1.0:
                result.add_violation(
                    ViolationType.POSITION_SIZE_EXCEEDED,
                    f"Kelly percentage {kelly_pct*100:.1f}% exceeds maximum allowed (100%)",
                    {'kelly_percentage': kelly_pct, 'max_kelly': 1.0}
                )

            # Check if Kelly position exceeds gate position limit
            if kelly_pct > config.max_position_pct:
                result.add_violation(
                    ViolationType.POSITION_SIZE_EXCEEDED,
                    f"Kelly position {kelly_pct*100:.1f}% exceeds gate limit {config.max_position_pct*100:.0f}%",
                    {
                        'kelly_percentage': kelly_pct,
                        'max_position_pct': config.max_position_pct,
                        'gate_level': config.level.value
                    }
                )

        # 7. Add warnings for approaching limits
        if side == 'BUY':
            cash_utilization = 1 - (current_portfolio.get('cash', 0) /
                                  current_portfolio.get('total_value', 1))

            warning_threshold = config.cash_floor_pct + 0.1  # 10% buffer
            if cash_utilization > warning_threshold:
                result.add_warning(
                    f"Approaching cash floor limit: {cash_utilization*100:.1f}% utilization",
                    {'cash_utilization': cash_utilization, 'threshold': warning_threshold}
                )

            # Kelly warning (if provided)
            if kelly_pct is not None and kelly_pct > config.max_position_pct * 0.8:
                result.add_warning(
                    f"Kelly position {kelly_pct*100:.1f}% approaching gate limit {config.max_position_pct*100:.0f}%",
                    {'kelly_percentage': kelly_pct, 'gate_limit': config.max_position_pct}
                )
        
        # Record violation if invalid
        if not result.is_valid:
            for violation in result.violations:
                self._record_violation(
                    ViolationType(violation['type']),
                    violation['message'],
                    violation.get('details', {})
                )
        
        return result
    
    def _record_violation(self, violation_type: ViolationType, message: str, 
                         details: Dict[str, Any]):
        """Record a gate violation."""
        violation = ViolationRecord(
            timestamp=datetime.now(),
            gate_level=self.current_gate,
            violation_type=violation_type,
            message=message,
            details=details
        )
        
        self.violation_history.append(violation)
        self.graduation_metrics.last_violation_date = violation.timestamp
        
        # Update 30-day violation count
        cutoff_date = datetime.now() - timedelta(days=30)
        self.graduation_metrics.total_violations_30d = len([
            v for v in self.violation_history 
            if v.timestamp >= cutoff_date and not v.resolved
        ])
        
        self._save_state()
        logger.warning(f"Gate violation recorded: {violation_type.value} - {message}")
    
    def check_graduation(self, portfolio_metrics: Dict[str, Any]) -> str:
        """
        Check if current gate should be graduated, held, or downgraded.
        
        Args:
            portfolio_metrics: Dict containing performance metrics
                - sharpe_ratio_30d: float
                - max_drawdown_30d: float
                - avg_cash_utilization_30d: float
                - total_return_30d: float
        
        Returns:
            str: 'GRADUATE', 'HOLD', or 'DOWNGRADE'
        """
        # Update graduation metrics
        self.graduation_metrics.sharpe_ratio_30d = portfolio_metrics.get('sharpe_ratio_30d')
        self.graduation_metrics.max_drawdown_30d = portfolio_metrics.get('max_drawdown_30d', 0)
        self.graduation_metrics.avg_cash_utilization_30d = portfolio_metrics.get('avg_cash_utilization_30d', 0)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(portfolio_metrics)
        self.graduation_metrics.performance_score = performance_score
        
        # Update consecutive compliant days
        if self.graduation_metrics.last_violation_date:
            days_since_violation = (datetime.now() - self.graduation_metrics.last_violation_date).days
            if days_since_violation >= 1:
                self.graduation_metrics.consecutive_compliant_days = min(
                    days_since_violation, self.graduation_metrics.consecutive_compliant_days + 1
                )
        else:
            self.graduation_metrics.consecutive_compliant_days += 1
        
        # Graduation criteria
        graduation_criteria = {
            GateLevel.G0: {
                'min_compliant_days': 14,
                'max_violations_30d': 2,
                'min_performance_score': 0.6,
                'min_capital': self.gate_configs[GateLevel.G1].capital_min
            },
            GateLevel.G1: {
                'min_compliant_days': 21,
                'max_violations_30d': 1,
                'min_performance_score': 0.7,
                'min_capital': self.gate_configs[GateLevel.G2].capital_min
            },
            GateLevel.G2: {
                'min_compliant_days': 30,
                'max_violations_30d': 0,
                'min_performance_score': 0.75,
                'min_capital': self.gate_configs[GateLevel.G3].capital_min
            }
        }
        
        # Downgrade criteria
        downgrade_criteria = {
            'max_violations_30d': 5,
            'min_performance_score': 0.3,
            'max_drawdown_threshold': 0.15  # 15% drawdown
        }
        
        current_criteria = graduation_criteria.get(self.current_gate)
        
        # Check for downgrade first
        if (self.graduation_metrics.total_violations_30d > downgrade_criteria['max_violations_30d'] or
            performance_score < downgrade_criteria['min_performance_score'] or
            self.graduation_metrics.max_drawdown_30d > downgrade_criteria['max_drawdown_threshold']):
            return 'DOWNGRADE'
        
        # Check for graduation
        if (current_criteria and
            self.graduation_metrics.consecutive_compliant_days >= current_criteria['min_compliant_days'] and
            self.graduation_metrics.total_violations_30d <= current_criteria['max_violations_30d'] and
            performance_score >= current_criteria['min_performance_score'] and
            self.current_capital >= current_criteria['min_capital']):
            return 'GRADUATE'
        
        return 'HOLD'
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a composite performance score (0-1)."""
        score = 0.0
        
        # Sharpe ratio component (0-0.4)
        sharpe = metrics.get('sharpe_ratio_30d', 0)
        if sharpe is not None:
            score += min(0.4, max(0, sharpe / 2))  # Normalize around 2.0 Sharpe
        
        # Drawdown component (0-0.3)
        max_drawdown = abs(metrics.get('max_drawdown_30d', 0))
        if max_drawdown <= 0.05:  # Less than 5% drawdown
            score += 0.3
        elif max_drawdown <= 0.10:  # Less than 10% drawdown
            score += 0.2
        elif max_drawdown <= 0.15:  # Less than 15% drawdown
            score += 0.1
        
        # Cash utilization component (0-0.2)
        cash_util = metrics.get('avg_cash_utilization_30d', 0)
        config = self.get_current_config()
        optimal_util = 1 - config.cash_floor_pct + 0.05  # Slightly above cash floor
        
        if abs(cash_util - optimal_util) <= 0.05:
            score += 0.2
        elif abs(cash_util - optimal_util) <= 0.10:
            score += 0.1
        
        # Compliance component (0-0.1)
        if self.graduation_metrics.total_violations_30d == 0:
            score += 0.1
        
        return min(1.0, score)
    
    def execute_graduation(self) -> bool:
        """Execute graduation to next gate level."""
        current_level = self.current_gate
        
        if current_level == GateLevel.G0:
            self.current_gate = GateLevel.G1
        elif current_level == GateLevel.G1:
            self.current_gate = GateLevel.G2
        elif current_level == GateLevel.G2:
            self.current_gate = GateLevel.G3
        else:
            logger.warning(f"Cannot graduate from {current_level.value} - already at maximum")
            return False
        
        # Reset metrics for new gate
        self.graduation_metrics = GraduationMetrics()
        self._save_state()
        
        logger.info(f"Successfully graduated from {current_level.value} to {self.current_gate.value}")
        return True
    
    def execute_downgrade(self) -> bool:
        """Execute downgrade to previous gate level."""
        current_level = self.current_gate
        
        if current_level == GateLevel.G1:
            self.current_gate = GateLevel.G0
        elif current_level == GateLevel.G2:
            self.current_gate = GateLevel.G1
        elif current_level == GateLevel.G3:
            self.current_gate = GateLevel.G2
        else:
            logger.warning(f"Cannot downgrade from {current_level.value} - already at minimum")
            return False
        
        # Reset metrics for new gate
        self.graduation_metrics = GraduationMetrics()
        self._save_state()
        
        logger.warning(f"Downgraded from {current_level.value} to {self.current_gate.value}")
        return True
    
    def get_violation_history(self, days: int = 30) -> List[ViolationRecord]:
        """Get violation history for specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [v for v in self.violation_history if v.timestamp >= cutoff_date]
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        config = self.get_current_config()
        
        return {
            'current_gate': self.current_gate.value,
            'current_capital': self.current_capital,
            'gate_config': {
                'capital_range': f"${config.capital_min:.0f}-${config.capital_max:.0f}",
                'allowed_assets': list(config.allowed_assets),
                'cash_floor_pct': config.cash_floor_pct,
                'options_enabled': config.options_enabled,
                'max_theta_pct': config.max_theta_pct,
                'max_position_pct': config.max_position_pct
            },
            'graduation_metrics': {
                'consecutive_compliant_days': self.graduation_metrics.consecutive_compliant_days,
                'total_violations_30d': self.graduation_metrics.total_violations_30d,
                'performance_score': self.graduation_metrics.performance_score,
                'last_violation': (self.graduation_metrics.last_violation_date.isoformat() 
                                 if self.graduation_metrics.last_violation_date else None)
            },
            'recent_violations': len(self.get_violation_history(7)),
            'total_violations': len(self.violation_history)
        }
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'current_gate': self.current_gate.value,
            'current_capital': self.current_capital,
            'graduation_metrics': {
                'consecutive_compliant_days': self.graduation_metrics.consecutive_compliant_days,
                'total_violations_30d': self.graduation_metrics.total_violations_30d,
                'avg_cash_utilization_30d': self.graduation_metrics.avg_cash_utilization_30d,
                'max_drawdown_30d': self.graduation_metrics.max_drawdown_30d,
                'sharpe_ratio_30d': self.graduation_metrics.sharpe_ratio_30d,
                'performance_score': self.graduation_metrics.performance_score,
                'last_violation_date': (self.graduation_metrics.last_violation_date.isoformat()
                                      if self.graduation_metrics.last_violation_date else None)
            },
            'violation_history': [
                {
                    'timestamp': v.timestamp.isoformat(),
                    'gate_level': v.gate_level.value,
                    'violation_type': v.violation_type.value,
                    'message': v.message,
                    'details': v.details,
                    'resolved': v.resolved
                }
                for v in self.violation_history
            ]
        }
        
        state_file = self.data_dir / 'gate_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        state_file = self.data_dir / 'gate_state.json'
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_gate = GateLevel(state.get('current_gate', 'G0'))
            self.current_capital = state.get('current_capital', 0.0)
            
            # Load graduation metrics
            metrics_data = state.get('graduation_metrics', {})
            self.graduation_metrics.consecutive_compliant_days = metrics_data.get('consecutive_compliant_days', 0)
            self.graduation_metrics.total_violations_30d = metrics_data.get('total_violations_30d', 0)
            self.graduation_metrics.avg_cash_utilization_30d = metrics_data.get('avg_cash_utilization_30d', 0.0)
            self.graduation_metrics.max_drawdown_30d = metrics_data.get('max_drawdown_30d', 0.0)
            self.graduation_metrics.sharpe_ratio_30d = metrics_data.get('sharpe_ratio_30d')
            self.graduation_metrics.performance_score = metrics_data.get('performance_score', 0.0)
            
            last_violation_str = metrics_data.get('last_violation_date')
            if last_violation_str:
                self.graduation_metrics.last_violation_date = datetime.fromisoformat(last_violation_str)
            
            # Load violation history
            self.violation_history = []
            for v_data in state.get('violation_history', []):
                violation = ViolationRecord(
                    timestamp=datetime.fromisoformat(v_data['timestamp']),
                    gate_level=GateLevel(v_data['gate_level']),
                    violation_type=ViolationType(v_data['violation_type']),
                    message=v_data['message'],
                    details=v_data['details'],
                    resolved=v_data.get('resolved', False)
                )
                self.violation_history.append(violation)
            
            logger.info(f"Loaded gate state: {self.current_gate.value} with ${self.current_capital:.2f} capital")
            
        except Exception as e:
            logger.error(f"Error loading gate state: {e}")
            # Continue with default state if load fails