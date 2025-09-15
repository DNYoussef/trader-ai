# Enhanced Implementation Plan: Gary×Taleb Autonomous Trading System

## System Architecture Overview

```
Desktop UI (Tauri/Next.js)
    │
    ├─ Local API (FastAPI/Rust)
    │   ├─ Trading Engine
    │   │   ├─ Gate Manager (G0-G12 progression)
    │   │   ├─ Position Manager (ULTY/AMDY → Full Macro)
    │   │   └─ Risk Engine (EVT, CVaR, Barbell)
    │   │
    │   ├─ Intelligence Layer
    │   │   ├─ DPI Calculator (distributional flows)
    │   │   ├─ NG Analyzer (narrative gaps)
    │   │   └─ Regime Detector (HMM/BoCPD)
    │   │
    │   ├─ Execution Layer
    │   │   ├─ Broker Adapters (Alpaca/IBKR)
    │   │   ├─ Order Manager (pre-trade checks)
    │   │   └─ Siphon Controller (50/50 weekly)
    │   │
    │   └─ Learning System
    │       ├─ Forecast Cards (predictions)
    │       ├─ Trade Plans (structures)
    │       └─ LoRA Training (local 7B model)
    │
    └─ Data Store (SQLite + Parquet)
        ├─ Market Data (prices, volumes, spreads)
        ├─ Audit Logs (WORM, immutable)
        └─ Training Artifacts (for ML improvement)
```

## Implementation Phases with SPEK Loops

### LOOP 1: Foundation (Weeks 1-2)

#### Research & Planning
- [x] Analyzed trading libraries and broker APIs
- [x] Researched ULTY/AMDY characteristics
- [ ] Design gate progression system
- [ ] Plan risk management framework

#### Core Implementation
```python
# src/gates.py
class GateManager:
    GATES = {
        'G0': {'min_nav': 200, 'max_nav': 499, 'assets': ['ULTY', 'AMDY']},
        'G1': {'min_nav': 500, 'max_nav': 999, 'assets': ['ULTY', 'AMDY', 'IAU', 'VTIP']},
        # ... through G12
    }

    def current_gate(self, nav: float) -> str:
        """Determine current gate based on NAV"""

    def check_graduation(self, metrics: dict) -> bool:
        """Check if ready to graduate to next gate"""

# src/trading_engine.py
class TradingEngine:
    def __init__(self, broker: Broker, gate_manager: GateManager):
        self.broker = broker
        self.gate_manager = gate_manager
        self.risk_engine = RiskEngine()

    def weekly_cycle(self):
        """Execute Friday buy and siphon cycle"""
        # 1. Calculate available cash
        # 2. Apply gate-appropriate allocation
        # 3. Execute trades with pre-checks
        # 4. Calculate weekly delta
        # 5. Execute 50/50 siphon
```

### LOOP 2: Quality & Execution (Weeks 3-4)

#### Development Tasks
```python
# src/risk_engine.py
class RiskEngine:
    def __init__(self):
        self.evt_model = EVTModel()
        self.barbell = BarbellAllocator()

    def pre_trade_checks(self, portfolio, proposed_trade):
        checks = {
            'cvar_99': self.check_cvar(portfolio, proposed_trade),
            'daily_loss': self.check_daily_loss(proposed_trade),
            'barbell': self.check_barbell_ratio(portfolio),
            'spread': self.check_spread(proposed_trade),
            'gamma_vega': self.check_greeks_near_events(portfolio)
        }
        return all(checks.values()), checks

# tests/test_risk_engine.py
def test_g0_constraints():
    """Test G0 gate constraints (ETFs only, 50% cash floor)"""

def test_weekly_siphon():
    """Test 50/50 profit split logic"""

def test_gate_progression():
    """Test graduation criteria and downgrades"""
```

#### CI/CD Setup
```yaml
# .github/workflows/trading-system.yml
name: Trading System CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src/ --cov-report=xml
      - name: Run risk validation
        run: python scripts/validate_risk_constraints.py
```

### LOOP 3: Verification (Weeks 5-6)

#### Theater Detection
```python
# scripts/theater_detection.py
class TheaterDetector:
    def scan_for_fake_metrics(self, trading_history):
        """Detect performative metrics that don't reflect real edge"""
        checks = {
            'cherry_picked_timeframes': self.check_timeframe_selection(),
            'overfitted_parameters': self.check_parameter_stability(),
            'unrealistic_assumptions': self.check_market_assumptions(),
            'hidden_risks': self.scan_for_hidden_leverage()
        }
        return checks

# scripts/reality_validation.py
class RealityValidator:
    def validate_performance(self, backtest_results, paper_results, live_results):
        """Ensure claimed performance is achievable in reality"""
        validations = {
            'slippage_realistic': self.check_slippage_assumptions(),
            'liquidity_sufficient': self.verify_liquidity_depth(),
            'costs_included': self.verify_all_costs_accounted(),
            'risk_metrics_stable': self.check_risk_metric_stability()
        }
        return validations
```

## Configuration Files

### gates.yaml
```yaml
gates:
  G0:
    nav_range: [200, 499]
    assets: ['ULTY', 'AMDY']
    constraints:
      cash_floor_pct: 0.50
      max_ticket_size: 25
      daily_loss_limit_pct: 0.05
      no_margin: true
      no_options: true
    graduation:
      min_weeks: 4
      max_violations: 0
      calibration_started: true

  G1:
    nav_range: [500, 999]
    assets: ['ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP']
    constraints:
      cash_floor_pct: 0.60
      # ... additional constraints
    graduation:
      min_weeks: 8
      brier_score_max: 0.30
      max_drawdown_pct: 3.0
```

### config.yaml
```yaml
broker:
  provider: alpaca  # or ibkr
  mode: paper      # paper or live
  credentials:
    key_id: ${ALPACA_KEY_ID}
    secret: ${ALPACA_SECRET}

trading:
  universe:
    G0: ['ULTY', 'AMDY']
    allocation: {'ULTY': 0.70, 'AMDY': 0.30}

  schedule:
    buy_time: "FRI 16:10 ET"
    siphon_time: "FRI 18:00 ET"

  siphon_policy:
    reinvest_pct: 0.50
    withdraw_pct: 0.50

risk:
  pre_trade_checks:
    - cvar_99_check
    - daily_loss_check
    - barbell_check
    - spread_check
    - liquidity_check

  limits:
    max_cvar_99_pct: 0.0125
    max_daily_loss_pct: 0.05
    min_cash_floor_pct: 0.50
```

## UI Components Structure

```typescript
// src/ui/components/Dashboard.tsx
interface DashboardProps {
  nav: number;
  currentGate: string;
  positions: Position[];
  weeklyDelta: number;
  nextRunTime: Date;
}

// src/ui/components/GateProgress.tsx
interface GateProgressProps {
  currentGate: string;
  graduationMetrics: GraduationMetrics;
  timeInGate: number;
  violations: number;
}

// src/ui/components/RiskMonitor.tsx
interface RiskMonitorProps {
  cvar99: number;
  dailyLoss: number;
  barbellRatio: number;
  netGamma: number;
  netVega: number;
}

// src/ui/components/KillSwitch.tsx
interface KillSwitchProps {
  isArmed: boolean;
  onActivate: () => void;
}
```

## Testing Strategy

### Unit Tests
- Gate progression logic
- Risk constraint calculations
- Weekly siphon mathematics
- Pre-trade validation checks

### Integration Tests
- Broker API connectivity
- End-to-end weekly cycle
- Gate graduation scenarios
- Risk engine with market data

### Simulation Tests
- Historical backtest validation
- Monte Carlo stress testing
- EVT tail model verification
- Regime transition handling

## Deployment Checklist

### Week 1-2: Foundation
- [ ] Setup project structure
- [ ] Implement G0 gate logic
- [ ] Create broker adapter (Alpaca)
- [ ] Build weekly buy/siphon cycle
- [ ] Create basic UI with kill switch

### Week 3-4: Risk & Quality
- [ ] Implement EVT tail modeling
- [ ] Add pre-trade check pipeline
- [ ] Create WORM audit logging
- [ ] Setup paper trading mode
- [ ] Add CI/CD workflows

### Week 5-6: Verification
- [ ] Run theater detection scans
- [ ] Perform reality validation
- [ ] Conduct stress testing
- [ ] Validate all constraints
- [ ] Document edge cases

### Week 7-8: Intelligence Layer
- [ ] Build DPI calculator
- [ ] Implement NG analyzer
- [ ] Add regime detection
- [ ] Create forecast cards
- [ ] Setup LoRA training pipeline

## Success Metrics

### Technical Metrics
- Test coverage > 90%
- Zero critical security issues
- API response time < 100ms
- UI render time < 50ms

### Trading Metrics
- Zero ruin events
- Weekly siphon success rate: 100%
- Gate progression: Every 8-16 weeks
- Calibration Brier score < 0.25

### Operational Metrics
- Uptime > 99.9%
- Audit log completeness: 100%
- Pre-trade check compliance: 100%
- Manual intervention: < 5%

## Next Steps

1. **Immediate**: Create project repository with this structure
2. **Day 1-3**: Implement G0 gate and basic trading engine
3. **Day 4-7**: Add risk engine and pre-trade checks
4. **Week 2**: Build UI and broker integration
5. **Week 3**: Add paper trading and testing suite
6. **Week 4**: Deploy to production with $200 seed capital