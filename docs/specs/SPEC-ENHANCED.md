# Specification: Gary×Taleb Autonomous Trading System

## Executive Summary

An autonomous trading system that starts with $200 and scales to millions through capital-gated progression, combining Gary's distributional edge (following money flows to detect mispriced narratives) with Taleb's antifragility (profiting from disorder while preventing ruin). The system operates mostly hands-off, executing weekly buy/siphon cycles while progressively unlocking capabilities as capital and competence grow.

## Core Philosophy

### Gary's Edge: "Follow Distribution, Fade Consensus"
- **Distributional Pressure Index (DPI)**: Track where money actually flows (by income cohort)
- **Narrative Gap (NG)**: Measure divergence between market pricing and distributional reality
- **Repricing Potential (RP)**: Identify assets where narrative correction is imminent

### Taleb's Antifragility: "Survive First, Then Profit from Chaos"
- **Barbell Strategy**: 80% safe assets, 20% convex bets (no mediocre middle)
- **Ruin Prevention**: Hard constraint p(ruin) < 10^-6/year via EVT modeling
- **Convexity Preference**: Systematic long volatility with bounded theta bleed

## System Requirements

### Functional Requirements

#### FR1: Capital-Gated Progression
- System starts at G0 ($200) with ULTY/AMDY ETFs only
- Unlocks capabilities through 12 gates (G0-G12) based on capital and behavior
- Each gate enables new assets, structures, and strategies
- Automatic graduation/downgrade based on performance metrics

#### FR2: Weekly Autonomous Cycle
- **Buy Phase**: Friday 4:10pm ET - deploy available cash per gate rules
- **Siphon Phase**: Friday 6:00pm ET - calculate weekly delta, split 50/50
- **Reinvestment**: 50% of profits stay for compounding
- **Withdrawal**: 50% of profits swept to protected treasury account

#### FR3: Risk Management Framework
- **Pre-trade Checks**: 8 validation gates before any order
- **Position Limits**: Per-ticket caps, concentration limits
- **Tail Protection**: EVT-based CVaR_99 ≤ 1.25% NAV
- **Event Safety**: Net gamma ≥ 0, net vega ≥ 0 near catalysts

#### FR4: Intelligence Layer
- **DPI Calculation**: Cohort cash flows, rent/debt ratios, capture rates
- **NG Detection**: Model-implied vs market-implied path divergence
- **Regime Classification**: 5-state HMM (disinflation, inflation, policy error, etc.)
- **Catalyst Tracking**: Countdown to CPI/FOMC/earnings events

#### FR5: Learning System
- **Forecast Cards**: Structured predictions with confidence and rationale
- **Trade Plans**: Documented structures, sizes, and hedges
- **Risk Plans**: CVaR limits, barbell ratios, tail policies
- **Local ML**: LoRA fine-tuning on 7B model using trading artifacts

### Non-Functional Requirements

#### NFR1: Automation Level
- **Hands-off Operation**: 95%+ autonomous after initial setup
- **Human Checkpoints**: Only for live mode arming and kill switch
- **Self-Healing**: Automatic recovery from transient failures

#### NFR2: Safety & Security
- **Paper-First**: Mandatory paper trading before live enablement
- **Hardware Key**: Required for live trading activation
- **24-Hour Delay**: Arming period for live mode transition
- **Kill Switch**: One-click emergency stop with position flattening

#### NFR3: Performance
- **Latency**: <100ms API response, <50ms UI render
- **Throughput**: Handle 100+ concurrent positions at scale
- **Uptime**: 99.9% availability during market hours

#### NFR4: Auditability
- **WORM Logs**: Immutable append-only audit trail
- **Decision Records**: Every trade with rationale archived
- **Compliance Ready**: SOC2/PCI-capable infrastructure

## Technical Architecture

### System Components

```
1. Desktop Application (Tauri + Next.js)
   - Dashboard: NAV, positions, gate progress
   - Trade Composer: Strategy selection, risk HUD
   - Risk Monitor: Real-time constraint tracking
   - Kill Switch: Emergency stop control

2. Trading Engine (Python/Rust)
   - Gate Manager: Progression logic, graduation checks
   - Position Manager: Allocation, rebalancing
   - Risk Engine: EVT modeling, pre-trade validation
   - Siphon Controller: Weekly profit distribution

3. Intelligence Layer
   - DPI Calculator: Distributional flow aggregation
   - NG Analyzer: Narrative gap measurement
   - Regime Detector: HMM state classification
   - Catalyst Clock: Event countdown tracking

4. Execution Layer
   - Broker Adapters: Alpaca, IBKR integration
   - Order Manager: Routing, slippage control
   - Microstructure: Spread monitoring, depth analysis

5. Learning System
   - Artifact Store: Forecast/Trade/Risk cards
   - LoRA Trainer: Local model fine-tuning
   - Calibration Scorer: Brier/log-loss tracking
   - Refutation Engine: Causal validation
```

### Data Requirements

#### Market Data
- Real-time prices for authorized assets per gate
- Options surfaces (unlocked at G3+)
- Order book depth and spread monitoring
- Historical data for backtesting

#### Distributional Data (DPI Components)
- Income cohort cash flows (wages, transfers)
- Rent-to-income ratios by decile
- Debt service metrics by balance sheet cohort
- Corporate margins, buybacks, dividend flows
- Housing affordability indices

#### Narrative Data (NG Components)
- Sell-side research notes and forecasts
- Central bank communications
- Media sentiment analysis
- Earnings call transcripts

## Capital Gates Specification

### Gate Progression Table

| Gate | NAV Range | New Capabilities | Risk Constraints | Graduation Criteria |
|------|-----------|------------------|------------------|-------------------|
| **G0** | $200-499 | ULTY, AMDY only | 50% cash floor, no leverage | 4 weeks clean |
| **G1** | $500-999 | +Gold (IAU), TIPS (VTIP) | 60% cash floor | 8 weeks, Brier<0.3 |
| **G2** | $1k-2.5k | +Factor ETFs, USD (UUP) | 65% cash floor | Zero overrides |
| **G3** | $2.5k-5k | +Long options (0.5% theta) | 70% cash, gamma≥0 | EVT stable |
| **G4** | $5k-10k | +Micro futures (hedge only) | 75% cash floor | Liquidity pass |
| **G5** | $10k-25k | +VIX calls, 1% theta | 80% cash (full barbell) | 3mo calibrated |
| **G6** | $25k-50k | +T-bill ladder, TWAP | Pattern day trader OK | Zero breaches |
| **G7** | $50k-100k | +Credit ETFs, FX flies | Event blackouts | Shadow profit |
| **G8** | $100k-250k | +Dynamic convexity | Per-asset CVaR | Quarterly review |
| **G9** | $250k-500k | +Multi-broker, SOFR | Concentration<15% | Counterfactual pass |
| **G10** | $500k-1M | Full macro suite | Daily VaR guard | External audit |
| **G11** | $1M-5M | +Direct treasuries | Model committee | Legal ready |
| **G12** | $5M+ | Optional ISDA/swaps | Capacity limits | Board review |

### Gate Transition Logic

```python
def check_graduation(current_gate, metrics):
    requirements = GATE_REQUIREMENTS[current_gate]

    checks = {
        'time_in_gate': metrics['weeks'] >= requirements['min_weeks'],
        'nav_threshold': metrics['nav'] >= requirements['next_gate_min'],
        'calibration': metrics['brier_score'] <= requirements['max_brier'],
        'violations': metrics['rule_breaches'] == 0,
        'drawdown': metrics['max_dd'] <= requirements['dd_limit'],
        'specific': gate_specific_checks(current_gate, metrics)
    }

    if all(checks.values()):
        return 'GRADUATE'
    elif metrics['critical_breach']:
        return 'DOWNGRADE'
    else:
        return 'HOLD'
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Objective**: Build core trading engine with G0 capabilities

**Deliverables**:
- Broker connection (Alpaca/IBKR)
- Weekly buy/siphon cycle
- Basic risk checks
- Desktop UI shell
- Audit logging

**Success Criteria**:
- Successfully execute paper trades
- Weekly cycle runs without intervention
- All trades logged immutably

### Phase 2: Risk & Quality (Weeks 3-4)
**Objective**: Implement comprehensive risk management

**Deliverables**:
- EVT tail modeling
- Pre-trade validation pipeline
- Barbell allocator
- Paper/live mode switching
- CI/CD pipeline

**Success Criteria**:
- Zero risk constraint violations
- 90%+ test coverage
- Successful stress testing

### Phase 3: Intelligence Layer (Weeks 5-6)
**Objective**: Add market intelligence capabilities

**Deliverables**:
- DPI calculator
- NG analyzer
- Regime detector
- Forecast card system
- Catalyst tracking

**Success Criteria**:
- Accurate regime classification
- Meaningful NG measurements
- Calibrated forecasts

### Phase 4: Learning System (Weeks 7-8)
**Objective**: Implement self-improvement mechanisms

**Deliverables**:
- Artifact storage
- LoRA training pipeline
- Calibration scoring
- Performance attribution
- Refutation engine

**Success Criteria**:
- Model improves with data
- Calibration scores increase
- Causal validation passes

## Operational Procedures

### Weekly Cycle Procedure
```
Every Friday:
1. 4:00pm - Market close
2. 4:05pm - Calculate NAV, available cash
3. 4:10pm - Execute buy orders per allocation
4. 4:15pm - Verify fills, update positions
5. 6:00pm - Calculate weekly delta
6. 6:05pm - Execute 50/50 siphon
7. 6:10pm - Update audit log
8. 6:15pm - Generate performance report
```

### Live Mode Activation Procedure
```
1. Complete 100+ paper trades with zero violations
2. Achieve Brier score < 0.25 for 30 days
3. Insert hardware key
4. Enter passphrase
5. Complete risk quiz (100% required)
6. Initiate 24-hour arming delay
7. Confirm activation after delay
8. Start with minimum capital ($200)
```

### Emergency Procedures

#### Kill Switch Activation
```
1. Click emergency stop button
2. System immediately:
   - Cancels all open orders
   - Liquidates options positions
   - Flattens futures positions
   - Converts to 100% cash
3. Generates incident report
4. Requires manual reactivation
```

#### Gate Downgrade
```
Automatic triggers:
- Critical risk breach
- Three violations in 7 days
- Drawdown exceeds limit
- Calibration degrades >20%

Actions:
- Immediately adopt lower gate constraints
- Reduce position sizes
- Increase cash floor
- Notify operator
```

## Success Metrics

### Growth Metrics
- **Capital Growth**: 15-20% annual return target
- **Gate Progression**: Advance every 8-16 weeks
- **Compound Timeline**: Reach $1,000 in 2.6 years

### Risk Metrics
- **Ruin Probability**: <10^-6/year maintained
- **Max Drawdown**: <6% rolling window
- **CVaR_99**: <1.25% of NAV
- **Violation Rate**: <1% of trades

### Operational Metrics
- **Automation Rate**: >95% hands-off
- **Siphon Success**: 100% weekly execution
- **Uptime**: >99.9% during market hours
- **Audit Completeness**: 100% decision coverage

### Learning Metrics
- **Calibration**: Brier score <0.25
- **Forecast Accuracy**: 60%+ directional
- **Model Improvement**: 5%+ monthly gain
- **Refutation Pass Rate**: >90%

## Risk Analysis

### Financial Risks
- **Market Risk**: Managed via position limits and diversification
- **Liquidity Risk**: Spread guards and depth requirements
- **Concentration Risk**: Maximum 15% per theme at scale
- **Operational Risk**: Kill switch and automatic downgrades

### Technical Risks
- **System Failure**: Redundant infrastructure, automatic recovery
- **Data Quality**: Multiple source validation, outlier detection
- **Model Risk**: Refutation engine, counterfactual testing
- **Security Risk**: Hardware keys, encryption, audit trails

### Mitigation Strategies
- Start with minimal capital ($200)
- Mandatory paper trading period
- Progressive capability unlocking
- Conservative default parameters
- Human oversight at critical points

## Compliance & Governance

### Regulatory Considerations
- Pattern day trader rules (addressed at G6)
- Tax reporting automation
- Jurisdiction-specific restrictions
- No advisory or discretionary management

### Audit & Documentation
- Every decision logged immutably
- Full rationale captured
- Performance attribution tracked
- Compliance reports generated

### Ethical Guidelines
- No market manipulation
- No insider information usage
- Transparent performance reporting
- Risk-first, returns second

## Conclusion

This specification defines a sophisticated yet practical autonomous trading system that:
1. Starts small ($200) and scales systematically
2. Combines proven edges (distributional flows + antifragility)
3. Operates mostly hands-off while maintaining safety
4. Learns and improves from its own experience
5. Prevents catastrophic failure through multiple safeguards

The system is designed to be buildable by a small team in 8 weeks, with clear phases, measurable success criteria, and comprehensive risk management throughout.