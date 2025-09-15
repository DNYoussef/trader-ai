# Research Synthesis: Gary×Taleb Trading System

## Executive Summary
Building a capital-gated autonomous trading system starting at $200, scaling to millions with progressive capability unlocks.

## Key Research Findings

### 1. Trading Philosophy Integration
- **Gary's Edge**: Distributional Pressure Index (DPI) + Narrative Gap (NG)
- **Taleb's Antifragility**: Barbell allocation (80% safe, 20% convex), EVT tail modeling, ruin prevention

### 2. Initial Setup Requirements ($200 Start)
- **Assets**: ULTY (weekly dividend ETF), AMDY (monthly option-income ETF)
- **Allocation**: 70% ULTY / 30% AMDY
- **Yield Estimates**: ~1.66% weekly combined (current run-rate)
- **Timeline to $1000**: ~133 weeks combined wealth (2.6 years)

### 3. Technology Stack Recommendations
- **Trading Libraries**:
  - alpaca-py (Apache 2.0) - Broker integration
  - TA-Lib + pandas-ta - Technical indicators
  - VectorBT - Backtesting (1000x faster)
  - Riskfolio-Lib - Risk management (13 measures)

### 4. Capital Gates System (12 levels)
- G0 ($200-499): ETFs only, 50% cash floor
- G3 ($2.5k-5k): Long options enabled (0.5% theta/yr)
- G5 ($10k-25k): Micro futures, 1% theta budget
- G10 ($500k-1M): Full macro suite, 2% theta budget
- G12 ($5M+): Institutional grade, ISDA access

### 5. Risk Management Framework
- **Hard Constraints**:
  - CVaR_99 ≤ 1.25% NAV
  - p_ruin < 10^-6/year
  - Net gamma ≥ 0 near events
  - Daily loss cap: 5% (initial), 1% (scaled)

### 6. Automation Architecture
- **Weekly Cycle**: Buy Friday 4:10pm, Siphon Friday 6pm
- **50/50 Rule**: Half profits reinvested, half withdrawn
- **Pre-trade Checks**: 8 validation gates before execution
- **Kill Switch**: One-click emergency stop

### 7. UI/UX Requirements
- Desktop app (Tauri/Next.js)
- Paper → Live progression with 24hr arming delay
- Hardware key + passphrase for live trading
- WORM audit logs for compliance

### 8. Data Intelligence Layer
- **DPI Components**: Rent-to-income, debt service by cohort, deposit flows
- **NG Calculation**: Model-implied vs market-implied paths
- **Regime Detection**: HMM/BoCPD with 5 states
- **Catalyst Clock**: Event countdown timers

## Implementation Priority

### Phase 1 (Weeks 1-2): Foundation
1. Setup broker connection (Alpaca/IBKR)
2. Implement G0 gate (ULTY/AMDY only)
3. Create weekly buy/siphon logic
4. Build basic UI with kill switch

### Phase 2 (Weeks 3-4): Risk Layer
1. Add EVT tail modeling
2. Implement pre-trade checks
3. Create audit logging system
4. Add paper trading mode

### Phase 3 (Weeks 5-8): Intelligence
1. Build DPI/NG calculators
2. Add regime detection
3. Create forecast cards system
4. Implement gate progression logic

## Success Metrics
- **Calibration**: Brier score < 0.25
- **Risk**: Zero ruin events
- **Growth**: Gate progression every 8-16 weeks
- **Automation**: 95%+ hands-off operation

## Recommendations
1. Start with Alpaca for simplicity
2. Use IBKR for internal transfers at scale
3. Implement gates.yaml for policy enforcement
4. Build SFT/LoRA training from day 1 artifacts
5. Maintain 50/50 siphon throughout all gates