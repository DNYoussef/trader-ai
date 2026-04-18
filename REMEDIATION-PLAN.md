# TRADER-AI COMPREHENSIVE REMEDIATION PLAN

Generated: 2025-12-15
Status: READY FOR EXECUTION

## EXECUTIVE SUMMARY

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Training Pipeline | 5 | 2 | 1 | 8 |
| Feature Engineering | 3 | 1 | 2 | 6 |
| MCPT Validation | 2 | 4 | 0 | 6 |
| Security | 3 | 5 | 2 | 10 |
| Safety Integration | 5 | 0 | 0 | 5 |
| Performance | 2 | 3 | 0 | 5 |
| **TOTAL** | **20** | **15** | **5** | **40** |

## BLOCKING DEPENDENCIES

```
PHASE 1: Training Pipeline -----> Blocks ALL model training
     |
     v
PHASE 2: Security -------------> Blocks production/paper trading
     |
     v
PHASE 3: Safety Integration ---> Blocks live trading
     |
     v
PHASE 4: Performance ----------> Blocks scale (1000+ MCPT perms)
```

---

## PHASE 1: TRAINING PIPELINE FIXES (BLOCKS ALL TRAINING)

### 1.1 Target Lookahead Fix (CRITICAL - DO FIRST)
**File:** `src/intelligence/training/trainer.py:178`
**Issue:** `shift(-1)` creates lookahead bias
**Fix:**
```python
# BEFORE
future_returns = df['price'].pct_change(periods=1).shift(-1)

# AFTER
future_returns = (df['price'].shift(-1) - df['price']) / df['price']
```
**Agent:** ML Training Specialist

### 1.2 Scaler Fit Order Fix (CRITICAL)
**File:** `src/intelligence/training/trainer.py:121`
**Issue:** Scaler fit on full data before split
**Fix:** Move scaler.fit_transform() to AFTER train/test split, fit only on training data
**Agent:** ML Training Specialist

### 1.3 TimeSeriesSplit Fix (CRITICAL)
**File:** `src/intelligence/training/trainer.py:200`
**Issue:** GridSearchCV uses random shuffle on time series
**Fix:**
```python
# BEFORE
grid_search = GridSearchCV(rf, param_grid, cv=5, ...)

# AFTER
from sklearn.model_selection import TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(rf, param_grid, cv=ts_cv, ...)
```
**Agent:** ML Training Specialist

### 1.4 LSTM Shuffle Fix (HIGH)
**File:** `src/intelligence/training/trainer.py:306`
**Issue:** DataLoader shuffle=True destroys sequence order
**Fix:** Set `shuffle=False`
**Agent:** ML Training Specialist

### 1.5 Imputer Fit Order Fix (HIGH)
**File:** `src/intelligence/data/processor.py:124`
**Issue:** Imputer fit on full data
**Fix:** Split imputation into train/test methods, fit only on training
**Agent:** ML Training Specialist

### 1.6 Pivot Lookahead Fix (CRITICAL)
**File:** `src/intelligence/breakout_sweep_signal.py:53`
**Issue:** Pivot confirmation uses i+7 (future bars)
**Fix:**
```python
# BEFORE
confirm_idx = i + window
pivots.iloc[confirm_idx] = 1

# AFTER
pivots.iloc[i] = 1  # Mark at actual pivot, accept natural lag
```
**Agent:** Feature Engineering Specialist

### 1.7 Walk-Forward Index Fix (CRITICAL)
**File:** `third_party/mcpt/bar_permute.py:74`
**Issue:** Off-by-one error
**Fix:**
```python
# BEFORE
perm_index = start_index + 1

# AFTER
perm_index = start_index
```
**Agent:** Quant Validation Specialist

### 1.8 Calmar Ratio Fix (CRITICAL)
**File:** `src/intelligence/validation/objectives.py:179-181`
**Issue:** Wrong annualization formula
**Fix:** Use CAGR formula with exp(sum(log_returns))
**Agent:** Quant Validation Specialist

---

## PHASE 2: SECURITY FIXES (BLOCKS PRODUCTION)

### 2.1 Remove Hardcoded Credentials (CRITICAL)
**File:** `.env` files across project
**Fix:**
1. Add .env to .gitignore
2. Remove all .env files from git history
3. Rotate all credentials: JWT_SECRET, DB passwords, HF_TOKEN
4. Use environment variable injection or secrets manager
**Agent:** Security Specialist

### 2.2 Add JWT Authentication to Trade Endpoints (CRITICAL)
**File:** `src/dashboard/backend/run_server_simple.py`
**Fix:** Add global authentication middleware for all /api/* endpoints
**Agent:** Security Specialist

### 2.3 Restrict CORS Origins (HIGH)
**File:** `settings.py`, `main.py`
**Fix:** Replace `allow_headers=["*"]` with explicit whitelist
**Agent:** Security Specialist

### 2.4 Apply Rate Limiting (HIGH)
**File:** `main.py`, all routers
**Fix:** Add @limiter decorators to all endpoints
**Agent:** Security Specialist

### 2.5 Add Input Validation Schemas (HIGH)
**Files:** All API routers
**Fix:** Use Pydantic models for all request bodies
**Agent:** Security Specialist

---

## PHASE 3: SAFETY INTEGRATION FIXES (BLOCKS LIVE TRADING)

### 3.1 Add Gate Validation Before Orders (CRITICAL)
**File:** `src/trading/trade_executor.py`
**Fix:** Call `gate_manager.validate_trade()` in buy_market_order() and sell_market_order()
**Agent:** Risk Management Specialist

### 3.2 Add Circuit Breaker Checks (CRITICAL)
**File:** `src/trading/trade_executor.py`, `src/trading_engine.py`
**Fix:** Check `circuit_manager.get_system_status()` before order submission
**Agent:** Risk Management Specialist

### 3.3 Fix Cash Floor Calculation (CRITICAL)
**File:** `src/gates/gate_manager.py:294-300`
**Fix:** Account for reduced portfolio value after trade
```python
# BEFORE
required_cash = current_portfolio.get('total_value', 0) * config.cash_floor_pct

# AFTER
post_trade_total = current_portfolio.get('total_value', 0) - trade_value
required_cash = post_trade_total * config.cash_floor_pct
```
**Agent:** Risk Management Specialist

### 3.4 Implement Daily Loss Limits (CRITICAL)
**Files:** `portfolio_manager.py`, `trading_engine.py`
**Fix:** Add daily P&L tracking with -2% hard stop
**Agent:** Risk Management Specialist

### 3.5 Block Trading on CRITICAL State (CRITICAL)
**File:** `src/trading_engine.py:625-630`
**Fix:** Replace logging with actual blocking logic + kill switch
**Agent:** Risk Management Specialist

---

## PHASE 4: PERFORMANCE FIXES (BLOCKS SCALE)

### 4.1 Parallelize MCPT (HIGH)
**File:** `src/intelligence/validation/mcpt_validator.py:152`
**Fix:** Implement ProcessPoolExecutor for n_permutations
**Expected:** 4-8x speedup on multi-core
**Agent:** Performance Specialist

### 4.2 Vectorize Bar Permutation (HIGH)
**File:** `third_party/mcpt/bar_permute.py:134-140`
**Fix:** Use NumPy cumsum + broadcasting instead of sequential loop
**Expected:** 10-30x speedup
**Agent:** Performance Specialist

### 4.3 Add Numba JIT (HIGH)
**Files:** bar_permute.py, objectives.py, signal_interface.py
**Fix:** Add @numba.jit(nopython=True, cache=True) decorators
**Expected:** 5-15x speedup
**Agent:** Performance Specialist

### 4.4 Fix list.extend() Inefficiency (MEDIUM)
**File:** `validation_battery.py:332`
**Fix:** Pre-allocate numpy array instead of list.extend()
**Expected:** 100x local speedup
**Agent:** Performance Specialist

### 4.5 Add Caching (MEDIUM)
**Files:** Objective functions, MCPT validator
**Fix:** Add @functools.lru_cache to pure functions
**Expected:** 2-3x additional speedup
**Agent:** Performance Specialist

---

## ADDITIONAL FIXES (PARALLEL TO PHASES)

### A.1 Profit Factor Infinity Fix
**File:** `objectives.py:39`
**Fix:** Return 1e6 instead of float('inf')
**Agent:** Quant Validation Specialist

### A.2 Sharpe Ratio Std Mismatch
**File:** `objectives.py:65-69`
**Fix:** Use std(excess_returns) not std(strategy_returns)
**Agent:** Quant Validation Specialist

### A.3 MCPT P-Value Inequality
**File:** `mcpt_validator.py:177`
**Fix:** Use `>` instead of `>=` for strict p-value
**Agent:** Quant Validation Specialist

### A.4 Position Cost Alignment
**File:** `signal_interface.py:117`
**Fix:** Remove `[:-1]` slice from position_changes
**Agent:** Quant Validation Specialist

### A.5 Feature Dimension Documentation
**File:** `enhanced_hrm_features.py:106,446`
**Fix:** Update all references from 32D to 38D
**Agent:** Feature Engineering Specialist

---

## IMPLEMENTATION SEQUENCE (DEPENDENCY-ORDERED)

```
Week 1: PHASE 1 - Training Pipeline
  Day 1-2: 1.1, 1.6, 1.7, 1.8 (CRITICAL data integrity)
  Day 3-4: 1.2, 1.3 (Scaler + CV fixes)
  Day 5: 1.4, 1.5 (LSTM + Imputer)

Week 2: PHASE 2 - Security
  Day 1-2: 2.1 (Credentials - MOST CRITICAL)
  Day 3: 2.2 (JWT auth)
  Day 4: 2.3, 2.4 (CORS + Rate limiting)
  Day 5: 2.5 (Validation schemas)

Week 3: PHASE 3 - Safety Integration
  Day 1: 3.1, 3.2 (Gate + Circuit breaker checks)
  Day 2: 3.3, 3.5 (Cash floor + Kill switch)
  Day 3-4: 3.4 (Daily loss limits)
  Day 5: Integration testing

Week 4: PHASE 4 - Performance
  Day 1-2: 4.1 (MCPT parallelization)
  Day 3: 4.2 (Vectorization)
  Day 4: 4.3 (Numba JIT)
  Day 5: 4.4, 4.5 (Caching + cleanup)
```

---

## TESTING REQUIREMENTS

### Per-Fix Testing
- Unit tests for each fix
- Regression tests to ensure no breaks
- Before/after performance benchmarks

### Phase Gate Testing
- **After Phase 1:** Walk-forward backtest must show realistic (lower) returns
- **After Phase 2:** Security scan (OWASP ZAP) must pass
- **After Phase 3:** Safety integration tests must block invalid trades
- **After Phase 4:** MCPT 1000 perms must complete in <5 minutes

---

## SUCCESS CRITERIA

1. All 20 CRITICAL issues resolved
2. All 15 HIGH issues resolved
3. Tests pass (pytest -v)
4. No data leakage in training pipeline
5. Security vulnerabilities patched
6. Safety systems actively blocking invalid trades
7. MCPT runs 10-50x faster

---

## AGENT ASSIGNMENTS

| Phase | Primary Agent | Support Agents |
|-------|--------------|----------------|
| Phase 1 | ml-training-specialist | feature-engineer, quant-analyst |
| Phase 2 | security-specialist | backend-dev, code-reviewer |
| Phase 3 | risk-manager | backend-dev, safety-auditor |
| Phase 4 | performance-specialist | backend-dev, code-optimizer |

---

## NOTES

- Each fix is independent within its phase
- Phases must complete in order due to blocking dependencies
- All fixes include rollback procedures
- Comprehensive audit trail maintained
