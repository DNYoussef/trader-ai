# k(L) Formula Implementation Audit Report

**Date**: 2025-12-16
**Auditor**: Code Quality Specialist
**File**: D:\Projects\trader-ai\src\training\k_formula.py
**Version**: Current (171 lines)
**Status**: CRITICAL ISSUES FOUND

---

## Executive Summary

The k(L) formula implementation has **5 CRITICAL and 3 HIGH severity issues** that fundamentally compromise its correctness. The most severe problem is a **SEMANTIC MISMATCH**: the coefficients were derived from spatial length scales in meters (physics) but are being applied to gradient norms, layer indices, and PnL magnitudes (ML/finance) without validation or recalibration.

**CRITICAL**: The formula may be mathematically correct but semantically wrong for its current use cases.

---

## 1. Coefficient Verification

### CRITICAL: Semantic Domain Mismatch

**Lines**: 26-27, Comments at 5-7
**Severity**: CRITICAL

```python
# Current code claims:
K_SLOPE = -0.0137
K_INTERCEPT = 0.1593
# Comment: "R^2 = 0.71, p = 0.008 (statistically significant)"
```

**Issue**: According to NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md:244-249:

```
Variables:
- k(L): Meta-weight at length scale L
- L: Length scale in meters  <--- PHYSICS DOMAIN
- slope: -0.0137
- intercept: 0.1593

Use Case for Trader-AI: k varies by portfolio size/gate level
Gate G1 ($200):    L ~ small -> k ~ 0.14 (less meta)
Gate G12 ($10M+):  L ~ large  -> k ~ 0.05 (more meta)
```

**Problem**: The formula was fitted to MOO optimization results where L represented **spatial length in meters** from multiscale physics simulations. Now it's being applied to:

1. **Gradient norms** (line 114): `L = torch.norm(grad)`
   - Units: Dimensionless gradient magnitude
   - Expected range: 1e-8 to 1e6
   - Physical interpretation: NONE

2. **Layer indices** (line 143): `L = (layer_idx + 1) / total_layers`
   - Units: Normalized position [0.01, 1.0]
   - Expected range: 0.01 to 1.0
   - Physical interpretation: NONE

3. **PnL magnitudes** (line 169): `L = abs(pnl) / scale`
   - Units: Normalized profit/loss
   - Expected range: eps to unbounded
   - Physical interpretation: NONE

**Impact**:
- k(L) may produce values that are statistically fitted to the wrong domain
- The R^2=0.71 significance ONLY applies to physics length scales
- No evidence that the formula transfers to gradient/layer/PnL domains

**Test Case Failure**:
```python
# Physics (original domain):
L = 1.0  # 1 meter
k = -0.0137 * log10(1.0) + 0.1593 = 0.1593  # Makes sense

# Gradient magnitude (current usage):
grad_norm = 0.001  # Small gradient
k = -0.0137 * log10(0.001) + 0.1593
k = -0.0137 * (-3) + 0.1593
k = 0.0411 + 0.1593 = 0.2004  # Higher k = more amplification

grad_norm = 100  # Large gradient
k = -0.0137 * log10(100) + 0.1593
k = -0.0137 * 2 + 0.1593
k = -0.0274 + 0.1593 = 0.1319  # Lower k = more dampening

# BUT: Where's the proof this relationship holds for gradients?
```

**Recommendation**:
1. Add disclaimer that coefficients are **NOT VALIDATED** for ML/finance domains
2. Implement empirical recalibration for trader-ai use cases
3. Add validation tests comparing k(L) behavior against known gradient dynamics

---

### HIGH: Missing Source Traceability

**Lines**: 4-5
**Severity**: HIGH

```python
# Comment claims:
# SOURCE: the-agent-maker/src/cross_phase/meta_calculus/k_formula.py
```

**Issue**:
- File path reference cannot be verified (external repository)
- No git commit hash or version number
- No link to MOO optimization results
- "Verified coefficients" claim lacks proof

**Impact**: Cannot verify correctness or reproduce derivation

**Recommendation**: Add:
```python
# SOURCE: the-agent-maker/src/cross_phase/meta_calculus/k_formula.py
# COMMIT: [hash]
# MOO RESULTS: [path to JSON files in results/]
# FITTED DOMAIN: Spatial length scales (meters) from physics simulations
# WARNING: Not validated for gradient/layer/PnL domains
```

---

## 2. Formula Correctness

### CRITICAL: log10 vs ln Ambiguity

**Lines**: 75, 83, 91
**Severity**: CRITICAL

```python
# Scalar version (line 75):
log_L = np.log10(L_safe)

# NumPy version (line 83):
log_L = np.log10(L_safe)

# Torch version (line 91):
log_L = torch.log10(L_safe)
```

**Issue**: The formula uses `log10` (base-10 logarithm), but:

1. **Meta-calculus context**: Bigeometric formulas typically use natural log (ln)
   - `D_BG[f](a) = exp(a * f'(a) / f(a))` uses natural exp/log
   - Gradient transforms use natural log for numerical stability

2. **No justification**: Why log10 instead of ln?
   - log10(L) changes by 1 per decade (10x change)
   - ln(L) changes by 1 per e-fold (2.718x change)
   - Coefficient would need adjustment if switching

3. **Consistency check**:
   ```python
   # Current: log10
   L = 100
   k = -0.0137 * log10(100) + 0.1593
   k = -0.0137 * 2 + 0.1593 = 0.1319

   # If it should be ln:
   k = -0.0137 * ln(100) + 0.1593
   k = -0.0137 * 4.605 + 0.1593 = 0.0961

   # 35% difference!
   ```

**Impact**: If wrong logarithm base, k values systematically wrong

**Recommendation**:
1. Verify with source material which logarithm base was used
2. Add comment explaining why log10 (if correct)
3. Add unit test comparing log10 vs ln behavior

---

### MEDIUM: Missing Formula Validation

**Lines**: 76, 84, 92
**Severity**: MEDIUM

```python
k = config.slope * log_L + config.intercept
```

**Issue**: No validation that formula produces sensible k values:
- What if slope and intercept are user-modified?
- What if they violate assumptions (e.g., positive slope)?
- No check that k behavior is monotonic

**Impact**: User error or config corruption silently produces garbage

**Recommendation**:
```python
# Add validation in KFormulaConfig:
def __post_init__(self):
    if self.slope >= 0:
        raise ValueError("slope must be negative for dampening behavior")
    if self.intercept < 0 or self.intercept > 1:
        warnings.warn("intercept outside [0,1] unusual")
    # Verify k(L=1) is reasonable
    k_at_1 = self.intercept
    if not (0.0 <= k_at_1 <= 1.0):
        raise ValueError(f"k(1)={k_at_1} outside [0,1]")
```

---

## 3. Boundary Conditions

### HIGH: L Near Epsilon (Very Small Values)

**Lines**: 74, 82, 90
**Severity**: HIGH

```python
# Scalar version (line 74):
L_safe = max(L, config.eps)  # eps = 1e-8

# If L = 1e-8:
log_L = log10(1e-8) = -8
k = -0.0137 * (-8) + 0.1593 = 0.1096 + 0.1593 = 0.2689
```

**Issue**: Very small L produces MODERATE k, not maximum:
- eps = 1e-8 -> k = 0.2689
- k_max = 1.0 (clamped later)
- Expected: Very small gradients should get HIGH k (more amplification)
- Actual: They get 0.27 (weak amplification)

**Test Case**:
```python
# Small gradient (should amplify):
grad_norm = 1e-10  # Clamped to 1e-8
k = compute_k(1e-10)  # Returns 0.2689
# Bigeometric: g_meta = g * |g|^(2*0.2689 - 1)
#                      = g * |g|^(-0.4622)
# This AMPLIFIES but weakly

# Expected behavior?
# If we want strong amplification for tiny gradients:
# k should be closer to 1.0, giving |g|^(2*1 - 1) = |g|^1 = |g|
```

**Impact**: Boundary behavior may not match intended amplification/dampening

**Recommendation**:
1. Clarify design intent for L near eps
2. Consider sigmoid clamping instead of linear clamp
3. Add test coverage for boundary cases

---

### HIGH: L Very Large (Explosion Cases)

**Lines**: Implicit (no explicit upper bound before clamping)
**Severity**: HIGH

```python
# Large gradient:
grad_norm = 1e6
k = -0.0137 * log10(1e6) + 0.1593
k = -0.0137 * 6 + 0.1593
k = -0.0822 + 0.1593 = 0.0771

# Extremely large gradient:
grad_norm = 1e10
k = -0.0137 * log10(1e10) + 0.1593
k = -0.0137 * 10 + 0.1593
k = -0.137 + 0.1593 = 0.0223

# Absurdly large:
grad_norm = 1e20
k = -0.0137 * 20 + 0.1593 = -0.274 + 0.1593 = -0.1147
# Clamped to k_min = 0.0
```

**Issue**: k goes NEGATIVE for very large L, then clamped to 0:
- k=0 means: `g_meta = g * |g|^(2*0 - 1) = g * |g|^(-1) = g / |g|`
- This is **sign-only** (gradient direction, magnitude=1)
- Is this intended? Extreme dampening or a bug?

**Test Case**:
```python
# Gradient explosion (bad training):
grad_norm = 1e50
k = compute_k(1e50)
print(k)  # 0.0 (clamped)

# Bigeometric transform:
# g_meta = g * |g|^(-1) = sign(g)
# All magnitude information LOST
```

**Impact**: Extreme gradients get COMPLETELY flattened (may be desired, but undocumented)

**Recommendation**:
1. Document that k=0 is intentional for explosion cases
2. Add explicit MAX_L threshold for numerical stability
3. Consider softer clamping (tanh-based) to preserve some magnitude info

---

### MEDIUM: Clamping Strategy

**Lines**: 77, 85, 93
**Severity**: MEDIUM

```python
# Hard clamp to [k_min, k_max]:
return max(config.k_min, min(config.k_max, k))  # Scalar
return np.clip(k, config.k_min, config.k_max)  # NumPy
return torch.clamp(k, config.k_min, config.k_max)  # Torch
```

**Issue**: Hard clamp loses information:
- If k would be 1.5, clamped to 1.0
- If k would be -0.2, clamped to 0.0
- No warning or logging of clamp events

**Impact**: Silent saturation hides pathological cases

**Recommendation**:
```python
# Add optional clamp tracking:
def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    L_safe = torch.clamp(L, min=config.eps)
    log_L = torch.log10(L_safe)
    k_raw = config.slope * log_L + config.intercept
    k = torch.clamp(k_raw, config.k_min, config.k_max)

    # Track saturation:
    if config.track_stats:
        n_clamped = ((k_raw < config.k_min) | (k_raw > config.k_max)).sum()
        if n_clamped > 0:
            warnings.warn(f"k clamped {n_clamped} times")

    return k
```

---

## 4. Usage Analysis

### CRITICAL: k_from_gradient() - Gradient Norm Abuse

**Lines**: 96-118
**Severity**: CRITICAL

```python
def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray],
    config: Optional[KFormulaConfig] = None
) -> Union[torch.Tensor, float]:
    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad) * config.gradient_scale
    else:
        L = np.linalg.norm(grad) * config.gradient_scale
    return compute_k(L, config)
```

**Issue 1**: Global norm loses per-parameter information:
- `torch.norm(grad)` computes L2 norm across **entire gradient tensor**
- For a 1000-element gradient: `norm = sqrt(sum(g_i^2))`
- Returns **single scalar** k for entire tensor
- Different parameters may need different k values

**Example**:
```python
# Gradient with mixed magnitudes:
grad = torch.tensor([1e-6, 1e-6, 1e-6, ..., 100.0])  # 999 small, 1 large
grad_norm = torch.norm(grad) = ~100.0  # Dominated by large element
k = k_from_gradient(grad) = 0.132  # Dampening

# Problem: Small elements (1e-6) need amplification (high k)
#          Large element (100) needs dampening (low k)
#          Single k cannot handle both
```

**Issue 2**: Dimensionality not accounted for:
- 10x10 matrix (100 params) with all values = 0.1:
  - norm = sqrt(100 * 0.01) = 1.0
- 1000x1000 matrix (1M params) with all values = 0.1:
  - norm = sqrt(1000000 * 0.01) = 100.0
- Same per-element magnitude, 100x different norm!

**Impact**: k_from_gradient() is **fundamentally broken** for per-parameter adaptation

**Recommendation**:
```python
def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray],
    config: Optional[KFormulaConfig] = None,
    per_element: bool = False  # NEW OPTION
) -> Union[torch.Tensor, float]:
    """
    Compute k from gradient.

    Args:
        grad: Gradient tensor
        config: Configuration
        per_element: If True, compute k per-element (preserves shape)
                     If False, use global norm (returns scalar)
    """
    if config is None:
        config = KFormulaConfig()

    if per_element:
        # Element-wise k (preserves spatial structure):
        L = torch.abs(grad) * config.gradient_scale
        L_safe = torch.clamp(L, min=config.eps)
        return compute_k(L_safe, config)
    else:
        # Global k (backward compatible):
        if isinstance(grad, torch.Tensor):
            L = torch.norm(grad) * config.gradient_scale
        else:
            L = np.linalg.norm(grad) * config.gradient_scale
        return compute_k(L, config)
```

---

### HIGH: k_from_layer_index() - Boundary Fencepost Error

**Lines**: 121-144
**Severity**: HIGH

```python
def k_from_layer_index(
    layer_idx: int,
    total_layers: int,
    config: Optional[KFormulaConfig] = None
) -> float:
    if config is None:
        config = KFormulaConfig()

    L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
    return compute_k(L, config)
```

**Issue 1**: Off-by-one potential:
```python
# Example: 10 layers (indices 0-9)
total_layers = 10

# First layer:
layer_idx = 0
L = (0 + 1) / 10 = 0.1
k = -0.0137 * log10(0.1) + 0.1593
k = -0.0137 * (-1) + 0.1593 = 0.1730

# Last layer:
layer_idx = 9
L = (9 + 1) / 10 = 1.0
k = -0.0137 * log10(1.0) + 0.1593
k = -0.0137 * 0 + 0.1593 = 0.1593

# But comment says (lines 129-130):
# "Early layers -> higher k -> more conservative"
# "Later layers -> lower k -> more aggressive"

# Actual: k(first) = 0.173, k(last) = 0.159
# Difference: Only 8.7% change!
```

**Issue 2**: Weak variation across layers:
- The [0.1, 1.0] range for L only changes k by ~0.014
- log10(0.1) = -1, log10(1.0) = 0
- Delta k = -0.0137 * 1 = 0.0137
- Relative change: 0.0137 / 0.16 = 8.6%

**Issue 3**: Wrong semantic interpretation:
- Comment claims "Early layers -> higher k -> more conservative"
- But bigeometric formula: **higher k = more amplification**
  - k=0.5 is identity
  - k>0.5 amplifies small gradients
  - k<0.5 dampens large gradients
- So "conservative" should mean k closer to 0.5 (less transformation)

**Impact**: Layer-wise k barely varies and semantic description is backwards

**Recommendation**:
1. Use wider L range: `L = 10^(layer_idx / (total_layers - 1) - 1)` gives [0.1, 10]
2. Fix comments to match actual behavior
3. Add validation that first/last layers have significantly different k

---

### MEDIUM: k_from_pnl() - Scale Parameter Magic Number

**Lines**: 147-170
**Severity**: MEDIUM

```python
def k_from_pnl(
    pnl: float,
    scale: float = 0.05,  # <--- Magic number
    config: Optional[KFormulaConfig] = None
) -> float:
    L = max(config.eps, abs(pnl) / scale)
    return compute_k(L, config)
```

**Issue**: scale=0.05 is arbitrary:
- No justification in comments
- No units specified (dollars? percentage? basis points?)
- For PnL = $100: L = 100 / 0.05 = 2000
- For PnL = $1: L = 1 / 0.05 = 20
- What's the expected PnL range?

**Example**:
```python
# Small trade:
pnl = 5.0  # $5 profit
L = 5.0 / 0.05 = 100
k = -0.0137 * log10(100) + 0.1593 = 0.1319

# Large trade:
pnl = 500.0  # $500 profit
L = 500.0 / 0.05 = 10000
k = -0.0137 * log10(10000) + 0.1593 = 0.1045

# Is this the intended behavior?
```

**Impact**: PnL-based k may be miscalibrated for actual trading magnitudes

**Recommendation**:
1. Document scale parameter with units and rationale
2. Make scale adaptive to account capital (e.g., scale = 0.01 * account_balance)
3. Add validation that PnL range produces sensible k values

---

## 5. Type Safety

### HIGH: Device Mismatch Risk (GPU/CPU)

**Lines**: 88-93
**Severity**: HIGH

```python
def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    L_safe = torch.clamp(L, min=config.eps)  # <--- config.eps is Python float
    log_L = torch.log10(L_safe)
    k = config.slope * log_L + config.intercept  # <--- config.slope is Python float
    return torch.clamp(k, config.k_min, config.k_max)
```

**Issue**: Broadcasting Python floats with GPU tensors:
- `config.eps`, `config.slope`, `config.intercept` are Python floats
- If `L` is on GPU, PyTorch broadcasts Python floats to GPU
- Usually works BUT:
  - Forces CPU->GPU memory transfers
  - Can cause dtype mismatches (float64 vs float32)
  - Breaks with mixed precision training

**Example**:
```python
# Training with mixed precision:
grad = torch.randn(1000, device='cuda', dtype=torch.float16)
grad_norm = torch.norm(grad)  # torch.float16 on CUDA

k = compute_k(grad_norm, config)  # config.eps is Python float64
# PyTorch must:
# 1. Convert float64 -> float16
# 2. Move to CUDA
# 3. Broadcast
# Overhead: ~0.1ms per call (adds up!)
```

**Impact**: Performance degradation and potential precision loss

**Recommendation**:
```python
def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    # Convert config to tensor ONCE:
    eps = torch.tensor(config.eps, device=L.device, dtype=L.dtype)
    slope = torch.tensor(config.slope, device=L.device, dtype=L.dtype)
    intercept = torch.tensor(config.intercept, device=L.device, dtype=L.dtype)
    k_min = torch.tensor(config.k_min, device=L.device, dtype=L.dtype)
    k_max = torch.tensor(config.k_max, device=L.device, dtype=L.dtype)

    L_safe = torch.clamp(L, min=eps)
    log_L = torch.log10(L_safe)
    k = slope * log_L + intercept
    return torch.clamp(k, k_min, k_max)
```

---

### MEDIUM: Scalar vs Tensor Return Type

**Lines**: 96-99
**Severity**: MEDIUM

```python
def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray],
    config: Optional[KFormulaConfig] = None
) -> Union[torch.Tensor, float]:  # <--- Inconsistent return type
```

**Issue**: Return type depends on norm implementation:
- `torch.norm()` returns 0-dim tensor (not scalar)
- `np.linalg.norm()` returns Python float
- Caller must handle both cases

**Example**:
```python
grad_torch = torch.randn(100)
k_torch = k_from_gradient(grad_torch)
print(type(k_torch))  # torch.Tensor (0-dim)

grad_numpy = np.random.randn(100)
k_numpy = k_from_gradient(grad_numpy)
print(type(k_numpy))  # float

# Problem: Cannot use in branching:
if k_torch > 0.5:  # Works
    ...
if k_numpy > 0.5:  # Works
# BUT: Incompatible in typed code or JIT compilation
```

**Impact**: Type system confusion and JIT compilation failures

**Recommendation**:
```python
def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray],
    config: Optional[KFormulaConfig] = None
) -> float:  # <--- Always return Python float
    if config is None:
        config = KFormulaConfig()

    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad).item() * config.gradient_scale  # .item()
    else:
        L = float(np.linalg.norm(grad)) * config.gradient_scale

    return float(compute_k(L, config))  # Ensure float
```

---

## 6. Edge Cases

### CRITICAL: All-Zero Gradient

**Lines**: 96-118
**Severity**: CRITICAL

```python
def k_from_gradient(grad, config):
    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad) * config.gradient_scale
    # ...
```

**Issue**: If `grad` is all zeros:
- `torch.norm(grad)` = 0.0
- Not clamped before passing to `compute_k()`
- Inside `compute_k()`: `L_safe = max(0.0, eps)` = eps
- Returns k for L=eps

**Test Case**:
```python
grad = torch.zeros(1000)
k = k_from_gradient(grad)
# L = 0.0 -> L_safe = 1e-8
# k = -0.0137 * log10(1e-8) + 0.1593 = 0.2689

# Expected: All-zero gradient should probably be no-op (k=0.5?)
# Actual: k=0.27 (amplification applied to zero = still zero, but wasteful)
```

**Impact**: Unnecessary computation and potential numerical instability

**Recommendation**:
```python
def k_from_gradient(grad, config):
    if config is None:
        config = KFormulaConfig()

    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad)
        if L == 0:
            return 0.5  # Identity transform for zero gradient
        L *= config.gradient_scale
    else:
        L = np.linalg.norm(grad)
        if L == 0:
            return 0.5
        L *= config.gradient_scale

    return compute_k(L, config)
```

---

### HIGH: NaN and Inf Handling

**Lines**: No explicit handling
**Severity**: HIGH

```python
# Current code has NO checks for:
# - NaN in gradients
# - Inf in gradients
# - NaN in L
# - Inf in L
```

**Issue**: Pathological values propagate:
```python
# NaN gradient:
grad = torch.tensor([1.0, float('nan'), 2.0])
L = torch.norm(grad)  # = NaN
k = compute_k(L, config)  # log10(NaN) = NaN
# Result: k = NaN (infects entire computation)

# Inf gradient:
grad = torch.tensor([float('inf')])
L = torch.norm(grad)  # = Inf
k = compute_k(L, config)  # log10(Inf) = Inf
# Result: k = Inf (clamped to k_max, but still broken)
```

**Impact**: NaN/Inf gradients cause silent failures or training crashes

**Recommendation**:
```python
def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    # Check for pathological values:
    if torch.isnan(L).any():
        raise ValueError("NaN detected in L")
    if torch.isinf(L).any():
        warnings.warn("Inf detected in L, clamping to max")
        L = torch.clamp(L, max=1e10)

    L_safe = torch.clamp(L, min=config.eps)
    log_L = torch.log10(L_safe)
    k = config.slope * log_L + config.intercept
    return torch.clamp(k, config.k_min, config.k_max)
```

---

### MEDIUM: Layer Index Out of Bounds

**Lines**: 121-144
**Severity**: MEDIUM

```python
def k_from_layer_index(
    layer_idx: int,
    total_layers: int,
    config: Optional[KFormulaConfig] = None
) -> float:
    L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
    # No validation!
```

**Issue**: No bounds checking:
```python
# Invalid inputs:
k = k_from_layer_index(-1, 10)  # Negative index
# L = max(0.01, (-1 + 1) / 10) = max(0.01, 0) = 0.01
# Returns k(0.01) without error

k = k_from_layer_index(100, 10)  # Index > total
# L = max(0.01, (100 + 1) / 10) = 10.1
# Returns k(10.1) without error

k = k_from_layer_index(5, 0)  # Zero layers
# L = (5 + 1) / 0  -> ZeroDivisionError!
```

**Impact**: Invalid inputs produce garbage or crash

**Recommendation**:
```python
def k_from_layer_index(
    layer_idx: int,
    total_layers: int,
    config: Optional[KFormulaConfig] = None
) -> float:
    # Validate inputs:
    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}")
    if not (0 <= layer_idx < total_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {total_layers})")

    if config is None:
        config = KFormulaConfig()

    L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
    return compute_k(L, config)
```

---

## 7. BUG HUNTING

### CRITICAL: gradient_scale and layer_scale NOT USED in Defaults

**Lines**: 41-42, 114, 143
**Severity**: CRITICAL

```python
# In KFormulaConfig (lines 41-42):
gradient_scale: float = 1.0
layer_scale: float = 1.0

# In k_from_gradient (line 114):
L = torch.norm(grad) * config.gradient_scale

# In k_from_layer_index (line 143):
L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
```

**Issue**: Scale factors default to 1.0 (no effect), but no guidance on when to change:
- What values should gradient_scale be?
- What values should layer_scale be?
- When should they be tuned?
- How do they interact with the MOO-derived coefficients?

**Example**:
```python
# Default behavior (scale=1.0):
grad_norm = 100
k = k_from_gradient(grad)  # L = 100, k = 0.132

# With gradient_scale=10.0:
k = k_from_gradient(grad, KFormulaConfig(gradient_scale=10.0))
# L = 1000, k = 0.118

# 10% difference, but no documentation on which is correct!
```

**Impact**: Users cannot tune scales without understanding original MOO fitting

**Recommendation**:
1. Add docstring explaining scale factors:
   ```python
   gradient_scale: float = 1.0
   """
   Scale factor for gradient magnitudes.

   Calibration guide:
   - 1.0: Default (assumes gradients in [1e-3, 1e3] range)
   - 0.1: If gradients typically in [1e-2, 1e4] (scale down)
   - 10.0: If gradients typically in [1e-4, 1e2] (scale up)

   Goal: Map typical gradient range to L in [0.1, 100]
   """
   ```

2. Add auto-calibration utility:
   ```python
   @staticmethod
   def calibrate_gradient_scale(grad_samples: List[torch.Tensor]) -> float:
       """
       Compute optimal gradient_scale from sample gradients.

       Target: Median gradient norm should map to L ~ 1.0
       """
       norms = [torch.norm(g).item() for g in grad_samples]
       median_norm = np.median(norms)
       return 1.0 / median_norm if median_norm > 0 else 1.0
   ```

---

### HIGH: Silent Failures in Type Dispatch

**Lines**: 64-69
**Severity**: HIGH

```python
def compute_k(
    L: Union[float, np.ndarray, torch.Tensor],
    config: Optional[KFormulaConfig] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    if isinstance(L, torch.Tensor):
        return _compute_k_torch(L, config)
    elif isinstance(L, np.ndarray):
        return _compute_k_numpy(L, config)
    else:
        return _compute_k_scalar(float(L), config)
```

**Issue**: `else` clause assumes L is scalar-like:
- What if L is a list?
- What if L is a tuple?
- What if L is a custom type?

**Example**:
```python
# Unexpected input:
L = [1.0, 2.0, 3.0]  # List
k = compute_k(L, config)
# Falls to else clause: float([1.0, 2.0, 3.0]) -> TypeError!

# Or worse:
L = "100"  # String
k = compute_k(L, config)
# float("100") = 100.0 (silently converts!)
```

**Impact**: Type errors or silent conversions

**Recommendation**:
```python
def compute_k(
    L: Union[float, np.ndarray, torch.Tensor],
    config: Optional[KFormulaConfig] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    if config is None:
        config = KFormulaConfig()

    if isinstance(L, torch.Tensor):
        return _compute_k_torch(L, config)
    elif isinstance(L, np.ndarray):
        return _compute_k_numpy(L, config)
    elif isinstance(L, (int, float)):
        return _compute_k_scalar(float(L), config)
    else:
        raise TypeError(f"L must be float/ndarray/Tensor, got {type(L)}")
```

---

### MEDIUM: Off-by-One in Layer Indexing (Potential)

**Lines**: 143
**Severity**: MEDIUM

```python
L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
```

**Issue**: `layer_idx + 1` suggests 1-indexed layers:
- If layers are 0-indexed (layer_idx=0 is first layer): Correct
- If layers are 1-indexed (layer_idx=1 is first layer): Off-by-one

**Example**:
```python
# Assuming 0-indexed (typical PyTorch):
total_layers = 10
layer_idx = 0  # First layer
L = (0 + 1) / 10 = 0.1  # Correct

# But if user thinks 1-indexed:
layer_idx = 1  # "First layer"
L = (1 + 1) / 10 = 0.2  # Wrong!
```

**Impact**: Semantic confusion leading to wrong k values

**Recommendation**:
```python
def k_from_layer_index(
    layer_idx: int,
    total_layers: int,
    config: Optional[KFormulaConfig] = None
) -> float:
    """
    Compute k from layer position.

    Args:
        layer_idx: 0-indexed layer position (0 = first layer)
        total_layers: Total number of layers
        config: Optional configuration

    Returns:
        k value for this layer
    """
    # Explicit 0-indexing check:
    assert 0 <= layer_idx < total_layers, \
        f"layer_idx must be in [0, {total_layers}), got {layer_idx}"

    # ...
```

---

## 8. Missing Functionality

### MEDIUM: No Inverse Function

**Severity**: MEDIUM

**Issue**: Cannot compute L from k:
- Given desired k, what L is needed?
- Useful for debugging and calibration

**Recommendation**:
```python
def compute_L_from_k(
    k: float,
    config: Optional[KFormulaConfig] = None
) -> float:
    """
    Inverse of k(L): Compute L that produces given k.

    Formula: k = slope * log10(L) + intercept
    Solve for L: log10(L) = (k - intercept) / slope
                 L = 10^((k - intercept) / slope)
    """
    if config is None:
        config = KFormulaConfig()

    # Clamp k to valid range:
    k = max(config.k_min, min(config.k_max, k))

    # Inverse formula:
    log_L = (k - config.intercept) / config.slope
    L = 10 ** log_L

    return L
```

---

### LOW: No Visualization Utilities

**Severity**: LOW

**Issue**: Hard to understand k(L) behavior without plotting

**Recommendation**:
```python
def plot_k_curve(
    L_range: tuple = (1e-3, 1e3),
    config: Optional[KFormulaConfig] = None,
    show_applications: bool = True
) -> None:
    """
    Plot k(L) curve and typical application ranges.
    """
    import matplotlib.pyplot as plt

    if config is None:
        config = KFormulaConfig()

    L_vals = np.logspace(np.log10(L_range[0]), np.log10(L_range[1]), 1000)
    k_vals = [compute_k(L, config) for L in L_vals]

    plt.figure(figsize=(10, 6))
    plt.semilogx(L_vals, k_vals, label='k(L)')
    plt.axhline(0.5, color='gray', linestyle='--', label='Identity (k=0.5)')

    if show_applications:
        # Show typical ranges:
        plt.axvspan(0.001, 0.1, alpha=0.2, color='blue', label='Small gradients')
        plt.axvspan(1, 10, alpha=0.2, color='green', label='Normal gradients')
        plt.axvspan(100, 1000, alpha=0.2, color='red', label='Large gradients')

    plt.xlabel('L (scale parameter)')
    plt.ylabel('k (meta-weight)')
    plt.title('k(L) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## Summary of Issues

| Severity | Count | Issues |
|----------|-------|--------|
| CRITICAL | 5 | Semantic domain mismatch, log10 vs ln, k_from_gradient() global norm, all-zero gradient, gradient_scale unused |
| HIGH | 5 | Missing source traceability, L near epsilon, L very large, device mismatch, NaN/Inf handling |
| MEDIUM | 5 | Missing formula validation, clamping strategy, k_from_pnl() magic number, scalar vs tensor return, layer index OOB |
| LOW | 2 | No inverse function, no visualization |

**Total**: 17 issues identified

---

## Recommendations Priority

### Immediate (Block Training):
1. Add disclaimer that coefficients are NOT VALIDATED for ML/finance domains
2. Fix k_from_gradient() to use per-element k or document global norm limitation
3. Add NaN/Inf checks to prevent training crashes
4. Fix device mismatch for GPU training
5. Document gradient_scale and layer_scale calibration

### Short-term (Next Sprint):
6. Verify log10 vs ln with source material
7. Add boundary condition tests (L near eps, L very large)
8. Fix layer-wise k to have meaningful variation
9. Validate k_from_pnl() scale for actual trading magnitudes
10. Add input validation to all public functions

### Long-term (Post-MVP):
11. Empirically recalibrate coefficients for trader-ai use cases
12. Add inverse function and visualization utilities
13. Implement auto-calibration for scale factors
14. Add comprehensive unit tests (currently NONE)
15. Consider adaptive k strategies (per-parameter, time-varying)

---

## Testing Recommendations

### Unit Tests (Missing):
```python
# tests/unit/test_k_formula.py

def test_k_at_reference_points():
    """Test k at documented reference points."""
    # From physics domain:
    assert abs(compute_k(1.0) - 0.1593) < 1e-6

def test_k_monotonicity():
    """k should decrease as L increases."""
    k1 = compute_k(0.1)
    k2 = compute_k(1.0)
    k3 = compute_k(10.0)
    assert k1 > k2 > k3

def test_k_boundary_conditions():
    """Test k at extreme L values."""
    # Very small L:
    k_small = compute_k(1e-10)
    assert 0.0 <= k_small <= 1.0

    # Very large L:
    k_large = compute_k(1e10)
    assert 0.0 <= k_large <= 1.0

def test_k_from_gradient_zero():
    """Zero gradient should handle gracefully."""
    grad = torch.zeros(100)
    k = k_from_gradient(grad)
    assert not torch.isnan(k)
    assert not torch.isinf(k)

def test_k_from_gradient_nan():
    """NaN gradient should raise error."""
    grad = torch.tensor([1.0, float('nan')])
    with pytest.raises(ValueError):
        k = k_from_gradient(grad)

def test_k_from_layer_index_bounds():
    """Layer index out of bounds should raise error."""
    with pytest.raises(ValueError):
        k_from_layer_index(-1, 10)
    with pytest.raises(ValueError):
        k_from_layer_index(10, 10)
    with pytest.raises(ValueError):
        k_from_layer_index(5, 0)

def test_k_device_consistency():
    """k computation should work on GPU."""
    if torch.cuda.is_available():
        grad = torch.randn(100, device='cuda')
        k = k_from_gradient(grad)
        assert k.device.type == 'cuda'

def test_k_dtype_consistency():
    """k should preserve dtype."""
    grad = torch.randn(100, dtype=torch.float16)
    k = k_from_gradient(grad)
    # Should not upcast to float64
    assert k.dtype == torch.float16

def test_log10_vs_ln():
    """Verify log10 is correct (not ln)."""
    # If formula derived with log10, this should match:
    L = 100
    k_log10 = -0.0137 * np.log10(L) + 0.1593
    k_computed = compute_k(L)
    assert abs(k_log10 - k_computed) < 1e-6

    # If formula should be ln, this would fail:
    k_ln = -0.0137 * np.log(L) + 0.1593
    assert abs(k_ln - k_computed) > 0.01  # Should differ significantly
```

### Integration Tests:
```python
def test_metagrokfast_integration():
    """Test k_formula with MetaGrokFast optimizer."""
    model = SimpleNet()
    optimizer = MetaGrokFast(model.parameters())

    # Run training step:
    loss = model(torch.randn(10, 10)).sum()
    loss.backward()
    optimizer.step()

    # Should not crash or produce NaN
    for param in model.parameters():
        assert not torch.isnan(param).any()

def test_k_calibration():
    """Test gradient_scale calibration."""
    # Collect sample gradients:
    samples = [torch.randn(100) * 0.01 for _ in range(100)]

    # Auto-calibrate:
    scale = KFormulaConfig.calibrate_gradient_scale(samples)

    # Median gradient should map to k ~ 0.16:
    median_grad = torch.stack(samples).median(dim=0).values
    k = k_from_gradient(median_grad, KFormulaConfig(gradient_scale=scale))
    assert 0.10 < k < 0.20
```

---

## Conclusion

The k(L) formula implementation is **mathematically correct** but suffers from:
1. **Semantic domain mismatch** (physics -> ML transfer not validated)
2. **Missing validation and error handling**
3. **Suboptimal design choices** (global norm, weak layer variation)
4. **Zero test coverage**

**CRITICAL**: Before using in production, the coefficients MUST be recalibrated for trader-ai's specific use cases (gradients, layers, PnL). The current R^2=0.71 applies ONLY to physics length scales.

**RECOMMENDATION**: Add large disclaimer and begin empirical validation campaign.

---

**Audit Complete**: 2025-12-16
**Next Steps**: Implement priority recommendations and add unit tests
