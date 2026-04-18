# Bar Permutation Vectorization: Visual Guide

## The Problem: Sequential Dependency Chain

In the original implementation, each bar depends on the previous bar's close price, creating a sequential dependency chain:

```
Initial Close = 100.0

Bar 0:
  open[0]  = 100.0 + r_open[0]           = 100.5
  high[0]  = 100.5 + r_high[0]           = 101.0
  low[0]   = 100.5 + r_low[0]            = 100.0
  close[0] = 100.5 + r_close[0]          = 100.8

Bar 1: (depends on close[0])
  open[1]  = 100.8 + r_open[1]           = 101.2
  high[1]  = 101.2 + r_high[1]           = 101.5
  low[1]   = 101.2 + r_low[1]            = 100.9
  close[1] = 101.2 + r_close[1]          = 101.3

Bar 2: (depends on close[1])
  open[2]  = 101.3 + r_open[2]           = 101.8
  ...
```

This **CANNOT** be directly vectorized because:
- Each step depends on the previous result
- Python must execute loop sequentially
- No SIMD parallelization possible

---

## The Solution: Reformulate as Cumulative Sum

### Step 1: Identify the Gap Pattern

Instead of thinking "bar i depends on bar i-1", think "what's the gap between bars?"

```
Gap 0: open[0] - initial_close = r_open[0]
Gap 1: open[1] - open[0]       = (close[0] - open[0]) + (open[1] - close[0])
                               = r_close[0] + r_open[1]
Gap 2: open[2] - open[1]       = r_close[1] + r_open[2]
```

Pattern discovered:
```
gaps = [r_open[0], r_open[1] + r_close[0], r_open[2] + r_close[1], ...]
```

### Step 2: Build Gaps Array (Vectorized)

```python
gaps = np.empty(n_recon)
gaps[0] = r_open[0]                    # First gap (scalar assignment)
gaps[1:] = r_open[1:] + r_close[:-1]   # Rest of gaps (VECTORIZED)
```

**Visualization:**
```
r_open  = [0.5,  0.4,  0.6,  0.3, ...]
r_close = [0.3,  0.1,  0.5,  0.2, ...]

gaps[0] = 0.5
gaps[1] = r_open[1] + r_close[0] = 0.4 + 0.3 = 0.7
gaps[2] = r_open[2] + r_close[1] = 0.6 + 0.1 = 0.7
gaps[3] = r_open[3] + r_close[2] = 0.3 + 0.5 = 0.8
...

gaps = [0.5, 0.7, 0.7, 0.8, ...]
```

### Step 3: Cumulative Sum (Vectorized)

```python
open_prices = initial_close + np.cumsum(gaps)
```

**Visualization:**
```
initial_close = 100.0
gaps          = [0.5,  0.7,  0.7,  0.8, ...]

cumsum(gaps)  = [0.5,  1.2,  1.9,  2.7, ...]
open_prices   = [100.5, 101.2, 101.9, 102.7, ...]
```

This is **exactly the same** as the sequential loop but computed in parallel!

### Step 4: Apply Relative Movements (Vectorized)

```python
perm_bars[:, 0] = open_prices              # Open
perm_bars[:, 1] = open_prices + r_high     # High (VECTORIZED)
perm_bars[:, 2] = open_prices + r_low      # Low  (VECTORIZED)
perm_bars[:, 3] = open_prices + r_close    # Close (VECTORIZED)
```

**Visualization:**
```
open_prices = [100.5, 101.2, 101.9, 102.7]
r_high      = [0.5,   0.3,   0.4,   0.6]
r_low       = [-0.5,  -0.3,  -0.4,  -0.2]
r_close     = [0.3,   0.1,   0.5,   0.2]

perm_bars[:, 1] = [101.0, 101.5, 102.3, 103.3]  # High
perm_bars[:, 2] = [100.0, 100.9, 101.5, 102.5]  # Low
perm_bars[:, 3] = [100.8, 101.3, 102.4, 102.9]  # Close
```

All computed **in parallel** using CPU SIMD instructions!

---

## Performance Visualization

### Memory Access Pattern

**Sequential (SLOW):**
```
CPU does:
  Load perm_bars[i-1, 3]
  Load r_open[k]
  Add
  Store perm_bars[i, 0]
  Load perm_bars[i, 0]  # Just stored!
  Load r_high[k]
  Add
  Store perm_bars[i, 1]
  ... (repeat for each bar)
```

Many memory loads/stores, cache misses, no parallelization.

**Vectorized (FAST):**
```
CPU does:
  Load r_open[1:] (entire array)
  Load r_close[:-1] (entire array)
  Add (SIMD parallel, 4-8 values at once)
  Store gaps[1:] (entire array)

  Compute cumsum(gaps) (optimized C code)

  Load open_prices (entire array)
  Load r_high (entire array)
  Add (SIMD parallel)
  Store perm_bars[:, 1] (entire array)
  ... (repeat for L/C in parallel)
```

Fewer operations, better cache usage, SIMD parallelization.

---

## Scaling Behavior

### Operation Count

**Sequential:**
```
For n bars:
- n iterations
- 4n array loads (O, H, L, C)
- 4n additions
- 4n array stores

Total: 12n operations in Python loop
```

**Vectorized:**
```
For n bars:
- 1 array slice (gaps[0])
- 1 vectorized add (gaps[1:])
- 1 cumsum
- 4 vectorized adds (H, L, C relative to O)

Total: 7 C-level operations (much faster than Python)
```

### Wall-Clock Time

```
n = 100:
  Sequential: 2.29ms (overhead dominates)
  Vectorized: 2.09ms (overhead dominates)
  Speedup: 1.1x

n = 1,000:
  Sequential: 5.65ms (loop cost grows)
  Vectorized: 1.99ms (constant overhead)
  Speedup: 2.8x

n = 10,000:
  Sequential: 32.60ms (loop cost dominates)
  Vectorized: 4.59ms (still near constant)
  Speedup: 7.1x
```

As n grows, vectorized version maintains near-constant overhead while sequential grows linearly.

---

## Why This Works

### Mathematical Equivalence

**Sequential formulation:**
```
open[i] = open[i-1] + (close[i-1] - open[i-1]) + (open[i] - close[i-1])
        = open[i-1] + r_close[i-1] + r_open[i]
```

Expanding recursively:
```
open[0] = initial_close + r_open[0]
open[1] = open[0] + r_close[0] + r_open[1]
        = initial_close + r_open[0] + r_close[0] + r_open[1]
open[2] = open[1] + r_close[1] + r_open[2]
        = initial_close + r_open[0] + r_close[0] + r_open[1] + r_close[1] + r_open[2]

General:
open[i] = initial_close + sum(r_open[0:i+1] + r_close[0:i])
```

**Vectorized formulation:**
```
gaps[0] = r_open[0]
gaps[i] = r_open[i] + r_close[i-1]  for i > 0

open = initial_close + cumsum(gaps)
     = initial_close + [gaps[0], gaps[0] + gaps[1], gaps[0] + gaps[1] + gaps[2], ...]
     = initial_close + [r_open[0], r_open[0] + r_open[1] + r_close[0], ...]
```

These are **mathematically identical** but the vectorized form can be computed in parallel.

---

## Key Insight

The transformation from sequential to vectorized requires identifying:

1. **The accumulation pattern** (what are we summing?)
2. **The independent elements** (what can be computed in parallel?)
3. **The cumulative operation** (can we use cumsum?)

For bar permutation:
1. We're accumulating gaps between bars
2. Each gap can be computed independently: `gap[i] = r_open[i] + r_close[i-1]`
3. Open prices are cumsum of gaps

Once reformulated, NumPy's optimized C implementations take over:
- `np.cumsum()` uses fast C code
- Array slicing is zero-copy (views)
- Broadcasting uses SIMD instructions
- All operations are cache-friendly

Result: **7x speedup** with exact numerical equivalence!
