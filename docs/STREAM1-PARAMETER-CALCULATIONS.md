# STREAM 1: Parameter Calculations

## Model Architecture

TinyRecursiveModel (TRM) with:
- Input dimension: 10 (market features)
- Hidden dimension: configurable
- Output dimension: 2 (binary classification) or 8 (multi-class)
- Recursion cycles: 3
- Latent steps: 6

## Parameter Count Formula

### Layer-by-Layer Breakdown

```python
# Given:
input_dim = 10
hidden_dim = H  # variable
output_dim = C  # 2 for binary, 8 for multi-class

# Input projection
input_proj = input_dim * hidden_dim + hidden_dim
           = 10H + H = 11H

# Reasoning layer 1 (combines x, y, z)
reasoning_layer1 = (hidden_dim * 3) * hidden_dim + hidden_dim
                 = 3H * H + H = 3H^2 + H

# Reasoning layer 2
reasoning_layer2 = hidden_dim * hidden_dim + hidden_dim
                 = H^2 + H

# Solution layer 1 (combines y, z)
solution_layer1 = (hidden_dim * 2) * hidden_dim + hidden_dim
                = 2H * H + H = 2H^2 + H

# Solution layer 2
solution_layer2 = hidden_dim * hidden_dim + hidden_dim
                = H^2 + H

# Output head
output_head = hidden_dim * output_dim + output_dim
            = HC + C

# Halt layer 1
halt_layer1 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
            = H * (H//2) + H//2 = (H//2)(H + 1)

# Halt layer 2
halt_layer2 = (hidden_dim // 2) * 1 + 1
            = H//2 + 1

# LayerNorm (3 instances: reasoning, solution, output)
# Each LayerNorm has 2*hidden_dim parameters (gamma and beta)
layer_norm = 3 * (2 * hidden_dim)
           = 6H

# Total
total_params = 11H + 3H^2 + H + H^2 + H + 2H^2 + H + H^2 + H + HC + C
               + (H//2)(H+1) + H//2 + 1 + 6H
```

### Simplified Formula

For integer division approximation (H//2 ≈ H/2):

```python
total_params ≈ 7H^2 + 19H + HC + C + 0.5H^2 + 0.5H + 0.5H + 1
             ≈ 7.5H^2 + 20H + HC + C + 1
```

For binary classification (C=2):
```python
total_params ≈ 7.5H^2 + 22H + 3
```

## Actual Calculations

### Configuration 1: ORIGINAL (BROKEN)
```
hidden_dim = 1024
output_dim = 8

Input projection:       11,264
Reasoning layers:    4,196,352  (3*1024^2 + 1024 + 1024^2 + 1024)
Solution layers:     3,147,776  (2*1024^2 + 1024 + 1024^2 + 1024)
Output head:             8,200  (1024*8 + 8)
Halt layers:           525,313  (512*1025 + 513)
LayerNorm:               6,144  (3 * 2 * 1024)
---------------------------------
TOTAL:               7,895,049

Dataset size: 1,201
Model-to-data ratio: 6,574:1
STATUS: FAIL (65.74x over limit)
```

### Configuration 2: FIRST FIX ATTEMPT
```
hidden_dim = 256
output_dim = 2

Input projection:        2,816
Reasoning layers:      262,656  (3*256^2 + 256 + 256^2 + 256)
Solution layers:       197,120  (2*256^2 + 256 + 256^2 + 256)
Output head:             2,056  (256*2 + 2)
Halt layers:            33,025  (128*257 + 129)
LayerNorm:               1,536  (3 * 2 * 256)
---------------------------------
TOTAL:                 499,209

Dataset size: 1,201
Model-to-data ratio: 415.7:1
STATUS: FAIL (4.16x over limit)
```

### Configuration 3: SECOND ATTEMPT
```
hidden_dim = 128
output_dim = 2

Input projection:        1,408
Reasoning layers:       65,792  (3*128^2 + 128 + 128^2 + 128)
Solution layers:        49,408  (2*128^2 + 128 + 128^2 + 128)
Output head:               258  (128*2 + 2)
Halt layers:             8,321  (64*129 + 65)
LayerNorm:                 768  (3 * 2 * 128)
---------------------------------
TOTAL:                 125,955

Dataset size: 1,201
Model-to-data ratio: 104.9:1
STATUS: FAIL (1.05x over limit)
```

### Configuration 4: FINAL FIX (SUCCESS)
```
hidden_dim = 96
output_dim = 2

Input projection:        1,056  (10*96 + 96)
Reasoning layers:       37,056  (3*96^2 + 96 + 96^2 + 96)
Solution layers:        27,840  (2*96^2 + 96 + 96^2 + 96)
Output head:               194  (96*2 + 2)
Halt layers:             4,705  (48*97 + 49)
LayerNorm:                 576  (3 * 2 * 96)
---------------------------------
TOTAL:                  71,427

Dataset size (train): 842
Model-to-data ratio: 84.8:1
STATUS: PASS (15.2 under limit)
```

## Ratio Targets

### Standard ML Guidelines
- Recommended: < 10:1 (ideal)
- Acceptable: < 100:1 (workable)
- Risky: 100-1000:1 (high overfitting risk)
- Broken: > 1000:1 (will not generalize)

### Our Results
- Original: 6,574:1 (BROKEN - 65.74x over acceptable limit)
- Fixed: 84.8:1 (PASS - 15.2 under limit, 77.5x improvement)

## Hidden Dimension Selection Table

For binary classification (C=2), dataset size N=842:

| hidden_dim | Parameters | Ratio  | Status | Safety Margin |
|------------|-----------|--------|--------|---------------|
| 64         | 32,003    | 38.0:1 | PASS   | 62.0          |
| 96         | 71,427    | 84.8:1 | PASS   | 15.2          |
| 128        | 125,955   | 149.6:1| FAIL   | -49.6         |
| 256        | 499,209   | 592.9:1| FAIL   | -492.9        |
| 512        | 1,985,033 | 2,357:1| FAIL   | -2,257        |
| 1024       | 7,895,049 | 9,375:1| FAIL   | -9,275        |

**Recommended:** hidden_dim = 96 (best balance of capacity and safety)
**Conservative:** hidden_dim = 64 (maximum safety, may underfit)
**Aggressive:** hidden_dim = 128 (slightly over limit, use heavy regularization)

## Verification Script

```python
def count_trm_parameters(hidden_dim, output_dim):
    """Count exact TRM parameters."""
    # Input projection
    input_proj = 10 * hidden_dim + hidden_dim

    # Reasoning layers
    reasoning_l1 = (hidden_dim * 3) * hidden_dim + hidden_dim
    reasoning_l2 = hidden_dim * hidden_dim + hidden_dim

    # Solution layers
    solution_l1 = (hidden_dim * 2) * hidden_dim + hidden_dim
    solution_l2 = hidden_dim * hidden_dim + hidden_dim

    # Output head
    output_head = hidden_dim * output_dim + output_dim

    # Halt layers
    halt_l1 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
    halt_l2 = (hidden_dim // 2) * 1 + 1

    # LayerNorm (3x)
    layer_norm = 3 * (2 * hidden_dim)

    total = (input_proj + reasoning_l1 + reasoning_l2 +
             solution_l1 + solution_l2 + output_head +
             halt_l1 + halt_l2 + layer_norm)

    return total

# Verify
assert count_trm_parameters(96, 2) == 71_427
assert count_trm_parameters(1024, 8) == 7_895_049
```

## Usage in Code

```python
# In train_until_grokking.py
model = TinyRecursiveModel(
    input_dim=10,
    hidden_dim=96,        # Optimized for <100:1 ratio
    output_dim=2,         # Binary classification
    num_latent_steps=6,
    num_recursion_cycles=3,
)

# Expected: 71,427 parameters
# Dataset: 842 training samples
# Ratio: 84.8:1 (PASS)
```
