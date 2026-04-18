# Deferred Issues Implementation Plan

## Overview

This plan addresses 3 deferred issues from the AI Training Debug MECE analysis:

| ID | Issue | Impact | Complexity | Priority |
|----|-------|--------|------------|----------|
| A3 | No temporal sequences | Model can't detect time patterns | HIGH | P1 |
| B6 | Complexity penalty stub | May overfit with max recursion | MEDIUM | P2 |
| C4 | Flash Crash 2 samples | Can't learn rare crash patterns | HIGH | P1 |

---

## A3: Temporal Sequences

### Problem
- Current data shape: `(batch, 10)` - each sample is independent
- TRM can't detect patterns like "VIX rising for 3 consecutive days"
- Model sees snapshots, not trends

### Solution: Sliding Window + TemporalEncoder

**Architecture Change:**
```
BEFORE: features (batch, 10) -> input_proj -> TRM reasoning -> output
AFTER:  sequences (batch, seq_len, 10) -> TemporalEncoder -> (batch, 512) -> TRM reasoning -> output
```

### Files to Modify

#### 1. `src/models/trm_model.py`
```python
# Add new class
class TemporalEncoder(nn.Module):
    """LSTM-based encoder for sequential market features"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (batch, hidden_dim) - last layer's hidden state

# Modify TinyRecursiveModel.__init__
def __init__(
    self,
    input_dim: int = 10,
    hidden_dim: int = 1024,
    output_dim: int = 8,
    num_latent_steps: int = 6,
    num_recursion_cycles: int = 3,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    seq_len: int = 1,              # NEW
    use_temporal: bool = False     # NEW
):
    # ... existing code ...

    self.use_temporal = use_temporal
    self.seq_len = seq_len

    if use_temporal and seq_len > 1:
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim)
        # input_proj now receives hidden_dim from temporal encoder
        self.input_proj = nn.Identity()
    else:
        self.input_proj = nn.Linear(input_dim, hidden_dim)

# Modify forward()
def forward(self, x: torch.Tensor, T: Optional[int] = None, n: Optional[int] = None, ...):
    # Handle temporal input
    if self.use_temporal and x.ndim == 3:
        # x: (batch, seq_len, input_dim) -> (batch, hidden_dim)
        x_proj = self.temporal_encoder(x)
    else:
        # x: (batch, input_dim) -> (batch, hidden_dim)
        x_proj = self.input_proj(x)

    # Rest of TRM reasoning unchanged...
```

#### 2. `src/training/trm_data_loader.py`
```python
class TRMDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        split: str,
        indices: np.ndarray,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
        seq_len: int = 1  # NEW: sequence length for temporal model
    ):
        self.seq_len = seq_len
        # ... existing loading code ...

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        if self.seq_len > 1:
            # Return sequence of features
            start_idx = max(0, idx - self.seq_len + 1)
            seq_features = self.features[start_idx:idx + 1]

            # Pad if at beginning of dataset
            if len(seq_features) < self.seq_len:
                padding = np.zeros((self.seq_len - len(seq_features), 10), dtype=np.float32)
                seq_features = np.vstack([padding, seq_features])

            features_tensor = torch.from_numpy(seq_features)  # (seq_len, 10)
        else:
            features_tensor = torch.from_numpy(self.features[idx])  # (10,)

        return features_tensor, int(self.strategy_labels[idx]), float(self.pnl_values[idx])
```

#### 3. `config/feature_flags.json`
```json
{
    "use_temporal_sequences": false,
    "temporal_seq_len": 5
}
```

### Testing
```python
# Test temporal model
model = TinyRecursiveModel(use_temporal=True, seq_len=5)
x = torch.randn(4, 5, 10)  # (batch, seq_len, features)
output = model(x)
assert output['strategy_logits'].shape == (4, 8)
```

---

## B6: Complexity Penalty

### Problem
- Line 71 in `trm_loss_functions.py`: "TODO: add complexity penalty"
- Model always uses max T=3 cycles even when T=1 would suffice
- No incentive to halt early when confident

### Solution: Halt-Based Complexity Penalty

**Loss Formula:**
```
L_total = lambda_halt * L_halt + lambda_profit * L_profit + lambda_complexity * L_complexity

where:
L_complexity = mean(1 - halt_probability)
             = 1 if model never confident (halt_prob=0)
             = 0 if model always confident (halt_prob=1)
```

### File to Modify

#### `src/training/trm_loss_functions.py`
```python
def compute_complexity_penalty(
    halt_probability: torch.Tensor,
    penalty_type: str = 'halt_based'
) -> torch.Tensor:
    """
    Compute complexity penalty to encourage early halting.

    Args:
        halt_probability: (batch, 1) halt confidence from model (after sigmoid)
        penalty_type: 'halt_based' penalizes low confidence

    Returns:
        penalty: Scalar penalty value in [0, 1]

    Intuition:
        - High halt_probability = confident = should stop early = LOW penalty
        - Low halt_probability = uncertain = needs more cycles = HIGH penalty
    """
    if halt_probability.ndim == 2:
        halt_probability = halt_probability.squeeze(-1)

    # Encourage high halt probability (confidence)
    penalty = (1.0 - halt_probability).mean()

    return penalty


class TRMLoss(nn.Module):
    def __init__(
        self,
        lambda_halt: float = 0.01,
        lambda_profit: float = 1.0,
        lambda_complexity: float = 0.001,  # NEW - very small
        class_weights: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.7,
        pnl_scale: float = 0.05
    ):
        super().__init__()
        self.lambda_halt = lambda_halt
        self.lambda_profit = lambda_profit
        self.lambda_complexity = lambda_complexity  # NEW
        # ... rest unchanged ...

    def forward(
        self,
        task_logits: torch.Tensor,
        halt_logits: torch.Tensor,
        labels: torch.Tensor,
        pnl: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        # Existing losses
        halt_loss = compute_halt_loss(halt_logits, task_logits, labels, ...)
        profit_weighted_task_loss = compute_profit_weighted_loss(task_logits, labels, pnl, ...)

        # NEW: Complexity penalty
        halt_prob = torch.sigmoid(halt_logits)
        complexity_penalty = compute_complexity_penalty(halt_prob)

        # Combined loss
        total_loss = (
            self.lambda_halt * halt_loss +
            self.lambda_profit * profit_weighted_task_loss +
            self.lambda_complexity * complexity_penalty  # NEW
        )

        if return_components:
            return {
                'total_loss': total_loss,
                'halt_loss': halt_loss,
                'profit_weighted_task_loss': profit_weighted_task_loss,
                'complexity_penalty': complexity_penalty,  # NEW
                'mean_halt_prob': halt_prob.mean().item()  # NEW - for monitoring
            }

        return total_loss
```

### Expected Behavior
- Early training: Low halt_prob (~0.5), complexity penalty ~0.5
- Late training: Higher halt_prob (~0.7-0.9), complexity penalty ~0.1-0.3
- Model learns WHEN to be confident vs uncertain

---

## C4: Flash Crash Data Augmentation

### Problem
- Flash Crash (2010-05-06) has only 2 training samples
- Model can't learn patterns from 2 examples
- May fail catastrophically on similar future events

### Solution: Controlled Synthetic Augmentation

**Approach:**
1. Define "crisis profiles" with characteristic ranges
2. Add controlled noise to original samples
3. Preserve domain constraints (VIX range, return bounds, etc.)
4. Generate 20-50 synthetic samples per underrepresented period

### New File to Create

#### `src/data/augmentation.py`
```python
"""
Black Swan Data Augmentation

Generates synthetic crisis samples for underrepresented periods
while preserving domain-specific constraints.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class CrisisProfile:
    """Characteristics of a crisis type for constrained augmentation"""
    name: str
    vix_range: Tuple[float, float]        # (min, max) VIX during crisis
    spy_return_range: Tuple[float, float]  # Daily return range
    volume_multiplier: Tuple[float, float] # Volume spike range
    correlation_floor: float               # Minimum correlation (crisis = high)

# Crisis profiles based on historical data
CRISIS_PROFILES = {
    'flash_crash': CrisisProfile(
        name='Flash Crash',
        vix_range=(25, 45),
        spy_return_range=(-0.10, 0.05),
        volume_multiplier=(2.0, 5.0),
        correlation_floor=0.85
    ),
    'liquidity_crisis': CrisisProfile(
        name='Liquidity Crisis (2008, 2020)',
        vix_range=(40, 80),
        spy_return_range=(-0.12, -0.02),
        volume_multiplier=(1.5, 3.0),
        correlation_floor=0.80
    ),
    'vol_spike': CrisisProfile(
        name='Volatility Spike (Feb 2018, Aug 2015)',
        vix_range=(20, 50),
        spy_return_range=(-0.05, 0.02),
        volume_multiplier=(1.5, 2.5),
        correlation_floor=0.75
    )
}

class BlackSwanAugmenter:
    """Generate synthetic black swan samples with domain constraints"""

    def __init__(self, noise_scale: float = 0.1, random_seed: int = 42):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(random_seed)

    def augment_sample(
        self,
        original_features: np.ndarray,
        profile: CrisisProfile,
        n_augmented: int = 10
    ) -> List[np.ndarray]:
        """
        Generate augmented samples from single original.

        Feature order (10 features):
        [0] vix_level
        [1] spy_returns_5d
        [2] spy_returns_20d
        [3] volume_ratio
        [4] market_breadth
        [5] correlation
        [6] put_call_ratio
        [7] gini_coefficient
        [8] sector_dispersion
        [9] signal_quality_score
        """
        augmented = []

        for _ in range(n_augmented):
            new_f = original_features.copy()

            # VIX: Stay within crisis range with noise
            new_f[0] = np.clip(
                original_features[0] * self.rng.uniform(0.9, 1.1),
                profile.vix_range[0],
                profile.vix_range[1]
            )

            # SPY returns: Within crisis return range
            for i in [1, 2]:
                noise = self.rng.uniform(-0.02, 0.02)
                new_f[i] = np.clip(
                    original_features[i] + noise,
                    profile.spy_return_range[0],
                    profile.spy_return_range[1]
                )

            # Volume: Elevated during crisis
            new_f[3] = original_features[3] * self.rng.uniform(*profile.volume_multiplier)

            # Correlation: High during crisis
            new_f[5] = max(
                profile.correlation_floor,
                min(0.99, original_features[5] + self.rng.uniform(0, 0.1))
            )

            # Other features: Small noise
            for i in [4, 6, 7, 8, 9]:
                noise = self.rng.normal(0, self.noise_scale * 0.5)
                new_f[i] = original_features[i] * (1 + noise)

            augmented.append(new_f)

        return augmented

    def augment_period(
        self,
        period_samples: List[Dict],
        target_count: int = 30,
        crisis_type: str = 'flash_crash'
    ) -> List[Dict]:
        """Augment underrepresented period to target sample count"""
        if len(period_samples) >= target_count:
            return period_samples

        profile = CRISIS_PROFILES.get(crisis_type, CRISIS_PROFILES['flash_crash'])
        n_needed = target_count - len(period_samples)
        n_per_original = max(1, n_needed // len(period_samples))

        all_samples = list(period_samples)

        for orig in period_samples:
            aug_features = self.augment_sample(
                np.array(orig['features']),
                profile,
                n_per_original
            )

            for features in aug_features:
                all_samples.append({
                    'features': features.tolist(),
                    'strategy_idx': orig['strategy_idx'],
                    'pnl': orig['pnl'] * self.rng.uniform(0.8, 1.2),
                    'period_name': orig['period_name'],
                    'date': orig.get('date'),
                    'is_synthetic': True  # Track for validation
                })

                if len(all_samples) >= target_count:
                    break

            if len(all_samples) >= target_count:
                break

        return all_samples[:target_count]


def detect_crisis_type(period_name: str) -> str:
    """Map period name to crisis type for profile selection"""
    name_lower = period_name.lower()

    if 'flash' in name_lower:
        return 'flash_crash'
    elif any(x in name_lower for x in ['2008', '2020', 'covid', 'lehman', 'gfc']):
        return 'liquidity_crisis'
    elif any(x in name_lower for x in ['vix', 'vol', 'spike']):
        return 'vol_spike'
    else:
        return 'liquidity_crisis'  # Default
```

### Integration with Data Pipeline

#### Modify `src/data/strategy_labeler.py`
```python
from src.data.augmentation import BlackSwanAugmenter, detect_crisis_type

def generate_labels_with_augmentation(
    self,
    min_samples_per_period: int = 20,
    augment: bool = True
) -> pd.DataFrame:
    """Generate labels with optional augmentation for rare periods"""
    # Generate original labels
    df = self.generate_all_labels()

    if not augment:
        return df

    # Group by period
    augmenter = BlackSwanAugmenter()
    augmented_records = []

    for period_name, group in df.groupby('period_name'):
        samples = group.to_dict('records')

        if len(samples) < min_samples_per_period:
            crisis_type = detect_crisis_type(period_name)
            augmented = augmenter.augment_period(
                samples, min_samples_per_period, crisis_type
            )
            augmented_records.extend(augmented)
            print(f"Augmented {period_name}: {len(samples)} -> {len(augmented)}")
        else:
            augmented_records.extend(samples)

    return pd.DataFrame(augmented_records)
```

---

## Implementation Timeline

| Phase | Task | Files | Est. Time |
|-------|------|-------|-----------|
| 1 | B6: Complexity penalty | trm_loss_functions.py | 1 hour |
| 2 | C4: Augmentation | NEW augmentation.py, strategy_labeler.py | 2 hours |
| 3 | Regenerate training data | Run labeler with augmentation | 30 min |
| 4 | A3: TemporalEncoder | trm_model.py, trm_data_loader.py | 3 hours |
| 5 | Testing & validation | All test files | 2 hours |

**Total estimated time: 8-9 hours**

---

## Success Metrics

| Issue | Metric | Target |
|-------|--------|--------|
| A3 | Model detects 3-day VIX trends | >70% accuracy on trend direction |
| B6 | Average halt_probability | Increases from ~0.5 to ~0.7 over training |
| C4 | Flash Crash test accuracy | >60% (vs current ~50% random) |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Temporal model too slow | Use small LSTM (2 layers), benchmark inference time |
| Complexity penalty hurts accuracy | Start with lambda_complexity=0.0001, tune on val set |
| Synthetic samples hurt generalization | Track is_synthetic flag, compare real vs synthetic performance |
| Breaking existing checkpoints | Use feature flags, default to backward-compatible mode |

---

## Feature Flags for Gradual Rollout

```json
{
    "use_temporal_sequences": false,
    "temporal_seq_len": 5,
    "use_complexity_penalty": true,
    "lambda_complexity": 0.001,
    "use_data_augmentation": true,
    "min_samples_per_period": 20
}
```
