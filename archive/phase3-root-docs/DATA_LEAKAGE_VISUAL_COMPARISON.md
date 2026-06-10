# Visual Comparison: Data Leakage Fix

## BEFORE (Data Leakage)

```
+--------------------------------------------------+
|  Load Data (10,000 samples)                     |
+--------------------------------------------------+
                    |
                    v
+--------------------------------------------------+
|  Feature Engineering                             |
+--------------------------------------------------+
                    |
                    v
+--------------------------------------------------+
|  FIT SCALER ON ALL DATA  <-- LEAKAGE!           |
|  scaler.fit_transform(X_all)                     |
|  - Learns mean, quartiles from test set         |
|  - Test statistics influence training           |
+--------------------------------------------------+
                    |
                    v
+--------------------------------------------------+
|  Split into Train (80%) and Test (20%)          |
|  - Already contaminated by test statistics      |
+--------------------------------------------------+
                    |
          +---------+---------+
          |                   |
          v                   v
   +-----------+       +-----------+
   | Train     |       | Test      |
   | (scaled)  |       | (scaled)  |
   +-----------+       +-----------+
```

**Problem**: Scaler was fit on ALL data (train + test) BEFORE splitting
**Result**: Test set statistics leaked into the scaler, affecting training

---

## AFTER (No Leakage)

```
+--------------------------------------------------+
|  Load Data (10,000 samples)                     |
+--------------------------------------------------+
                    |
                    v
+--------------------------------------------------+
|  Feature Engineering                             |
+--------------------------------------------------+
                    |
                    v
+--------------------------------------------------+
|  Split into Train (80%) and Test (20%)          |
|  - Raw features, NO scaling yet                 |
+--------------------------------------------------+
                    |
          +---------+---------+
          |                   |
          v                   v
   +-----------+       +-----------+
   | X_train   |       | X_test    |
   | y_train   |       | y_test    |
   | (raw)     |       | (raw)     |
   +-----------+       +-----------+
          |                   |
          |                   |
          v                   |
   +-----------+              |
   | FIT       |              |
   | scaler on |              |
   | TRAIN     |              |
   | ONLY      |              |
   +-----------+              |
          |                   |
          v                   v
   +-----------+       +-----------+
   | TRANSFORM |       | TRANSFORM |
   | train     |       | test      |
   | (fit_     |       | (transform|
   | transform)|       | ONLY)     |
   +-----------+       +-----------+
          |                   |
          v                   v
   +-----------+       +-----------+
   | X_train   |       | X_test    |
   | _scaled   |       | _scaled   |
   +-----------+       +-----------+
```

**Solution**: Split FIRST, then fit scaler ONLY on training data
**Result**: Test set has ZERO influence on scaler parameters

---

## Key Differences

| Aspect | BEFORE (Leakage) | AFTER (Fixed) |
|--------|------------------|---------------|
| **Order** | Scale -> Split | Split -> Scale |
| **Scaler Fit** | All data (10,000 samples) | Training only (8,000 samples) |
| **Test Data** | Used in scaler.fit() | Never seen by scaler.fit() |
| **Test Transform** | fit_transform(all) | transform(test_only) |
| **Data Leakage** | YES | NO |
| **Production Ready** | NO | YES |

---

## Code Comparison

### BEFORE
```python
# Load data
X = processed_df[feature_cols].values
y = processed_df.get('target', ...).values

# LEAKAGE: Scale ALL data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)  # Fits on ALL 10,000 samples

# Then split (too late!)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, ...)
```

### AFTER
```python
# Load data
X = processed_df[feature_cols].values
y = processed_df.get('target', ...).values

# Split FIRST (8,000 train, 2,000 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # Time series
)

# FIX: Fit scaler ONLY on training data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fits on 8,000 samples
X_test_scaled = scaler.transform(X_test)  # Transform only, NO FIT
```

---

## Impact on Model Training

### Random Forest & Gradient Boosting
```python
# BEFORE: Received already-scaled data, split internally
def train_random_forest(self, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)  # Duplicate split!
    # Train model...

# AFTER: Receives pre-split data
def train_random_forest(self, X_train, X_test, y_train, y_test):
    # No split needed - data already properly prepared
    # Train model...
```

### LSTM
```python
# BEFORE: Split after sequence preparation
X_seq, y_seq = self._prepare_sequences(X, y, sequence_length=60)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, ...)

# AFTER: Split before, then prepare sequences maintaining split
X_combined = np.vstack([X_train, X_test])  # Temporarily combine
X_seq, y_seq = self._prepare_sequences(X_combined, y_combined, ...)
split_idx = int(len(X_seq) * 0.8)  # Re-split at same ratio
X_train_seq = X_seq[:split_idx]
X_test_seq = X_seq[split_idx:]
```

---

## Expected Metric Changes

| Metric | Before (Leakage) | After (Fixed) | Change |
|--------|------------------|---------------|--------|
| **MSE** | Lower (optimistic) | Slightly higher (realistic) | +5-10% |
| **MAE** | Lower (optimistic) | Slightly higher (realistic) | +5-10% |
| **R2** | Higher (optimistic) | Slightly lower (realistic) | -0.02 to -0.05 |

**Note**: The "worse" metrics are actually CORRECT. Previous metrics were artificially inflated due to test data leakage.

---

## Verification Checklist

- [x] Scaler.fit() called ONLY on training data
- [x] Scaler.transform() (no fit) called on test data
- [x] prepare_data() returns 4 values (train/test splits)
- [x] Training methods receive pre-split data
- [x] No duplicate splits in training methods
- [x] Temporal ordering preserved (shuffle=False)
- [x] All models use consistent data preparation
- [x] Saved scaler is production-ready (never saw test data)

---

## References

- **File Modified**: `src/intelligence/training/trainer.py`
- **Lines Changed**: 90-141 (prepare_data), 198-437 (training methods)
- **Commits**: Data leakage fix applied
