# MCPT Parallelization Architecture

## Before: Sequential Execution

```
Main Thread
    |
    +-- Get real score from optimizer_fn(data)
    |
    +-- For i in range(n_permutations):  [SEQUENTIAL BOTTLENECK]
    |       |
    |       +-- Generate permuted data
    |       +-- Score = optimizer_fn(perm_data)
    |       +-- Compare to real score
    |       +-- (Wait for completion before next iteration)
    |
    +-- Calculate p-value
    +-- Return result

Execution Time: N * T (where T = time per permutation)
```

## After: Parallel Execution

```
Main Thread
    |
    +-- Get real score from optimizer_fn(data)
    |
    +-- Generate deterministic seeds: [seed+0, seed+1, ..., seed+N]
    |
    +-- Create ProcessPoolExecutor (n_workers = CPU cores)
    |       |
    |       +-- Worker 1: _run_single_permutation(seed+0, data, fn, 0)
    |       |       |
    |       |       +-- np.random.seed(seed+0)
    |       |       +-- perm_data = get_permutation(data)
    |       |       +-- return optimizer_fn(perm_data)
    |       |
    |       +-- Worker 2: _run_single_permutation(seed+1, data, fn, 1)
    |       |       |
    |       |       +-- np.random.seed(seed+1)
    |       |       +-- perm_data = get_permutation(data)
    |       |       +-- return optimizer_fn(perm_data)
    |       |
    |       +-- Worker 3: _run_single_permutation(seed+2, data, fn, 2)
    |       |       [... parallel execution ...]
    |       |
    |       +-- Worker N: _run_single_permutation(seed+N-1, data, fn, N-1)
    |       |
    |       +-- [All workers execute concurrently]
    |       +-- [Main thread collects results via executor.map()]
    |
    +-- Calculate p-value from collected results
    +-- Return result

Execution Time: (N * T) / n_workers + overhead
Expected Speedup: 4-8x on typical systems
```

## Data Flow Diagram

```
                                [Main Process]
                                      |
                    +-----------------+------------------+
                    |                                    |
              Get Real Score                    Generate Seeds
            optimizer_fn(data)          [seed+0, seed+1, ..., seed+N]
                    |                                    |
                    |                                    |
                    v                                    v
              real_score = X                    ProcessPoolExecutor
                                                         |
                    +------------------------------------+
                    |
        +-----------+-----------+-----------+------------+
        |           |           |           |            |
    Worker 1    Worker 2    Worker 3    Worker 4    Worker N
        |           |           |           |            |
   Perm 0,N,2N  Perm 1,N+1  Perm 2,N+2  Perm 3,N+3   ...
        |           |           |           |            |
    score_0     score_1     score_2     score_3      score_N
        |           |           |           |            |
        +-----------+-----------+-----------+------------+
                                |
                        [Collect Results]
                                |
                    +-----------+------------+
                    |                        |
              Count where            Track all scores
           score >= real_score       [score_0, score_1, ...]
                    |                        |
                    +------------------------+
                                |
                        p_value = better_count / N
                                |
                                v
                          [Return Result]
```

## Seed Management Strategy

```
DETERMINISTIC SEED GENERATION:

Input: seed=42, n_permutations=1000

base_seed = 42
seeds = [42+0, 42+1, 42+2, ..., 42+999]
      = [42, 43, 44, ..., 1041]

Worker 1: receives seed=42  -> Produces deterministic perm_0
Worker 2: receives seed=43  -> Produces deterministic perm_1
Worker 3: receives seed=44  -> Produces deterministic perm_2
...
Worker N: receives seed=1041 -> Produces deterministic perm_999

REPRODUCIBILITY GUARANTEE:
- Same base_seed always produces same seeds array
- Same seed always produces same permutation
- Same permutation always produces same score
- Therefore: Same input seed always produces same p-value
```

## Worker Function Architecture

```
Module Level (Picklable):

def _run_single_permutation(args):
    """
    MUST be module-level for pickling.
    Cannot be nested function or lambda.
    """
    seed, data, optimizer_fn, perm_idx = args

    # Step 1: Set seed for reproducibility
    np.random.seed(seed)

    try:
        # Step 2: Generate permuted data
        perm_data = get_permutation(data, start_index=0)

        # Step 3: Score permuted data
        perm_score = optimizer_fn(perm_data)

        # Step 4: Return score
        return perm_score

    except Exception as e:
        # Step 5: Handle errors gracefully
        logger.debug(f"Permutation {perm_idx} failed: {e}")
        return None  # Exclude from analysis


Class Level (Not Picklable):
X def _run_perm(self, args):  # FAILS: Cannot pickle 'self'
X lambda args: fn(*args)       # FAILS: Cannot pickle lambda
```

## Performance Comparison

```
SYSTEM: 8-core CPU, 1000 permutations

Sequential Mode (parallel=False):
------------------------------------------------------------
Permutation 0    [====] 0.2s
Permutation 1    [====] 0.2s
Permutation 2    [====] 0.2s
...
Permutation 999  [====] 0.2s
------------------------------------------------------------
Total Time: 1000 * 0.2s = 200 seconds


Parallel Mode (parallel=True, n_workers=8):
------------------------------------------------------------
Core 1: Perm 0,8,16,24...992   [================================] 25s
Core 2: Perm 1,9,17,25...993   [================================] 25s
Core 3: Perm 2,10,18,26...994  [================================] 25s
Core 4: Perm 3,11,19,27...995  [================================] 25s
Core 5: Perm 4,12,20,28...996  [================================] 25s
Core 6: Perm 5,13,21,29...997  [================================] 25s
Core 7: Perm 6,14,22,30...998  [================================] 25s
Core 8: Perm 7,15,23,31...999  [================================] 25s
------------------------------------------------------------
Total Time: (1000 * 0.2s) / 8 cores + 5s overhead = 30 seconds

Speedup: 200s / 30s = 6.7x
```

## Error Handling Flow

```
Worker Process:
    |
    +-- Try:
    |     |
    |     +-- Set seed
    |     +-- Generate permutation
    |     +-- Score permutation
    |     +-- Return score
    |
    +-- Except Exception:
    |     |
    |     +-- Log error (debug level)
    |     +-- Return None
    |
    v

Main Process:
    |
    +-- For each result in executor.map():
    |     |
    |     +-- If result is not None:
    |     |     |
    |     |     +-- Add to perm_scores
    |     |     +-- Compare to real_score
    |     |
    |     +-- If result is None:
    |           |
    |           +-- Skip (don't count in p-value)
    |
    +-- p_value = better_count / len(perm_scores)
    |              (only valid permutations)
    |
    v
```

## Memory Usage Comparison

```
Sequential Mode:
- Main Process: 1x data size
- Total Memory: ~1x data size

Parallel Mode (8 workers):
- Main Process: 1x data size
- Worker 1: 1x data size
- Worker 2: 1x data size
- ...
- Worker 8: 1x data size
- Total Memory: ~9x data size

Trade-off:
- Memory: 9x higher
- Time: 6.7x faster
- Net Benefit: Positive (memory is cheap, time is valuable)
```

## Configuration Matrix

```
| Scenario           | parallel | n_workers | Use Case                    |
|--------------------|----------|-----------|----------------------------|
| Default            | True     | None      | Best performance           |
| Debugging          | False    | N/A       | Step-through debugging     |
| Shared System      | True     | 4         | Don't monopolize resources |
| Memory Constrained | True     | 2         | Reduce memory footprint    |
| Maximum Speed      | True     | 16        | High-end workstation       |
| CI/CD              | False    | N/A       | Deterministic timing       |
```

## Walk-Forward Specific Architecture

```
Walk-Forward MCPT:
    |
    +-- Get real score from walkforward_fn(data)
    |
    +-- CRITICAL: Preserve training window
    |     |
    |     +-- train_window = 252 bars
    |     +-- Only permute bars AFTER index 252
    |     +-- Bars 0-251 remain unchanged
    |
    +-- Parallel Execution:
    |     |
    |     +-- Each worker receives:
    |           - seed (for reproducibility)
    |           - data (full DataFrame)
    |           - walkforward_fn
    |           - train_window (252)
    |           - perm_idx (for logging)
    |     |
    |     +-- Worker logic:
    |           perm_data = get_permutation(data, start_index=train_window)
    |           # Bars 0-251: unchanged
    |           # Bars 252+: permuted
    |
    +-- Calculate p-value
    +-- Return result
```

## Summary

The parallelization architecture:

1. **Preserves Semantics**: Same logic, just distributed
2. **Maintains Reproducibility**: Deterministic seeds guarantee identical results
3. **Scales Efficiently**: Near-linear speedup with core count
4. **Handles Errors Gracefully**: Failed permutations don't crash entire run
5. **Backward Compatible**: Sequential mode still available
6. **Memory Conscious**: Users can limit workers if needed
7. **Production Ready**: Context manager ensures proper cleanup

The design follows Python multiprocessing best practices and achieves the target 4-8x performance improvement on multi-core systems.
