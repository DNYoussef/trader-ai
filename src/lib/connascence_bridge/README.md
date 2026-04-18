# Integration Bridges for Cognitive Architecture

Provides bridges between loop telemetry and external systems.

## Components

### TelemetryBridge

Syncs loop telemetry with Memory MCP storage using WHO/WHEN/PROJECT/WHY tagging protocol.

```python
from cognitive_architecture.integration import TelemetryBridge

bridge = TelemetryBridge(loop_dir=Path(".loop"))

# Sync specific iteration
record = bridge.sync_iteration(iteration=5)

# Sync all iterations
records = bridge.sync_all()

# Export to Memory MCP format
mcp_data = bridge.export_to_memory_mcp()

# Store to Memory MCP
result = bridge.store_to_memory_mcp(iteration=5)
print(f"Stored: {result['stored_count']}")
```

### ConnascenceBridge

Integrates with the 7-Analyzer Quality Suite:

1. **Connascence** - 9 coupling types (CoN, CoT, CoM, CoP, CoA, CoE, CoT2, CoV, CoI)
2. **NASA Safety** - Power of 10 rules compliance
3. **MECE** - Duplication detection
4. **Clarity Linter** - Cognitive load analysis
5. **Six Sigma** - Quality metrics (DPMO, sigma level)
6. **Theater Detection** - Fake quality indicators
7. **Safety Violations** - God objects, parameter bombs

```python
from cognitive_architecture.integration import ConnascenceBridge

bridge = ConnascenceBridge()

# Analyze a file
result = bridge.analyze_file(Path("src/module.py"))
print(f"Sigma Level: {result.sigma_level}")
print(f"DPMO: {result.dpmo}")
print(f"NASA Compliance: {result.nasa_compliance:.0%}")

# Check quality gate
if result.passes_gate(strict=True):
    print("PASSED Six Sigma quality gate")
else:
    print("FAILED quality gate")
```

## Quality Gate Thresholds

### Strict Mode (Six Sigma)
| Metric | Threshold |
|--------|-----------|
| Sigma Level | >= 4.0 |
| DPMO | <= 6,210 |
| NASA Compliance | >= 95% |
| MECE Score | >= 80% |
| Theater Risk | < 20% |
| Critical Violations | 0 |

### Lenient Mode
- Critical Violations: 0

## Memory MCP Format

Records use WHO/WHEN/PROJECT/WHY tagging protocol:

```json
{
  "x-who": "ralph_loop_iteration_5",
  "x-when": "2026-01-10T12:00:00Z",
  "x-project": "cognitive-architecture",
  "x-why": "loop-telemetry",
  "task_accuracy": 0.85,
  "token_efficiency": 0.78,
  "_schema_version": "3.0"
}
```

## Invocation Modes

ConnascenceBridge supports multiple modes:
1. **direct** - Direct Python import (preferred)
2. **cli** - CLI subprocess
3. **mock** - Heuristic fallback (when analyzer unavailable)

```python
bridge = ConnascenceBridge()
print(f"Mode: {bridge.mode}")
print(f"Available: {bridge.is_available()}")
```

## VERIX Notation

```
[assert|neutral] TelemetryBridge syncs loop state to Memory MCP
[ground:architecture-spec] [conf:0.90] [state:confirmed]
```
