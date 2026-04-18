"""
Telemetry Bridge - Bridges loop telemetry with Memory MCP

Converts between loop formats and Memory MCP storage format,
maintaining v3.0 schema compliance with WHO/WHEN/PROJECT/WHY tagging.

VERIX Example:
    [assert|neutral] TelemetryBridge syncs loop state to Memory MCP
    [ground:architecture-spec] [conf:0.90] [state:confirmed]

Usage:
    from cognitive_architecture.integration import TelemetryBridge

    bridge = TelemetryBridge(loop_dir=Path(".loop"))
    result = bridge.store_to_memory_mcp(iteration=5)
    print(f"Stored {result['stored_count']} records")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopTelemetryRecord:
    """
    A telemetry record combining loop state with cognitive architecture metrics.

    This is the unified format that flows to Memory-MCP.
    Uses WHO/WHEN/PROJECT/WHY tagging protocol.
    """

    # Identity
    task_id: str
    timestamp: str
    session_id: Optional[str] = None

    # Loop state
    iteration: int = 0
    plane: str = "execution"
    timescale: str = "micro"

    # Configuration (from runtime_config.json)
    mode: str = "balanced"
    vector14: List[float] = field(default_factory=list)
    active_frames: List[str] = field(default_factory=list)
    verix_strictness: str = "MODERATE"
    compression_level: str = "L1"

    # Metrics (from eval_report.json - harness only)
    task_accuracy: float = 0.0
    token_efficiency: float = 0.0
    edge_robustness: float = 0.0
    epistemic_consistency: float = 0.0
    overall_score: float = 0.0

    # Decision
    decision: str = "continue"
    reason: str = ""

    # Git state
    git_head: Optional[str] = None

    def to_memory_mcp_format(self) -> Dict[str, Any]:
        """
        Convert to Memory-MCP v3.0 format with x- prefixes.

        Uses WHO/WHEN/PROJECT/WHY tagging protocol.
        """
        return {
            # Identity
            "task_id": self.task_id,
            "timestamp": self.timestamp,

            # WHO/WHEN/PROJECT/WHY tags
            "x-who": f"ralph_loop_iteration_{self.iteration}",
            "x-when": self.timestamp,
            "x-project": "cognitive-architecture-integration",
            "x-why": "loop-telemetry",

            # Loop state
            "x-iteration": self.iteration,
            "x-plane": self.plane,
            "x-timescale": self.timescale,

            # Configuration
            "x-mode": self.mode,
            "x-vector14": self.vector14,
            "x-active-frames": self.active_frames,
            "x-verix-strictness": self.verix_strictness,
            "x-compression-level": self.compression_level,

            # Metrics (harness-graded only)
            "task_accuracy": self.task_accuracy,
            "token_efficiency": self.token_efficiency,
            "edge_robustness": self.edge_robustness,
            "epistemic_consistency": self.epistemic_consistency,
            "overall_score": self.overall_score,

            # Decision
            "decision": self.decision,
            "reason": self.reason,

            # Git
            "git_head": self.git_head,

            # Schema version
            "_schema_version": "3.0",
        }

    @classmethod
    def from_loop_state(
        cls,
        eval_report: Dict[str, Any],
        runtime_config: Dict[str, Any],
        event: Dict[str, Any],
    ) -> "LoopTelemetryRecord":
        """
        Create record from loop state files.

        Args:
            eval_report: Contents of eval_report.json
            runtime_config: Contents of runtime_config.json
            event: A UnifiedEvent dict

        Returns:
            LoopTelemetryRecord
        """
        metrics = eval_report.get("metrics", {})
        frames = runtime_config.get("frames", {})

        return cls(
            task_id=event.get("task_id", f"ralph_{event.get('iteration', 0)}"),
            timestamp=event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            iteration=event.get("iteration", 0),
            plane=event.get("plane", "execution"),
            timescale=event.get("timescale", "micro"),
            mode=runtime_config.get("mode", "balanced"),
            vector14=runtime_config.get("vector14", []),
            active_frames=[k for k, v in frames.items() if v],
            verix_strictness=runtime_config.get("verix", {}).get("strictness", "MODERATE"),
            compression_level=runtime_config.get("verix", {}).get("compression", "L1"),
            task_accuracy=metrics.get("task_accuracy", 0.0),
            token_efficiency=metrics.get("token_efficiency", 0.0),
            edge_robustness=metrics.get("edge_robustness", 0.0),
            epistemic_consistency=metrics.get("epistemic_consistency", 0.0),
            overall_score=metrics.get("overall", 0.0),
            decision=event.get("decision", "continue"),
            reason=event.get("reason", ""),
            git_head=event.get("git_head"),
        )


class TelemetryBridge:
    """
    Bridge that syncs loop telemetry with Memory MCP storage.

    Maintains consistency between:
    - .loop/ contract files
    - Cognitive architecture telemetry
    - Memory-MCP storage
    """

    def __init__(self, loop_dir: Path):
        """
        Initialize TelemetryBridge.

        Args:
            loop_dir: Path to .loop/ directory
        """
        self.loop_dir = Path(loop_dir)

    def sync_iteration(self, iteration: int) -> Optional[LoopTelemetryRecord]:
        """
        Sync a specific iteration's telemetry.

        Reads from .loop/ files and creates a record.
        """
        eval_report = self._load_json(self.loop_dir / "eval_report.json")
        runtime_config = self._load_json(self.loop_dir / "runtime_config.json")
        events = self._load_events()

        # Find event for this iteration
        event = None
        for e in events:
            if e.get("iteration") == iteration:
                event = e
                break

        if event is None:
            return None

        return LoopTelemetryRecord.from_loop_state(eval_report, runtime_config, event)

    def sync_all(self) -> List[LoopTelemetryRecord]:
        """Sync all iterations from events.jsonl."""
        events = self._load_events()
        records = []

        eval_report = self._load_json(self.loop_dir / "eval_report.json")
        runtime_config = self._load_json(self.loop_dir / "runtime_config.json")

        for event in events:
            record = LoopTelemetryRecord.from_loop_state(eval_report, runtime_config, event)
            records.append(record)

        return records

    def export_to_memory_mcp(self) -> List[Dict[str, Any]]:
        """Export all records in Memory-MCP v3.0 format."""
        records = self.sync_all()
        return [r.to_memory_mcp_format() for r in records]

    def store_to_memory_mcp(
        self,
        iteration: Optional[int] = None,
        mcp_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Store telemetry record(s) to Memory MCP.

        Args:
            iteration: Specific iteration to store (None = all)
            mcp_client: Optional MCP client (placeholder for integration)

        Returns:
            Summary of storage operations
        """
        events = self._load_events()
        eval_report = self._load_json(self.loop_dir / "eval_report.json")
        runtime_config = self._load_json(self.loop_dir / "runtime_config.json")

        stored = []
        errors = []

        for event in events:
            event_iteration = event.get("iteration", 0)
            if iteration is not None and event_iteration != iteration:
                continue

            record = LoopTelemetryRecord.from_loop_state(eval_report, runtime_config, event)
            mcp_format = record.to_memory_mcp_format()

            key = f"iteration_{event_iteration}_{record.timestamp}"

            # Build WHO/WHEN/PROJECT/WHY metadata
            metadata = {
                "WHO": f"telemetry-bridge:ralph_iteration_{event_iteration}",
                "WHEN": record.timestamp,
                "PROJECT": "cognitive-architecture",
                "WHY": "loop-telemetry",
                "x-iteration": str(event_iteration),
                "x-overall-score": str(record.overall_score),
                "x-decision": record.decision,
            }

            # Store to Memory MCP (if client provided)
            if mcp_client:
                try:
                    result = mcp_client.memory_store(
                        key=key,
                        value=mcp_format,
                        metadata=metadata,
                    )
                    if result.success:
                        stored.append({
                            "iteration": event_iteration,
                            "key": key,
                            "location": result.data.get("location", "unknown"),
                        })
                    else:
                        errors.append({
                            "iteration": event_iteration,
                            "error": result.error,
                        })
                except Exception as e:
                    errors.append({
                        "iteration": event_iteration,
                        "error": str(e),
                    })
            else:
                # No client - just prepare the data
                stored.append({
                    "iteration": event_iteration,
                    "key": key,
                    "data": mcp_format,
                })

        return {
            "stored_count": len(stored),
            "error_count": len(errors),
            "stored": stored,
            "errors": errors,
            "mcp_available": mcp_client is not None,
        }

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file or return empty dict."""
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _load_events(self) -> List[Dict[str, Any]]:
        """Load events from events.jsonl."""
        events_path = self.loop_dir / "events.jsonl"
        events = []

        if events_path.exists():
            for line in events_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return events


def bridge_loop_to_telemetry(loop_dir: str) -> Dict[str, Any]:
    """
    Convenience function to bridge loop state to telemetry.

    Args:
        loop_dir: Path to .loop/ directory

    Returns:
        Summary of bridged records
    """
    bridge = TelemetryBridge(Path(loop_dir))
    records = bridge.sync_all()

    return {
        "records_synced": len(records),
        "iterations": [r.iteration for r in records],
        "latest_score": records[-1].overall_score if records else 0.0,
    }
