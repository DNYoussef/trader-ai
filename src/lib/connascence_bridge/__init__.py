"""
Integration Bridges for Cognitive Architecture

Provides bridges between loop telemetry and external systems:
- TelemetryBridge: Syncs with Memory MCP
- ConnascenceBridge: Integrates 7-Analyzer Quality Suite

Example:
    from cognitive_architecture.integration import (
        TelemetryBridge,
        ConnascenceBridge,
        ConnascenceResult,
    )

    # Sync telemetry to Memory MCP
    telemetry = TelemetryBridge(Path(".loop"))
    result = telemetry.store_to_memory_mcp(iteration=5)

    # Analyze code quality
    quality = ConnascenceBridge()
    analysis = quality.analyze_file(Path("src/module.py"))
"""

from .telemetry_bridge import (
    TelemetryBridge,
    LoopTelemetryRecord,
    bridge_loop_to_telemetry,
)

from .connascence_bridge import (
    ConnascenceBridge,
    ConnascenceResult,
    analyze_artifact,
    quality_gate,
)

__all__ = [
    # Telemetry
    "TelemetryBridge",
    "LoopTelemetryRecord",
    "bridge_loop_to_telemetry",

    # Connascence
    "ConnascenceBridge",
    "ConnascenceResult",
    "analyze_artifact",
    "quality_gate",
]

__version__ = "1.0.0"
