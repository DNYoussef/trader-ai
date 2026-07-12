"""Universal component wiring for trader-ai.

The optional `library.components` package provides tagging/telemetry/memory
bridges. It is not required to run trader-ai; when it is absent every init_*
function returns None and callers must tolerate that (main.py only holds the
return values). This keeps the entry point booting without the phantom package.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from library.components.cognitive_architecture.integration.connascence_bridge import ConnascenceBridge
    from library.components.cognitive_architecture.integration.telemetry_bridge import TelemetryBridge
    from library.components.memory.memory_mcp_client import create_memory_mcp_client
    from library.components.observability.tagging_protocol import TaggingProtocol, create_simple_tagger
    _LIBRARY_AVAILABLE = True
except ImportError:
    _LIBRARY_AVAILABLE = False


def init_tagger():
    if _LIBRARY_AVAILABLE:
        return create_simple_tagger(agent_id="trader-ai", project_id="trader-ai")
    return None   # no-op: callers must tolerate None


def init_memory_client():
    if _LIBRARY_AVAILABLE:
        endpoint = os.getenv("MEMORY_MCP_URL", "http://localhost:3000")
        return create_memory_mcp_client(
            project_id="trader-ai",
            project_name="trader-ai",
            agent_id="trader-ai",
            agent_category="backend",
            capabilities=["trading", "risk", "monitoring"],
            mcp_endpoint=endpoint,
        )
    return None


def init_telemetry_bridge(loop_dir: Optional[str] = None):
    if _LIBRARY_AVAILABLE:
        resolved = Path(loop_dir) if loop_dir else Path(".loop")
        return TelemetryBridge(loop_dir=resolved)
    return None


def init_connascence_bridge():
    if _LIBRARY_AVAILABLE:
        return ConnascenceBridge()
    return None
