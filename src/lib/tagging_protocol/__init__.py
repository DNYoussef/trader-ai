"""
WHO/WHEN/PROJECT/WHY Tagging Protocol

A reusable observability component for generating structured metadata tags
for memory operations, logs, events, and any system requiring attribution
and provenance tracking.

Quick Start:
    from tagging_protocol import TaggingProtocol, Intent, AgentCategory

    # Create a tagger
    tagger = TaggingProtocol(
        agent_id="my-agent",
        agent_category=AgentCategory.BACKEND,
        capabilities=["REST API", "PostgreSQL"],
        project_id="my-project",
        project_name="My Project"
    )

    # Generate tags
    tags = tagger.generate_tags(intent=Intent.IMPLEMENTATION)

    # Or create a complete payload
    payload = tagger.create_payload(
        content="Implemented feature X",
        intent=Intent.IMPLEMENTATION
    )

Factory Functions:
    - create_tagger(): Full configuration
    - create_simple_tagger(): Minimal configuration with defaults

For full documentation, see tagging_protocol.py
"""

from .tagging_protocol import (
    # Main class
    TaggingProtocol,

    # Enums
    Intent,
    AgentCategory,

    # Factory functions
    create_tagger,
    create_simple_tagger,

    # Constants
    INTENT_DESCRIPTIONS,
)

__all__ = [
    # Main class
    "TaggingProtocol",

    # Enums
    "Intent",
    "AgentCategory",

    # Factory functions
    "create_tagger",
    "create_simple_tagger",

    # Constants
    "INTENT_DESCRIPTIONS",
]

__version__ = "1.0.0"
__author__ = "David Youssef"
