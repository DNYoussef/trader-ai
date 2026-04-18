"""
WHO/WHEN/PROJECT/WHY Tagging Protocol

A reusable observability component for generating structured metadata tags
on memory operations, logs, events, and any system that requires attribution
and provenance tracking.

This protocol ensures all operations are tagged with:
- WHO: Agent/user identification and capabilities
- WHEN: Temporal information in multiple formats
- PROJECT: Project and task context
- WHY: Intent and purpose classification

Usage:
    from tagging_protocol import TaggingProtocol, Intent, AgentCategory

    # Initialize for your agent/service
    tagger = TaggingProtocol(
        agent_id="my-agent",
        agent_category=AgentCategory.BACKEND,
        capabilities=["REST API", "PostgreSQL"],
        project_id="my-project",
        project_name="My Project"
    )

    # Generate tags for an operation
    tags = tagger.generate_tags(
        intent=Intent.IMPLEMENTATION,
        user_id="user-123",
        task_id="TASK-001"
    )

    # Or create a complete payload
    payload = tagger.create_payload(
        content="Implemented feature X",
        intent=Intent.IMPLEMENTATION
    )

Author: David Youssef
License: MIT
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import uuid


class Intent(str, Enum):
    """
    Intent categories for classifying the purpose of operations.

    These categories answer the "WHY" question in the tagging protocol,
    providing semantic meaning to what type of work is being performed.

    Attributes:
        IMPLEMENTATION: Creating new features or functionality
        BUGFIX: Fixing bugs or errors in existing code
        REFACTOR: Improving code quality without changing behavior
        TESTING: Writing or executing tests
        DOCUMENTATION: Creating or updating documentation
        ANALYSIS: Analyzing code, data, or system behavior
        PLANNING: Planning architecture, approach, or strategy
        RESEARCH: Researching solutions, libraries, or best practices
        DEPLOYMENT: Deploying or releasing to environments
        MONITORING: Observing system health or performance
        MAINTENANCE: Routine maintenance and updates
        SECURITY: Security-related operations and audits
    """
    IMPLEMENTATION = "implementation"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    RESEARCH = "research"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    SECURITY = "security"


class AgentCategory(str, Enum):
    """
    Agent categories for classifying the type of agent or service.

    These categories answer the "WHO" question by providing context
    about what type of component is performing the operation.

    Attributes:
        CORE_DEVELOPMENT: Core development and coding agents
        TESTING_VALIDATION: Testing and validation agents
        FRONTEND: Frontend/UI development agents
        BACKEND: Backend/API development agents
        DATABASE: Database and data management agents
        DOCUMENTATION: Documentation generation agents
        SWARM_COORDINATION: Multi-agent coordination agents
        PERFORMANCE: Performance optimization agents
        SECURITY: Security audit and hardening agents
        RESEARCH: Research and discovery agents
        DEVOPS: DevOps and infrastructure agents
        MONITORING: Monitoring and observability agents
        ORCHESTRATION: Workflow orchestration agents
        QUALITY: Quality assurance and code review agents
    """
    CORE_DEVELOPMENT = "core-development"
    TESTING_VALIDATION = "testing-validation"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DOCUMENTATION = "documentation"
    SWARM_COORDINATION = "swarm-coordination"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESEARCH = "research"
    DEVOPS = "devops"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"
    QUALITY = "quality"


# Intent descriptions for human-readable output
INTENT_DESCRIPTIONS: Dict[Intent, str] = {
    Intent.IMPLEMENTATION: "Implementing new feature or functionality",
    Intent.BUGFIX: "Fixing bugs or errors in existing code",
    Intent.REFACTOR: "Refactoring code for better quality or performance",
    Intent.TESTING: "Writing or executing tests",
    Intent.DOCUMENTATION: "Creating or updating documentation",
    Intent.ANALYSIS: "Analyzing code, performance, or system behavior",
    Intent.PLANNING: "Planning architecture or approach",
    Intent.RESEARCH: "Researching solutions or best practices",
    Intent.DEPLOYMENT: "Deploying or releasing to environments",
    Intent.MONITORING: "Monitoring system health or performance",
    Intent.MAINTENANCE: "Performing routine maintenance and updates",
    Intent.SECURITY: "Performing security operations or audits",
}


class TaggingProtocol:
    """
    WHO/WHEN/PROJECT/WHY Tagging Protocol Implementation.

    This class provides a standardized way to generate metadata tags for
    any operation that requires attribution, provenance, and context tracking.

    The protocol generates four categories of metadata:

    1. WHO - Agent and user identification
       - agent_id: Unique identifier for the agent/service
       - agent_category: Type classification (backend, frontend, etc.)
       - capabilities: List of agent capabilities
       - user_id: Optional user who initiated the operation

    2. WHEN - Temporal information
       - iso_timestamp: ISO 8601 formatted timestamp
       - unix_timestamp: Unix epoch timestamp (seconds)
       - readable: Human-readable formatted timestamp

    3. PROJECT - Project and task context
       - project_id: Unique project identifier
       - project_name: Human-readable project name
       - task_id: Optional task/ticket identifier

    4. WHY - Intent and purpose
       - intent: Intent category (implementation, bugfix, etc.)
       - description: Human-readable description of the intent

    Example:
        >>> tagger = TaggingProtocol(
        ...     agent_id="api-service",
        ...     agent_category=AgentCategory.BACKEND,
        ...     capabilities=["REST API", "Authentication"],
        ...     project_id="user-service",
        ...     project_name="User Service"
        ... )
        >>> tags = tagger.generate_tags(Intent.IMPLEMENTATION)
        >>> print(tags["who"]["agent_id"])
        'api-service'

    Attributes:
        agent_id: Unique identifier for this agent/service
        agent_category: Category classification of the agent
        capabilities: List of capabilities this agent has
        project_id: Current project identifier
        project_name: Human-readable project name
    """

    def __init__(
        self,
        agent_id: str,
        agent_category: Union[AgentCategory, str],
        capabilities: List[str],
        project_id: str,
        project_name: str
    ):
        """
        Initialize the tagging protocol for an agent or service.

        Args:
            agent_id: Unique identifier for the agent (e.g., "backend-dev",
                     "test-runner", "security-auditor")
            agent_category: Category of the agent. Can be an AgentCategory enum
                           value or a string matching an enum name.
            capabilities: List of capabilities this agent possesses
                         (e.g., ["REST API", "PostgreSQL", "Redis"])
            project_id: Identifier for the project (e.g., "my-project",
                       "PROJ-001")
            project_name: Human-readable name of the project
                         (e.g., "My Awesome Project")

        Raises:
            ValueError: If agent_category string doesn't match any AgentCategory

        Example:
            >>> tagger = TaggingProtocol(
            ...     agent_id="code-reviewer",
            ...     agent_category=AgentCategory.QUALITY,
            ...     capabilities=["static analysis", "security scanning"],
            ...     project_id="core-api",
            ...     project_name="Core API Service"
            ... )
        """
        self.agent_id = agent_id

        # Handle string input for agent_category
        if isinstance(agent_category, str):
            try:
                self.agent_category = AgentCategory(agent_category)
            except ValueError:
                # Try matching by name (case-insensitive)
                for cat in AgentCategory:
                    if cat.name.lower() == agent_category.lower().replace("-", "_"):
                        self.agent_category = cat
                        break
                else:
                    raise ValueError(
                        f"Unknown agent category: {agent_category}. "
                        f"Valid categories: {[c.value for c in AgentCategory]}"
                    )
        else:
            self.agent_category = agent_category

        self.capabilities = self._validate_capabilities(capabilities)
        self.project_id = project_id
        self.project_name = project_name

    def _validate_capabilities(self, capabilities: List[str]) -> List[str]:
        """
        Validate that capabilities list contains only non-empty strings.

        Args:
            capabilities: List of capability strings to validate

        Returns:
            The validated capabilities list

        Raises:
            ValueError: If any capability is not a non-empty string
        """
        if not isinstance(capabilities, list):
            raise ValueError("capabilities must be a list")

        validated = []
        for i, cap in enumerate(capabilities):
            if not isinstance(cap, str):
                raise ValueError(
                    f"Capability at index {i} must be a string, got {type(cap).__name__}"
                )
            stripped = cap.strip()
            if not stripped:
                raise ValueError(
                    f"Capability at index {i} cannot be empty or whitespace-only"
                )
            validated.append(stripped)

        return validated

    def __repr__(self) -> str:
        """
        Return a string representation of the TaggingProtocol instance.

        Returns:
            A string showing agent_id, agent_category, and project_name
            for debugging purposes.
        """
        return (
            f"TaggingProtocol("
            f"agent_id={self.agent_id!r}, "
            f"agent_category={self.agent_category.value!r}, "
            f"project_name={self.project_name!r})"
        )

    def generate_tags(
        self,
        intent: Union[Intent, str],
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        custom_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate complete WHO/WHEN/PROJECT/WHY tags.

        This method creates a comprehensive metadata dictionary containing
        all required tagging information for an operation.

        Args:
            intent: The intent/purpose of the operation. Can be an Intent
                   enum value or a string matching an intent name.
            user_id: Optional identifier for the user who initiated the
                    operation. Defaults to "system" if not provided.
            task_id: Optional task/ticket identifier. If not provided,
                    an auto-generated ID will be created.
            additional_metadata: Optional dictionary of additional metadata
                                to include in the tags.
            custom_timestamp: Optional custom timestamp. If not provided,
                             the current UTC time is used.

        Returns:
            A dictionary with the following structure:
            {
                "who": {
                    "agent_id": str,
                    "agent_category": str,
                    "capabilities": List[str],
                    "user_id": str
                },
                "when": {
                    "iso_timestamp": str,
                    "unix_timestamp": int,
                    "readable": str
                },
                "project": {
                    "project_id": str,
                    "project_name": str,
                    "task_id": str
                },
                "why": {
                    "intent": str,
                    "description": str
                },
                "additional": Dict[str, Any]  # Only if additional_metadata provided
            }

        Raises:
            ValueError: If intent string doesn't match any Intent enum value

        Example:
            >>> tagger = TaggingProtocol(...)
            >>> tags = tagger.generate_tags(
            ...     intent=Intent.BUGFIX,
            ...     user_id="user-456",
            ...     task_id="BUG-123",
            ...     additional_metadata={"severity": "high"}
            ... )
        """
        # Handle string input for intent
        if isinstance(intent, str):
            try:
                intent = Intent(intent)
            except ValueError:
                # Try matching by name (case-insensitive)
                for i in Intent:
                    if i.name.lower() == intent.lower():
                        intent = i
                        break
                else:
                    raise ValueError(
                        f"Unknown intent: {intent}. "
                        f"Valid intents: {[i.value for i in Intent]}"
                    )

        # Use custom timestamp or current UTC time
        now = custom_timestamp if custom_timestamp is not None else datetime.now(timezone.utc)
        # Ensure custom timestamp is timezone-aware (UTC) if provided
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        tags: Dict[str, Any] = {
            # WHO - Agent and user identification
            "who": {
                "agent_id": self.agent_id,
                "agent_category": self.agent_category.value,
                "capabilities": self.capabilities,
                "user_id": user_id or "system"
            },

            # WHEN - Temporal information
            "when": {
                "iso_timestamp": now.isoformat(),
                "unix_timestamp": int(now.timestamp()),
                "readable": now.strftime("%Y-%m-%d %H:%M:%S UTC")
            },

            # PROJECT - Project and task context
            "project": {
                "project_id": self.project_id,
                "project_name": self.project_name,
                "task_id": task_id or f"auto-{uuid.uuid4().hex[:8]}"
            },

            # WHY - Intent and purpose
            "why": {
                "intent": intent.value,
                "description": self._get_intent_description(intent)
            }
        }

        # Add any additional metadata
        if not additional_metadata:
            return tags
        tags["additional"] = additional_metadata
        return tags

    def _get_intent_description(self, intent: Intent) -> str:
        """
        Get human-readable description of an intent.

        Args:
            intent: The Intent enum value

        Returns:
            Human-readable description string
        """
        return INTENT_DESCRIPTIONS.get(intent, "Unknown intent")

    def create_payload(
        self,
        content: str,
        intent: Union[Intent, str],
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        custom_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create a complete payload with content and tags.

        This is a convenience method that combines content with generated
        tags into a single payload structure suitable for storage operations.

        Args:
            content: The content to include in the payload (log message,
                    memory content, event data, etc.)
            intent: The intent/purpose of the operation
            user_id: Optional user identifier
            task_id: Optional task/ticket identifier
            additional_metadata: Optional additional metadata
            custom_timestamp: Optional custom timestamp

        Returns:
            A dictionary with the following structure:
            {
                "content": str,
                "metadata": Dict[str, Any],  # The generated tags
                "timestamp": int  # Unix timestamp for quick sorting
            }

        Example:
            >>> tagger = TaggingProtocol(...)
            >>> payload = tagger.create_payload(
            ...     content="Fixed authentication bug in login flow",
            ...     intent=Intent.BUGFIX,
            ...     task_id="BUG-789"
            ... )
            >>> # payload can now be stored in memory, logs, etc.
        """
        tags = self.generate_tags(
            intent=intent,
            user_id=user_id,
            task_id=task_id,
            additional_metadata=additional_metadata,
            custom_timestamp=custom_timestamp
        )

        return {
            "content": content,
            "metadata": tags,
            "timestamp": tags["when"]["unix_timestamp"]
        }

    def create_flat_tags(
        self,
        intent: Union[Intent, str],
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        Create flattened tags suitable for logging systems.

        Some logging and metrics systems prefer flat key-value pairs
        rather than nested structures. This method provides that format.

        Args:
            intent: The intent/purpose of the operation
            user_id: Optional user identifier
            task_id: Optional task/ticket identifier
            prefix: Optional prefix for all keys (e.g., "tag_" or "meta.")

        Returns:
            A flat dictionary with string keys and values:
            {
                "who_agent_id": str,
                "who_agent_category": str,
                "who_user_id": str,
                "when_iso": str,
                "when_unix": str,
                "project_id": str,
                "project_name": str,
                "task_id": str,
                "why_intent": str
            }

        Example:
            >>> tagger = TaggingProtocol(...)
            >>> flat = tagger.create_flat_tags(
            ...     intent=Intent.ANALYSIS,
            ...     prefix="meta_"
            ... )
            >>> print(flat["meta_who_agent_id"])
        """
        tags = self.generate_tags(intent, user_id, task_id)

        flat: Dict[str, str] = {
            f"{prefix}who_agent_id": tags["who"]["agent_id"],
            f"{prefix}who_agent_category": tags["who"]["agent_category"],
            f"{prefix}who_user_id": tags["who"]["user_id"],
            f"{prefix}when_iso": tags["when"]["iso_timestamp"],
            f"{prefix}when_unix": str(tags["when"]["unix_timestamp"]),
            f"{prefix}project_id": tags["project"]["project_id"],
            f"{prefix}project_name": tags["project"]["project_name"],
            f"{prefix}task_id": tags["project"]["task_id"],
            f"{prefix}why_intent": tags["why"]["intent"],
        }

        return flat

    def update_context(
        self,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ) -> "TaggingProtocol":
        """
        Update the context and return self for method chaining.

        This allows updating project context without creating a new instance.

        Args:
            project_id: New project ID (if provided)
            project_name: New project name (if provided)
            capabilities: New capabilities list (if provided)

        Returns:
            Self, for method chaining

        Example:
            >>> tagger = TaggingProtocol(...)
            >>> tagger.update_context(
            ...     project_id="new-project",
            ...     project_name="New Project"
            ... ).generate_tags(Intent.PLANNING)
        """
        if project_id is not None:
            self.project_id = project_id
        if project_name is not None:
            self.project_name = project_name
        if capabilities is not None:
            self.capabilities = capabilities
        return self


def create_tagger(
    agent_id: str,
    agent_category: Union[AgentCategory, str],
    capabilities: List[str],
    project_id: str,
    project_name: str
) -> TaggingProtocol:
    """
    Factory function to create a TaggingProtocol instance.

    This is a convenience function that provides a simpler interface
    for creating taggers.

    Args:
        agent_id: Unique identifier for the agent
        agent_category: Category of the agent
        capabilities: List of agent capabilities
        project_id: Project identifier
        project_name: Human-readable project name

    Returns:
        Configured TaggingProtocol instance

    Example:
        >>> tagger = create_tagger(
        ...     agent_id="my-service",
        ...     agent_category="backend",
        ...     capabilities=["API", "Database"],
        ...     project_id="my-proj",
        ...     project_name="My Project"
        ... )
    """
    return TaggingProtocol(
        agent_id=agent_id,
        agent_category=agent_category,
        capabilities=capabilities,
        project_id=project_id,
        project_name=project_name
    )


def create_simple_tagger(
    agent_id: str,
    project_id: str,
    project_name: Optional[str] = None
) -> TaggingProtocol:
    """
    Create a minimal tagger with sensible defaults.

    This is useful for quick setup when you don't need to specify
    all the details.

    Args:
        agent_id: Unique identifier for the agent
        project_id: Project identifier
        project_name: Optional project name (defaults to project_id)

    Returns:
        Configured TaggingProtocol instance with default category
        and empty capabilities

    Example:
        >>> tagger = create_simple_tagger("my-agent", "my-project")
        >>> tags = tagger.generate_tags(Intent.IMPLEMENTATION)
    """
    return TaggingProtocol(
        agent_id=agent_id,
        agent_category=AgentCategory.CORE_DEVELOPMENT,
        capabilities=[],
        project_id=project_id,
        project_name=project_name or project_id
    )


# Example usage and self-test
if __name__ == "__main__":
    import json

    # Create a tagger
    tagger = create_tagger(
        agent_id="example-agent",
        agent_category=AgentCategory.BACKEND,
        capabilities=["REST API", "PostgreSQL", "Redis"],
        project_id="example-project",
        project_name="Example Project"
    )

    # Generate tags
    tags = tagger.generate_tags(
        intent=Intent.IMPLEMENTATION,
        user_id="user-123",
        task_id="TASK-001",
        additional_metadata={
            "features": ["feature-a", "feature-b"],
            "priority": "high"
        }
    )

    print("Generated tags:")
    print(json.dumps(tags, indent=2))

    # Create a payload
    payload = tagger.create_payload(
        content="Implemented new API endpoint for user management",
        intent=Intent.IMPLEMENTATION,
        task_id="TASK-002"
    )

    print("\nGenerated payload:")
    print(json.dumps(payload, indent=2))

    # Create flat tags
    flat = tagger.create_flat_tags(
        intent=Intent.ANALYSIS,
        prefix="meta_"
    )

    print("\nFlat tags:")
    print(json.dumps(flat, indent=2))
