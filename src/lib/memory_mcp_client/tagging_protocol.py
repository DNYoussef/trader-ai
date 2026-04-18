"""
Memory MCP Tagging Protocol
WHO/WHEN/PROJECT/WHY metadata for all memory_store operations

NOTE: This is a LOCAL COPY bundled with memory-mcp-client for convenience.
The canonical standalone component is at:
    C:/Users/17175/.claude/library/components/memory/tagging-protocol/

If updating this file, consider whether the standalone component should also
be updated to maintain consistency across the library ecosystem.

Implements mandatory tagging as per Memory MCP specification:
- WHO (agent_id, user_id, agent_category, capabilities)
- WHEN (ISO timestamp, Unix timestamp, readable format)
- PROJECT (project_id, project_name, task_id)
- WHY (intent: implementation/bugfix/refactor/testing/documentation/analysis/planning/research)

Example:
    >>> config = TaggingConfig(
    ...     agent_id="my-agent",
    ...     agent_category=AgentCategory.BACKEND,
    ...     capabilities=["api-development", "database"],
    ...     project_id="my-project",
    ...     project_name="My Project"
    ... )
    >>> tagger = TaggingProtocol(config)
    >>> payload = tagger.create_memory_store_payload(
    ...     content="Implemented new feature",
    ...     intent=Intent.IMPLEMENTATION,
    ...     task_id="TASK-001"
    ... )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid


class Intent(str, Enum):
    """Intent categories for Memory MCP operations"""
    IMPLEMENTATION = "implementation"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    RESEARCH = "research"


class AgentCategory(str, Enum):
    """Agent categories for WHO tagging"""
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
    ORCHESTRATION = "orchestration"
    DEVOPS = "devops"
    CUSTOM = "custom"


# Intent descriptions mapping
INTENT_DESCRIPTIONS: Dict[Intent, str] = {
    Intent.IMPLEMENTATION: "Implementing new feature or functionality",
    Intent.BUGFIX: "Fixing bugs or errors in existing code",
    Intent.REFACTOR: "Refactoring code for better quality or performance",
    Intent.TESTING: "Writing or executing tests",
    Intent.DOCUMENTATION: "Creating or updating documentation",
    Intent.ANALYSIS: "Analyzing code, performance, or system behavior",
    Intent.PLANNING: "Planning architecture or approach",
    Intent.RESEARCH: "Researching solutions or best practices"
}


@dataclass
class TaggingConfig:
    """
    Configuration for TaggingProtocol

    Attributes:
        agent_id: Unique agent identifier (e.g., "backend-dev", "test-runner")
        agent_category: Agent category from AgentCategory enum
        capabilities: List of agent capabilities/skills
        project_id: Project identifier (e.g., "my-project-123")
        project_name: Human-readable project name
        default_user_id: Default user ID when not specified (defaults to "system")
    """
    agent_id: str
    agent_category: AgentCategory
    capabilities: List[str]
    project_id: str
    project_name: str
    default_user_id: str = "system"


@dataclass
class WhoTag:
    """WHO metadata tag"""
    agent_id: str
    agent_category: str
    capabilities: List[str]
    user_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_category": self.agent_category,
            "capabilities": self.capabilities,
            "user_id": self.user_id
        }


@dataclass
class WhenTag:
    """WHEN metadata tag"""
    iso_timestamp: str
    unix_timestamp: int
    readable: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iso_timestamp": self.iso_timestamp,
            "unix_timestamp": self.unix_timestamp,
            "readable": self.readable
        }

    @classmethod
    def now(cls) -> "WhenTag":
        """Create WhenTag for current time"""
        now = datetime.now(timezone.utc)
        return cls(
            iso_timestamp=now.isoformat(),
            unix_timestamp=int(now.timestamp()),
            readable=now.strftime("%Y-%m-%d %H:%M:%S UTC")
        )


@dataclass
class ProjectTag:
    """PROJECT metadata tag"""
    project_id: str
    project_name: str
    task_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "task_id": self.task_id
        }


@dataclass
class WhyTag:
    """WHY metadata tag"""
    intent: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "description": self.description
        }


@dataclass
class MemoryTags:
    """Complete set of WHO/WHEN/PROJECT/WHY tags"""
    who: WhoTag
    when: WhenTag
    project: ProjectTag
    why: WhyTag
    additional: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "who": self.who.to_dict(),
            "when": self.when.to_dict(),
            "project": self.project.to_dict(),
            "why": self.why.to_dict()
        }
        if not self.additional:
            return result
        result["additional"] = self.additional
        return result


@dataclass
class MemoryStorePayload:
    """Complete payload for memory_store operation"""
    content: str
    metadata: MemoryTags
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "timestamp": self.timestamp
        }


class TaggingProtocol:
    """
    Memory MCP Tagging Protocol Implementation

    Automatically generates WHO/WHEN/PROJECT/WHY metadata for all memory operations.
    Ensures compliance with Memory MCP tagging requirements.

    Example:
        >>> config = TaggingConfig(
        ...     agent_id="code-reviewer",
        ...     agent_category=AgentCategory.CORE_DEVELOPMENT,
        ...     capabilities=["code-review", "security-analysis"],
        ...     project_id="my-app",
        ...     project_name="My Application"
        ... )
        >>> tagger = TaggingProtocol(config)
        >>> tags = tagger.generate_tags(Intent.ANALYSIS, task_id="PR-123")
    """

    def __init__(self, config: TaggingConfig):
        """
        Initialize tagging protocol

        Args:
            config: TaggingConfig with agent and project information
        """
        self._config = config

    @property
    def config(self) -> TaggingConfig:
        """Get tagging configuration"""
        return self._config

    @property
    def agent_id(self) -> str:
        """Get agent ID from config"""
        return self._config.agent_id

    @property
    def project_id(self) -> str:
        """Get project ID from config"""
        return self._config.project_id

    def generate_tags(
        self,
        intent: Intent,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryTags:
        """
        Generate complete WHO/WHEN/PROJECT/WHY tags

        Args:
            intent: Intent category (implementation, bugfix, etc.)
            user_id: Optional user identifier (defaults to config.default_user_id)
            task_id: Optional task identifier (auto-generated if not provided)
            additional_metadata: Optional additional metadata to include

        Returns:
            MemoryTags with all required tags
        """
        when = WhenTag.now()

        who = WhoTag(
            agent_id=self._config.agent_id,
            agent_category=self._config.agent_category.value,
            capabilities=self._config.capabilities,
            user_id=user_id or self._config.default_user_id
        )

        project = ProjectTag(
            project_id=self._config.project_id,
            project_name=self._config.project_name,
            task_id=task_id or f"auto-{uuid.uuid4().hex[:8]}"
        )

        why = WhyTag(
            intent=intent.value,
            description=INTENT_DESCRIPTIONS.get(intent, "Unknown intent")
        )

        return MemoryTags(
            who=who,
            when=when,
            project=project,
            why=why,
            additional=additional_metadata
        )

    def create_memory_store_payload(
        self,
        content: str,
        intent: Intent,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryStorePayload:
        """
        Create complete memory_store payload with tags

        Args:
            content: Content to store in memory
            intent: Intent category
            user_id: Optional user identifier
            task_id: Optional task identifier
            additional_metadata: Optional additional metadata

        Returns:
            MemoryStorePayload ready for memory_store operation
        """
        tags = self.generate_tags(intent, user_id, task_id, additional_metadata)

        return MemoryStorePayload(
            content=content,
            metadata=tags,
            timestamp=tags.when.unix_timestamp
        )


# Factory functions for common agent types

def create_backend_tagger(
    project_id: str,
    project_name: str,
    agent_id: str = "backend-dev",
    capabilities: Optional[List[str]] = None
) -> TaggingProtocol:
    """
    Create tagging protocol for backend development agent

    Args:
        project_id: Project identifier
        project_name: Human-readable project name
        agent_id: Agent identifier (default: "backend-dev")
        capabilities: Agent capabilities (default: backend-related)

    Returns:
        TaggingProtocol instance configured for backend development
    """
    return TaggingProtocol(TaggingConfig(
        agent_id=agent_id,
        agent_category=AgentCategory.BACKEND,
        capabilities=capabilities or [
            "REST API development",
            "Database operations",
            "Backend services",
            "Integration patterns"
        ],
        project_id=project_id,
        project_name=project_name
    ))


def create_frontend_tagger(
    project_id: str,
    project_name: str,
    agent_id: str = "frontend-dev",
    capabilities: Optional[List[str]] = None
) -> TaggingProtocol:
    """
    Create tagging protocol for frontend development agent

    Args:
        project_id: Project identifier
        project_name: Human-readable project name
        agent_id: Agent identifier (default: "frontend-dev")
        capabilities: Agent capabilities (default: frontend-related)

    Returns:
        TaggingProtocol instance configured for frontend development
    """
    return TaggingProtocol(TaggingConfig(
        agent_id=agent_id,
        agent_category=AgentCategory.FRONTEND,
        capabilities=capabilities or [
            "UI development",
            "React/Vue/Angular",
            "CSS/Styling",
            "Frontend optimization"
        ],
        project_id=project_id,
        project_name=project_name
    ))


def create_testing_tagger(
    project_id: str,
    project_name: str,
    agent_id: str = "test-runner",
    capabilities: Optional[List[str]] = None
) -> TaggingProtocol:
    """
    Create tagging protocol for testing/QA agent

    Args:
        project_id: Project identifier
        project_name: Human-readable project name
        agent_id: Agent identifier (default: "test-runner")
        capabilities: Agent capabilities (default: testing-related)

    Returns:
        TaggingProtocol instance configured for testing
    """
    return TaggingProtocol(TaggingConfig(
        agent_id=agent_id,
        agent_category=AgentCategory.TESTING_VALIDATION,
        capabilities=capabilities or [
            "Unit testing",
            "Integration testing",
            "E2E testing",
            "Test automation"
        ],
        project_id=project_id,
        project_name=project_name
    ))


def create_custom_tagger(
    agent_id: str,
    agent_category: AgentCategory,
    capabilities: List[str],
    project_id: str,
    project_name: str,
    default_user_id: str = "system"
) -> TaggingProtocol:
    """
    Create tagging protocol with custom configuration

    Args:
        agent_id: Unique agent identifier
        agent_category: Agent category
        capabilities: List of agent capabilities
        project_id: Project identifier
        project_name: Human-readable project name
        default_user_id: Default user ID

    Returns:
        TaggingProtocol instance with custom configuration
    """
    return TaggingProtocol(TaggingConfig(
        agent_id=agent_id,
        agent_category=agent_category,
        capabilities=capabilities,
        project_id=project_id,
        project_name=project_name,
        default_user_id=default_user_id
    ))
