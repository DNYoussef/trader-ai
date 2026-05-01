"""Unified severity and category mapping for linter outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class UnifiedSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ViolationCategory(Enum):
    SECURITY = "security"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    COMPLEXITY = "complexity"


@dataclass
class SeverityRule:
    tool_name: str
    rule_pattern: str
    rule_codes: List[str]
    unified_severity: UnifiedSeverity
    category: ViolationCategory
    description: str
    rationale: str
    examples: List[str] = field(default_factory=list)


class UnifiedSeverityMapper:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.tool_mappings = self._default_mappings()
        if config_path:
            self._load_config(Path(config_path))

    def _default_mappings(self) -> Dict[str, Dict[str, str]]:
        return {
            "flake8": {
                "E9": "critical",
                "F821": "critical",
                "F822": "critical",
                "F631": "high",
                "F405": "high",
                "F401": "medium",
                "F841": "medium",
                "E501": "medium",
                "E5": "medium",
                "E1": "low",
                "E2": "low",
                "W1": "low",
                "W291": "low",
            },
            "pylint": {
                "F": "critical",
                "E1120": "high",
                "E": "high",
                "W0611": "medium",
                "W": "medium",
                "R0903": "low",
                "R": "low",
                "C0103": "low",
                "C": "low",
                "I0011": "info",
                "I": "info",
            },
            "ruff": {
                "S": "high",
                "B9": "high",
                "B": "high",
                "PERF": "medium",
                "ASYNC": "high",
                "I": "medium",
                "E501": "low",
                "W291": "low",
                "D100": "low",
                "D": "low",
            },
            "mypy": {
                "assignment": "high",
                "return-value": "high",
                "attr-defined": "high",
                "unused-ignore": "medium",
                "redundant-cast": "medium",
                "note": "info",
            },
            "bandit": {
                "HIGH": "critical",
                "MEDIUM": "high",
                "LOW": "medium",
                "B602": "critical",
                "B101": "critical",
                "B303": "high",
                "B104": "high",
                "B105": "medium",
                "B106": "medium",
            },
        }

    def _load_config(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())
        for tool, mappings in data.get("tool_mappings", {}).items():
            self.tool_mappings.setdefault(tool, {}).update(mappings)

    def map_severity(
        self, tool_name: str, rule_code: str, tool_severity: str = ""
    ) -> UnifiedSeverity:
        tool = tool_name.lower()
        mappings = self.tool_mappings.get(tool)
        if not mappings:
            return UnifiedSeverity.MEDIUM

        candidates = [rule_code, tool_severity, rule_code[:4], rule_code[:3], rule_code[:2], rule_code[:1]]
        for candidate in candidates:
            if candidate in mappings:
                return UnifiedSeverity(mappings[candidate])

        upper_severity = tool_severity.upper()
        if upper_severity in mappings:
            return UnifiedSeverity(mappings[upper_severity])

        return UnifiedSeverity.MEDIUM

    def categorize_violation(
        self, tool_name: str, rule_code: str, message: str
    ) -> ViolationCategory:
        text = f"{tool_name} {rule_code} {message}".lower()
        rule = rule_code.upper()

        if "security" in text or "password" in text or "shell" in text or tool_name == "bandit" or rule.startswith("S"):
            return ViolationCategory.SECURITY
        if "undefined" in text or "syntax" in text or "exception" in text or rule.startswith("F") or tool_name == "mypy":
            return ViolationCategory.CORRECTNESS
        if "slow" in text or "inefficient" in text or rule.startswith("PERF"):
            return ViolationCategory.PERFORMANCE
        if "line too long" in text or "whitespace" in text or "naming" in text or rule.startswith(("E", "W", "C")):
            return ViolationCategory.STYLE
        if "docstring" in text or "documentation" in text or rule.startswith("D"):
            return ViolationCategory.DOCUMENTATION
        if "too many" in text or "complexity" in text or rule.startswith("R"):
            return ViolationCategory.COMPLEXITY
        if "test" in text or "mock" in text or "assert" in text:
            return ViolationCategory.TESTING
        return ViolationCategory.MAINTAINABILITY

    def get_severity_distribution(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        distribution = {severity.value: 0 for severity in UnifiedSeverity}
        for violation in violations:
            severity = self.map_severity(
                violation.get("tool_name", violation.get("tool", "")),
                violation.get("rule_code", violation.get("rule", "")),
                violation.get("tool_severity", violation.get("severity", "")),
            )
            distribution[severity.value] += 1
        return distribution

    def get_category_distribution(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        distribution = {category.value: 0 for category in ViolationCategory}
        for violation in violations:
            category = self.categorize_violation(
                violation.get("tool_name", violation.get("tool", "")),
                violation.get("rule_code", violation.get("rule", "")),
                violation.get("message", ""),
            )
            distribution[category.value] += 1
        return distribution

    def calculate_quality_score(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not violations:
            return {
                "quality_score": 100.0,
                "grade": "A",
                "total_violations": 0,
                "severity_distribution": self.get_severity_distribution([]),
                "category_distribution": self.get_category_distribution([]),
                "recommendations": [],
            }

        severity_distribution = self.get_severity_distribution(violations)
        category_distribution = self.get_category_distribution(violations)
        penalties = {
            "critical": 12,
            "high": 7,
            "medium": 3,
            "low": 1,
            "info": 0.25,
        }
        penalty = sum(severity_distribution[key] * value for key, value in penalties.items())
        score = max(0.0, 100.0 - penalty)
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 50 else "F"

        recommendations = []
        if severity_distribution["critical"]:
            recommendations.append("URGENT: resolve critical violations before release.")
        if severity_distribution["high"]:
            recommendations.append("Prioritize high-severity correctness and security issues.")
        if category_distribution["style"] > 10:
            recommendations.append("Use an auto-formatter to reduce style noise.")

        return {
            "quality_score": score,
            "grade": grade,
            "total_violations": len(violations),
            "severity_distribution": severity_distribution,
            "category_distribution": category_distribution,
            "recommendations": recommendations,
        }

    def export_config(self, path: str, format: str = "yaml") -> None:
        payload = {"tool_mappings": self.tool_mappings}
        output = Path(path)
        if format == "json":
            output.write_text(json.dumps(payload, indent=2))
        else:
            output.write_text(yaml.safe_dump(payload))


unified_mapper = UnifiedSeverityMapper()
