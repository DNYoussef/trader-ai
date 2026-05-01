"""Dependency inventory analyzer for enterprise supply-chain security."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class DependencyAnalyzer:
    """Collect direct dependency names from common manifest files."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.dependencies: List[Dict[str, Any]] = []

    async def analyze_dependencies(self) -> Dict[str, Any]:
        dependencies: List[Dict[str, Any]] = []

        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            dependencies.extend(self._parse_requirements(requirements))

        package_json = self.project_root / "package.json"
        if package_json.exists():
            dependencies.extend(self._parse_package_json(package_json))

        self.dependencies = dependencies
        report_path = self.project_root / "dependency-report.json"
        return {
            "project_root": str(self.project_root),
            "dependencies": dependencies,
            "report_path": str(report_path),
        }

    def _parse_requirements(self, path: Path) -> List[Dict[str, Any]]:
        dependencies: List[Dict[str, Any]] = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            dependencies.append({"name": name, "ecosystem": "pypi", "specifier": line})
        return dependencies

    def _parse_package_json(self, path: Path) -> List[Dict[str, Any]]:
        dependencies: List[Dict[str, Any]] = []
        data = json.loads(path.read_text())
        for section in ("dependencies", "devDependencies"):
            for name, version in data.get(section, {}).items():
                dependencies.append(
                    {
                        "name": name,
                        "ecosystem": "npm",
                        "specifier": version,
                        "scope": section,
                    }
                )
        return dependencies

    def get_status(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "dependencies_count": len(self.dependencies),
        }
