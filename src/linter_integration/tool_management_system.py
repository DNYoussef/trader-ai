"""Tool lifecycle management primitives for linter integration tests."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolEnvironment:
    nodeVersion: Optional[str] = None
    pythonVersion: Optional[str] = None
    environmentVariables: Dict[str, str] = field(default_factory=dict)
    workingDirectory: str = ""
    pathExtensions: List[str] = field(default_factory=list)


@dataclass
class ToolConfiguration:
    configFile: Optional[str] = None
    rules: Dict[str, Any] = field(default_factory=dict)
    ignore: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    customArgs: List[str] = field(default_factory=list)
    environment: Optional[ToolEnvironment] = None


@dataclass
class ResourceAllocation:
    cpuLimit: Optional[float] = None
    memoryLimit: Optional[int] = None
    concurrencyLimit: int = 1
    priorityWeight: float = 0.5
    executionQuota: int = 50
    throttleInterval: int = 1000


@dataclass
class ToolHealth:
    isHealthy: bool = True
    healthScore: int = 100
    failureRate: float = 0.0
    averageExecutionTime: float = 0.0
    successfulExecutions: int = 0
    failedExecutions: int = 0
    lastError: Optional[str] = None


@dataclass
class ToolMetrics:
    totalExecutions: int = 0
    successfulExecutions: int = 0
    failedExecutions: int = 0
    averageExecutionTime: float = 0.0
    minExecutionTime: float = float("inf")
    maxExecutionTime: float = 0.0
    totalViolationsFound: int = 0
    uniqueRulesTriggered: set[str] = field(default_factory=set)


@dataclass
class ToolExecutionOptions:
    additionalArgs: List[str] = field(default_factory=list)


@dataclass
class ToolExecutionResult:
    success: bool
    output: str
    stderr: str
    executionTime: float
    memoryUsed: int
    exitCode: int
    violationsFound: int


@dataclass
class RecoveryProcedures:
    resetConfiguration: bool = True
    clearCache: bool = True
    customRecoverySteps: List[str] = field(
        default_factory=lambda: ["restart_process", "refresh_environment"]
    )


@dataclass
class CircuitBreaker:
    isOpen: bool = False
    failureCount: int = 0
    lastFailureTime: float = 0.0
    successCount: int = 0
    nextAttemptTime: float = 0.0


@dataclass
class ToolStatus:
    tool: Any
    health: ToolHealth
    metrics: ToolMetrics
    circuitBreaker: Any
    allocation: ResourceAllocation
    isRunning: bool
    queueLength: int


class ToolManagementSystem:
    def __init__(self, workspaceRoot: str) -> None:
        self.workspaceRoot = workspaceRoot
        self.maxGlobalConcurrency = 10
        self.environments = self._default_environments(workspaceRoot)
        self.resourceAllocations = self._default_allocations()
        self.tools: Dict[str, Any] = {}
        self.configurations: Dict[str, ToolConfiguration] = {}
        self.healthStatus: Dict[str, ToolHealth] = {}
        self.metrics: Dict[str, ToolMetrics] = {}
        self.circuitBreakers: Dict[str, Any] = {}
        self.recoveryProcedures: Dict[str, RecoveryProcedures] = {}
        self.runningProcesses: Dict[str, Any] = {}
        self.executionQueue: Dict[str, List[Any]] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

    def _default_environments(self, workspace: str) -> Dict[str, ToolEnvironment]:
        return {
            "nodejs": ToolEnvironment(
                nodeVersion="18",
                workingDirectory=workspace,
                environmentVariables={"NODE_ENV": "test"},
                pathExtensions=["node_modules/.bin"],
            ),
            "python": ToolEnvironment(
                pythonVersion="3",
                workingDirectory=workspace,
                environmentVariables={"PYTHONPATH": workspace},
                pathExtensions=[".venv/bin", ".venv/Scripts"],
            ),
            "system": ToolEnvironment(workingDirectory=workspace),
        }

    def _default_allocations(self) -> Dict[str, ResourceAllocation]:
        return {
            "eslint": ResourceAllocation(concurrencyLimit=2, priorityWeight=0.7),
            "tsc": ResourceAllocation(concurrencyLimit=1, priorityWeight=0.8),
            "flake8": ResourceAllocation(
                concurrencyLimit=2,
                priorityWeight=0.7,
                executionQuota=80,
                throttleInterval=1500,
            ),
            "pylint": ResourceAllocation(
                concurrencyLimit=1,
                priorityWeight=0.6,
                executionQuota=30,
                throttleInterval=3000,
            ),
            "ruff": ResourceAllocation(
                concurrencyLimit=4,
                priorityWeight=0.9,
                executionQuota=150,
                throttleInterval=500,
            ),
            "mypy": ResourceAllocation(concurrencyLimit=2, priorityWeight=0.8),
            "bandit": ResourceAllocation(concurrencyLimit=2, priorityWeight=0.8),
        }

    async def registerTool(
        self, tool: Any, config: Optional[ToolConfiguration] = None
    ) -> None:
        await self.validateToolInstallation(tool)
        self.tools[tool.id] = tool
        self.configurations[tool.id] = config or ToolConfiguration()
        self.initializeToolHealth(tool.id)
        self.initializeToolMetrics(tool.id)
        self.circuitBreakers[tool.id] = CircuitBreaker()
        self.recoveryProcedures[tool.id] = RecoveryProcedures()
        allocation = self.resourceAllocations.get(tool.id, ResourceAllocation())
        self._semaphores[tool.id] = asyncio.Semaphore(allocation.concurrencyLimit)

    async def validateToolInstallation(self, tool: Any) -> None:
        return None

    async def executeTool(
        self, tool_id: str, file_paths: List[str], options: Optional[ToolExecutionOptions] = None
    ) -> ToolExecutionResult:
        breaker = self.circuitBreakers.get(tool_id)
        if self._breaker_is_open(breaker):
            raise Exception("Circuit breaker open")

        semaphore = self._semaphores.setdefault(
            tool_id,
            asyncio.Semaphore(
                self.resourceAllocations.get(tool_id, ResourceAllocation()).concurrencyLimit
            ),
        )
        async with semaphore:
            start = time.time()
            try:
                result = await self.executeWithMonitoring(tool_id, file_paths, options)
                self.updateSuccessMetrics(tool_id, result.executionTime, result)
                self._record_breaker_success(tool_id)
                return result
            except Exception as exc:
                self.updateFailureMetrics(tool_id, time.time() - start, exc)
                self._record_breaker_failure(tool_id)
                raise

    async def executeWithMonitoring(
        self, tool_id: str, file_paths: List[str], options: Optional[ToolExecutionOptions] = None
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            success=True,
            output="[]",
            stderr="",
            executionTime=0.0,
            memoryUsed=0,
            exitCode=0,
            violationsFound=0,
        )

    async def performToolHealthCheck(self, tool_id: str) -> None:
        health = self.healthStatus[tool_id]
        try:
            await self.validateToolInstallation(self.tools[tool_id])
            health.isHealthy = True
            health.healthScore = 100
            health.lastError = None
        except Exception as exc:
            health.isHealthy = False
            health.healthScore = max(0, health.healthScore - 20)
            health.lastError = str(exc)

    async def attemptToolRecovery(self, tool_id: str) -> None:
        procedures = self.recoveryProcedures[tool_id]
        if procedures.resetConfiguration:
            await self.resetToolConfiguration(tool_id)
        if procedures.clearCache:
            await self.clearToolCache(tool_id)
        for step in procedures.customRecoverySteps:
            await self.executeRecoveryStep(tool_id, step)
        self.initializeToolHealth(tool_id)
        self.circuitBreakers[tool_id] = CircuitBreaker()

    async def resetToolConfiguration(self, tool_id: str) -> None:
        return None

    async def clearToolCache(self, tool_id: str) -> None:
        return None

    async def executeRecoveryStep(self, tool_id: str, step: str) -> None:
        return None

    def initializeToolHealth(self, tool_id: str) -> None:
        self.healthStatus[tool_id] = ToolHealth()

    def initializeToolMetrics(self, tool_id: str) -> None:
        self.metrics[tool_id] = ToolMetrics()

    def getToolStatus(self, tool_id: str) -> ToolStatus:
        return ToolStatus(
            tool=self.tools[tool_id],
            health=self.healthStatus[tool_id],
            metrics=self.metrics[tool_id],
            circuitBreaker=self.circuitBreakers[tool_id],
            allocation=self.resourceAllocations.get(tool_id, ResourceAllocation()),
            isRunning=self.getRunningProcessCount(tool_id) > 0,
            queueLength=len(self.executionQueue.get(tool_id, [])),
        )

    def getAllToolStatus(self) -> Dict[str, ToolStatus]:
        return {tool_id: self.getToolStatus(tool_id) for tool_id in self.tools}

    def getToolEnvironment(self, tool: Any) -> ToolEnvironment:
        if tool.id in {"eslint", "tsc"}:
            return self.environments["nodejs"]
        if tool.id in {"flake8", "pylint", "ruff", "mypy", "bandit"}:
            return self.environments["python"]
        return self.environments["system"]

    def prepareExecutionArgs(
        self,
        tool: Any,
        file_paths: List[str],
        config: Optional[ToolConfiguration] = None,
        options: Optional[ToolExecutionOptions] = None,
    ) -> List[str]:
        args = list(getattr(tool, "args", []))
        if config:
            args.extend(config.customArgs)
        if options:
            args.extend(options.additionalArgs)
        args.extend(file_paths)
        return args

    def getRunningProcessCount(self, tool_id: str) -> int:
        prefix = f"{tool_id}_"
        return sum(1 for process_id in self.runningProcesses if process_id.startswith(prefix))

    def countViolationsInOutput(self, tool: Any, output: str) -> int:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                return sum(len(item.get("messages", [])) if isinstance(item, dict) else 1 for item in parsed)
            if isinstance(parsed, dict):
                return len(parsed.get("messages", parsed.get("violations", [])))
        except Exception:
            pass
        return len([line for line in output.splitlines() if line.strip()])

    def processExecutionQueue(self, tool_id: str) -> None:
        queue = self.executionQueue.get(tool_id, [])
        allocation = self.resourceAllocations.get(tool_id, ResourceAllocation())
        if queue and self.getRunningProcessCount(tool_id) < allocation.concurrencyLimit:
            callback = queue.pop(0)
            callback()

    def updateSuccessMetrics(
        self, tool_id: str, execution_time: float, result: ToolExecutionResult
    ) -> None:
        metrics = self.metrics[tool_id]
        health = self.healthStatus[tool_id]

        metrics.totalExecutions += 1
        metrics.successfulExecutions += 1
        metrics.averageExecutionTime = self._running_average(
            metrics.averageExecutionTime, metrics.totalExecutions, execution_time
        )
        metrics.minExecutionTime = min(metrics.minExecutionTime, execution_time)
        metrics.maxExecutionTime = max(metrics.maxExecutionTime, execution_time)
        metrics.totalViolationsFound += result.violationsFound

        health.successfulExecutions += 1
        health.averageExecutionTime = metrics.averageExecutionTime
        total = health.successfulExecutions + health.failedExecutions
        health.failureRate = health.failedExecutions / total if total else 0.0
        health.isHealthy = True

    def updateFailureMetrics(self, tool_id: str, execution_time: float, error: Exception) -> None:
        metrics = self.metrics[tool_id]
        health = self.healthStatus[tool_id]

        metrics.totalExecutions += 1
        metrics.failedExecutions += 1
        health.failedExecutions += 1
        total = health.successfulExecutions + health.failedExecutions
        health.failureRate = health.failedExecutions / total if total else 0.0
        health.lastError = str(error)
        health.isHealthy = False

    def _running_average(self, previous: float, count: int, value: float) -> float:
        if count <= 1:
            return value
        return ((previous * (count - 1)) + value) / count

    def _breaker_is_open(self, breaker: Any) -> bool:
        if isinstance(breaker, dict):
            return bool(breaker.get("isOpen"))
        return bool(getattr(breaker, "isOpen", False))

    def _record_breaker_success(self, tool_id: str) -> None:
        breaker = self.circuitBreakers.get(tool_id)
        if isinstance(breaker, dict):
            breaker["successCount"] = breaker.get("successCount", 0) + 1
            return
        if breaker:
            breaker.successCount += 1

    def _record_breaker_failure(self, tool_id: str) -> None:
        breaker = self.circuitBreakers.get(tool_id)
        if isinstance(breaker, dict):
            breaker["failureCount"] = breaker.get("failureCount", 0) + 1
            breaker["lastFailureTime"] = time.time()
            if breaker["failureCount"] >= 5:
                breaker["isOpen"] = True
            return
        if breaker:
            breaker.failureCount += 1
            breaker.lastFailureTime = time.time()
            if breaker.failureCount >= 5:
                breaker.isOpen = True
