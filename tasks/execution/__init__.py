"""Task execution core."""

from tasks.execution.base import BaseExecutor, ExecutionStats
from tasks.execution.browser_adapter import BrowserEnvAdapter, ActionConverter, StateExtractor
from tasks.execution.execution_agent import ExecutionAgent, StopActionResult, TaskExecutionResult
from tasks.execution.collaborative_agent import CollaborativeAgent, CollaborativeOptimizationResult
from tasks.execution.evaluator import TaskEvaluator, evaluator_router, evaluate_task_result
from tasks.execution.multi_executor import MultiExecutor, WorkerResult

__all__ = [
    "ActionConverter", "BaseExecutor", "BrowserEnvAdapter", "CollaborativeAgent",
    "CollaborativeOptimizationResult", "ExecutionAgent", "ExecutionStats",
    "MultiExecutor", "StateExtractor", "StopActionResult", "TaskEvaluator",
    "TaskExecutionResult", "WorkerResult",
    "evaluate_task_result", "evaluator_router",
]
