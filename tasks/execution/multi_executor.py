"""
Multi-Executor Module

This module implements multi-process task execution for:
1. Parallel task execution across multiple browser instances
2. Task distribution and load balancing
3. Result aggregation
4. Resource management and cleanup

Based on SynthAgent's multi-process execution pattern.
"""

import logging
import os
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
from pathlib import Path

from logging_utils import create_file_handler
from tasks.execution.base import ExecutionStats

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """
    Result from a worker process.
    
    Simplified version - only contains essential info for result aggregation.
    Detailed trajectory data is saved separately by each worker.
    """
    worker_id: int
    task_id: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


def _setup_worker_logging(worker_id: int, output_dir: str) -> logging.Logger:
    """
    Setup dedicated logging for a worker process.
    
    Creates a separate log file for each worker in its own directory.
    
    Args:
        worker_id: Worker identifier
        output_dir: Worker output directory
        
    Returns:
        Configured logger for this worker
    """
    # Create worker log file
    log_file = os.path.join(output_dir, f"worker_{worker_id}.log")
    
    # Worker processes use the shared handler factory and keep their own files.
    file_handler = create_file_handler(log_file, logging.DEBUG)
    
    # Get or create worker-specific logger
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.DEBUG)
    worker_logger.propagate = False  # Don't propagate to root logger
    
    # Replace inherited handlers in the child process without duplicating output.
    worker_logger.handlers.clear()
    worker_logger.addHandler(file_handler)

    for module_name in (
        "tasks.execution.execution_agent",
        "tasks.execution.collaborative_agent",
        "tasks.execution.browser_adapter",
        "tasks.execution.base",
    ):
        module_logger = logging.getLogger(module_name)
        module_logger.handlers.clear()
        module_logger.addHandler(file_handler)
        module_logger.setLevel(logging.DEBUG)
        module_logger.propagate = False
    
    return worker_logger


def _worker_process(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    config_dict: Dict[str, Any],
    enable_collaborative: bool,
) -> None:
    """
    Worker process for executing tasks.
    
    Args:
        worker_id: Unique worker identifier
        task_queue: Queue of tasks to process
        result_queue: Queue for results
        config_dict: Configuration dictionary
        enable_collaborative: Enable T2 refinement
    """
    # Import here to avoid issues with multiprocessing
    from tasks.execution.collaborative_agent import CollaborativeAgent
    from tasks.execution.execution_agent import ExecutionAgent

    try:
        # Receive a picklable structural configuration object.
        config = config_dict
        base_output = getattr(config, "output_dir", None)
        if base_output is None:
            output = getattr(config, "output", None)
            base_output = getattr(output, "root_dir", "./outputs")
        worker_output = os.path.join(str(base_output), f"worker_{worker_id}")
        if hasattr(config, "output_dir"):
            config.output_dir = worker_output
        elif hasattr(getattr(config, "output", None), "root_dir"):
            config.output.root_dir = worker_output
        os.makedirs(worker_output, exist_ok=True)

        # Setup worker-specific logging AFTER output dir is created
        logger = _setup_worker_logging(worker_id, getattr(config, "output_dir", getattr(getattr(config, "output", None), "root_dir", "./outputs")))
        logger.info(f"Worker {worker_id} starting...")
        
        # Create agent
        if enable_collaborative:
            agent = CollaborativeAgent(config)
        else:
            agent = ExecutionAgent(config)
        
        # Browser will be initialized on first task execution
        
        while True:
            try:
                # Get task (non-blocking with timeout)
                task_data = task_queue.get(timeout=5)
                
                if task_data is None:
                    # Poison pill - shutdown
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                task_id = task_data.get('id', 'unknown')
                logger.info(f"Worker {worker_id} processing task {task_id}")
                
                start_time = time.time()
                max_retries = 2  # Allow 1 retry for timeout/navigation errors
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        # Execute task
                        result = agent.execute_task(task_data)
                        
                        execution_time = time.time() - start_time
                        
                        # Use TaskExecutionResult.success property
                        worker_result = WorkerResult(
                            worker_id=worker_id,
                            task_id=task_id,
                            success=result.success,
                            error=None if result.success else f"Task ended: {result.end_reason}",
                            execution_time=execution_time,
                        )
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        last_error = e
                        error_str = str(e).lower()
                        
                        # Check if this is a recoverable error (timeout, navigation, connection)
                        is_recoverable = any(keyword in error_str for keyword in [
                            'timeout', 'navigation', 'connection', 'target closed',
                            'page closed', 'context closed', 'browser closed'
                        ])
                        
                        if is_recoverable and attempt < max_retries:
                            logger.warning(
                                f"Task {task_id} failed with recoverable error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                            )
                            # Reset browser for retry
                            try:
                                agent.close_browser()
                            except Exception as close_error:
                                logger.warning(f"Error closing browser: {close_error}")
                            time.sleep(2)  # Brief pause before retry
                            continue
                        
                        # Non-recoverable error or max retries reached
                        execution_time = time.time() - start_time
                        worker_result = WorkerResult(
                            worker_id=worker_id,
                            task_id=task_id,
                            success=False,
                            error=f"{type(e).__name__}: {str(e)}",
                            execution_time=execution_time,
                        )
                        logger.error(f"Task {task_id} failed: {e}")
                        break
                
                # Persist this task's data (screenshots + trajectories) immediately
                # so results are not lost if the worker dies mid-run. save_state is
                # incremental: jsonl files append, exploration_trajs is cleared, and
                # the screenshot counter persists across calls to avoid name clashes.
                try:
                    agent.save_state()
                except Exception as save_error:
                    logger.error(f"Worker {worker_id} failed to save state for task {task_id}: {save_error}")

                result_queue.put(worker_result)
                
            except Exception as e:
                if "Empty" in str(type(e).__name__):
                    # Queue timeout, check if more tasks
                    continue
                logger.error(f"Worker {worker_id} error: {e}")
        
        # Cleanup. State is already persisted per-task above; this final call only
        # flushes token-usage/stats/config (trajectories/screenshots were saved and
        # exploration_trajs cleared after each task, so the guard makes it a no-op there).
        agent.save_state()
        agent.close()
        
    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        traceback.print_exc()
    
    logger.info(f"Worker {worker_id} finished")


class MultiExecutor:
    """
    Multi-process task executor.
    
    Manages parallel execution across multiple browser instances:
    - Task queue management
    - Worker process lifecycle
    - Result aggregation
    - Progress logging
    """
    
    def __init__(
        self,
        config: Any,
        num_workers: int = 4,
        enable_collaborative: bool = True,
    ):
        """
        Initialize multi-executor.
        
        Args:
            config: Configuration for all workers
            num_workers: Number of parallel workers
            enable_collaborative: Enable T2 collaborative refinement
        """
        self.config = config
        self.num_workers = num_workers
        self.enable_collaborative = enable_collaborative
        
        self.manager: Optional[SyncManager] = None
        self.task_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.workers: List = []
        
        self.stats = ExecutionStats()
        self.results: List[WorkerResult] = []
        
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Setup the run output directory."""
        output_dir = getattr(self.config, "output_dir", None)
        if output_dir is None:
            output_dir = getattr(getattr(self.config, "output", None), "root_dir", "./outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Worker directories are created by workers when they start.
        self._worker_output_root = output_dir
    
    def execute_tasks(
        self,
        tasks: List[Dict[str, Any]],
        callback: Optional[Callable[[WorkerResult], None]] = None,
    ) -> List[WorkerResult]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of task dictionaries
            callback: Optional callback for each completed task
            
        Returns:
            List of WorkerResult objects
        """
        self.stats = ExecutionStats(total_tasks=len(tasks))
        self.results = []
        
        start_time = time.time()
        
        logger.info(f"Starting execution of {len(tasks)} tasks with {self.num_workers} workers")
        
        try:
            # Initialize multiprocessing
            self.manager = Manager()
            self.task_queue = self.manager.Queue()
            self.result_queue = self.manager.Queue()
            
            # Fill task queue
            for task in tasks:
                self.task_queue.put(task)
            
            # Add poison pills for shutdown
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            
            # Convert config to dict for pickling
            config_dict = self.config
            
            # Start workers
            self.workers = []
            for i in range(self.num_workers):
                p = Process(
                    target=_worker_process,
                    args=(
                        i,
                        self.task_queue,
                        self.result_queue,
                        config_dict,
                        self.enable_collaborative,
                    ),
                )
                p.start()
                self.workers.append(p)
            
            # Collect results (no monitoring/timeout heuristics; rely on worker poison-pill shutdown).
            results_received = 0
            while results_received < len(tasks):
                try:
                    # Short timeout only so we can periodically check for the
                    # "all workers dead" edge case; not a stall/monitoring heuristic.
                    result = self.result_queue.get(timeout=30)
                except Exception as e:
                    if "Empty" in type(e).__name__ or "empty" in str(e).lower():
                        # No result yet. Only stop if every worker has exited, which
                        # means no further results will ever arrive (crash guard).
                        if not any(p.is_alive() for p in self.workers):
                            logger.warning(
                                f"All workers exited with {results_received}/{len(tasks)} "
                                "results collected; stopping."
                            )
                            break
                        continue
                    raise

                results_received += 1
                self.results.append(result)
                self._update_stats(result)

                logger.info(
                    f"Progress: {results_received}/{len(tasks)} "
                    f"(Success: {self.stats.successful_tasks}, "
                    f"Failed: {self.stats.failed_tasks})"
                )

                if callback:
                    callback(result)
            
            # Wait for workers to finish gracefully
            logger.info("Waiting for workers to finish...")
            for i, p in enumerate(self.workers):
                p.join(timeout=60)  # Give more time for cleanup
                if p.is_alive():
                    logger.warning(f"Worker {i} did not finish in time, terminating...")
                    p.terminate()
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self._cleanup()
            raise
        
        finally:
            self._cleanup()
        
        self.stats.total_time = time.time() - start_time
        
        logger.info(f"Execution completed in {self.stats.total_time:.1f}s")
        logger.info(f"Success rate: {self.stats.successful_tasks}/{len(tasks)}")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _update_stats(self, result: WorkerResult):
        """Update execution statistics."""
        self.stats.completed_tasks += 1
        if result.success:
            self.stats.successful_tasks += 1
        else:
            self.stats.failed_tasks += 1
    
    def _cleanup(self):
        """Cleanup resources."""
        for p in self.workers:
            if p.is_alive():
                p.terminate()
        
        if self.manager:
            self.manager.shutdown()
    
    def _save_results(self):
        """Save execution results."""
        output_dir = getattr(self.config, "output_dir", None)
        if output_dir is None:
            output_dir = getattr(getattr(self.config, "output", None), "root_dir", "./outputs")
        results_path = os.path.join(output_dir, "execution_results.json")
        
        # Convert results to serializable format
        results_data = []
        for r in self.results:
            results_data.append({
                'worker_id': r.worker_id,
                'task_id': r.task_id,
                'success': r.success,
                'error': r.error,
                'execution_time': r.execution_time,
            })
        
        with open(results_path, 'w') as f:
            json.dump({
                'results': results_data,
                'stats': {
                    'total_tasks': self.stats.total_tasks,
                    'completed_tasks': self.stats.completed_tasks,
                    'successful_tasks': self.stats.successful_tasks,
                    'failed_tasks': self.stats.failed_tasks,
                    'total_time': self.stats.total_time,
                },
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")

