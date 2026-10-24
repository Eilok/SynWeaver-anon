"""
Collaborative Agent Module

This module implements the T2 Collaborative Optimization Agent that:
1. Inherits from T1 ExecutionAgent for basic execution
2. Overrides STOP handling with task-trajectory collaborative optimization
3. Uses T2 model for comprehensive analysis and refinement

Key Features:
- Two optimization strategies:
  1. Task-only: Refine task to fit existing trajectory
  2. Co-optimize: Edit both task AND trajectory (reorder/delete steps)
  
- Trajectory editing capabilities:
  - Reorder steps
  - Delete steps  
  - Update reasoning for affected steps

Based on SynthAgent's refine patterns with task-trajectory collaborative
optimization enhancements.
"""

import logging
from typing import Optional, Dict, Any, List

from tasks.models import (
    ActionType, Action, StateInfo,
    LowLevelTask, HighLevelTask, ExplorationTraj,
    LowTaskStatus,
)
from tasks.execution.execution_agent import ExecutionAgent, TaskExecutionResult, StopActionResult
from tasks.execution.browser_adapter import format_page_context

logger = logging.getLogger(__name__)


class CollaborativeOptimizationResult:
    """Result of collaborative optimization."""
    
    def __init__(
        self,
        success: bool,
        strategy: str,
        refined_task: Optional[str] = None,
        trajectory_modified: bool = False,
        step_order: Optional[List[int]] = None,
        modified_reasonings: Optional[Dict[str, str]] = None,
        analysis: str = "",
        error: Optional[str] = None,
    ):
        self.success = success
        self.strategy = strategy  # "task_only" or "co_optimize"
        self.refined_task = refined_task
        self.trajectory_modified = trajectory_modified
        self.step_order = step_order or []
        self.modified_reasonings = modified_reasonings or {}
        self.analysis = analysis
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'strategy': self.strategy,
            'refined_task': self.refined_task,
            'trajectory_modified': self.trajectory_modified,
            'step_order': self.step_order,
            'modified_reasonings': self.modified_reasonings,
            'analysis': self.analysis,
            'error': self.error,
        }


class CollaborativeAgent(ExecutionAgent):
    """
    T2 Collaborative Optimization Agent.
    
    Extends ExecutionAgent with collaborative optimization capabilities:
    - When T1 returns STOP, triggers T2 for task-trajectory co-optimization
    - Supports two strategies:
      1. Task-only: Only modify task description
      2. Co-optimize: Modify both task and trajectory
    
    The key improvement over SynthAgent:
    - Instead of only refining the task (which leads to task bloat),
      we can also edit the trajectory to better align with user intent.
    """
    
    def __init__(
        self,
        config: Any,
        enable_vision: bool = True,
        history_last_k: Optional[int] = None,
        enable_collaborative_refine: bool = True,
    ):
        """
        Initialize collaborative agent.
        
        Args:
            config: Full configuration
            enable_vision: Whether to include screenshots
            history_last_k: History steps for T1 (simplified)
            enable_collaborative_refine: Whether to enable T2 refinement
        """
        super().__init__(config, enable_vision, history_last_k)
        
        self.enable_collaborative_refine = enable_collaborative_refine
        
        # Store optimization history for analysis
        self.optimization_history: List[Dict[str, Any]] = []
    
    def _handle_stop_action(
        self,
        task: str,
        current_state: StateInfo,
        previous_traj: List[LowLevelTask],
        exp_traj: ExplorationTraj,
    ) -> StopActionResult:
        """
        Handle STOP action with collaborative optimization.
        
        Overrides base implementation to use T2 model for:
        1. Analyzing why the task failed
        2. Choosing optimization strategy (task-only vs co-optimize)
        3. Refining task and/or editing trajectory
        
        Args:
            task: Current high-level task
            current_state: Current page state
            previous_traj: Previous low-level task steps
            exp_traj: Full exploration trajectory
            
        Returns:
            StopActionResult with refinement details
        """
        if not self.enable_collaborative_refine:
            return StopActionResult(refined_task=None)
        
        logger.info("Triggering collaborative optimization...")
        
        # Call T2 for collaborative optimization
        opt_result = self._collaborative_refine_step(
            task=task,
            current_state=current_state,
            previous_traj=previous_traj,
            exp_traj=exp_traj,
        )
        
        if not opt_result.success:
            logger.warning(f"Collaborative optimization failed: {opt_result.error}")
            return StopActionResult(refined_task=None)
        
        # Build StopActionResult with trajectory modification info
        # Note: We no longer apply modifications here - execute_task handles it
        should_modify = (
            opt_result.strategy == 'co_optimize' and 
            opt_result.trajectory_modified
        )
        
        return StopActionResult(
            refined_task=opt_result.refined_task,
            should_modify_trajectory=should_modify,
            step_order=opt_result.step_order if should_modify else None,
            modified_reasonings=opt_result.modified_reasonings if should_modify else None
        )
    
    def _collaborative_refine_step(
        self,
        task: str,
        current_state: StateInfo,
        previous_traj: List[LowLevelTask],
        exp_traj: ExplorationTraj,
    ) -> CollaborativeOptimizationResult:
        """
        Perform collaborative task-trajectory optimization using T2 model.
        
        This method:
        1. Formats numbered history for editing
        2. Calls T2 model with collaborative optimization prompt
        3. Parses response for task and trajectory changes
        4. Records optimization attempt
        
        Args:
            task: Current high-level task
            current_state: Current page state
            previous_traj: All executed low-level tasks
            exp_traj: Full exploration trajectory
            
        Returns:
            CollaborativeOptimizationResult with optimization details
        """
        from prompts.collaborative_optimize import (
            prompt_t2_collaborative_optimize,
            format_numbered_history,
            parse_collaborative_response,
            extract_refined_task,
            get_optimization_summary,
        )
        
        # Format numbered history for editing (step number + summary + reasoning + action)
        history_str = format_numbered_history(
            previous_traj,
            last_k=None,  # Include all history
        )
        total_steps = len(previous_traj)
        
        # Format previous high-level tasks
        prev_tasks = [t.task for t in exp_traj.high_level_tasks]
        prev_tasks_str = "\n".join(
            f"{i+1}. {t}" for i, t in enumerate(prev_tasks)
        ) if prev_tasks else "None"
        
        # Get page context
        page_context = format_page_context(current_state)
        
        # Get screenshot if enabled
        screenshot = None
        if self.enable_vision and hasattr(current_state.raw_state, 'screenshot'):
            screenshot = current_state.raw_state.screenshot
        
        # Build prompt
        messages = prompt_t2_collaborative_optimize(
            current_task=task,
            current_url=current_state.raw_state.url,
            page_context=page_context,
            previous_actions=history_str,
            previous_tasks=prev_tasks_str,
            total_steps=total_steps,
            screenshot=screenshot,
        )
        
        # Call T2 model
        try:
            response = self.t2_client.complete(
                user=messages[-1]["content"],
                json_mode=True,
            )

            if not response:
                return CollaborativeOptimizationResult(
                    success=False,
                    strategy="none",
                    error="Empty response from T2 model",
                )
            
            # The parser accepts both unified-client dictionaries and text fallbacks.
            parsed = parse_collaborative_response(response)
            if not parsed:
                return CollaborativeOptimizationResult(
                    success=False,
                    strategy="none",
                    error=f"Failed to parse T2 response: {str(response)[:200]}",
                )
            
            # Get optimization summary
            summary = get_optimization_summary(parsed, total_steps)
            
            # Log analysis
            logger.info(f"T2 Strategy: {parsed.get('strategy', 'unknown')}")
            logger.info(f"T2 Analysis: {parsed.get('analysis', '')[:200]}...")
            logger.info(f"Need refinement: {parsed.get('need_refine', False)}")
            
            # Check for trajectory changes
            step_order = parsed.get('step_order', [])
            modified_reasonings = parsed.get('modified_reasonings', {})
            original_order = list(range(1, total_steps + 1))
            trajectory_modified = (
                step_order != original_order or 
                len(modified_reasonings) > 0
            )
            
            if trajectory_modified:
                logger.info(f"Trajectory modified: step_order={step_order}")
                if modified_reasonings:
                    logger.info(f"Reasoning updates: {list(modified_reasonings.keys())}")
            
            # Record optimization attempt
            record = {
                'original_task': task,
                'strategy': parsed.get('strategy', 'task_only'),
                'analysis': parsed.get('analysis', ''),
                'need_refine': parsed.get('need_refine', False),
                'refined_task': parsed.get('refined_task', ''),
                'step_order': step_order,
                'modified_reasonings': modified_reasonings,
                'trajectory_modified': trajectory_modified,
                'original_steps': total_steps,
                'final_steps': len(step_order) if step_order else total_steps,
                'completion_estimate': parsed.get('completion_estimate', ''),
            }
            self.optimization_history.append(record)
            
            # Extract refined task
            refined_task = extract_refined_task(parsed)
            
            if refined_task:
                logger.info(f"Task refined: '{task}' -> '{refined_task}'")
            
            return CollaborativeOptimizationResult(
                success=True,
                strategy=parsed.get('strategy', 'task_only'),
                refined_task=refined_task,
                trajectory_modified=trajectory_modified,
                step_order=step_order,
                modified_reasonings=modified_reasonings,
                analysis=parsed.get('analysis', ''),
            )
            
        except Exception as e:
            logger.error(f"Collaborative optimization failed: {e}")
            return CollaborativeOptimizationResult(
                success=False,
                strategy="none",
                error=str(e),
            )
    
    def execute_task(
        self,
        task_dict: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> TaskExecutionResult:
        """
        Execute task with collaborative optimization.
        
        Extends base execute_task with additional tracking.
        
        Args:
            task_dict: Task dictionary
            max_steps: Maximum steps
            
        Returns:
            TaskExecutionResult with additional refinement info
        """
        # Track optimization history for this task
        optimization_start = len(self.optimization_history)
        
        # Execute using parent method
        result = super().execute_task(task_dict, max_steps)
        
        # Get task-specific optimizations
        task_optimizations = self.optimization_history[optimization_start:]
        if task_optimizations:
            logger.info(
                f"Task had {len(task_optimizations)} optimization attempts"
            )
            
            # Log summary
            for i, opt in enumerate(task_optimizations):
                logger.info(
                    f"  Optimization {i+1}: strategy={opt['strategy']}, "
                    f"trajectory_modified={opt['trajectory_modified']}, "
                    f"steps: {opt['original_steps']} -> {opt['final_steps']}"
                )
        
        return result
    
    def save_state(self):
        """Save state including optimization history."""
        super().save_state()
         
        # Save optimization history
        import json
        import os
        
        history_path = os.path.join(self.output_dir, 'optimization_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.optimization_history)} optimization records")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization attempts.
        
        Returns:
            Summary statistics
        """
        if not self.optimization_history:
            return {
                'total_optimizations': 0,
                'task_only_count': 0,
                'co_optimize_count': 0,
                'successful_refinements': 0,
            }
        
        task_only = sum(1 for o in self.optimization_history if o['strategy'] == 'task_only')
        co_optimize = sum(1 for o in self.optimization_history if o['strategy'] == 'co_optimize')
        successful = sum(1 for o in self.optimization_history if o.get('need_refine') and o.get('refined_task'))
        traj_modified = sum(1 for o in self.optimization_history if o.get('trajectory_modified'))
        
        # Calculate average steps removed
        steps_removed = [
            o['original_steps'] - o['final_steps']
            for o in self.optimization_history
            if o.get('trajectory_modified')
        ]
        avg_steps_removed = sum(steps_removed) / len(steps_removed) if steps_removed else 0
        
        return {
            'total_optimizations': len(self.optimization_history),
            'task_only_count': task_only,
            'co_optimize_count': co_optimize,
            'successful_refinements': successful,
            'trajectory_modifications': traj_modified,
            'avg_steps_removed': avg_steps_removed,
            'refinement_rate': successful / len(self.optimization_history),
        }
