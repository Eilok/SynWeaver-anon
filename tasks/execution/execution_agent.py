"""
Execution Agent Module

This module implements the T1 Execution Agent that performs:
1. Single-step CoT reasoning for action generation
2. Action execution in browser environment
3. Trajectory tracking and state management

Based on SynthAgent's ExeAgent with optimized history context.
"""

import os
import json
import time
import copy
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from tasks.models import (
    ActionType, Action, Element, StateInfo,
    LowLevelTask, HighLevelTask, ExplorationTraj,
    LowTaskStatus,
)
from tasks.execution.base import BaseExecutor
from tasks.execution.browser_adapter import format_page_context

logger = logging.getLogger(__name__)


@dataclass
class StopActionResult:
    """
    Result of STOP action handling.
    
    Contains refinement information to be applied in execute_task.
    This design separates analysis from modification, allowing execute_task
    to control when and how trajectory modifications are applied.
    
    Attributes:
        refined_task: New task description if refinement needed
        should_modify_trajectory: Whether trajectory needs modification
        step_order: New step order (1-indexed) for trajectory reordering
        modified_reasonings: Dict of step_idx -> new reasoning
    """
    refined_task: Optional[str] = None
    should_modify_trajectory: bool = False
    step_order: Optional[List[int]] = None
    modified_reasonings: Optional[Dict[str, str]] = None
    
    @property
    def has_refinement(self) -> bool:
        """Check if any refinement is needed."""
        return self.refined_task is not None


@dataclass
class TaskExecutionResult:
    """
    Result of executing a single task.
    
    Simplified version - trajectory data is saved separately via save_state().
    """
    task: str  # Final task (may be refined)
    original_task: str  # Original task from input
    start_url: str
    steps: int
    end_reason: str  # 'completed', 'not_achievable', 'exceeded_max_steps'
    refine_count: int
    auto_eval: str = 'NA'
    
    @property
    def success(self) -> bool:
        """Check if task was completed successfully."""
        return self.end_reason == 'completed'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'original_task': self.original_task,
            'start_url': self.start_url,
            'steps': self.steps,
            'end_reason': self.end_reason,
            'refine_count': self.refine_count,
            'auto_eval': self.auto_eval,
        }


class ExecutionAgent(BaseExecutor):
    """
    T1 Execution Agent for web task execution.
    
    This agent:
    1. Takes a high-level task and executes it step by step
    2. Uses CoT reasoning to determine next actions
    3. Tracks trajectory for later analysis
    4. Handles NONE (completion) and STOP (failure) actions
    
    History format is optimized to only include:
    - state_observation_summary
    - action_str
    """
    
    def __init__(
        self,
        config: Any,
        enable_vision: bool = True,
        history_last_k: Optional[int] = None,
    ):
        """
        Initialize execution agent.
        
        Args:
            config: Full configuration
            enable_vision: Whether to include screenshots in prompts
            history_last_k: Number of history steps to include (None = all)
        """
        super().__init__(config)
        
        self.enable_vision = enable_vision
        self.history_last_k = history_last_k or config.execution.max_steps
    
    def _cot_step(
        self,
        task: str,
        current_state: StateInfo,
        previous_traj: List[LowLevelTask],
    ) -> LowLevelTask:
        """
        Perform single CoT step to get next action.
        
        Uses T1 model with simplified history format.
        
        Args:
            task: High-level task description
            current_state: Current page state
            previous_traj: Previous trajectory steps
            
        Returns:
            LowLevelTask with determined action
        """
        from prompts.execution import (
            prompt_t1_execution,
            format_simplified_history,
            parse_action_response,
            validate_action_response,
        )
        
        # Format inputs
        page_context = format_page_context(current_state)
        
        # Format history - simplified format
        history_str = format_simplified_history(
            previous_traj,
            last_k=self.history_last_k,
        )
        
        # Get screenshot
        screenshot = None
        if self.enable_vision and hasattr(current_state.raw_state, 'screenshot'):
            screenshot = current_state.raw_state.screenshot
        
        # Build prompt (elements removed from prompt, but still used for element validation)
        messages = prompt_t1_execution(
            url=current_state.raw_state.url,
            page_context=page_context,
            high_level_task=task,
            previous_actions=history_str,
            screenshot=screenshot,
            history_last_k=self.history_last_k,
        )
        
        # Create failed task template
        failed_task = LowLevelTask(
            task="failed during cot_step",
            action=Action(
                action_type=ActionType.STOP,
                value="error during cot_step",
            ),
            curr_state=current_state,
            task_status=LowTaskStatus.IN_PROGRESS,
        )
        
        # Call LLM
        try:
            logger.info(f"Calling T1 model for CoT reasoning...")
            logger.debug(f"  Task: {task}")
            logger.debug(f"  URL: {current_state.raw_state.url}")
            logger.debug(f"  Elements count: {len(current_state.elements)}")
            logger.debug(f"  Vision enabled: {screenshot is not None}")
            
            response = self.t1_client.complete(
                user=messages[-1]["content"],
                json_mode=True,
            )

            if not response:
                failed_task.action.value = "Empty response from LLM"
                logger.error(failed_task.action.value)
                return failed_task
            
            logger.debug("LLM response type: %s", type(response).__name__)

            # The parser accepts both unified-client dictionaries and text fallbacks.
            data = parse_action_response(response)
            if not data:
                response_preview = str(response)[:200]
                failed_task.action.value = f"Failed to parse response: {response_preview}"
                logger.error(failed_task.action.value)
                return failed_task
            
            # Validate response
            error = validate_action_response(data)
            if error:
                failed_task.action.value = error
                logger.error(f"Invalid response: {error}")
                return failed_task
            
            # Extract action
            next_action = data['next_action']
            action_dict = next_action['action']
            action_type = ActionType(action_dict['type'].lower())
            
            # Get state summary
            state_summary = data.get('state_observation_summary', '')
            current_state.summary = str(state_summary)
            
            # Log model decision details
            reasoning = data.get('reasoning', '')
            logger.info(f"=== Model Decision ===")
            logger.info(f"  State Summary: {state_summary}")
            logger.info(f"  Reasoning: {reasoning}")
            logger.info(f"  Action: {action_dict}")
            
            # Get target element if needed
            target_element = None
            if ActionType.requires_element(action_type):
                eleid = str(action_dict['element_id'])
                eleid_to_element = {str(e.id): e for e in current_state.elements}
                
                if eleid not in eleid_to_element:
                    error_msg = f"Element ID {eleid} not found. Available: {list(eleid_to_element.keys())[:10]}"
                    failed_task.action.value = error_msg
                    logger.error(error_msg)
                    return failed_task
                
                target_element = eleid_to_element[eleid]
                logger.info(f"  Target Element: [{eleid}] {target_element.role} - {target_element.name[:50]}")
            
            # Create action
            action = Action(
                action_type=action_type,
                target_element=target_element,
                value=action_dict.get('value'),
            )
            
            # Create low-level task (task field uses action string with element content)
            low_level_task = LowLevelTask(
                task=action.get_action_str_with_element(),
                curr_state=current_state,
                action=action,
                task_status=LowTaskStatus.IN_PROGRESS,
                reasoning=reasoning,
            )
            
            logger.info(f"  Generated Action: {low_level_task.task}")
            logger.info(f"======================")
            
            return low_level_task
            
        except Exception as e:
            failed_task.action.value = f"Exception during cot_step: {str(e)}"
            logger.error(failed_task.action.value)
            return failed_task
    
    def execute_task(
        self,
        task_dict: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> TaskExecutionResult:
        """
        Execute a single high-level task.
        
        Args:
            task_dict: Task dictionary with 'task', 'start_url'
            max_steps: Maximum steps (overrides config)
            
        Returns:
            TaskExecutionResult with execution details
        """
        max_steps = max_steps or self.config.execution.max_steps
        
        task = task_dict['task']
        start_url = task_dict['start_url']  # Required field from task dict
        sites = task_dict.get('sites', [])  # Site names for auto-login
        
        logger.info(f"Executing task: {task}")
        logger.info(f"Start URL: {start_url}")
        logger.info(f"Sites: {sites}")

        # Reset browser to start URL
        current_state = self.reset_browser(start_url, sites=sites)
        
        # Initialize high-level task
        high_level_task = HighLevelTask(
            task=task,
            start_url=start_url,
            trajectories=[],
        )
        
        # Initialize exploration trajectory
        exp_traj = ExplorationTraj(
            curr_state=current_state,
            high_level_tasks=[high_level_task],
        )
        
        # Execution loop
        step_idx = 0
        failed_attempts = 0
        max_failed = self.config.execution.failed_retry
        
        while step_idx < max_steps and failed_attempts < max_failed:
            logger.info(
                f"Step {step_idx + 1}/{max_steps}, "
                f"URL: {current_state.raw_state.url}, "
                f"failed_attempts: {failed_attempts}/{max_failed}"
            )
            
            # Get next action via CoT
            next_task = self._cot_step(
                high_level_task.task,
                current_state,
                high_level_task.trajectories,
            )
            
            # Add to trajectory
            exp_traj.add_low_level_task(next_task)
            
            # Handle different action types
            if next_task.action.action_type == ActionType.NONE:
                # Task completed
                next_task.task_status = LowTaskStatus.END
                exp_traj.end_exploration()
                logger.info(f"Task completed: {high_level_task.task}")
                break
                
            elif next_task.action.action_type == ActionType.STOP:
                # Task cannot be achieved
                next_task.task_status = LowTaskStatus.NOTACHIEVEABLE
                failed_attempts += 1
                logger.info(f"STOP action, reason: {next_task.action.value}")
                
                # Get refinement result (will be overridden in collaborative agent)
                stop_result = self._handle_stop_action(
                    task=high_level_task.task,
                    current_state=current_state,
                    previous_traj=high_level_task.trajectories,
                    exp_traj=exp_traj,
                )
                
                if stop_result.has_refinement and stop_result.refined_task != high_level_task.task:
                    # Task was refined - apply trajectory modifications
                    old_task = high_level_task.task
                    
                    # Step 1: Deep copy current trajectories before modification
                    current_traj = copy.deepcopy(high_level_task.trajectories)

                    # Step 2: Apply trajectory modifications if any
                    if stop_result.should_modify_trajectory:
                        current_traj = self._apply_trajectory_modifications(
                            trajectories=current_traj,
                            step_order=stop_result.step_order,
                            modified_reasonings=stop_result.modified_reasonings,
                        )
                    
                    # Step 3: Remove last STOP step if requested
                    if current_traj:
                        last_step = current_traj[-1]
                        if last_step.action and last_step.action.action_type == ActionType.STOP:
                            current_traj = current_traj[:-1]
                            logger.info("Removed last STOP step from trajectory")
                    
                    # Step 4: Add new high-level task with modified trajectory
                    exp_traj.add_high_level_task(stop_result.refined_task, current_state)
                    high_level_task = exp_traj.high_level_tasks[-1]
                    
                    # Step 5: Transfer copied trajectory to new high-level task
                    high_level_task.trajectories = current_traj
                    
                    logger.info(f"Task refined: {old_task} -> {stop_result.refined_task}")
                    failed_attempts = 0  # Reset on successful refinement
                else:
                    # No refinement, add reflection
                    if failed_attempts < max_failed:
                        next_task.action.action_type = ActionType.REFLECT
                        next_task.action.value = (
                            f"**Failed Analysis**: {next_task.action.value}. "
                            f"**Reflection**: Consider trying a different approach."
                        )
                
            else:
                # Execute action
                next_state = self.execute_action(next_task.action, current_state)
                next_task.state_after = next_state
                current_state = next_state
                exp_traj.curr_state = current_state
            
            step_idx += 1
        
        # Determine end reason
        if high_level_task.trajectories:
            last_task = high_level_task.trajectories[-1]
            if last_task.task_status == LowTaskStatus.END:
                end_reason = 'completed'
            elif last_task.task_status == LowTaskStatus.NOTACHIEVEABLE:
                end_reason = 'not_achievable'
            else:
                end_reason = 'exceeded_max_steps'
        else:
            end_reason = 'no_steps'
        
        # Create result (simplified - trajectory saved separately via save_state)
        result = TaskExecutionResult(
            task=high_level_task.task,
            original_task=task,
            start_url=start_url,
            steps=len(high_level_task.trajectories),
            end_reason=end_reason,
            refine_count=len(exp_traj.high_level_tasks) - 1,
        )
        
        # Perform automatic evaluation if eval config exists
        eval_result = self._evaluate_task_result(task_dict, high_level_task)
        result.auto_eval = eval_result.get('auto_eval', 'NA')
        self.evaluation_results.append({"task": task_dict.get("task", ""), **eval_result})
        
        # Update stats
        self.stats.total_tasks += 1
        self.stats.total_steps += result.steps
        self.stats.total_refinements += result.refine_count
        if end_reason == 'completed':
            self.stats.completed_tasks += 1
        else:
            self.stats.failed_tasks += 1
        
        # Store trajectory for later saving
        self.exploration_trajs.append(exp_traj)
        
        logger.info(
            f"Task finished: {end_reason}, "
            f"steps={result.steps}, refines={result.refine_count}"
        )
        return result
    
    def _handle_stop_action(
        self,
        task: str,
        current_state: StateInfo,
        previous_traj: List[LowLevelTask],
        exp_traj: ExplorationTraj,
    ) -> StopActionResult:
        """
        Handle STOP action - base implementation returns no refinement.
        
        Override in collaborative agent for task refinement.
        
        Args:
            task: Current task
            current_state: Current state
            previous_traj: Previous trajectory
            exp_traj: Full exploration trajectory
            
        Returns:
            StopActionResult with refinement details
        """
        # Base implementation: no refinement
        return StopActionResult(refined_task=None)
    
    def _apply_trajectory_modifications(
        self,
        trajectories: List[LowLevelTask],
        step_order: Optional[List[int]],
        modified_reasonings: Optional[Dict[str, str]],
    ) -> List[LowLevelTask]:
        """
        Apply trajectory modifications (reorder/filter steps, update reasonings).
        
        This method is called by execute_task when STOP action handling
        returns trajectory modifications.
        
        CRITICAL: Apply reasoning modifications BEFORE reordering to avoid index mismatch.
        The modified_reasonings uses 1-indexed step numbers from the original trajectory,
        so we must update reasonings on the original trajectory first, then reorder.
        
        Args:
            trajectories: List of low-level tasks to modify (already deep-copied)
            step_order: New order of step numbers (1-indexed), None if no reorder
            modified_reasonings: Dict of step_idx (str) -> new reasoning
            
        Returns:
            Modified list of LowLevelTask
        """
        if not trajectories:
            return trajectories
        
        # Apply reasoning modifications to original trajectories FIRST
        # This must be done before reordering to avoid index mismatch
        if modified_reasonings:
            for step_key, new_reasoning in modified_reasonings.items():
                try:
                    step_idx = int(step_key) - 1  # Convert 1-indexed to 0-indexed
                    if 0 <= step_idx < len(trajectories):
                        trajectories[step_idx].reasoning = new_reasoning
                        logger.info(f"Updated reasoning for step {step_key}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to update reasoning for step {step_key}: {e}")
        
        # Then apply step reordering if specified
        if step_order:
            # step_order contains 1-indexed step numbers
            new_traj = []
            for step_num in step_order:
                idx = step_num - 1  # Convert to 0-indexed
                if 0 <= idx < len(trajectories):
                    new_traj.append(trajectories[idx])
                else:
                    logger.warning(f"Invalid step number in step_order: {step_num}")
            result_traj = new_traj
            logger.info(
                f"Trajectory reordered: {len(trajectories)} -> {len(result_traj)} steps"
            )
        else:
            result_traj = trajectories
        
        return result_traj
    
    def _evaluate_task_result(
        self,
        task_dict: Dict[str, Any],
        high_level_task: HighLevelTask,
    ) -> Dict[str, Any]:
        """
        Evaluate task result using SynthAgent-style evaluation.
        
        Supports string_match (exact, must_include, fuzzy), url_match,
        and program_html evaluation types.
        
        Args:
            task_dict: Task configuration with optional 'eval' field
            high_level_task: Executed high-level task
            
        Returns:
            Dict with 'auto_eval' status ('success', 'failed', 'NA')
        """
        result = {'auto_eval': 'NA', 'score': None, 'error': None}
        
        # Check if eval config exists
        eval_config = task_dict.get('eval')
        if not eval_config or not isinstance(eval_config, dict):
            logger.debug(f"No eval config for task: {task_dict.get('task', '')[:50]}")
            return result
        
        # Get page from browser adapter
        if not self.browser_adapter or not self.browser_adapter.page:
            logger.warning("Browser not available for evaluation")
            return result
        
        page = self.browser_adapter.page
        
        try:
            # Use the new evaluator module (SynthAgent-style)
            from tasks.execution.evaluator import TaskEvaluator
            
            # Configure LLM for evaluation
            llm_eval_config = {
                
                "temperature": 0.0,
                "max_tokens": 768,
            }
            
            evaluator = TaskEvaluator(
                llm_client=self.t2_client,
                llm_eval_config=llm_eval_config,
            )
            result = evaluator.evaluate(
                task_dict=task_dict,
                high_level_task=high_level_task,
                page=page,
                url_mapping=(self.config.environment if hasattr(self.config, "environment") else self.config.env).get_url_mapping(),
            )
            
            logger.info(f"Auto-eval result: {result.get('auto_eval', 'NA')}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result['error'] = str(e)
        
        return result
    
    def _get_last_action_summary(self, high_level_task: HighLevelTask) -> str:
        """Extract last action summary from trajectory."""
        if not high_level_task.trajectories:
            return "No actions executed"
        
        last_task = high_level_task.trajectories[-1]
        
        # Try to get summary from NONE/STOP action value
        if last_task.action and last_task.action.action_type in {ActionType.NONE, ActionType.STOP}:
            if last_task.action.value and last_task.action.value.strip():
                return last_task.action.value
        
        # Fall back to state summary
        if last_task.state_after and last_task.state_after.summary:
            return last_task.state_after.summary
        
        if last_task.curr_state and last_task.curr_state.summary:
            return last_task.curr_state.summary
        
        # Fall back to task description
        if last_task.task:
            return last_task.task
        
        return "No summary available"
