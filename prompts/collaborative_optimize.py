"""
Collaborative Optimization Prompts Module

This module provides prompt templates for T2 collaborative optimization agent.

Key Features:
1. Two optimization modes:
   - Task-only optimization: Only refine task to fit existing trajectory
   - Task-trajectory co-optimization: Edit both task and trajectory together

2. Trajectory editing operations:
   - Reorder: Change the order of steps
   - Delete: Remove specific steps
   - Edit reasoning: Update reasoning for affected steps

3. History format includes step number, state summary, reasoning, and action

Based on SynthAgent's prompt_refine_during_execution and prompt_refine_trajectory
with collaborative optimization enhancements.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .execution import ndarray_to_base64_image


def format_numbered_history(
    trajectories: List[Any],
    last_k: Optional[int] = None,
) -> str:
    """
    Format trajectory history with step numbers for editing.
    
    Each step includes:
    - Step number (1-indexed)
    - State observation summary
    - Reasoning
    - Action string
    
    Args:
        trajectories: List of LowLevelTask objects
        last_k: Number of recent steps to include
        
    Returns:
        Formatted history string with step numbers
    """
    if not trajectories:
        return "No previous actions."
    
    if last_k is None:
        last_k = len(trajectories)
    
    history_items = []
    
    for i, task in enumerate(trajectories[-last_k:]):
        step_num = i + 1
        
        # Extract fields
        summary = ""
        if hasattr(task, 'curr_state') and task.curr_state:
            summary = task.curr_state.summary or ""
        
        reasoning = task.reasoning if hasattr(task, 'reasoning') else ""
        
        # Get action string with element content for better context
        action_str = ""
        if hasattr(task, 'action') and task.action:
            if hasattr(task.action, 'get_action_str_with_element'):
                action_str = task.action.get_action_str_with_element()
            else:
                action_str = str(task.action)
        
        item = f"""[Step {step_num}]
State Summary: {summary}
Reasoning: {reasoning}
Action: {action_str}"""
        
        history_items.append(item)
    
    return "\n\n".join(history_items)


def prompt_t2_collaborative_optimize(
    current_task: str,
    current_url: str,
    page_context: str,
    previous_actions: str,
    previous_tasks: str,
    total_steps: int,
    screenshot: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Generate prompt for T2 collaborative optimization.
    
    This prompt supports two optimization modes:
    1. Task-only: Only refine the task description
    2. Co-optimize: Refine both task and trajectory (reorder/delete steps)
    
    Args:
        current_task: Current high-level task
        current_url: Current page URL
        page_context: Accessibility tree content
        previous_actions: Numbered history (format_numbered_history output)
        previous_tasks: List of previous high-level tasks
        total_steps: Total number of steps in trajectory
        screenshot: Optional screenshot array
        
    Returns:
        List of message dicts for LLM
    """
    img_info = "<image is provided in the attachment>" if screenshot is not None else "<image is not available>"
    
    # Generate original order string
    original_order = ', '.join(str(i) for i in range(1, total_steps + 1))
    
    prompt = f"""You are a GUI Web Agent expert specializing in task and trajectory optimization. The execution agent has encountered a STOP action, indicating it cannot complete the current task with the existing trajectory.

**Your Primary Goal**: Refine the task description so the execution agent can overcome its current obstacle and complete the task within 2-3 additional steps.

## Current Situation

**High-Level Task**:
"{current_task}"

**Previous High-Level Tasks Attempted**:
{previous_tasks}

**Execution Trajectory** (steps 1-{total_steps}):
{previous_actions}

**Current Page URL**:
{current_url}

**Current Page Context (Accessibility Tree)**:
{page_context}

**Screenshot**:
{img_info}

**Important Note**: The last step is a STOP action which produced no actual operation and will be automatically discarded. Focus your optimization on making the trajectory effective before that point.

---

## Optimization Strategies

Your goal is to create a refined task that enables the agent to complete its objective within 2-3 more steps. Choose between:

### Strategy 1: Task-Only Optimization
**Use when**: The executed trajectory represents meaningful progress, but the original task was too ambiguous or unachievable given the current context.

**Goal**: **Clarify user intent and align with the current trajectory, BUT DO NOT describe steps.**
The refined task should focus on *WHAT* to achieve (the goal state), not *HOW* to do it (the actions).

**Example**: 
- Bad (Instructional): "Click the blue 'Add to Cart' button and then proceed to checkout." (Too specific, loses intent)
- Good (Intent-Driven): "Add the selected item to the shopping cart." (Clear goal, lets agent decide actions)

**How to apply**:
- Change the original intent if necessary to fit the trajectory's progress.
- Clarify the *target entity* or *specific condition* required for success.
- Keep `step_order` as the original order [{original_order}] (excluding the STOP step)
- Set `modified_reasonings` to empty dict {{}}

### Strategy 2: Task-Trajectory Co-Optimization  
**Use when**: The trajectory contains redundant, incorrect, or out-of-order steps that prevent task completion, even with intent clarification.

**Example scenario**: Agent clicked wrong tabs, navigated back and forth, or executed steps in illogical order.

**How to apply**:
- Refine the task description to strictly reflect the core user intent.
- Change the original intent if necessary to fit the trajectory's progress and make it achievable in 2-3 steps.
- Remove or reorder trajectory steps to create a coherent path:
  - **Delete**: Omit step numbers (e.g., [1, 3, 5] removes steps 2, 4)
  - **Reorder**: Change sequence (e.g., [1, 4, 3, 5] swaps steps 3 and 4)
  - **Combined**: [1, 5, 6] means delete steps 2-4 and keep rest
- Update reasoning for affected steps in `modified_reasonings` to ensure logical flow

**Critical**: After modifications, the trajectory must be coherent:
- Each step should logically follow the previous one
- The reasoning should reflect the new context (e.g., "Since we removed the navigation step, now proceed directly to...")
- The modified trajectory should lead naturally to task completion in 2-3 more steps

---

## Decision Framework

Ask yourself these questions in order:

1. **Root Cause Analysis**: Why did the agent issue a STOP action? 
   - Task was impossible/invalid?
   - Task was unclear/ambiguous?
   - Trajectory led to wrong page/state?
   - Agent got stuck in a loop?

2. **Can Task-Only Optimization Succeed?**
   - Does the current page state enable task completion?
   - Will clarifying or changing the **goal state** guide the agent to success in 2-3 steps?
   - Are the executed steps (before STOP) generally correct?
   
   If YES → Use Strategy 1 (Task-Only)
   If NO → Proceed to question 3

3. **What Trajectory Issues Exist?**
   - Are there redundant/unnecessary steps?
   - Are steps in the wrong order?
   - Did early mistakes prevent success?
   
   If YES → Use Strategy 2 (Co-Optimization)

4. **Coherence Check** (for Strategy 2):
   - After reordering/deleting, does the trajectory tell a logical story?
   - Do you need to update reasoning to maintain coherence?
   - Will the modified trajectory naturally lead to completion in 2-3 steps?

---

## Output Requirements

Return a JSON object with EXACTLY this structure:
{{
    "analysis": "Detailed analysis: (1) Why STOP occurred (2) Root cause (3) Which strategy is appropriate (4) How refined task enables 2-3 step completion",
    "strategy": "task_only" or "co_optimize",
    "need_refine": true or false,
    "refined_task": "The optimized task description that enables completion in 2-3 more steps (or empty string if no refinement needed)",
    "step_order": [1, 2, ...],
    "modified_reasonings": {{
        // ONLY include steps whose reasoning needs updates to maintain trajectory coherence after reordering/deletion
        // Key: step number (as string), Value: updated reasoning text
        // Example: {{"3": "Now that we've navigated back, proceed directly to click the search button"}}
    }},
    "completion_estimate": "Specific description of the 1-3 steps needed to complete the refined task from current state"
}}

**Critical Rules**:
1. The refined task MUST be achievable within 2-3 more steps from the current page state.
2. **INTENT over INSTRUCTION**: The refined task must describe the **Goal State** (e.g., "Find the pricing page"), NOT a list of actions (e.g., "Click menu, then click pricing").
3. `step_order` must only contain integers from 1 to {total_steps} (the STOP step should be removed).
4. Each step number can appear at most once in `step_order`.
5. If `strategy` is "task_only", `step_order` must be [{original_order}].
6. For "co_optimize" strategy, ensure modified trajectory is logically coherent.
7. Update `modified_reasonings` for any step whose context changed due to reordering/deletion.

**Focus**: Prioritize enabling the agent to complete the task quickly (2-3 steps) over preserving all trajectory steps. Ensure the task remains a high-level goal, not a low-level script.

RETURN ONLY THE JSON OBJECT WITHOUT ANY COMMENTARY."""

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    if screenshot is not None:
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": ndarray_to_base64_image(screenshot)},
            },
        )
    
    return message


def prompt_simple_task_refine(
    current_task: str,
    current_url: str,
    page_context: str,
    previous_actions: str,
    previous_tasks: str,
    screenshot: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Generate simpler prompt for task-only refinement.
    
    Based on SynthAgent's prompt_refine_during_execution.
    Use this for fallback when co-optimization is not needed.
    
    Args:
        current_task: Current high-level task
        current_url: Current page URL
        page_context: Accessibility tree content
        previous_actions: Previous actions string
        previous_tasks: Previous high-level tasks
        screenshot: Optional screenshot
        
    Returns:
        List of message dicts
    """
    img_info = "<image is provided in the attachment>" if screenshot is not None else "<image is not available>"
    
    prompt = f"""You are a GUI Web Agent expert. Analyze whether the high-level task needs refinement based on the execution history.

## High-Level Task Categories
1. **Information Seeking** — Retrieve specific information
2. **Site Navigation** — Reach a specific page or state  
3. **Content Modification** — Change site content or settings

## When to REFINE
1. **Invalid Goal** - Target entity/page/action doesn't exist or conflicts with observations
2. **Insufficient Details** - Task lacks necessary parameters (username, product name, etc.)
3. **Stalled Execution** - ≥3 consecutive actions with no meaningful progress

## When NOT to REFINE
- Goal is valid and consistent with observations
- Necessary parameters are available or can be inferred
- Actions are making progress toward completion

## Current Situation

**High-Level Task**: "{current_task}"

**Current URL**: {current_url}

**Page Context**: {page_context}

**Screenshot**: {img_info}

**Previous Tasks**: {previous_tasks}

**Action History**:
{previous_actions}

## Output Requirements
{{
    "analysis": "Step-by-step analysis of whether refinement is needed",
    "need_refine": true or false,
    "refined_task": "Refined task if needed, otherwise empty string"
}}

The refined task must be:
- Context-independent (no references to "current page" or temporary observations)
- Self-contained and actionable
- Achievable in 2-3 more steps

RETURN ONLY THE JSON OBJECT."""

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    if screenshot is not None:
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": ndarray_to_base64_image(screenshot)},
            },
        )
    
    return message


# ==============================================================================
# Response Parsing
# ==============================================================================

def parse_collaborative_response(response_text: Any) -> Optional[Dict[str, Any]]:
    """
    Parse collaborative optimization response from LLM.
    
    Handles various response formats and validates structure.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Parsed dict with fields:
        - analysis: str
        - strategy: "task_only" | "co_optimize"
        - need_refine: bool
        - refined_task: str
        - step_order: List[int]
        - modified_reasonings: Dict[str, str]
        - completion_estimate: str
    """
    if not response_text:
        return None

    # Unified JSON extraction (fences / // comments / repair), then validate.
    if isinstance(response_text, dict):
        return _validate_and_normalize(response_text)
    from llm.json_parsing import parse_json
    data = parse_json(response_text)
    if isinstance(data, dict):
        return _validate_and_normalize(data)

    return None


def _validate_and_normalize(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize parsed response."""
    if not isinstance(data, dict):
        return None
    
    # Normalize need_refine
    need_refine = data.get('need_refine', False)
    if isinstance(need_refine, str):
        need_refine = need_refine.lower() in ('true', 'yes', '1')
    
    # Normalize strategy
    strategy = data.get('strategy', 'task_only')
    if strategy not in ('task_only', 'co_optimize'):
        strategy = 'task_only'
    
    # Normalize step_order
    step_order = data.get('step_order', [])
    if not isinstance(step_order, list):
        step_order = []
    step_order = [int(s) for s in step_order if isinstance(s, (int, float, str)) and str(s).replace('.', '').isdigit()]
    
    # Normalize modified_reasonings
    modified_reasonings = data.get('modified_reasonings', {})
    if not isinstance(modified_reasonings, dict):
        modified_reasonings = {}
    # Convert keys to strings
    modified_reasonings = {str(k): str(v) for k, v in modified_reasonings.items()}
    
    return {
        'analysis': str(data.get('analysis', '')),
        'strategy': strategy,
        'need_refine': need_refine,
        'refined_task': str(data.get('refined_task', '')),
        'step_order': step_order,
        'modified_reasonings': modified_reasonings,
        'completion_estimate': str(data.get('completion_estimate', '')),
    }


def extract_refined_task(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract refined task from parsed response.
    
    Args:
        response: Parsed response dict
        
    Returns:
        Refined task string or None if no refinement
    """
    if not response:
        return None
    
    if not response.get('need_refine', False):
        return None
    
    refined = response.get('refined_task', '').strip()
    return refined if refined else None


def apply_trajectory_optimization(
    trajectories: List[Any],
    step_order: List[int],
    modified_reasonings: Dict[str, str],
) -> Tuple[List[Any], List[str]]:
    """
    Apply trajectory optimization based on LLM response.
    
    This function:
    1. Reorders/filters steps according to step_order
    2. Updates reasoning for affected steps
    3. Returns the modified trajectory
    
    Args:
        trajectories: Original list of LowLevelTask objects
        step_order: New order of step numbers (1-indexed)
        modified_reasonings: Dict mapping step number (str) to new reasoning
        
    Returns:
        Tuple of (modified_trajectories, change_log)
    """
    import copy
    
    if not trajectories:
        return [], ["No trajectories to modify"]
    
    if not step_order:
        return trajectories, ["No step order provided, keeping original"]
    
    change_log = []
    original_count = len(trajectories)
    
    # Validate step_order
    valid_steps = set(range(1, original_count + 1))
    step_order = [s for s in step_order if s in valid_steps]
    
    if len(step_order) == 0:
        change_log.append("All steps filtered out, keeping original")
        return trajectories, change_log
    
    # Check if order changed
    original_order = list(range(1, original_count + 1))
    is_reordered = step_order != original_order[:len(step_order)]
    is_deleted = len(step_order) < original_count
    
    if not is_reordered and not is_deleted:
        change_log.append("No changes to trajectory order")
        # But still check for reasoning updates
        for step_key, new_reasoning in modified_reasonings.items():
            step_idx = int(step_key) - 1
            if 0 <= step_idx < len(trajectories):
                if hasattr(trajectories[step_idx], 'reasoning'):
                    trajectories[step_idx].reasoning = new_reasoning
                    change_log.append(f"Updated reasoning for step {step_key}")
        return trajectories, change_log
    
    if is_deleted:
        deleted_steps = set(original_order) - set(step_order)
        change_log.append(f"Deleted steps: {sorted(deleted_steps)}")
    if is_reordered:
        change_log.append(f"Reordered steps to: {step_order}")
    
    # Build new trajectory
    new_trajectories = []
    for new_idx, step_num in enumerate(step_order):
        # Get original task (convert to 0-indexed)
        # Create a copy to avoid mutating original
        original_task = copy.deepcopy(trajectories[step_num - 1])
        
        # Check if reasoning needs update
        step_key = str(step_num)
        if step_key in modified_reasonings:
            if hasattr(original_task, 'reasoning'):
                original_task.reasoning = modified_reasonings[step_key]
                change_log.append(f"Updated reasoning for step {step_num}")
        
        new_trajectories.append(original_task)
    
    change_log.append(f"Final trajectory: {len(new_trajectories)} steps (was {original_count})")
    
    return new_trajectories, change_log


def get_optimization_summary(response: Dict[str, Any], original_steps: int = 0) -> Dict[str, Any]:
    """
    Get a summary of the optimization decision.
    
    Args:
        response: Parsed response dict
        original_steps: Original number of steps
        
    Returns:
        Summary dict
    """
    if not response:
        return {'valid': False}
    
    strategy = response.get('strategy', 'task_only')
    need_refine = response.get('need_refine', False)
    step_order = response.get('step_order', [])
    modified_reasonings = response.get('modified_reasonings', {})
    
    # Check if trajectory actually changed
    original_order = list(range(1, original_steps + 1)) if original_steps > 0 else []
    trajectory_changed = (
        strategy == 'co_optimize' and 
        (step_order != original_order or len(modified_reasonings) > 0)
    )
    
    return {
        'valid': True,
        'strategy': strategy,
        'need_refine': need_refine,
        'task_changed': bool(response.get('refined_task', '').strip()),
        'trajectory_changed': trajectory_changed,
        'steps_deleted': original_steps - len(step_order) if original_steps > 0 else 0,
        'reasoning_updated': len(modified_reasonings),
        'analysis_preview': response.get('analysis', '')[:200],
    }


def extract_recovery_suggestions(response: Dict[str, Any]) -> List[str]:
    """
    Extract recovery suggestions from response.
    
    Args:
        response: Parsed response dict
        
    Returns:
        List of suggestion strings
    """
    if not response:
        return []
    
    suggestions = []
    
    # From completion estimate
    estimate = response.get('completion_estimate', '')
    if estimate:
        suggestions.append(f"Next steps: {estimate}")
    
    # From analysis
    analysis = response.get('analysis', '')
    if 'should' in analysis.lower() or 'recommend' in analysis.lower():
        # Extract sentences with recommendations
        sentences = analysis.split('.')
        for s in sentences:
            if 'should' in s.lower() or 'recommend' in s.lower():
                suggestions.append(s.strip())
    
    return suggestions[:5]  # Limit to 5 suggestions
