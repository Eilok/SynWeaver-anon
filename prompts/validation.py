"""
Validation Prompts Module

This module provides prompt templates for trajectory validation and LLM reconstruction.

Key Features:
1. LLM-based trajectory reconstruction for failed validations
2. Support for different failure types: MISSING_END, LOOP_DETECTED, failed trajectories
3. Trajectory editing: only deletion, reordering, and reasoning modification (no new steps)
"""

import json
from typing import List, Dict, Any, Optional

# TODO：no use
def format_trajectory_for_validation(
    trajectories: List[Any],
) -> str:
    """
    Format trajectory history for validation/reconstruction.
    
    Each step includes:
    - Step number (1-indexed)
    - State observation summary
    - Reasoning
    - Action string
    
    Args:
        trajectories: List of LowLevelTask objects or dicts
        
    Returns:
        Formatted history string with step numbers
    """
    if not trajectories:
        return "No steps in trajectory."
    
    history_items = []
    
    for i, task in enumerate(trajectories):
        step_num = i + 1
        
        # Handle both dict and object formats
        if isinstance(task, dict):
            # Extract from dict format
            curr_state = task.get('curr_state', {})
            if isinstance(curr_state, dict):
                summary = curr_state.get('summary', '')
            else:
                summary = getattr(curr_state, 'summary', '') or ''
            
            reasoning = task.get('reasoning', '')
            action = task.get('action', {})
            if isinstance(action, dict):
                action_type = action.get('action_type', '')
                target = action.get('target_element', {})
                element_id = target.get('id', '') if isinstance(target, dict) else ''
                value = action.get('value', '')
                
                # Get element's accessibility tree content for better context
                element_content = ''
                if isinstance(target, dict):
                    element_content = target.get('accessibility_tree_content', '')
                    # Remove the [id] prefix if present
                    if element_content and element_content.startswith(f"[{element_id}]"):
                        element_content = element_content[len(f"[{element_id}]"):].strip()
                    # Truncate if too long
                    if len(element_content) > 80:
                        element_content = element_content[:77] + "..."
                
                if element_id:
                    if element_content:
                        action_str = f"{action_type} [{element_id}] {element_content}"
                    else:
                        action_str = f"{action_type} [{element_id}]"
                    if action_type == 'type' and value and value.strip():
                        action_str += f" '{value}'"
                else:
                    action_str = f"{action_type}"
                    if value and value.strip():
                        action_str += f" '{value}'"
            else:
                action_str = str(action)
        else:
            # Handle object format
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


def format_validation_issues(issues: List[Dict[str, Any]]) -> str:
    """
    Format validation issues for LLM prompt.
    
    Args:
        issues: List of validation issue dicts
        
    Returns:
        Formatted issues string
    """
    if not issues:
        return "No issues detected."
    
    lines = []
    for i, issue in enumerate(issues, 1):
        issue_type = issue.get('issue_type', 'UNKNOWN')
        description = issue.get('description', '')
        step_index = issue.get('step_index')
        
        if step_index is not None:
            lines.append(f"{i}. [{issue_type}] Step {step_index + 1}: {description}")
        else:
            lines.append(f"{i}. [{issue_type}] {description}")
    
    return "\n".join(lines)


def prompt_llm_reconstruction(
    high_level_task: str,
    trajectory_history: str,
    validation_issues: str,
    total_steps: int,
) -> List[Dict[str, Any]]:
    """
    Generate prompt for LLM trajectory reconstruction.
    
    The LLM analyzes the task and trajectory, then:
    1. Identifies which steps to keep and in what order
    2. Optionally rewrites task description to match extracted trajectory
    3. Optionally updates reasoning for affected steps
    
    Args:
        high_level_task: Original high-level task description
        trajectory_history: Formatted trajectory
        validation_issues: Formatted validation issues
        total_steps: Total number of steps in trajectory
        
    Returns:
        List of message dicts for LLM
    """
    original_order = ', '.join(str(i) for i in range(1, total_steps + 1))
    
    prompt = f"""You are an expert at analyzing and reconstructing web agent trajectories. A trajectory has failed validation and needs to be reconstructed.

## Original High-Level Task
"{high_level_task}"

## Trajectory ({total_steps} steps)
{trajectory_history}

## Validation Issues Detected
{validation_issues}

---

## Your Task

Analyze the trajectory and validation issues, then reconstruct a valid task-trajectory pair by:

1. **Understanding the Context**: What was the agent trying to accomplish? What meaningful progress was made before the issues?

2. **Extracting Valid Steps**: Identify which steps represent meaningful progress and should be kept. You may:
   - **Delete** steps that are redundant, erroneous, or part of a loop
   - **Reorder** steps if needed for logical coherence
   - **Update reasoning** for steps whose context changes due to deletion/reordering
   - You **CANNOT** add new steps - only select from existing ones

3. **Handling Different Issues**:
   - **MISSING_END**: The trajectory lacks a proper NONE (completion) action. If the existing steps accomplish a meaningful sub-goal, rewrite the task to match what was achieved, then indicate we need to append a NONE action.
   - **LOOP_DETECTED**: Remove the redundant loop iterations, keeping only the first occurrence of each unique action in the pattern.
   - **Failed/STOP trajectories**: Extract the meaningful portion of the trajectory that accomplishes a coherent sub-task. Rewrite the task description to match this extracted trajectory.

4. **Ensuring Coherence**: The final trajectory must:
   - Have steps that logically follow each other
   - Match the (possibly rewritten) task description
   - End with a completion state (we'll add NONE if needed)

---

## Output Requirements

Return a JSON object with EXACTLY this structure:
{{
    "analysis": "Detailed analysis of the trajectory issues and your reconstruction strategy",
    "can_reconstruct": true or false,
    "reconstructed_task": "The task description (original or rewritten to match extracted trajectory)",
    "step_order": [list of step numbers to keep, in order, e.g., [1, 2, 4, 5] means delete step 3],
    "modified_reasonings": {{
        // Only include steps whose reasoning needs update
        // Key: original step number (as string), Value: new reasoning
        // Example: {{"4": "After clicking the search button, now viewing results..."}}
    }},
    "needs_none_action": true or false,
    "none_action_value": "Summary/answer for NONE action if needs_none_action is true",
    "reconstruction_summary": "Brief summary of what was changed and why"
}}

**Critical Rules**:
1. `step_order` must only contain integers from 1 to {total_steps}
2. Each step number can appear at most once
3. You can only delete or reorder existing steps, NOT add new ones
4. If `can_reconstruct` is false, explain why in `analysis`
5. The reconstructed trajectory should represent a coherent, completable task
6. If the trajectory is too broken to salvage, set `can_reconstruct` to false

**Example for LOOP_DETECTED**:
If steps 3-4-5-6 form a loop pattern (3-4 repeated), keep only [1, 2, 3, 4] and update reasoning if needed.

**Example for Failed Trajectory**:
If a 10-step trajectory ended in STOP but steps 1-6 accomplished "navigating to product page", rewrite task to "Navigate to the product page" and keep [1, 2, 3, 4, 5, 6].

RETURN ONLY THE JSON OBJECT WITHOUT ANY COMMENTARY."""

    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


def parse_reconstruction_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM reconstruction response.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Parsed dict with reconstruction info, or None on failure
    """
    if not response_text:
        return None

    # Unified JSON extraction (fences / // comments / repair), then validate.
    from llm.json_parsing import parse_json
    data = parse_json(response_text)
    if isinstance(data, dict):
        return _validate_reconstruction_response(data)

    return None


def _validate_reconstruction_response(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize reconstruction response."""
    if not isinstance(data, dict):
        return None
    
    # Normalize can_reconstruct
    can_reconstruct = data.get('can_reconstruct', False)
    if isinstance(can_reconstruct, str):
        can_reconstruct = can_reconstruct.lower() in ('true', 'yes', '1')
    
    # Normalize step_order
    step_order = data.get('step_order', [])
    if not isinstance(step_order, list):
        step_order = []
    step_order = [
        int(s) for s in step_order 
        if isinstance(s, (int, float, str)) and str(s).replace('.', '').isdigit()
    ]
    
    # Normalize modified_reasonings
    modified_reasonings = data.get('modified_reasonings', {})
    if not isinstance(modified_reasonings, dict):
        modified_reasonings = {}
    modified_reasonings = {str(k): str(v) for k, v in modified_reasonings.items()}
    
    # Normalize needs_none_action
    needs_none = data.get('needs_none_action', False)
    if isinstance(needs_none, str):
        needs_none = needs_none.lower() in ('true', 'yes', '1')
    
    return {
        'analysis': str(data.get('analysis', '')),
        'can_reconstruct': can_reconstruct,
        'reconstructed_task': str(data.get('reconstructed_task', '')),
        'step_order': step_order,
        'modified_reasonings': modified_reasonings,
        'needs_none_action': needs_none,
        'none_action_value': str(data.get('none_action_value', '')),
        'reconstruction_summary': str(data.get('reconstruction_summary', '')),
    }


# ==============================================================================
# ShareGPT Format Utilities
# ==============================================================================

def build_sharegpt_system_prompt() -> str:
    """
    Build the system prompt for ShareGPT training data.
    
    This should match the execution prompt format exactly.
    """
    return """You are a GUI (Graphical User Interface) Web Agent expert capable of long-horizon planning and executing high-level tasks on a website. Based on the observations and the high-level task to complete, generate the next low-level instruction.

**Available Low-Level Actions (exact JSON formats)**

Page Operation Actions:
- CLICK: {"type": "CLICK", "element_id": <int>, "value": ""}
- TYPE: {"type": "TYPE", "element_id": <int>, "value": "text to type"}
- HOVER: {"type": "HOVER", "element_id": <int>, "value": ""}
- PRESS: {"type": "PRESS", "element_id": "", "value": "key_comb"}
- SCROLL: {"type": "SCROLL", "element_id": "", "value": "up" or "down"}

Tab Management Actions:
- NEW_TAB: {"type": "NEW_TAB", "element_id": "", "value": ""}
- TAB_FOCUS: {"type": "TAB_FOCUS", "element_id": "", "value": "<tab_index>"}
- CLOSE_TAB: {"type": "CLOSE_TAB", "element_id": "", "value": ""}

URL Navigation Actions:
- GOTO: {"type": "GOTO", "element_id": "", "value": "url"}
- GO_BACK: {"type": "GO_BACK", "element_id": "", "value": ""}
- GO_FORWARD: {"type": "GO_FORWARD", "element_id": "", "value": ""}

Completion Actions:
- NONE: {"type": "NONE", "element_id": "", "value": "summary or final answer"}
- STOP: {"type": "STOP", "element_id": "", "value": "reason for stopping"}

**Output Format**
Return a JSON dictionary:
{
    "state_observation_summary": "1-3 sentence summary of current state",
    "reasoning": "step-by-step reasoning for next action",
    "next_action": {
        "action": {"type": "...", "element_id": ..., "value": "..."}
    }
}"""


def build_sharegpt_human_turn(
    high_level_task: str,
    url: str,
    page_context: str,
    previous_actions: str,
    has_image: bool = True,
) -> str:
    """
    Build human turn content for ShareGPT format.
    
    Args:
        high_level_task: The task description
        url: Current page URL
        page_context: Accessibility tree content
        previous_actions: Formatted history
        has_image: Whether image placeholder should be included
        
    Returns:
        Human turn content string
    """
    img_placeholder = "<image>\n" if has_image else ""
    
    return f"""{img_placeholder}**Information**

1. High-Level Task (your ultimate goal to finish):
"{high_level_task}"

2. History of Actions:
{previous_actions}

3. Current Page:
    - URL: {url}
    - Accessibility Tree (Page Context):
      {page_context}"""


def build_sharegpt_gpt_turn(
    state_summary: str,
    reasoning: str,
    action_dict: Dict[str, Any],
) -> str:
    """
    Build GPT turn content for ShareGPT format.
    
    Args:
        state_summary: State observation summary
        reasoning: Reasoning for the action
        action_dict: Action dictionary
        
    Returns:
        GPT turn content string (JSON format)
    """
    response = {
        "state_observation_summary": state_summary or "",
        "reasoning": reasoning or "",
        "next_action": {
            "action": action_dict
        }
    }
    return json.dumps(response, ensure_ascii=False, indent=2)


def build_validation_execution_prompt(
    high_level_task: str,
    url: str,
    page_context: str,
    previous_actions: str,
    has_image: bool = True,
) -> str:
    """Build the execution prompt used by validation ShareGPT export."""
    image_information = (
        "<image is provided in the attachment>"
        if has_image
        else "<image is not available>"
    )
    image_placeholder = "<image>\n" if has_image else ""
    return f"""{image_placeholder}You are a GUI (Graphical User Interface) Web Agent expert capable of long-horizon planning and executing high-level tasks on a website. Based on the observations and the high-level task to complete, generate the next low-level instruction.

**Information**

1. High-Level Task (your ultimate goal to finish):
"{high_level_task}"

2. History of Actions (from oldest to most recent):
{previous_actions}

3. Current Page (only current view, not full page, you may need to scroll to see more):
    - URL:
      {url}

    - Accessibility Tree (Page Context):
      {page_context}
      
    - Screenshot (only current view, not full page):
      {image_information}
      
**Available Low-Level Actions (exact JSON formats)**

Page Operation Actions:
- CLICK:
  {{"type": "CLICK", "element_id": <int>, "value": ""}}

- TYPE (default behavior is to press Enter after typing unless you explicitly set press_enter_after to 0):
  {{"type": "TYPE", "element_id": <int>, "value": "text to type <string>"}}

- HOVER:
  {{"type": "HOVER", "element_id": <int>, "value": ""}}

- PRESS (keyboard shortcut, e.g., Ctrl+V / Cmd+V):
  {{"type": "PRESS", "element_id": "", "value": "key_comb <string>"}}

- SCROLL:
  {{"type": "SCROLL", "element_id": "", "value": "up" or "down"}}

Tab Management Actions:
- NEW_TAB (open a new, empty browser tab):
  {{"type": "NEW_TAB", "element_id": "", "value": ""}}

- TAB_FOCUS (switch focus to a specific tab by its index, starting from 0):
  {{"type": "TAB_FOCUS", "element_id": "", "value": "<tab_index as int>"}}

- CLOSE_TAB (close the currently active tab):
  {{"type": "CLOSE_TAB", "element_id": "", "value": ""}}

URL Navigation Actions:
- GOTO (navigate directly to a URL):
  {{"type": "GOTO", "element_id": "", "value": "url <string>"}}

- GO_BACK:
  {{"type": "GO_BACK", "element_id": "", "value": ""}}

- GO_FORWARD:
  {{"type": "GO_FORWARD", "element_id": "", "value": ""}}

Completion Actions:
- NONE (use ONLY when the task is completed and you have the final answer):
  {{"type": "NONE", "element_id": "", "value": "summary content or final answer <string>"}}

- STOP (use ONLY when the task is impossible to complete):
  {{"type": "STOP", "element_id": "", "value": "the reason for the STOP <string>"}}

**Critical Rules for Success**
1. You must issue only actions that are valid given the current observation (accessibility tree and screenshot).
2. Only propose ONE atomic action; actions must be executable independently.
3. Prefer actions grounded by element IDs present in the accessibility tree when using CLICK/TYPE/HOVER.
4. You MUST provide meaningful and non-empty value if the action type is in {{TYPE, SCROLL, GOTO, NONE, STOP, TAB_FOCUS}}.
5. When you believe the task is complete (e.g., you have the answer), use NONE with the final answer in value.
6. Be concise and avoid redundant or risky actions; ensure each action clearly advances the task.
7. Use STOP only when:
   - The task lacks necessary information
   - The target does not exist (hallucination)
   - The task is harmful or inappropriate
   - Multiple attempts (>=3) have failed to make progress

8. You MUST first generate a "state_observation_summary" to observe the current environment, then take a step-by-step "reasoning" to decide the next action.
9. The high-level task often requires MULTIPLE steps to complete. Do NOT expect to finish in a single action.
10. You MUST actively decide the next step. Do NOT choose "NONE" or "STOP" unless you are sure the task is finished or impossible.
11. Choose element IDs from the accessibility tree and use them directly in CLICK/TYPE/HOVER actions.
12. If the page does not change after an action, try scrolling to see more elements.
13. When typing dates, use the format "MM/DD/YYYY".

**Output Requirements**
You MUST return a JSON dictionary with the following format (no extra commentary):
{{
    "state_observation_summary": "your 1-3 sentence summary of the current state relevant to the task",
    "reasoning": "your step-by-step reasoning to decide the next action",
    "next_action": {{
        "action": {{"type": "XXXX", "element_id": <int or "">, "value": <string or "">}}
    }}
}}
RETURN ONLY THE DICTIONARY WITHOUT ANY COMMENTARY."""
