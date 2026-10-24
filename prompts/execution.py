"""
Execution Prompts Module

This module provides prompt templates for T1 execution agent.
Key features:
1. Simplified history format (only state_observation_summary + action_str)
2. Action generation with CoT reasoning
3. Support for both vision and text-only modes

Based on SynthAgent's prompts.py with optimizations for reduced context length.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union


def ndarray_to_base64_image(img: np.ndarray) -> str:
    """
    Convert numpy array to base64 image URL.

    Thin wrapper delegating to common.llm.image_encoding.to_data_url.
    Returns "" for a missing/unencodable image (unchanged behaviour).
    """
    from llm.image_encoding import to_data_url
    if img is None:
        return ""
    return to_data_url(img) or ""


def prompt_t1_execution(
    url: str,
    page_context: str,
    high_level_task: str,
    previous_actions: str,
    screenshot: Optional[np.ndarray] = None,
    history_last_k: Optional[int] = None,
    elements: str = "",  # Deprecated, kept for backward compatibility
) -> List[Dict[str, Any]]:
    """
    Generate prompt for T1 execution agent.
    
    This prompt is used for single-step CoT reasoning and action generation.
    History format is simplified to only include state_observation_summary and action_str.
    
    Args:
        url: Current page URL
        page_context: Accessibility tree content
        high_level_task: High-level task description
        previous_actions: Formatted history (simplified format)
        screenshot: Optional screenshot array
        history_last_k: Number of history steps shown
        elements: Deprecated, kept for backward compatibility (not used in prompt)
        
    Returns:
        List of message dicts for LLM
    """
    if history_last_k is None:
        hint_for_history = "from oldest to most recent"
    else:
        hint_for_history = f"showing only the {history_last_k} most recent, listed from oldest to newest"
    
    img_info = "<image is provided in the attachment>" if screenshot is not None else "<image is not available>"
    
    prompt = f"""You are a GUI (Graphical User Interface) Web Agent expert capable of long-horizon planning and executing high-level tasks on a website. Based on the observations and the high-level task to complete, generate the next low-level instruction.

**Information**

1. High-Level Task (your ultimate goal to finish):
"{high_level_task}"

2. History of Actions ({hint_for_history}):
{previous_actions}

3. Current Page (only current view, not full page, you may need to scroll to see more):
    - URL:
      {url}

    - Accessibility Tree (Page Context):
      {page_context}
      
    - Screenshot (only current view, not full page):
      {img_info}
      
**Available Low-Level Actions (exact JSON formats)**

Page Operation Actions:
- CLICK:
  {{"type": "CLICK", "element_id": <int>, "value": ""}}

- TYPE (default behavior is to press Enter after typing unless you explicitly set press_enter_after to 0):
  {{"type": "TYPE", "element_id": <int>, "value": "text to type <string>"}}

- HOVER:
  {{"type": "HOVER", "element_id": <int>, "value": ""}}

- PRESS (keyboard shortcut. The value MUST be either exactly ONE key, e.g., ArrowDown, or ONE key chord joined by "+", e.g., Ctrl+V / Cmd+V):
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


def format_simplified_history(
    trajectories: List[Any],
    last_k: Optional[int] = None,
) -> str:
    """
    Format trajectory history in simplified format.
    
    Only includes:
    - state_observation_summary: Current state summary
    - action_str: Executed action string
    
    This reduces context length significantly compared to full history.
    
    Args:
        trajectories: List of LowLevelTask objects
        last_k: Number of recent steps to include
        
    Returns:
        Formatted history string
    """
    if not trajectories:
        return "No previous actions."
    
    if last_k is None:
        last_k = len(trajectories)
    
    history_items = []
    
    for task in trajectories[-last_k:]:
        # Only include executed steps (with state_after)
        if hasattr(task, 'state_after') and task.state_after is not None:
            # Get action string with element content for better context
            action_str = ""
            if hasattr(task, 'action') and task.action:
                if hasattr(task.action, 'get_action_str_with_element'):
                    action_str = task.action.get_action_str_with_element()
                else:
                    action_str = str(task.action)
            
            item = {
                "state_observation_summary": (
                    task.curr_state.summary if hasattr(task.curr_state, 'summary') else ""
                ) or "",
                "action_str": action_str,
            }
            history_items.append(item)
    
    if not history_items:
        return "No previous actions."
    
    return "\n\n".join([str(item) for item in history_items])


def format_detailed_history(
    trajectories: List[Any],
    last_k: Optional[int] = None,
    include_all: bool = False,
) -> str:
    """
    Format trajectory history with full details.
    
    Used for T2 collaborative optimization where more context is needed.
    
    Args:
        trajectories: List of LowLevelTask objects
        last_k: Number of recent steps to include
        include_all: Include steps even if not executed
        
    Returns:
        Formatted history string with full details
    """
    if not trajectories:
        return "No previous actions."
    
    if last_k is None:
        last_k = len(trajectories)
    
    history_items = []
    
    for i, task in enumerate(trajectories[-last_k:]):
        # Check if step was executed
        executed = hasattr(task, 'state_after') and task.state_after is not None
        
        if executed or include_all:
            # Get action string with element content for better context
            action_str = ""
            if hasattr(task, 'action') and task.action:
                if hasattr(task.action, 'get_action_str_with_element'):
                    action_str = task.action.get_action_str_with_element()
                else:
                    action_str = str(task.action)
            
            item = {
                "step": i + 1,
                "state_observation_summary": (
                    task.curr_state.summary if hasattr(task.curr_state, 'summary') else ""
                ) or "",
                "current_url": (
                    task.curr_state.raw_state.url 
                    if hasattr(task.curr_state, 'raw_state') and task.curr_state.raw_state 
                    else ""
                ),
                "low_level_instruction": task.task if hasattr(task, 'task') else "",
                "action_str": action_str,
                "reasoning": task.reasoning if hasattr(task, 'reasoning') else "",
            }
            history_items.append(item)
    
    if not history_items:
        return "No previous actions."
    
    return "\n\n".join([str(item) for item in history_items])


# ==============================================================================
# Action Parsing Utilities
# ==============================================================================

def parse_action_response(response_text: Any) -> Optional[Dict[str, Any]]:
    """
    Parse action response from LLM.

    Delegates JSON extraction to common.llm.json_parsing.parse_json (unified
    implementation). Extracts state_observation_summary / reasoning / next_action.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed dict or None on failure
    """
    if isinstance(response_text, dict):
        return response_text
    from llm.json_parsing import parse_json
    result = parse_json(response_text)
    return result if isinstance(result, dict) else None


def validate_action_response(data: Dict[str, Any]) -> Optional[str]:
    """
    Validate parsed action response.
    
    Args:
        data: Parsed response dict
        
    Returns:
        Error message if invalid, None if valid
    """
    if not isinstance(data, dict):
        return f"Expected dict, got {type(data)}"
    
    if 'next_action' not in data:
        return "Missing 'next_action' field"
    
    next_action = data['next_action']
    if not isinstance(next_action, dict):
        return f"Expected 'next_action' to be dict, got {type(next_action)}"
    
    if 'action' not in next_action:
        return "Missing 'action' in 'next_action'"
    
    action = next_action['action']
    if not isinstance(action, dict):
        return f"Expected 'action' to be dict, got {type(action)}"
    
    required_fields = ['type', 'element_id', 'value']
    for field in required_fields:
        if field not in action:
            return f"Missing '{field}' in action"
    
    return None
