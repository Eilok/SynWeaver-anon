from typing import Dict, Any

def extract_step_essentials_from_trajectory(step_num: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential information from a single step."""
    simplified_step = {
        "step": int(step_num),
        "evaluation": None,
        "memory": None,
        "goal": None,
        "action_details": None,
    }
    
    # Extract goal, evaluation, and memory from output messages
    output_messages = step_data.get("output_messages", {})
    tool_call_message = output_messages.get("tool_call_message", {})
    
    if "tool_calls" in tool_call_message:
        tool_calls = tool_call_message["tool_calls"]
        if tool_calls and len(tool_calls) > 0:
            last_tool_call = tool_calls[-1]
            args = last_tool_call.get("args", {})
            
            current_state = args.get("current_state", {})
            simplified_step["goal"] = current_state.get("next_goal")
            simplified_step["evaluation"] = current_state.get("evaluation_previous_goal")
            simplified_step["memory"] = current_state.get("memory")
            
            # Extract actions and their arguments
            actions = args.get("action", [])
            if actions:
                action_details = []
                for action in actions:
                    if isinstance(action, dict):
                        for action_name, action_params in action.items():
                            action_details.append({
                                "action": action_name,
                                "params": action_params
                            })
                simplified_step["action_details"] = action_details
    
    return simplified_step

