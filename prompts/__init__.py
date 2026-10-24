"""Centralized prompt management for s3-webagent."""

# Task Synthesizer - Execution Prompts
from prompts.execution import (
    prompt_t1_execution,
    format_simplified_history,
    format_detailed_history,
    parse_action_response,
    validate_action_response,
)

# Task Synthesizer - Collaborative Optimization Prompts
from prompts.collaborative_optimize import (
    prompt_t2_collaborative_optimize,
    prompt_simple_task_refine,
    format_numbered_history,
    parse_collaborative_response,
    extract_refined_task,
    extract_recovery_suggestions,
    apply_trajectory_optimization,
    get_optimization_summary,
)

__all__ = [
    # Execution prompts
    'prompt_t1_execution',
    'format_simplified_history',
    'format_detailed_history',
    'parse_action_response',
    'validate_action_response',
    # Collaborative optimization prompts
    'prompt_t2_collaborative_optimize',
    'prompt_simple_task_refine',
    'format_numbered_history',
    'parse_collaborative_response',
    'extract_refined_task',
    'extract_recovery_suggestions',
    'apply_trajectory_optimization',
    'get_optimization_summary',
]

