"""LLM-based trajectory reconstruction."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from llm import get_client
from prompts.collaborative_optimize import format_numbered_history
from prompts.validation import (
    format_validation_issues,
    parse_reconstruction_response,
    prompt_llm_reconstruction,
)
from tasks.models import (
    Action,
    ActionExecuteStatus,
    ActionType,
    HighLevelTask,
    LowLevelTask,
    LowTaskStatus,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class LLMReconstructor:
    """Reconstruct invalid trajectories through the unified T2 client."""

    def __init__(
        self,
        *,
        conf_path: str = "./config.yaml",
        model_id: str | None = None,
        enable_reconstruction: bool = True,
    ) -> None:
        self.enable_reconstruction = enable_reconstruction
        self.client = None
        if enable_reconstruction:
            overrides = {"model": model_id} if model_id is not None else None
            self.client = get_client(
                "t2",
                conf_path=conf_path,
                overrides=overrides,
                fresh=True,
            )

    def reconstruct(
        self,
        high_level_task: HighLevelTask,
        validation_result: ValidationResult,
    ) -> HighLevelTask | None:
        """Attempt to reconstruct one invalid trajectory."""
        if not self.enable_reconstruction or self.client is None:
            logger.info("LLM reconstruction disabled")
            return None
        if not validation_result.issues:
            logger.info("No issues to reconstruct")
            return high_level_task

        messages = prompt_llm_reconstruction(
            high_level_task=high_level_task.task,
            trajectory_history=format_numbered_history(high_level_task.trajectories),
            validation_issues=format_validation_issues(
                [issue.to_dict() for issue in validation_result.issues]
            ),
            total_steps=len(high_level_task.trajectories),
        )
        try:
            logger.info("Calling LLM for trajectory reconstruction")
            response = self.client.complete(
                user=messages[0]["content"],
                json_mode=True,
            )
            if not response:
                logger.error("Empty response from LLM")
                return None
            response_text = (
                json.dumps(response, ensure_ascii=False)
                if isinstance(response, (dict, list))
                else str(response)
            )
            result = parse_reconstruction_response(response_text)
            if not result:
                logger.error(
                    "Failed to parse reconstruction response: %s",
                    response_text[:500],
                )
                return None
            if not result.get("can_reconstruct", False):
                logger.info(
                    "LLM determined trajectory cannot be reconstructed: %s",
                    result.get("analysis", "")[:200],
                )
                return None
            reconstructed = self._apply_reconstruction(high_level_task, result)
            logger.info(
                "Trajectory reconstructed: %s",
                result.get("reconstruction_summary", ""),
            )
            return reconstructed
        except Exception as error:
            logger.error("LLM reconstruction failed: %s", error)
            return None

    def _apply_reconstruction(
        self,
        high_level_task: HighLevelTask,
        result: dict[str, Any],
    ) -> HighLevelTask:
        trajectories_copy = [
            copy.deepcopy(step) for step in high_level_task.trajectories
        ]
        for step_key, new_reasoning in result.get("modified_reasonings", {}).items():
            try:
                step_index = int(step_key) - 1
                if 0 <= step_index < len(trajectories_copy):
                    trajectories_copy[step_index].reasoning = new_reasoning
            except (ValueError, IndexError) as error:
                logger.warning(
                    "Failed to apply reasoning for step %s: %s",
                    step_key,
                    error,
                )

        new_trajectories: list[LowLevelTask] = []
        for step_number in result.get("step_order", []):
            index = step_number - 1
            if 0 <= index < len(trajectories_copy):
                new_trajectories.append(trajectories_copy[index])
            else:
                logger.warning(
                    "Invalid step number in reconstruction: %s",
                    step_number,
                )

        none_value = result.get("none_action_value", "")
        if result.get("needs_none_action", False) and new_trajectories:
            last_step = new_trajectories[-1]
            if last_step.action.action_type != ActionType.NONE:
                new_trajectories.append(
                    LowLevelTask(
                        task="Complete the task",
                        curr_state=last_step.state_after or last_step.curr_state,
                        action=Action(
                            action_type=ActionType.NONE,
                            value=none_value or "Task completed",
                            status=ActionExecuteStatus.SUCCESS,
                        ),
                        state_after=None,
                        task_status=LowTaskStatus.END,
                        reasoning=f"The task has been completed. {none_value}",
                    )
                )

        return HighLevelTask(
            task=result.get("reconstructed_task", high_level_task.task),
            trajectories=new_trajectories,
            start_url=high_level_task.start_url,
        )

    def close(self) -> None:
        """Close the unified LLM client once when it exists."""
        client, self.client = self.client, None
        if client is not None:
            client.close()
