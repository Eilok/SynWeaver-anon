"""ShareGPT trajectory export."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompts.validation import build_validation_execution_prompt
from storage.io import jsonl_save
from tasks.models import HighLevelTask, LowLevelTask

logger = logging.getLogger(__name__)


@dataclass
class ShareGPTSample:
    """Represent one ShareGPT training sample."""

    sample_id: str
    images: list[str] = field(default_factory=list)
    conversations: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the sample to its output schema."""
        result: dict[str, Any] = {
            "id": self.sample_id,
            "conversations": self.conversations,
        }
        if self.images:
            result["images"] = self.images
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class ShareGPTConverter:
    """Convert every trajectory step into one ShareGPT sample."""

    def __init__(
        self,
        output_dir: str | Path,
        images_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir) if images_dir else self.output_dir / "images"
        self._sample_counter = 0

    def _resolve_screenshot_path(
        self,
        saved_path: str,
        source_screenshots_dir: str | Path | None = None,
    ) -> Path | None:
        source_path = Path(saved_path)
        if source_path.exists():
            return source_path
        if source_screenshots_dir:
            alternative = Path(source_screenshots_dir) / source_path.name
            if alternative.exists():
                return alternative
        logger.warning(
            "Screenshot not found: %s (also checked %s)",
            saved_path,
            source_screenshots_dir or "no fallback dir",
        )
        return None

    def convert_trajectory(
        self,
        high_level_task: HighLevelTask,
        traj_uuid: str,
        source_screenshots_dir: str | Path | None = None,
    ) -> list[ShareGPTSample]:
        """Convert each step while preserving its preceding trajectory context."""
        samples: list[ShareGPTSample] = []
        trajectories = high_level_task.trajectories
        for step_index, step in enumerate(trajectories):
            self._sample_counter += 1
            sample_id = f"{traj_uuid}_step_{step_index + 1}"
            history = self._format_history(trajectories[:step_index])
            current_state = step.curr_state
            raw_state = current_state.raw_state if current_state else None
            url = raw_state.url if raw_state else ""
            page_context = raw_state.accessibility_tree if raw_state else ""

            image_path = None
            saved_screenshot = raw_state.screenshot if raw_state else None
            if saved_screenshot and isinstance(saved_screenshot, str):
                source_path = self._resolve_screenshot_path(
                    saved_screenshot,
                    source_screenshots_dir,
                )
                if source_path and source_path.exists():
                    destination_name = f"{traj_uuid}_step_{step_index + 1}.png"
                    destination_path = self.images_dir / destination_name
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                    image_path = f"images/{destination_name}"

            human_content = self._build_execution_prompt(
                high_level_task=high_level_task.task,
                url=url,
                page_context=page_context,
                previous_actions=history,
                has_image=image_path is not None,
            )
            response = {
                "state_observation_summary": (
                    current_state.summary or "" if current_state else ""
                ),
                "reasoning": step.reasoning or "",
                "next_action": {"action": step.action.to_json_dict()},
            }
            samples.append(
                ShareGPTSample(
                    sample_id=sample_id,
                    images=[image_path] if image_path else [],
                    conversations=[
                        {"from": "human", "value": human_content},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                response,
                                ensure_ascii=False,
                                indent=2,
                            ),
                        },
                    ],
                    metadata={
                        "task": high_level_task.task,
                        "step_index": step_index,
                        "total_steps": len(trajectories),
                        "action_type": step.action.action_type.value,
                        "traj_uuid": traj_uuid,
                    },
                )
            )
        return samples

    def _format_history(self, previous_steps: list[LowLevelTask]) -> str:
        if not previous_steps:
            return "No previous actions."
        items = []
        for step in previous_steps:
            summary = step.curr_state.summary or "" if step.curr_state else ""
            if hasattr(step.action, "get_action_str_with_element"):
                action_string = step.action.get_action_str_with_element()
            elif hasattr(step.action, "get_action_str"):
                action_string = step.action.get_action_str()
            else:
                action_string = str(step.action)
            items.append(
                {
                    "state_observation_summary": summary,
                    "action_str": action_string,
                }
            )
        return "\n\n".join(str(item) for item in items)

    def _build_execution_prompt(
        self,
        high_level_task: str,
        url: str,
        page_context: str,
        previous_actions: str,
        has_image: bool = True,
    ) -> str:
        """Build the execution prompt through the centralized prompt helper."""
        return build_validation_execution_prompt(
            high_level_task=high_level_task,
            url=url,
            page_context=page_context,
            previous_actions=previous_actions,
            has_image=has_image,
        )

    def save_samples(
        self,
        samples: list[ShareGPTSample],
        filename: str = "train_sharegpt.jsonl",
        append: bool = True,
    ) -> str:
        """Write samples using the shared JSONL storage helper."""
        output_path = jsonl_save(
            [sample.to_dict() for sample in samples],
            self.output_dir / filename,
            append=append,
        )
        logger.info("Saved %s samples to %s", len(samples), output_path)
        return str(output_path)
