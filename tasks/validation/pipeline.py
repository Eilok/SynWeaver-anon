"""Trajectory validation, reconstruction, and export pipeline."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.tasks import ValidationRunConfig
from storage.io import write_json
from tasks.models import (
    Action,
    ActionExecuteStatus,
    ActionType,
    Element,
    HighLevelTask,
    LowLevelTask,
    LowTaskStatus,
    RawState,
    StateInfo,
)
from tasks.validation.export import ShareGPTConverter, ShareGPTSample
from tasks.validation.heuristic import HeuristicValidator
from tasks.validation.reconstruct import LLMReconstructor

logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Store thread-safe statistics for one validation run."""

    total_trajectories: int = 0
    passed_validation: int = 0
    failed_validation: int = 0
    reconstructed: int = 0
    reconstruction_failed: int = 0
    total_samples_generated: int = 0
    total_reconstruction_iterations: int = 0
    max_iterations_reached: int = 0
    skipped_validation: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def increment(self, field_name: str, value: int = 1) -> None:
        """Increment one statistic while holding the shared lock."""
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + value)

    def to_dict(self) -> dict[str, Any]:
        """Return the validation report statistics schema."""
        successful = (
            self.passed_validation
            + self.reconstructed
            + self.skipped_validation
        )
        return {
            "total_trajectories": self.total_trajectories,
            "passed_validation": self.passed_validation,
            "failed_validation": self.failed_validation,
            "reconstructed": self.reconstructed,
            "reconstruction_failed": self.reconstruction_failed,
            "total_samples_generated": self.total_samples_generated,
            "total_reconstruction_iterations": self.total_reconstruction_iterations,
            "max_iterations_reached": self.max_iterations_reached,
            "skipped_validation": self.skipped_validation,
            "success_rate": (
                successful / self.total_trajectories
                if self.total_trajectories > 0
                else 0
            ),
        }


class TrajectoryValidationPipeline:
    """Validate execution trajectories and export ShareGPT samples."""

    def __init__(self, config: ValidationRunConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.num_workers = config.validation.num_workers
        self.validator = HeuristicValidator(
            config=config.validation,
            max_steps=config.max_steps,
        )
        reconstruction_enabled = config.validation.mode == "reconstruct"
        self.reconstructor = (
            LLMReconstructor(
                conf_path=config.validation.config_path,
                model_id=config.model_override,
                enable_reconstruction=True,
            )
            if reconstruction_enabled
            else None
        )
        self.converter = ShareGPTConverter(
            output_dir=self.output_dir,
            images_dir=self.output_dir / "images",
        )
        self.stats = ValidationStats()

    def close(self) -> None:
        """Close resources owned by the validation pipeline."""
        if self.reconstructor is not None:
            self.reconstructor.close()

    def __enter__(self) -> "TrajectoryValidationPipeline":
        """Return this pipeline as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close owned resources when leaving a context."""
        self.close()

    def run(
        self,
        input_dir: str | Path | None = None,
        output_filename: str | None = None,
    ) -> ValidationStats:
        """Run discovery, validation, reconstruction, and export."""
        input_path = Path(input_dir) if input_dir is not None else self.config.input_dir
        filename = output_filename or self.config.validation.output_filename
        logger.info("Starting validation pipeline on %s", input_path)
        logger.info("Using %s parallel worker(s)", self.num_workers)

        worker_dirs = sorted(
            directory
            for directory in input_path.iterdir()
            if directory.is_dir() and directory.name.startswith("worker_")
        )
        if not worker_dirs:
            if (input_path / "trajectories.jsonl").exists():
                worker_dirs = [input_path]
            else:
                logger.warning("No worker directories found in %s", input_path)
                return self.stats
        logger.info("Found %s worker directories", len(worker_dirs))

        failed_task_ids = self._load_failed_task_ids(input_path)
        if failed_task_ids:
            logger.info(
                "Loaded %s failed task IDs from execution_results.json",
                len(failed_task_ids),
            )

        validation_tasks: list[dict[str, Any]] = []
        for worker_dir in worker_dirs:
            trajectories_file = worker_dir / "trajectories.jsonl"
            screenshots_dir = worker_dir / "screenshots"
            if not trajectories_file.exists():
                logger.warning("No trajectories.jsonl in %s", worker_dir)
                continue
            trajectories = self._load_trajectories(trajectories_file)
            logger.info(
                "Loaded %s trajectories from %s",
                len(trajectories),
                worker_dir.name,
            )
            for trajectory_data in trajectories:
                validation_tasks.append(
                    {
                        "traj_data": trajectory_data,
                        "screenshots_dir": screenshots_dir,
                        "failed_task_ids": failed_task_ids,
                    }
                )

        total_tasks = len(validation_tasks)
        logger.info("Total trajectories to validate: %s", total_tasks)
        all_samples: list[ShareGPTSample] = []
        validation_report: list[dict[str, Any]] = []

        if self.num_workers > 1 and total_tasks > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._validate_single_trajectory, task): task
                    for task in validation_tasks
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                        if result:
                            samples, report_entry = result
                            all_samples.extend(samples)
                            validation_report.append(report_entry)
                        if completed % 10 == 0 or completed == total_tasks:
                            logger.info(
                                "Progress: %s/%s trajectories validated",
                                completed,
                                total_tasks,
                            )
                    except Exception as error:
                        logger.error("Validation failed with exception: %s", error)
        else:
            for index, task in enumerate(validation_tasks, 1):
                result = self._validate_single_trajectory(task)
                if result:
                    samples, report_entry = result
                    all_samples.extend(samples)
                    validation_report.append(report_entry)
                if index % 10 == 0 or index == total_tasks:
                    logger.info(
                        "Progress: %s/%s trajectories validated",
                        index,
                        total_tasks,
                    )

        if all_samples:
            self.converter.save_samples(all_samples, filename, append=False)
            self.stats.increment("total_samples_generated", len(all_samples))

        write_json(
            {
                "stats": self.stats.to_dict(),
                "trajectories": validation_report,
                "timestamp": datetime.now().isoformat(),
            },
            self.output_dir / "validation_report.json",
        )
        logger.info("Validation complete: %s", self.stats.to_dict())
        return self.stats

    def _validate_single_trajectory(
        self,
        task: dict[str, Any],
    ) -> tuple[list[ShareGPTSample], dict[str, Any]] | None:
        trajectory_data = task["traj_data"]
        screenshots_dir: Path = task["screenshots_dir"]
        failed_task_ids: set[str] = task["failed_task_ids"]
        self.stats.increment("total_trajectories")

        task_id = trajectory_data.get("task_id", "")
        is_failed_task = task_id in failed_task_ids
        high_level_tasks = trajectory_data.get("high_level_tasks", [])
        if not high_level_tasks:
            logger.warning(
                "No high-level tasks in trajectory %s",
                trajectory_data.get("uuid", "unknown"),
            )
            return None

        last_task_data = high_level_tasks[-1]
        trajectory_uuid = trajectory_data.get(
            "uuid",
            f"traj_{id(trajectory_data)}",
        )
        high_level_task = self._parse_high_level_task(last_task_data)
        if not high_level_task or not high_level_task.trajectories:
            logger.warning("Empty trajectory in %s", trajectory_uuid)
            return None
        screenshot_source = (
            str(screenshots_dir) if screenshots_dir.exists() else None
        )

        if self.config.validation.mode == "export":
            self.stats.increment("skipped_validation")
            samples = self.converter.convert_trajectory(
                high_level_task=high_level_task,
                traj_uuid=trajectory_uuid,
                source_screenshots_dir=screenshot_source,
            )
            return samples, {
                "uuid": trajectory_uuid,
                "task_id": task_id,
                "task": high_level_task.task,
                "num_steps": len(high_level_task.trajectories),
                "is_failed_task": is_failed_task,
                "passed": None,
                "issues": [],
                "action": "converted_unvalidated",
            }

        result = self.validator.validate(
            high_level_task,
            is_failed_task=is_failed_task,
        )
        report_entry: dict[str, Any] = {
            "uuid": trajectory_uuid,
            "task_id": task_id,
            "task": high_level_task.task,
            "num_steps": len(high_level_task.trajectories),
            "is_failed_task": is_failed_task,
            "passed": result.passed,
            "issues": [issue.to_dict() for issue in result.issues],
        }
        samples: list[ShareGPTSample] = []

        if result.passed:
            self.stats.increment("passed_validation")
            samples = self.converter.convert_trajectory(
                high_level_task=high_level_task,
                traj_uuid=trajectory_uuid,
                source_screenshots_dir=screenshot_source,
            )
            report_entry["action"] = "converted"
            return samples, report_entry

        self.stats.increment("failed_validation")
        current_task = high_level_task
        current_result = result
        iteration = 0
        max_loops = self.config.validation.max_validation_loops
        for iteration in range(1, max_loops + 1):
            logger.info(
                "[%s] Reconstruction iteration %s/%s",
                trajectory_uuid,
                iteration,
                max_loops,
            )
            reconstructed = (
                self.reconstructor.reconstruct(
                    high_level_task=current_task,
                    validation_result=current_result,
                )
                if self.reconstructor is not None
                else None
            )
            if reconstructed is None:
                logger.info(
                    "[%s] Reconstruction returned None at iteration %s, stopping",
                    trajectory_uuid,
                    iteration,
                )
                break

            self.stats.increment("total_reconstruction_iterations")
            current_result = self.validator.validate(
                reconstructed,
                is_failed_task=False,
            )
            if current_result.passed:
                self.stats.increment("reconstructed")
                samples = self.converter.convert_trajectory(
                    high_level_task=reconstructed,
                    traj_uuid=f"{trajectory_uuid}_reconstructed",
                    source_screenshots_dir=screenshot_source,
                )
                report_entry["action"] = "reconstructed"
                report_entry["reconstructed_task"] = reconstructed.task
                report_entry["reconstructed_steps"] = len(
                    reconstructed.trajectories
                )
                report_entry["iterations"] = iteration
                break

            current_task = reconstructed
            logger.info(
                "[%s] Still failing after iteration %s: %s",
                trajectory_uuid,
                iteration,
                [issue.description for issue in current_result.issues],
            )
        else:
            self.stats.increment("reconstruction_failed")
            self.stats.increment("max_iterations_reached")
            report_entry["action"] = "failed"
            report_entry["iterations"] = max_loops
            report_entry["final_issues"] = [
                issue.to_dict() for issue in current_result.issues
            ]

        if report_entry.get("action") is None:
            self.stats.increment("reconstruction_failed")
            report_entry["action"] = "failed"
            report_entry["iterations"] = iteration
        return samples, report_entry

    def _load_failed_task_ids(self, input_path: Path) -> set[str]:
        results_file = input_path / "execution_results.json"
        if not results_file.exists():
            logger.debug("No execution_results.json found at %s", results_file)
            return set()
        try:
            with results_file.open("r", encoding="utf-8") as handle:
                results_data = json.load(handle)
            failed_ids = set()
            for result in results_data.get("results", []):
                if not result.get("success", True):
                    task_id = result.get("task_id", "")
                    if task_id:
                        failed_ids.add(task_id)
            logger.info(
                "Found %s failed tasks in execution_results.json",
                len(failed_ids),
            )
            return failed_ids
        except json.JSONDecodeError as error:
            logger.warning("Failed to parse execution_results.json: %s", error)
        except Exception as error:
            logger.warning("Error loading execution_results.json: %s", error)
        return set()

    def _load_trajectories(self, file_path: Path) -> list[dict[str, Any]]:
        trajectories = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    trajectories.append(json.loads(line))
                except json.JSONDecodeError as error:
                    logger.warning(
                        "Failed to parse line in %s: %s",
                        file_path,
                        error,
                    )
        return trajectories

    def _parse_high_level_task(
        self,
        task_data: dict[str, Any],
    ) -> HighLevelTask | None:
        try:
            trajectories = []
            for step_data in task_data.get("trajectories", []):
                step = self._parse_low_level_task(step_data)
                if step:
                    trajectories.append(step)
            return HighLevelTask(
                task=task_data.get("task", ""),
                trajectories=trajectories,
                start_url=task_data.get("start_url"),
            )
        except Exception as error:
            logger.error("Failed to parse high-level task: %s", error)
            return None

    def _parse_low_level_task(
        self,
        step_data: dict[str, Any],
    ) -> LowLevelTask | None:
        try:
            action_data = step_data.get("action", {})
            target_data = action_data.get("target_element")
            target_element = None
            if target_data:
                target_element = Element(
                    id=str(target_data.get("id", "")),
                    name=target_data.get("name", ""),
                    role=target_data.get("role", ""),
                    union_bound=tuple(
                        target_data.get("union_bound", [0, 0, 0, 0])
                    ),
                    accessibility_tree_content=target_data.get(
                        "accessibility_tree_content",
                        "",
                    ),
                )
            action = Action(
                action_type=ActionType(
                    action_data.get("action_type", "none")
                ),
                target_element=target_element,
                value=action_data.get("value"),
                status=ActionExecuteStatus(
                    action_data.get("status", "NOT_EXECUTED")
                ),
            )
            current_state = self._parse_state(step_data.get("curr_state", {}))
            state_after_data = step_data.get("state_after")
            state_after = (
                self._parse_state(state_after_data)
                if state_after_data
                else None
            )
            return LowLevelTask(
                task=step_data.get("task"),
                curr_state=current_state,
                action=action,
                state_after=state_after,
                task_status=LowTaskStatus(
                    step_data.get("task_status", "IN_PROGRESS")
                ),
                reasoning=step_data.get("reasoning"),
            )
        except Exception as error:
            logger.error("Failed to parse low-level task: %s", error)
            return None

    @staticmethod
    def _parse_state(state_data: dict[str, Any]) -> StateInfo:
        raw_state_data = state_data.get("raw_state", {})
        return StateInfo(
            raw_state=RawState(
                url=raw_state_data.get("url", ""),
                accessibility_tree=raw_state_data.get(
                    "accessibility_tree",
                    "",
                ),
                observation_metadata=raw_state_data.get(
                    "observation_metadata",
                    {},
                ),
                screenshot=raw_state_data.get("screenshot", ""),
            ),
            elements=[],
            summary=state_data.get("summary"),
        )
