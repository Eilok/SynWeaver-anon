"""Heuristic trajectory validation."""

from __future__ import annotations

import logging
from typing import Optional

from config.tasks import ValidationSettings
from tasks.models import (
    Action,
    ActionType,
    HighLevelTask,
    LowLevelTask,
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


class HeuristicValidator:
    """Validate trajectory endings, repeated patterns, and execution validity."""

    def __init__(
        self,
        config: ValidationSettings | None = None,
        max_steps: int = 30,
        min_loop_repeats: int | None = None,
    ) -> None:
        self.config = config or ValidationSettings()
        self.max_steps = max_steps
        self.min_loop_repeats = (
            self.config.min_loop_repeats
            if min_loop_repeats is None
            else min_loop_repeats
        )

    def validate(
        self,
        high_level_task: HighLevelTask,
        is_failed_task: bool = False,
    ) -> ValidationResult:
        """Validate one high-level task and return all detected issues."""
        issues: list[ValidationIssue] = []
        trajectories = high_level_task.trajectories
        if not trajectories:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationStatus.INVALID_ACTION,
                    description="Empty trajectory - no steps executed",
                    severity="error",
                )
            )
            return ValidationResult(
                passed=False,
                issues=issues,
                needs_reconstruction=True,
            )

        if self.config.check_end_action:
            end_issue = self._check_end_action(trajectories)
            if end_issue:
                issues.append(end_issue)
        if self.config.check_loops:
            issues.extend(self._check_loop_pattern(trajectories))
        if self.config.check_coherence:
            issues.extend(self._check_trajectory_validity(trajectories, is_failed_task))

        has_errors = any(issue.severity == "error" for issue in issues)
        return ValidationResult(
            passed=not has_errors,
            issues=issues,
            needs_reconstruction=has_errors,
        )

    def _check_end_action(
        self,
        trajectories: list[LowLevelTask],
    ) -> Optional[ValidationIssue]:
        if not trajectories:
            return None
        last_action_type = trajectories[-1].action.action_type
        if last_action_type != ActionType.NONE:
            return ValidationIssue(
                issue_type=ValidationStatus.MISSING_END,
                description=(
                    f"Trajectory ends with {last_action_type.value} instead of NONE"
                ),
                step_index=len(trajectories) - 1,
                severity="error",
            )
        return None

    def _check_loop_pattern(
        self,
        trajectories: list[LowLevelTask],
    ) -> list[ValidationIssue]:
        if len(trajectories) < 4:
            return []
        signatures = [self._get_action_signature(step.action) for step in trajectories]
        loop_info = self._find_loop_pattern_adaptive(signatures)
        if not loop_info:
            return []
        start_index, pattern_length, repeat_count = loop_info
        coverage = pattern_length * repeat_count / len(signatures)
        return [
            ValidationIssue(
                issue_type=ValidationStatus.LOOP_DETECTED,
                description=(
                    f"Detected repeating pattern of length {pattern_length} "
                    f"starting at step {start_index + 1}, repeating {repeat_count} times "
                    f"(covers {coverage:.0%} of trajectory)"
                ),
                step_index=start_index,
                severity="error",
            )
        ]

    def _get_action_signature(self, action: Action) -> str:
        action_type = action.action_type.value
        element_id = action.target_element.id if action.target_element else ""
        if ActionType.requires_value(action.action_type):
            return f"{action_type}:{element_id}:{action.value or ''}"
        return f"{action_type}:{element_id}"

    def _find_loop_pattern_adaptive(
        self,
        action_signatures: list[str],
    ) -> Optional[tuple[int, int, int]]:
        count = len(action_signatures)
        if count < 4:
            return None
        best_loop: Optional[tuple[int, int, int]] = None
        best_weighted_coverage = 0.0
        max_pattern_length = min(count // 2, 10)
        for pattern_length in range(1, max_pattern_length + 1):
            for start in range(count - pattern_length * 2 + 1):
                pattern = action_signatures[start : start + pattern_length]
                repeat_count = 1
                position = start + pattern_length
                while position + pattern_length <= count:
                    segment = action_signatures[position : position + pattern_length]
                    if segment != pattern:
                        break
                    repeat_count += 1
                    position += pattern_length
                if repeat_count >= self.min_loop_repeats:
                    coverage = pattern_length * repeat_count / count
                    weighted_coverage = coverage * (1 + 1 / pattern_length)
                    if weighted_coverage > best_weighted_coverage:
                        best_weighted_coverage = weighted_coverage
                        best_loop = (start, pattern_length, repeat_count)
        if best_loop:
            _, pattern_length, repeat_count = best_loop
            coverage = pattern_length * repeat_count / count
            if coverage > 0.4 or repeat_count >= 4:
                return best_loop
        return None

    def _check_trajectory_validity(
        self,
        trajectories: list[LowLevelTask],
        is_failed_task: bool = False,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if len(trajectories) >= self.max_steps:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationStatus.INVALID_ACTION,
                    description=(
                        f"Trajectory has {len(trajectories)} steps, "
                        f"exceeding max {self.max_steps}"
                    ),
                    severity="warning",
                )
            )
        if trajectories and trajectories[-1].action.action_type == ActionType.STOP:
            last_action = trajectories[-1].action
            issues.append(
                ValidationIssue(
                    issue_type=ValidationStatus.INVALID_ACTION,
                    description=(
                        f"Trajectory ended with STOP: "
                        f"{last_action.value or 'unknown reason'}"
                    ),
                    step_index=len(trajectories) - 1,
                    severity="error",
                )
            )
        if is_failed_task:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationStatus.INVALID_ACTION,
                    description="Task is marked as failed",
                    severity="error",
                )
            )
        return issues
