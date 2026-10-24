"""Generate high-level tasks from sampled state transitions."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from llm import UnifiedLLMClient
from prompts.task_synthesis import (
    format_task_examples,
    get_website_description,
    prompt_generate_high_level_task,
)
from tasks.generation.sampler import GraphSampler
from tasks.models import GraphTriplet

logger = logging.getLogger(__name__)


@dataclass
class TransitionContext:
    """Track generated tasks and output context for one transition."""

    transition_id: str
    from_state_id: str
    to_state_id: str
    action_type: str
    generated_tasks: list[str] = field(default_factory=list)
    generation_timestamps: list[datetime] = field(default_factory=list)
    start_url: str | None = None
    sites: list[str] = field(default_factory=list)

    def add_task(
        self,
        task: str,
        start_url: str | None = None,
        sites: list[str] | None = None,
    ) -> None:
        """Append a task and retain the source node URL for its transition."""
        normalized_url = (start_url or "").strip()
        if not normalized_url:
            raise ValueError("Source node URL is required for generated tasks")
        if self.start_url and self.start_url != normalized_url:
            raise ValueError("A transition cannot contain multiple source node URLs")
        self.generated_tasks.append(task)
        self.generation_timestamps.append(datetime.now())
        self.start_url = normalized_url
        if sites and not self.sites:
            self.sites = list(sites)

    @property
    def task_count(self) -> int:
        """Return the number of generated tasks."""
        return len(self.generated_tasks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "transition_id": self.transition_id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "action_type": self.action_type,
            "generated_tasks": self.generated_tasks,
            "generation_timestamps": [
                timestamp.isoformat() for timestamp in self.generation_timestamps
            ],
            "start_url": self.start_url,
            "sites": self.sites,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionContext:
        """Build a context from persisted data, accepting legacy omissions."""
        context = cls(
            transition_id=data["transition_id"],
            from_state_id=data["from_state_id"],
            to_state_id=data["to_state_id"],
            action_type=data["action_type"],
        )
        context.generated_tasks = data.get("generated_tasks", [])
        context.generation_timestamps = [
            datetime.fromisoformat(timestamp)
            for timestamp in data.get("generation_timestamps", [])
        ]
        context.start_url = data.get("start_url")
        context.sites = data.get("sites", [])
        return context


@dataclass
class GenerationResult:
    """Result of task generation for one triplet."""

    triplet_id: str
    task: str | None = None
    sub_instruction: str | None = None
    start_url: str | None = None
    sites: list[str] = field(default_factory=list)
    analysis: str | None = None
    success: bool = False
    error: str | None = None
    generation_time: float = 0.0

    def to_synthagent_format(self) -> dict[str, Any] | None:
        """Return canonical task output, or None for an unsuccessful result."""
        if not self.success or not self.task:
            return None
        if not self.start_url:
            raise ValueError("Successful generation result is missing source node URL")
        return {
            "task": self.task,
            "start_url": self.start_url,
            "sites": self.sites,
        }

    def to_extended_format(self) -> dict[str, Any]:
        """Return all generation fields and metadata."""
        return {
            "triplet_id": self.triplet_id,
            "task": self.task,
            "sub_instruction": self.sub_instruction,
            "start_url": self.start_url,
            "sites": self.sites,
            "analysis": self.analysis,
            "success": self.success,
            "error": self.error,
            "generation_time": self.generation_time,
        }


class TaskGenerator:
    """Generate high-level tasks with an injected unified LLM client."""

    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        graph_sampler: GraphSampler,
        website_name: str = "shopping",
        target_sites: list[str] | None = None,
        context_save_path: Path | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.graph_sampler = graph_sampler
        self.website_name = website_name
        self.target_sites = target_sites or [website_name]
        self.context_save_path = context_save_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_store: dict[str, TransitionContext] = {}
        self._context_lock = threading.Lock()

        if context_save_path and context_save_path.exists():
            self._load_context()

        logger.info(
            "TaskGenerator initialized for %s, loaded %d transition contexts",
            website_name,
            len(self.context_store),
        )

    def _load_context(self) -> None:
        """Load context leniently, retaining an empty store on failure."""
        if not self.context_save_path:
            return
        try:
            with self.context_save_path.open("r") as file:
                data = json.load(file)
            for transition_id, context_data in data.items():
                self.context_store[transition_id] = TransitionContext.from_dict(context_data)
            logger.info(
                "Loaded %d contexts from %s",
                len(self.context_store),
                self.context_save_path,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to load context: %s", error)

    def save_context(self) -> None:
        """Persist context leniently with a thread-safe snapshot."""
        if not self.context_save_path:
            return
        try:
            with self._context_lock:
                data = {
                    transition_id: context.to_dict()
                    for transition_id, context in self.context_store.items()
                }
            self.context_save_path.parent.mkdir(parents=True, exist_ok=True)
            with self.context_save_path.open("w") as file:
                json.dump(data, file, indent=2)
            logger.info("Saved %d contexts to %s", len(data), self.context_save_path)
        except Exception as error:  # noqa: BLE001
            logger.error("Failed to save context: %s", error)

    def _get_transition_id(self, triplet: GraphTriplet) -> str:
        """Return the original stable 16-character transition hash."""
        key = f"{triplet.from_state_id}|{triplet.to_state_id}|{triplet.action_type}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def task_count_for(self, triplet: GraphTriplet) -> int:
        """Return the existing task count without creating a context."""
        transition_id = self._get_transition_id(triplet)
        with self._context_lock:
            context = self.context_store.get(transition_id)
            return context.task_count if context else 0

    def _get_or_create_context(self, triplet: GraphTriplet) -> TransitionContext:
        """Get or atomically create a transition context."""
        transition_id = self._get_transition_id(triplet)
        with self._context_lock:
            if transition_id not in self.context_store:
                self.context_store[transition_id] = TransitionContext(
                    transition_id=transition_id,
                    from_state_id=triplet.from_state_id,
                    to_state_id=triplet.to_state_id,
                    action_type=triplet.action_type,
                )
            return self.context_store[transition_id]

    def _load_screenshot(self, path: str | None) -> np.ndarray | None:
        """Load a screenshot as an ndarray, returning None on failure."""
        if not path:
            return None
        try:
            image = Image.open(path)
            return np.array(image)
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to load screenshot %s: %s", path, error)
            return None

    def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Call the unified client using the prompt's sole user content list."""
        response = self.llm_client.complete(
            user=messages[0]["content"],
            json_mode=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response if isinstance(response, dict) else None

    def generate_task(
        self,
        triplet: GraphTriplet,
        website_description: str | None = None,
    ) -> GenerationResult:
        """Generate one high-level task from a state-transition triplet."""
        start_time = time.time()
        result = GenerationResult(triplet_id=triplet.id)
        source_url = (triplet.from_url or "").strip()
        if not source_url:
            result.error = "Source node URL is required for task generation"
            result.generation_time = time.time() - start_time
            logger.error("Task generation skipped for %s: %s", triplet.id, result.error)
            return result
        try:
            context = self._get_or_create_context(triplet)
            previous_tasks = context.generated_tasks.copy()
            action_description = triplet.get_action_description()
            website_intro = website_description or get_website_description(self.website_name)
            task_examples = format_task_examples(self.website_name)
            if previous_tasks:
                task_examples += (
                    "\n\nPreviously generated tasks for this transition "
                    "(avoid duplicating these):\n"
                )
                task_examples += "\n".join(f"- {task}" for task in previous_tasks)

            if triplet.is_scroll_action():
                before_screenshot = self._load_screenshot(triplet.from_screenshot_path)
            else:
                before_screenshot = self._load_screenshot(
                    triplet.screenshot_op_path or triplet.from_screenshot_path
                )
            after_screenshot = self._load_screenshot(triplet.to_screenshot_path)

            messages = prompt_generate_high_level_task(
                website_name=self.website_name,
                website_intro=website_intro,
                current_action_str=action_description,
                before_screenshot=before_screenshot,
                after_screenshot=after_screenshot,
                task_examples=task_examples,
            )
            response = self._call_llm(messages)
            if response:
                result.task = response.get("High-Level-Instruction")
                result.start_url = source_url
                result.sites = self.target_sites
                result.analysis = response.get("Analysis")
                result.sub_instruction = response.get("Sub-Instruction")
                result.success = bool(result.task)
                if result.success and result.task:
                    with self._context_lock:
                        context.add_task(
                            result.task,
                            start_url=result.start_url,
                            sites=result.sites,
                        )
                        task_count = context.task_count
                    task_preview = result.task[:50] if len(result.task) > 50 else result.task
                    logger.info(
                        "Generated task for %s: %s... (#%d for this transition)",
                        triplet.id,
                        task_preview,
                        task_count,
                    )
            else:
                result.error = "Failed to get valid response from LLM"
        except Exception as error:  # noqa: BLE001
            result.error = str(error)
            logger.error("Task generation failed for %s: %s", triplet.id, error)

        result.generation_time = time.time() - start_time
        return result

    def generate_tasks_batch(
        self,
        triplets: list[GraphTriplet],
        website_description: str | None = None,
        save_interval: int = 10,
    ) -> list[GenerationResult]:
        """Generate a batch and periodically persist context."""
        results: list[GenerationResult] = []
        for index, triplet in enumerate(triplets):
            result = self.generate_task(triplet, website_description)
            results.append(result)
            if (index + 1) % save_interval == 0:
                self.save_context()
                logger.info("Processed %d/%d triplets", index + 1, len(triplets))
        self.save_context()
        return results

    def generate_from_sampler(
        self,
        max_count: int | None = None,
        tasks_per_transition: int = 1,
        website_description: str | None = None,
        output_path: Path | None = None,
    ) -> Iterator[GenerationResult]:
        """Generate tasks from sampled transitions and optionally stream JSONL."""
        if max_count:
            triplets = self.graph_sampler.sample(max_count)
        else:
            triplets = self.graph_sampler.sample_all()
        logger.info(
            "Processing %d triplets, %d tasks each",
            len(triplets),
            tasks_per_transition,
        )

        output_file = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path.open("w")
        try:
            total_generated = 0
            for triplet in triplets:
                for _ in range(tasks_per_transition):
                    result = self.generate_task(triplet, website_description)
                    if output_file and result.success:
                        output_file.write(json.dumps(result.to_synthagent_format()) + "\n")
                        output_file.flush()
                    total_generated += 1
                    yield result
                    if total_generated % 10 == 0:
                        self.save_context()
            self.save_context()
        finally:
            if output_file:
                output_file.close()

    def get_context_summary(self) -> dict[str, Any]:
        """Return aggregate context statistics."""
        total_transitions = len(self.context_store)
        total_tasks = sum(context.task_count for context in self.context_store.values())
        distribution: dict[int, int] = {}
        for context in self.context_store.values():
            count = context.task_count
            distribution[count] = distribution.get(count, 0) + 1
        return {
            "total_transitions": total_transitions,
            "total_tasks": total_tasks,
            "average_tasks_per_transition": (
                total_tasks / total_transitions if total_transitions > 0 else 0
            ),
            "tasks_per_transition_distribution": distribution,
        }

    def iter_context_rows(self) -> Iterator[dict[str, Any]]:
        """Yield canonical rows from loaded history and newly generated tasks."""
        with self._context_lock:
            contexts = list(self.context_store.values())
        for context in contexts:
            start_url = (context.start_url or "").strip()
            if not start_url:
                raise ValueError(
                    f"Transition {context.transition_id} is missing its source node URL"
                )
            sites = context.sites or self.target_sites
            for task in context.generated_tasks:
                if task:
                    yield {
                        "task": task,
                        "start_url": start_url,
                        "sites": sites,
                    }

    def write_context_jsonl(self, output_path: Path) -> int:
        """Validate context and rebuild the canonical task JSONL."""
        rows = list(self.iter_context_rows())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Rebuilt %d tasks from context to %s", len(rows), output_path)
        return len(rows)


def write_synthagent_jsonl(
    results: list[GenerationResult], output_path: Path
) -> int:
    """Write successful results in canonical SynthAgent JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as file:
        for result in results:
            if result.success:
                file.write(json.dumps(result.to_synthagent_format()) + "\n")
                count += 1
    logger.info("Wrote %d tasks to %s", count, output_path)
    return count


def write_extended_jsonl(
    results: list[GenerationResult], output_path: Path
) -> int:
    """Write all results with generation metadata as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for result in results:
            file.write(json.dumps(result.to_extended_format()) + "\n")
    logger.info("Wrote %d results to %s", len(results), output_path)
    return len(results)
