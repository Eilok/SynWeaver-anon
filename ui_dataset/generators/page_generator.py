"""Generate page description and page QA datasets."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts.ui_dataset import get_student_instruction
from ui_dataset.extractors import StateData, StateExtractor
from ui_dataset.sharegpt_writer import MultiDatasetWriter, ShareGPTSample
from ui_dataset.vision import VisionGenerator

logger = logging.getLogger(__name__)


@dataclass
class PageDatasetResult:
    """Hold the result of generating data for one state."""

    state_id: str
    url: str
    page_description: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    screenshot_path: Optional[str] = None
    accessibility_tree: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class PageDatasetGenerator:
    """Generate page-level datasets from Neo4j state nodes."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.llm_client = VisionGenerator(
            temperature=getattr(config, "temperature", None),
            max_tokens=getattr(config, "max_tokens", None),
            retry_attempts=getattr(config, "retry_attempts", 3),
            config_path=getattr(config, "config_path", "./config.yaml"),
        )
        self._processed_states: set[str] = set()
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()

    def _output_path(self) -> Path:
        """Return the configured output directory."""
        return Path(self.config.output_dir).resolve()

    def _load_processed_states(self, output_dir: Path) -> set[str]:
        """Load state IDs already present in page description JSONL."""
        processed = set()
        prefix = f"page_desc_{self.config.neo4j.namespace}_"
        path = output_dir / "page_description.jsonl"
        if not path.exists():
            return processed
        try:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    sample_id = data.get("id", "")
                    if sample_id.startswith(prefix):
                        processed.add(sample_id[len(prefix):])
        except Exception as exc:
            logger.warning("Error loading processed states from %s: %s", path, exc)
        return processed

    def _create_page_description_sample(self, state: StateData, description: str, tree: str) -> ShareGPTSample:
        """Create the legacy page-description sample."""
        namespace = self.config.neo4j.namespace
        sample = ShareGPTSample(
            sample_id=f"page_desc_{namespace}_{state.state_id}",
            metadata={"namespace": namespace, "state_id": state.state_id, "url": state.url, "task_type": "page_description"},
        )
        sample.add_image(state.screenshot_path)
        instruction = get_student_instruction("page_description", accessibility_tree=tree)
        sample.add_human_turn(f"<image>\n{instruction}")
        sample.add_gpt_turn(description)
        return sample

    def _create_page_qa_sample(self, state: StateData, question: str, answer: str, tree: str) -> ShareGPTSample:
        """Create the legacy page-QA sample."""
        namespace = self.config.neo4j.namespace
        sample = ShareGPTSample(
            sample_id=f"page_qa_{namespace}_{state.state_id}",
            metadata={"namespace": namespace, "state_id": state.state_id, "url": state.url, "task_type": "page_qa"},
        )
        sample.add_image(state.screenshot_path)
        instruction = get_student_instruction("page_qa", question=question, accessibility_tree=tree)
        sample.add_human_turn(f"<image>\n{instruction}")
        sample.add_gpt_turn(answer)
        return sample

    def generate_for_state(self, state: StateData) -> PageDatasetResult:
        """Generate both page outputs for one state."""
        result = PageDatasetResult(state.state_id, state.url, screenshot_path=state.screenshot_path)
        if not state.has_valid_screenshot():
            result.error = f"No valid screenshot for state {state.state_id}"
            logger.warning(result.error)
            return result
        if state.accessibility_tree_content is None:
            state.load_accessibility_tree()
        result.accessibility_tree = state.accessibility_tree_content or ""
        try:
            generated = self.llm_client.generate_page_description_and_qa(
                [state.screenshot_path], result.accessibility_tree, state.url
            )
            if generated is None:
                result.error = f"LLM call failed for state {state.state_id}"
                logger.warning(result.error)
                return result
            result.page_description, result.question, result.answer = generated
            if not result.page_description:
                result.error = f"Empty page description for state {state.state_id}"
                logger.warning(result.error)
                return result
            if not result.question or not result.answer:
                result.error = f"Empty QA for state {state.state_id}"
                logger.warning(result.error)
            result.success = True
            logger.info("Generated page data for %s", state.state_id)
        except Exception as exc:
            result.error = f"Error processing state {state.state_id}: {exc}"
            logger.error(result.error, exc_info=True)
        return result

    def _write_state_result(self, state: StateData, result: PageDatasetResult, writer: MultiDatasetWriter) -> None:
        """Write successful outputs and update counters."""
        if not result.success:
            with self._counter_lock:
                self._failure_count += 1
            return
        if self.config.generate_page_description and result.page_description:
            writer.write_sample("page_description", self._create_page_description_sample(state, result.page_description, result.accessibility_tree or ""))
        if self.config.generate_page_qa and result.question and result.answer:
            writer.write_sample("page_qa", self._create_page_qa_sample(state, result.question, result.answer, result.accessibility_tree or ""))
        with self._counter_lock:
            self._success_count += 1

    def _dataset_names(self) -> List[str]:
        names = []
        if self.config.generate_page_description:
            names.append("page_description")
        if self.config.generate_page_qa:
            names.append("page_qa")
        return names

    def _process_states_parallel(self, states: List[StateData], writer: MultiDatasetWriter, num_workers: int) -> int:
        """Generate states concurrently and write results as they complete."""
        processed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.generate_for_state, state): state for state in states}
            for future in as_completed(futures):
                state = futures[future]
                try:
                    result = future.result()
                    with self._lock:
                        self._write_state_result(state, result, writer)
                except Exception as exc:
                    logger.error("Error processing state: %s", exc, exc_info=True)
                    with self._counter_lock:
                        self._failure_count += 1
                processed += 1
        return processed

    def _process_states_sequential(self, states: List[StateData], writer: MultiDatasetWriter) -> int:
        """Generate and write states in source order."""
        for state in states:
            self._write_state_result(state, self.generate_for_state(state), writer)
        return len(states)

    def generate_all(self, skip_existing: bool = True, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Generate datasets for all eligible states."""
        output_dir = self._output_path()
        self._success_count = 0
        self._failure_count = 0
        self._processed_states = self._load_processed_states(output_dir) if skip_existing else set()
        names = self._dataset_names()
        if not names:
            logger.warning("No page datasets enabled in configuration")
            return {"error": "No datasets enabled"}
        states = []
        total_states = 0
        writer = MultiDatasetWriter(str(output_dir), self.config.images_subdir)
        try:
            writer.open(names, overwrite=False)
            with StateExtractor(self.config.neo4j) as extractor:
                total_states = extractor.count_states()
                for state in extractor.extract_all(True, self.config.sampling.dedup_states_by_url):
                    if skip_existing and state.state_id in self._processed_states:
                        continue
                    states.append(state)
                    if max_samples is not None and len(states) >= max_samples:
                        break
                workers = self.config.num_workers
                processed = self._process_states_parallel(states, writer, workers) if workers > 1 else self._process_states_sequential(states, writer)
            writer.save_statistics()
        finally:
            writer.close()
        return {"total_states": total_states, "processed": processed, "success": self._success_count, "failures": self._failure_count, "output_dir": str(output_dir), "datasets": names, "num_workers": self.config.num_workers}

    def generate_for_states(self, state_ids: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate exactly the requested state IDs."""
        if output_dir:
            self.config.output_dir = output_dir
        output_path = self._output_path()
        self._success_count = 0
        self._failure_count = 0
        names = self._dataset_names()
        if not names:
            logger.warning("No page datasets enabled in configuration")
            return {"error": "No datasets enabled"}
        writer = MultiDatasetWriter(str(output_path), self.config.images_subdir)
        try:
            writer.open(names, overwrite=False)
            with StateExtractor(self.config.neo4j) as extractor:
                for state_id in state_ids:
                    state = extractor.extract_by_id(state_id)
                    if state is None:
                        logger.warning("State not found: %s", state_id)
                        self._failure_count += 1
                        continue
                    self._write_state_result(state, self.generate_for_state(state), writer)
            writer.save_statistics()
        finally:
            writer.close()
        return {"requested": len(state_ids), "success": self._success_count, "failures": self._failure_count, "output_dir": str(output_path)}
