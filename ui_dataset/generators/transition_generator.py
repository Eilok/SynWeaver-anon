"""Generate element, state-transition, and action-prediction datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts.ui_dataset import get_student_instruction
from ui_dataset.extractors import TransitionData, TransitionExtractor
from ui_dataset.sharegpt_writer import MultiDatasetWriter, ShareGPTSample
from ui_dataset.vision import VisionGenerator

logger = logging.getLogger(__name__)


@dataclass
class TransitionDatasetResult:
    """Hold generated outputs for one transition."""

    transition_id: str
    from_state_id: str
    to_state_id: str
    event_type: str
    element: str
    element_description: Optional[str] = None
    state_transition: Optional[str] = None
    success: bool = False
    skipped_duplicate: bool = False
    error: Optional[str] = None


class TransitionDatasetGenerator:
    """Generate transition-level UI datasets from graph edges."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.llm_client = VisionGenerator(
            temperature=getattr(config, "temperature", None),
            max_tokens=getattr(config, "max_tokens", None),
            retry_attempts=getattr(config, "retry_attempts", 3),
            config_path=getattr(config, "config_path", "./config.yaml"),
        )
        self._processed_transitions: set[str] = set()
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._reservation_lock = threading.Lock()
        self._reserved_transition_keys: set[str] = set()

    def _load_processed_transitions(self, output_dir: Path) -> set[str]:
        """Load transition keys already present in any transition JSONL."""
        processed = set()
        prefixes = (
            "elem_desc_",
            "state_trans_",
            "action_pred_",
        )
        for dataset_name in (
            "element_description",
            "state_transition",
            "action_prediction",
        ):
            path = output_dir / f"{dataset_name}.jsonl"
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as file:
                    for line in file:
                        sample_id = json.loads(line).get("id", "")
                        for prefix in prefixes:
                            if sample_id.startswith(prefix):
                                processed.add(sample_id[len(prefix):])
                                break
            except Exception as exc:
                logger.warning("Error loading processed transitions from %s: %s", path, exc)
        return processed

    def _transition_key(self, transition: TransitionData) -> str:
        """Return a run-scoped key that distinguishes parallel graph edges."""
        signature = {
            "from_state_id": transition.from_state_id,
            "to_state_id": transition.to_state_id,
            "event_type": transition.event_type,
            "element": transition.element,
            "xpath": transition.xpath,
            "filled_value": transition.filled_value,
            "new_page_opened": transition.new_page_opened,
            "is_composite": transition.is_composite,
            "composite_type": transition.composite_type,
            "sub_actions": transition.sub_actions,
        }
        payload = json.dumps(
            signature,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        run_id = Path(self.config.output_dir).resolve().name
        namespace = self.config.neo4j.namespace
        return (
            f"{run_id}_{namespace}_{transition.from_state_id}_to_"
            f"{transition.to_state_id}_{digest}"
        )

    def _reserve_transition(self, transition: TransitionData) -> bool:
        """Reserve a transition before any screenshot or LLM work begins."""
        key = self._transition_key(transition)
        with self._reservation_lock:
            if key in self._processed_transitions or key in self._reserved_transition_keys:
                logger.info("Skipping duplicate UI transition sample %s", key)
                return False
            self._reserved_transition_keys.add(key)
            return True

    def _create_element_description_sample(self, transition: TransitionData, description: str) -> ShareGPTSample:
        """Create the original element-description sample."""
        key = self._transition_key(transition)
        element = transition.get_element_tree_representation()
        sample = ShareGPTSample(
            f"elem_desc_{key}",
            metadata={"namespace": self.config.neo4j.namespace, "from_state_id": transition.from_state_id, "to_state_id": transition.to_state_id, "event_type": transition.event_type, "element": element, "task_type": "element_description"},
        )
        image = transition.operation_screenshot_path or transition.from_screenshot_path
        if image and Path(image).exists():
            sample.add_image(image)
        instruction = get_student_instruction("element_description", accessibility_tree=transition.from_accessibility_tree_content or "", element=element)
        sample.add_human_turn(f"<image>\n{instruction}")
        sample.add_gpt_turn(description)
        return sample

    def _create_state_transition_sample(self, transition: TransitionData, description: str) -> ShareGPTSample:
        """Create the original state-transition sample."""
        key = self._transition_key(transition)
        action = transition.get_action_description()
        sample = ShareGPTSample(
            f"state_trans_{key}",
            metadata={"namespace": self.config.neo4j.namespace, "from_state_id": transition.from_state_id, "to_state_id": transition.to_state_id, "from_url": transition.from_url, "to_url": transition.to_url, "action": action, "task_type": "state_transition"},
        )
        image = transition.operation_screenshot_path or transition.from_screenshot_path
        if image and Path(image).exists():
            sample.add_image(image)
        instruction = get_student_instruction("state_prediction", accessibility_tree=transition.from_accessibility_tree_content or "", action=action)
        sample.add_human_turn(f"<image>\n{instruction}")
        sample.add_gpt_turn(description)
        return sample

    def _create_action_prediction_sample(self, transition: TransitionData, description: str) -> ShareGPTSample:
        """Create the original action-prediction sample coupled to transition generation."""
        key = self._transition_key(transition)
        sample = ShareGPTSample(
            f"action_pred_{key}",
            metadata={"namespace": self.config.neo4j.namespace, "from_state_id": transition.from_state_id, "to_state_id": transition.to_state_id, "event_type": transition.event_type, "element": transition.get_element_tree_representation(), "task_type": "action_prediction"},
        )
        if transition.from_screenshot_path and Path(transition.from_screenshot_path).exists():
            sample.add_image(transition.from_screenshot_path)
        if transition.to_screenshot_path and Path(transition.to_screenshot_path).exists():
            sample.add_image(transition.to_screenshot_path)
        instruction = get_student_instruction("action_prediction", from_accessibility_tree=transition.from_accessibility_tree_content or "", to_accessibility_tree=transition.to_accessibility_tree_content or "")
        sample.add_human_turn(f"<image>\n<image>\n{instruction}")
        sample.add_gpt_turn(f"Changes observed: {description}\n\nPredicted action: {transition.get_action_description()}")
        return sample

    def generate_for_transition(self, transition: TransitionData) -> TransitionDatasetResult:
        """Generate both LLM outputs for one transition."""
        result = TransitionDatasetResult(transition.transition_id, transition.from_state_id, transition.to_state_id, transition.event_type, transition.get_element_tree_representation())
        if not self._reserve_transition(transition):
            result.skipped_duplicate = True
            return result
        if not transition.has_valid_screenshots():
            result.error = f"Missing screenshots for transition {transition.transition_id}"
            logger.warning(result.error)
            return result
        if transition.from_accessibility_tree_content is None or transition.to_accessibility_tree_content is None:
            transition.load_accessibility_trees()
        try:
            generated = self.llm_client.generate_element_and_transition(
                transition.operation_screenshot_path or transition.from_screenshot_path,
                transition.to_screenshot_path,
                transition.get_action_description(),
                transition.get_element_tree_representation(),
                transition.from_url,
                transition.to_url,
            )
            if generated is None:
                result.error = f"LLM call failed for transition {transition.transition_id}"
                logger.warning(result.error)
                return result
            result.element_description, result.state_transition = generated
            if not result.element_description:
                result.error = f"Empty element description for transition {transition.transition_id}"
                logger.warning(result.error)
                return result
            if not result.state_transition:
                result.error = f"Empty state transition for transition {transition.transition_id}"
                logger.warning(result.error)
            result.success = True
            logger.info("Generated transition data for %s -> %s", transition.from_state_id, transition.to_state_id)
        except Exception as exc:
            result.error = f"Error processing transition {transition.transition_id}: {exc}"
            logger.error(result.error, exc_info=True)
        return result

    def _dataset_names(self) -> List[str]:
        names = []
        if self.config.generate_element_description:
            names.append("element_description")
        if self.config.generate_state_transition:
            names.extend(["state_transition", "action_prediction"])
        return names

    def _write_transition_result(self, transition: TransitionData, result: TransitionDatasetResult, writer: MultiDatasetWriter) -> None:
        """Write all enabled outputs for a successful transition."""
        if not result.success:
            if result.skipped_duplicate:
                return
            with self._counter_lock:
                self._failure_count += 1
            return
        if self.config.generate_element_description and result.element_description:
            writer.write_sample("element_description", self._create_element_description_sample(transition, result.element_description))
        if self.config.generate_state_transition and result.state_transition:
            writer.write_sample("state_transition", self._create_state_transition_sample(transition, result.state_transition))
            writer.write_sample("action_prediction", self._create_action_prediction_sample(transition, result.state_transition))
        with self._counter_lock:
            self._success_count += 1

    def _process_parallel(self, transitions: List[TransitionData], writer: MultiDatasetWriter, workers: int) -> int:
        """Generate transitions concurrently and write completed results safely."""
        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.generate_for_transition, transition): transition for transition in transitions}
            for future in as_completed(futures):
                transition = futures[future]
                try:
                    with self._lock:
                        self._write_transition_result(transition, future.result(), writer)
                except Exception as exc:
                    logger.error("Error processing transition: %s", exc, exc_info=True)
                    with self._counter_lock:
                        self._failure_count += 1
                processed += 1
        return processed

    def generate_all(self, skip_existing: bool = True, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Generate datasets for all eligible graph transitions."""
        output_dir = Path(self.config.output_dir).resolve()
        self._success_count = 0
        self._failure_count = 0
        self._reserved_transition_keys = set()
        self._processed_transitions = self._load_processed_transitions(output_dir) if skip_existing else set()
        names = self._dataset_names()
        if not names:
            logger.warning("No transition datasets enabled in configuration")
            return {"error": "No datasets enabled"}
        transitions = []
        total_transitions = 0
        writer = MultiDatasetWriter(str(output_dir), self.config.images_subdir)
        try:
            writer.open(names, overwrite=False)
            with TransitionExtractor(self.config.neo4j) as extractor:
                total_transitions = extractor.count_transitions()
                for transition in extractor.extract_all(True, self.config.sampling.sample_transitions, self.config.sampling.scroll_cap):
                    if skip_existing and self._transition_key(transition) in self._processed_transitions:
                        continue
                    transitions.append(transition)
                    if max_samples is not None and len(transitions) >= max_samples:
                        break
                workers = self.config.num_workers
                if workers > 1:
                    processed = self._process_parallel(transitions, writer, workers)
                else:
                    for transition in transitions:
                        self._write_transition_result(transition, self.generate_for_transition(transition), writer)
                    processed = len(transitions)
            writer.save_statistics()
        finally:
            writer.close()
        return {"total_transitions": total_transitions, "processed": processed, "success": self._success_count, "failures": self._failure_count, "output_dir": str(output_dir), "datasets": names, "num_workers": self.config.num_workers}

    def generate_for_transitions(self, transition_pairs: List[Tuple[str, str]], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate exactly the requested transition pairs."""
        if output_dir:
            self.config.output_dir = output_dir
        output_path = Path(self.config.output_dir).resolve()
        self._success_count = 0
        self._failure_count = 0
        self._reserved_transition_keys = set()
        self._processed_transitions = self._load_processed_transitions(output_path)
        names = self._dataset_names()
        if not names:
            logger.warning("No transition datasets enabled in configuration")
            return {"error": "No datasets enabled"}
        writer = MultiDatasetWriter(str(output_path), self.config.images_subdir)
        try:
            writer.open(names, overwrite=False)
            with TransitionExtractor(self.config.neo4j) as extractor:
                for from_state_id, to_state_id in transition_pairs:
                    transition = extractor.extract_by_states(from_state_id, to_state_id)
                    if transition is None:
                        logger.warning("Transition not found: %s -> %s", from_state_id, to_state_id)
                        self._failure_count += 1
                        continue
                    self._write_transition_result(transition, self.generate_for_transition(transition), writer)
            writer.save_statistics()
        finally:
            writer.close()
        return {"requested": len(transition_pairs), "success": self._success_count, "failures": self._failure_count, "output_dir": str(output_path)}
