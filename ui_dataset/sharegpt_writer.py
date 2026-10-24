"""Readable ShareGPT JSONL writers with lazy image materialization."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """Represent one ShareGPT conversation turn."""

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Return the legacy ShareGPT turn mapping."""
        return {"from": self.role, "value": self.content}


@dataclass
class ShareGPTSample:
    """Represent one ShareGPT training sample."""

    sample_id: str
    images: List[str] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def add_human_turn(self, content: str) -> None:
        """Append a human turn."""
        self.conversations.append(Conversation(role="human", content=content))

    def add_gpt_turn(self, content: str) -> None:
        """Append a GPT turn."""
        self.conversations.append(Conversation(role="gpt", content=content))

    def add_image(self, image_path: str) -> None:
        """Append an image source path."""
        self.images.append(image_path)

    def to_dict(self) -> Dict[str, Any]:
        """Return the legacy ShareGPT structure."""
        result = {
            "id": self.sample_id,
            "images": self.images,
            "conversations": [conversation.to_dict() for conversation in self.conversations],
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def validate(self) -> bool:
        """Validate the sample and retain the original warning-only behavior."""
        if not self.conversations:
            return False
        for index, conversation in enumerate(self.conversations):
            expected_role = "human" if index % 2 == 0 else "gpt"
            if conversation.role != expected_role:
                logger.warning(
                    "Sample %s: expected %s at turn %s, got %s",
                    self.sample_id, expected_role, index, conversation.role,
                )
        placeholders = sum(turn.content.count("<image>") for turn in self.conversations)
        if placeholders != len(self.images):
            logger.warning(
                "Sample %s: %s image placeholders but %s images",
                self.sample_id, placeholders, len(self.images),
            )
        return True


class ShareGPTWriter:
    """Write one ShareGPT dataset while creating images only when needed."""

    def __init__(self, output_dir: str, dataset_name: str = "dataset", images_subdir: str = "images") -> None:
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.images_subdir = images_subdir
        self.images_dir = self.output_dir / images_subdir
        self.jsonl_path = self.output_dir / f"{dataset_name}.jsonl"
        self._file_handle = None
        self._sample_count = 0
        self._image_count = 0
        self._is_open = False
        self._sample_ids: set[str] = set()

    def open(self, overwrite: bool = False) -> None:
        """Open the JSONL file, appending unless overwrite is requested."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_count = 0
        self._sample_ids.clear()
        mode = "w" if overwrite else "a"
        if not overwrite and self.jsonl_path.exists():
            with open(self.jsonl_path, encoding="utf-8") as file:
                for line in file:
                    self._sample_count += 1
                    try:
                        sample_id = json.loads(line).get("id")
                    except json.JSONDecodeError:
                        continue
                    if isinstance(sample_id, str):
                        self._sample_ids.add(sample_id)
            logger.info("Appending to existing dataset with %s samples", self._sample_count)
        self._file_handle = open(self.jsonl_path, mode, encoding="utf-8")
        self._is_open = True
        logger.info("Opened ShareGPT writer: %s", self.jsonl_path)

    def close(self) -> None:
        """Close the JSONL file."""
        if self._file_handle is not None:
            self._file_handle.close()
        self._file_handle = None
        self._is_open = False
        logger.info("Closed ShareGPT writer: %s samples, %s images", self._sample_count, self._image_count)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _copy_image(self, src_path: str, new_filename: Optional[str] = None) -> str:
        """Copy an image and return its output-relative path."""
        source = Path(src_path)
        if not source.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        filename = new_filename or source.name
        base, extension = os.path.splitext(filename)
        destination = self.images_dir / filename
        counter = 1
        while destination.exists():
            filename = f"{base}_{counter}{extension}"
            destination = self.images_dir / filename
            counter += 1
        shutil.copy(source, destination)
        self._image_count += 1
        return f"{self.images_subdir}/{filename}"

    def write_sample(self, sample: ShareGPTSample, copy_images: bool = True, validate: bool = True) -> None:
        """Validate, materialize images, and append one sample."""
        if not self._is_open:
            raise RuntimeError("Writer is not open. Call open() first.")
        if sample.sample_id in self._sample_ids:
            logger.warning("Skipping duplicate sample id %s", sample.sample_id)
            return
        if validate:
            sample.validate()
        if copy_images and sample.images:
            copied_paths = []
            for index, image_path in enumerate(sample.images):
                extension = Path(image_path).suffix or ".png"
                try:
                    copied_paths.append(self._copy_image(image_path, f"{sample.sample_id}_img{index}{extension}"))
                except FileNotFoundError as exc:
                    logger.warning("Skipping missing image: %s", exc)
                    copied_paths.append(image_path)
            sample.images = copied_paths
        self._file_handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        self._file_handle.flush()
        self._sample_count += 1
        self._sample_ids.add(sample.sample_id)

    def has_sample_id(self, sample_id: str) -> bool:
        """Return whether this dataset already contains a sample ID."""
        return sample_id in self._sample_ids

    def write_samples(self, samples: List[ShareGPTSample], copy_images: bool = True, validate: bool = True) -> None:
        """Write multiple samples in order."""
        for sample in samples:
            self.write_sample(sample, copy_images=copy_images, validate=validate)

    def get_statistics(self) -> Dict[str, Any]:
        """Return current dataset statistics."""
        return {
            "dataset_name": self.dataset_name,
            "output_dir": str(self.output_dir),
            "sample_count": self._sample_count,
            "image_count": self._image_count,
            "jsonl_path": str(self.jsonl_path),
            "images_dir": str(self.images_dir),
        }

    def save_statistics(self) -> None:
        """Write the dataset statistics JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        statistics = {**self.get_statistics(), "created_at": datetime.now().isoformat()}
        with open(self.output_dir / f"{self.dataset_name}_stats.json", "w", encoding="utf-8") as file:
            json.dump(statistics, file, indent=2, ensure_ascii=False)
        logger.info("Saved statistics to %s", self.output_dir / f"{self.dataset_name}_stats.json")


class MultiDatasetWriter:
    """Write multiple datasets while sharing one lazily-created image directory."""

    def __init__(self, output_dir: str, images_subdir: str = "images") -> None:
        self.output_dir = Path(output_dir)
        self.images_subdir = images_subdir
        self.images_dir = self.output_dir / images_subdir
        self._writers: Dict[str, ShareGPTWriter] = {}
        self._is_open = False
        self._copied_images: Dict[str, str] = {}
        self._image_count = 0

    def open(self, dataset_names: List[str], overwrite: bool = False) -> None:
        """Open each requested dataset without creating images prematurely."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in dataset_names:
            writer = ShareGPTWriter(str(self.output_dir), name, self.images_subdir)
            writer.open(overwrite=overwrite)
            self._writers[name] = writer
        self._is_open = True
        logger.info("Opened multi-dataset writer with %s datasets", len(dataset_names))

    def close(self) -> None:
        """Close all contained writers."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        self._is_open = False
        logger.info("Closed multi-dataset writer")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def copy_image(self, src_path: str, sample_id: str, image_index: int = 0) -> str:
        """Copy directly to the shared directory and deduplicate source paths."""
        if src_path in self._copied_images:
            return self._copied_images[src_path]
        source = Path(src_path)
        if not source.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        extension = source.suffix or ".png"
        filename = f"{sample_id}_img{image_index}{extension}"
        base, extension = os.path.splitext(filename)
        destination = self.images_dir / filename
        counter = 1
        while destination.exists():
            filename = f"{base}_{counter}{extension}"
            destination = self.images_dir / filename
            counter += 1
        shutil.copy(source, destination)
        relative_path = f"{self.images_subdir}/{filename}"
        self._copied_images[src_path] = relative_path
        self._image_count += 1
        return relative_path

    def write_sample(self, dataset_name: str, sample: ShareGPTSample, copy_images: bool = True) -> None:
        """Write a sample to one of the opened datasets."""
        if not self._is_open:
            raise RuntimeError("Writer is not open. Call open() first.")
        if dataset_name not in self._writers:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        writer = self._writers[dataset_name]
        if writer.has_sample_id(sample.sample_id):
            logger.warning("Skipping duplicate sample id %s", sample.sample_id)
            return
        if copy_images and sample.images:
            copied_paths = []
            for index, image_path in enumerate(sample.images):
                try:
                    copied_paths.append(self.copy_image(image_path, sample.sample_id, index))
                except FileNotFoundError as exc:
                    logger.warning("Skipping missing image: %s", exc)
                    copied_paths.append(image_path)
            sample.images = copied_paths
        writer.write_sample(sample, copy_images=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Return combined statistics."""
        return {
            "output_dir": str(self.output_dir),
            "total_images": self._image_count,
            "datasets": {name: writer.get_statistics() for name, writer in self._writers.items()},
        }

    def save_statistics(self) -> None:
        """Write combined statistics."""
        statistics = {**self.get_statistics(), "created_at": datetime.now().isoformat()}
        with open(self.output_dir / "dataset_stats.json", "w", encoding="utf-8") as file:
            json.dump(statistics, file, indent=2, ensure_ascii=False)
        logger.info("Saved combined statistics to %s", self.output_dir / "dataset_stats.json")
