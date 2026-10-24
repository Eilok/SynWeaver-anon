"""Runtime configuration for UI dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.shared import Neo4jSettings, SamplingSettings, ui_sampling_defaults


@dataclass
class UiDatasetSettings:
    """Configure UI dataset generation behavior."""

    batch_size: int = 10
    skip_existing: bool = True
    num_workers: int = 1
    generate_page_description: bool = True
    generate_page_qa: bool = True
    generate_element_description: bool = True
    generate_state_transition: bool = True
    retry_attempts: int = 3
    config_path: str = "./config.yaml"
    images_subdir: str = "images"
    create_separate_files: bool = True
    log_level: str = "INFO"
    log_dir: str | None = None
    console_output: bool = True
    file_output: bool = True


@dataclass
class UiDatasetRunConfig:
    """Compose shared graph and sampling settings with UI dataset options."""

    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    sampling: SamplingSettings = field(default_factory=ui_sampling_defaults)
    dataset: UiDatasetSettings = field(default_factory=UiDatasetSettings)
    output_dir: Path | str = "./outputs/ui_dataset"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def get_output_path(self) -> Path:
        """Return the resolved dataset output path."""
        return Path(self.output_dir).resolve()

    def validate(self) -> bool:
        """Validate graph, output, and dataset settings."""
        if not self.neo4j.url:
            raise ValueError("Neo4j URL is required")
        if not self.neo4j.namespace:
            raise ValueError("Graph namespace is required")
        if not self.output_dir:
            raise ValueError("Output directory is required")
        if not any((self.dataset.generate_page_description, self.dataset.generate_page_qa,
                    self.dataset.generate_element_description, self.dataset.generate_state_transition)):
            raise ValueError("At least one dataset type must be enabled")
        return True

    def __getattr__(self, name: str):
        """Expose dataset options directly for generator call-site compatibility."""
        dataset = self.__dict__.get("dataset")
        if dataset is not None and hasattr(dataset, name):
            return getattr(dataset, name)
        sampling = self.__dict__.get("sampling")
        if sampling is not None and hasattr(sampling, name):
            return getattr(sampling, name)
        raise AttributeError(name)
