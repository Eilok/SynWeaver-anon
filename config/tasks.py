"""Task generation, execution, and validation configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.shared import Neo4jSettings, OutputSettings, SamplingSettings


@dataclass
class TaskGenSettings:
    """Configure high-level task generation."""

    website_name: str | None = None
    target_sites: list[str] = field(default_factory=list)
    tasks_per_transition: int = 1
    batch_size: int = 8
    num_workers: int = 1
    save_extended: bool = False
    resume_dir: str | None = None
    config_path: str = "./config.yaml"
    log_level: str = "INFO"
    log_dir: str | None = None
    console_output: bool = True
    file_output: bool = True


@dataclass
class TaskGenRunConfig:
    """Compose authoritative settings for one task generation run."""

    neo4j: Neo4jSettings
    sampling: SamplingSettings
    generation: TaskGenSettings
    output_dir: Path
    model: str | None = None
    temperature: float | None = None

    def validate(self) -> None:
        """Validate fields required before connecting or generating tasks."""
        if not self.neo4j.namespace:
            raise ValueError("Neo4j namespace is required")
        if not self.generation.website_name:
            raise ValueError("website_name is required")


@dataclass
class ExecutionSettings:
    """Configure task execution and collaborative refinement."""

    max_steps: int = 30
    failed_retry: int = 3
    refine_enabled: bool = True
    sleep_after_action: float = 0.5
    action_timeout: int = 30
    enable_vision: bool = True
    num_workers: int = 1
    temperature: float = 0.7
    t1_temperature: float | None = None
    t2_temperature: float | None = None
    config_path: str = "./config.yaml"
    log_level: str = "INFO"
    log_dir: str | None = None
    console_output: bool = True
    file_output: bool = True


@dataclass
class BrowserSettings:
    """Configure the browser used for task execution."""

    headless: bool = True
    slow_mo: int = 0
    observation_type: str = "accessibility_tree"
    viewport_width: int = 1280
    viewport_height: int = 720
    current_viewport_only: bool = True
    screenshot_full_page: bool = False

    @property
    def viewport_size(self) -> dict[str, int]:
        """Return the browser viewport in the expected mapping format."""
        return {"width": self.viewport_width, "height": self.viewport_height}


@dataclass
class EnvironmentSettings:
    """Configure WebArena service addresses and URL placeholder resolution."""

    env_domain: str = "127.0.0.1"
    auto_login: bool = True
    url_overrides: dict[str, str] = field(default_factory=dict)
    allowed_domains: list[str] = field(default_factory=list)
    _url_mapping: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def pre_process(self) -> None:
        """Populate service environment variables and the URL mapping."""
        ports = {
            "SHOPPING": 7770,
            "SHOPPING_ADMIN": 7780,
            "REDDIT": 9999,
            "GITLAB": 8023,
            "WIKIPEDIA": 8888,
            "MAP": 3000,
            "HOMEPAGE": 4399,
        }
        os.environ["ENV_DOMAIN"] = self.env_domain
        self._url_mapping.clear()

        for name, port in ports.items():
            os.environ[f"{name}_PORT"] = str(port)
            if name == "SHOPPING_ADMIN":
                url = f"http://{self.env_domain}:{port}/admin"
            elif name == "GITLAB":
                url = f"http://{self.env_domain}:{port}/explore"
            else:
                url = f"http://{self.env_domain}:{port}"
            os.environ[name] = url
            self._url_mapping[f"__{name}__"] = url

        for placeholder, url in self.url_overrides.items():
            normalized = placeholder if placeholder.startswith("__") else f"__{placeholder}__"
            self._url_mapping[normalized] = url
            os.environ[normalized.strip("_")] = url

    def get_url_mapping(self) -> dict[str, str]:
        """Return a copy of the current URL placeholder mapping."""
        if not self._url_mapping:
            self.pre_process()
        return self._url_mapping.copy()

    def resolve_url(self, url: str) -> str:
        """Replace the first known service placeholder in a URL."""
        if not url:
            return url
        if not self._url_mapping:
            self.pre_process()
        for placeholder, base_url in self._url_mapping.items():
            if placeholder in url:
                return url.replace(placeholder, base_url)
        return url


@dataclass
class DebugSettings:
    """Configure diagnostic output shared by task workflows."""

    debug: bool = False
    enable_stat_time_logging: bool = False
    verbose: bool = False
    save_intermediate: bool = False


@dataclass
class ExecutionRunConfig:
    """Compose authoritative settings for one trajectory execution run."""

    execution: ExecutionSettings
    browser: BrowserSettings
    environment: EnvironmentSettings
    debug: DebugSettings
    output: OutputSettings
    output_dir: Path

    @property
    def config_path(self) -> str:
        """Return the root LLM role configuration path."""
        return self.execution.config_path

    @property
    def temperature(self) -> float:
        """Return the legacy shared T1/T2 temperature fallback."""
        return self.execution.temperature

    @property
    def t1_temperature(self) -> float:
        """Return the explicit T1 override or legacy shared fallback."""
        value = self.execution.t1_temperature
        return self.execution.temperature if value is None else value

    @property
    def t2_temperature(self) -> float:
        """Return the explicit T2 override or legacy shared fallback."""
        value = self.execution.t2_temperature
        return self.execution.temperature if value is None else value

    def validate(self) -> None:
        """Validate execution settings before browser startup."""
        if self.execution.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.execution.failed_retry < 0:
            raise ValueError("failed_retry cannot be negative")
        if self.execution.num_workers <= 0:
            raise ValueError("num_workers must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return the source-compatible execution configuration snapshot."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class ValidationSettings:
    """Configure trajectory validation, reconstruction, and export."""

    check_end_action: bool = True
    check_loops: bool = True
    check_coherence: bool = True
    max_consecutive_same_action: int = 3
    min_loop_repeats: int = 2
    mode: str = "validate"
    max_validation_loops: int = 3
    num_workers: int = 1
    output_filename: str = "train_sharegpt.jsonl"
    config_path: str = "./config.yaml"
    log_level: str = "INFO"
    log_dir: str | None = None
    console_output: bool = True
    file_output: bool = True


@dataclass
class ValidationRunConfig:
    """Combine effective validation settings for one standard run."""

    validation: ValidationSettings
    output: OutputSettings
    input_dir: Path
    output_dir: Path
    max_steps: int
    model_override: str | None = None

    def validate(self) -> None:
        """Validate values required by the validation pipeline."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.validation.mode not in {"validate", "reconstruct", "export"}:
            raise ValueError("validation mode must be validate, reconstruct, or export")
        if self.validation.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.validation.max_validation_loops <= 0:
            raise ValueError("max_validation_loops must be positive")
        output_filename = self.validation.output_filename
        output_basename = Path(output_filename).name
        if (
            not output_filename.strip()
            or output_basename != output_filename
            or Path(output_basename).suffix.lower() != ".jsonl"
        ):
            raise ValueError("output_filename must be a local .jsonl filename")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable effective configuration snapshot."""
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["output_dir"] = str(self.output_dir)
        return data
