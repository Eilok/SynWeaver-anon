"""Single entry point for all non-LLM runtime configuration."""

from dataclasses import dataclass, field

from config.crawler import CrawlerSettings, LoginSettings
from config.shared import (
    Neo4jSettings,
    OutputSettings,
    SamplingSettings,
    task_sampling_defaults,
    ui_sampling_defaults,
)
from config.tasks import (
    BrowserSettings,
    EnvironmentSettings,
    DebugSettings,
    ExecutionRunConfig,
    ExecutionSettings,
    TaskGenRunConfig,
    TaskGenSettings,
    ValidationRunConfig,
    ValidationSettings,
)
from config.ui_dataset import UiDatasetRunConfig, UiDatasetSettings


@dataclass
class Settings:
    """Aggregate each configuration section without redefining shared fields."""

    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings.from_env)
    output: OutputSettings = field(default_factory=OutputSettings)
    ui_sampling: SamplingSettings = field(default_factory=ui_sampling_defaults)
    task_sampling: SamplingSettings = field(default_factory=task_sampling_defaults)
    crawler: CrawlerSettings = field(default_factory=CrawlerSettings)
    ui_dataset: UiDatasetSettings = field(default_factory=UiDatasetSettings)
    task_generation: TaskGenSettings = field(default_factory=TaskGenSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

    @classmethod
    def from_env(cls, namespace: str = "State") -> "Settings":
        """Create settings while resolving the shared Neo4j connection from the environment."""
        return cls(neo4j=Neo4jSettings.from_env(namespace))


def get_settings(namespace: str = "State") -> Settings:
    """Return an isolated settings object for one module invocation."""
    return Settings.from_env(namespace)


__all__ = [
    "BrowserSettings",
    "CrawlerSettings",
    "DebugSettings",
    "EnvironmentSettings",
    "ExecutionRunConfig",
    "ExecutionSettings",
    "LoginSettings",
    "Neo4jSettings",
    "OutputSettings",
    "SamplingSettings",
    "Settings",
    "TaskGenRunConfig",
    "TaskGenSettings",
    "UiDatasetRunConfig",
    "UiDatasetSettings",
    "ValidationRunConfig",
    "ValidationSettings",
    "get_settings",
]
