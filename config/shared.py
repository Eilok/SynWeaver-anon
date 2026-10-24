"""Configuration shared across multiple functional modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Neo4jSettings:
    """Define the single Neo4j connection configuration used by all modules."""

    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str | None = None
    database: str = "neo4j"
    namespace: str = "State"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60

    @classmethod
    def from_env(cls, namespace: str = "State") -> "Neo4jSettings":
        """Create settings from the shared Neo4j environment variables."""
        defaults = cls(namespace=namespace)
        return cls(
            url=os.environ.get("NEO4J_URL", defaults.url),
            username=os.environ.get("NEO4J_USERNAME", defaults.username),
            password=os.environ.get("NEO4J_PASSWORD", defaults.password),
            database=os.environ.get("NEO4J_DATABASE", defaults.database),
            namespace=namespace,
        )

    def get_driver_config(self) -> dict[str, Any]:
        """Return connection pool options accepted by the Neo4j driver."""
        return {
            "max_connection_lifetime": self.max_connection_lifetime,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout": self.connection_acquisition_timeout,
        }


@dataclass
class SamplingSettings:
    """Define the shared graph sampling schema used by dataset and task generation."""

    max_samples: int | None = None
    depth_priority: bool = True
    degree_priority: bool = True
    min_depth: int = 0
    max_depth: int = -1
    exclude_self_loops: bool = True
    deduplicate_by_action: bool = True
    dedup_states_by_url: bool = False
    sample_transitions: bool = False
    scroll_cap: int = 25


@dataclass
class OutputSettings:
    """Define the common output root and output behavior."""

    root_dir: str = "./outputs"
    save_screenshots: bool = True
    note: str | None = None


def ui_sampling_defaults() -> SamplingSettings:
    """Return the original UI dataset sampling defaults."""
    return SamplingSettings(max_samples=None)


def task_sampling_defaults() -> SamplingSettings:
    """Return the original task generation sampling defaults."""
    return SamplingSettings(max_samples=None)
