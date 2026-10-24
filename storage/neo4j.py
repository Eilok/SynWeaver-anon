"""Shared Neo4j driver construction."""

from __future__ import annotations

from typing import Any

from config.shared import Neo4jSettings


def create_driver(settings: Neo4jSettings) -> Any:
    """Create a Neo4j driver using the shared connection and pool settings."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        settings.url,
        auth=(settings.username, settings.password),
        **settings.get_driver_config(),
    )


def create_session(driver: Any, settings: Neo4jSettings) -> Any:
    """Create a database session for the configured Neo4j database."""
    return driver.session(database=settings.database)


__all__ = ["Neo4jSettings", "create_driver", "create_session"]
