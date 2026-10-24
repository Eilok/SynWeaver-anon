"""Shared command-line argument helpers and explicit override utilities."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import fields, replace
from typing import Any, Mapping

SUPPRESS = argparse.SUPPRESS
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL")


def add_neo4j_arguments(
    group: argparse._ArgumentGroup,
    *,
    legacy_user_alias: bool = False,
) -> None:
    """Add shared Neo4j overrides without defining runtime defaults."""
    group.add_argument("--neo4j-url", default=SUPPRESS)
    group.add_argument("--neo4j-username", default=SUPPRESS)
    if legacy_user_alias:
        group.add_argument("--neo4j-user", dest="legacy_neo4j_user", default=SUPPRESS)
    group.add_argument("--neo4j-password", default=SUPPRESS)
    group.add_argument("--neo4j-database", default=SUPPRESS)


def add_output_arguments(
    group: argparse._ArgumentGroup,
    *,
    legacy_options: tuple[str, ...] = (),
    allow_resume: bool = False,
    include_note: bool = True,
) -> None:
    """Add canonical output-root options and separately tracked legacy names."""
    group.add_argument("--output-root", default=SUPPRESS)
    for option in legacy_options:
        dest = f"legacy_{option.removeprefix('--').replace('-', '_')}"
        group.add_argument(option, dest=dest, default=SUPPRESS)
    if include_note:
        group.add_argument("--note", default=SUPPRESS)
    if allow_resume:
        group.add_argument("--resume-dir", default=SUPPRESS)


def add_logging_arguments(
    group: argparse._ArgumentGroup,
    *,
    legacy_negative_options: bool = True,
) -> None:
    """Add common logging overrides without parser-derived defaults."""
    group.add_argument("--log-level", choices=LOG_LEVELS, default=SUPPRESS)
    group.add_argument("--log-dir", default=SUPPRESS)
    add_boolean_override(group, "--console", dest="console_output")
    add_boolean_override(group, "--file-log", dest="file_output")
    if legacy_negative_options:
        group.add_argument("--no-file", dest="legacy_no_file", action="store_true", default=SUPPRESS)


def add_llm_arguments(
    group: argparse._ArgumentGroup,
    *,
    include_model: bool = True,
    include_temperature: bool = True,
    include_max_tokens: bool = False,
    legacy_config_alias: bool = True,
) -> None:
    """Add canonical LLM role configuration and optional debug overrides."""
    group.add_argument("--llm-config", default=SUPPRESS)
    if legacy_config_alias:
        group.add_argument("--config", "-c", dest="legacy_config", default=SUPPRESS)
    if include_model:
        group.add_argument("--model", default=SUPPRESS)
    if include_temperature:
        group.add_argument("--temperature", type=float, default=SUPPRESS)
    if include_max_tokens:
        group.add_argument("--max-tokens", type=int, default=SUPPRESS)


def add_worker_argument(group: argparse._ArgumentGroup, *, help_text: str) -> None:
    """Add a worker override while leaving the centralized default untouched."""
    group.add_argument("--workers", "-w", type=int, default=SUPPRESS, help=help_text)


def add_boolean_override(
    group: argparse._ArgumentGroup,
    name: str,
    *,
    dest: str | None = None,
    help: str | None = None,
) -> None:
    """Add a symmetric boolean override with an omitted state."""
    group.add_argument(
        name,
        dest=dest,
        action=argparse.BooleanOptionalAction,
        default=SUPPRESS,
        help=help,
    )


def provided(args: argparse.Namespace, name: str) -> bool:
    """Return whether an optional CLI value was explicitly supplied."""
    return hasattr(args, name)


def value_or(args: argparse.Namespace, name: str, default: Any) -> Any:
    """Return an explicitly supplied value or its centralized default."""
    return getattr(args, name, default)


def apply_explicit_overrides(
    target: Any,
    args: argparse.Namespace,
    field_map: Mapping[str, str] | None = None,
) -> Any:
    """Apply only explicitly supplied flat dataclass fields."""
    field_names = {item.name for item in fields(target)}
    mapping = field_map or {name: name for name in field_names}
    changes = {
        field_name: getattr(args, arg_name)
        for arg_name, field_name in mapping.items()
        if field_name in field_names and hasattr(args, arg_name)
    }
    return replace(target, **changes) if changes else target


def warn_deprecated_option(
    args: argparse.Namespace,
    dest: str,
    option: str,
    replacement: str,
) -> None:
    """Emit a visible warning when a separately tracked old option was used."""
    if hasattr(args, dest):
        warnings.warn(
            f"{option} is deprecated; use {replacement}",
            FutureWarning,
            stacklevel=2,
        )


def resolve_output_arguments(
    args: argparse.Namespace,
    *,
    default_root: str,
) -> tuple[str, str | None]:
    """Resolve canonical tool output options while honoring the legacy name."""
    if hasattr(args, "legacy_output_dir"):
        warn_deprecated_option(
            args,
            "legacy_output_dir",
            "--output-dir",
            "--output-root",
        )
    output_root = getattr(
        args,
        "output_root",
        getattr(args, "legacy_output_dir", default_root),
    )
    return output_root, getattr(args, "note", None)


__all__ = [
    "LOG_LEVELS",
    "SUPPRESS",
    "add_boolean_override",
    "add_llm_arguments",
    "add_logging_arguments",
    "add_neo4j_arguments",
    "add_output_arguments",
    "add_worker_argument",
    "apply_explicit_overrides",
    "provided",
    "resolve_output_arguments",
    "value_or",
    "warn_deprecated_option",
]
