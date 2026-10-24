"""Shared serialization, JSONL, and YAML configuration helpers."""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import yaml


_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def replace_env_vars(value: str) -> str:
    """Replace a value beginning with ``$`` using its environment variable."""
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    env_var = value[1:]
    return os.getenv(env_var, env_var)


def process_config(value: Any) -> Any:
    """Recursively replace environment variables in YAML values."""
    if isinstance(value, dict):
        return {key: process_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [process_config(item) for item in value]
    if isinstance(value, str):
        return replace_env_vars(value)
    return value


def load_yaml_config(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with environment substitution and caching."""
    cache_key = str(Path(file_path).expanduser().resolve())
    if not os.path.exists(cache_key):
        return {}
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    with open(cache_key, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    processed_config = process_config(config)
    _CONFIG_CACHE[cache_key] = processed_config
    return processed_config


def serialize_dataclass(obj: Any) -> Any:
    """Recursively convert supported project objects to JSON-compatible values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return serialize_dataclass(obj.value)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: serialize_dataclass(getattr(obj, field.name))
            for field in fields(obj)
        }
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return serialize_dataclass(obj.to_dict())
    if isinstance(obj, dict):
        return {key: serialize_dataclass(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialize_dataclass(item) for item in obj]
    if obj.__class__.__module__.startswith("numpy") and hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def jsonl_save(
    data: Iterable[Any],
    path: str | Path,
    append: bool = False,
) -> Path:
    """Write records to JSONL while preserving the legacy append behavior."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for item in data:
            serialized = serialize_dataclass(item)
            output_file.write(json.dumps(serialized, ensure_ascii=False) + "\n")
    return output_path


def jsonl_load(path: str | Path) -> list[Any]:
    """Load JSONL records, returning an empty list when the file does not exist."""
    input_path = Path(path)
    if not input_path.exists():
        return []

    records: list[Any] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text artifact."""
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Write a text artifact and create its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding=encoding)
    return output


def read_bytes(path: str | Path) -> bytes:
    """Read a binary artifact."""
    return Path(path).read_bytes()


def write_bytes(path: str | Path, content: bytes) -> Path:
    """Write a binary artifact and create its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return output


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write one JSON document with project serialization semantics."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        json.dump(serialize_dataclass(data), output_file, ensure_ascii=False, indent=indent)
        output_file.write("\n")
    return output


__all__ = [
    "jsonl_load",
    "jsonl_save",
    "read_bytes",
    "read_text",
    "load_yaml_config",
    "process_config",
    "replace_env_vars",
    "serialize_dataclass",
    "write_bytes",
    "write_json",
    "write_text",
]
