"""Unified runtime artifact directories and manifests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from storage.io import serialize_dataclass


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_SENSITIVE_KEYS = ("api_key", "password", "secret", "token", "credential")


@dataclass(frozen=True)
class RunDir:
    """Describe one module run and its standard artifact directories."""

    module: str
    run_id: str
    path: Path
    created_at: str

    @property
    def images_dir(self) -> Path:
        """Return the standard image directory for this run."""
        return self.path / "images"

    @property
    def screenshots_dir(self) -> Path:
        """Return the standard screenshot directory for this run."""
        return self.path / "screenshots"

    @property
    def manifest_path(self) -> Path:
        """Return the manifest path for this run."""
        return self.path / "manifest.json"


def _safe_component(value: str, field_name: str) -> str:
    """Normalize a path component and reject empty values."""
    normalized = _SAFE_COMPONENT.sub("_", value.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one safe character")
    return normalized


def new_run(
    module: str,
    note: str | None = None,
    *,
    root_dir: str | Path = "./outputs",
) -> RunDir:
    """Create a timestamped output directory for one independently runnable module."""
    module_name = _safe_component(module, "module")
    timestamp = datetime.now().astimezone()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    if note:
        run_id = f"{run_id}_{_safe_component(note, 'note')}"

    module_dir = Path(root_dir).expanduser() / module_name
    base_run_id = run_id
    collision_index = 0
    while True:
        path = module_dir / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            collision_index += 1
            run_id = f"{base_run_id}_{collision_index}"

    return RunDir(
        module=module_name,
        run_id=run_id,
        path=path,
        created_at=timestamp.isoformat(),
    )


def resolve_run_dir(
    module: str,
    cli_output: str | Path | None,
    *,
    note: str | None = None,
    root_dir: str | Path = "./outputs",
) -> RunDir:
    """Resolve an explicit output path or create a new standard module run."""
    if cli_output is None:
        return new_run(module, note, root_dir=root_dir)

    output_path = Path(cli_output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    module_name = _safe_component(module, "module")
    run_id = _safe_component(output_path.name, "run_id")
    created_at = datetime.fromtimestamp(output_path.stat().st_mtime).astimezone().isoformat()
    return RunDir(module_name, run_id, output_path, created_at)


def resolve_output_dir(
    module: str,
    cli_output: str | Path | None,
    *,
    note: str | None = None,
    root_dir: str | Path = "./outputs",
) -> Path:
    """Resolve an explicit CLI output path or create a standard module run directory."""
    return resolve_run_dir(module, cli_output, note=note, root_dir=root_dir).path


def _redact_secrets(value: Any) -> Any:
    """Redact sensitive values before persisting an effective configuration snapshot."""
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in _SENSITIVE_KEYS):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [_redact_secrets(item) for item in value]
    return value


def write_manifest(
    run: RunDir,
    *,
    config_snapshot: Any,
    inputs: Mapping[str, Any] | None = None,
    outputs: Iterable[str | Path] | None = None,
    upstream_runs: Iterable[str | Path] | None = None,
) -> Path:
    """Write the standard manifest for a completed or checkpointed module run."""
    serialized_config = serialize_dataclass(config_snapshot)
    manifest = {
        "module": run.module,
        "run_id": run.run_id,
        "created_at": run.created_at,
        "config": _redact_secrets(serialized_config),
        "inputs": serialize_dataclass(dict(inputs or {})),
        "outputs": [str(Path(item)) for item in outputs or []],
        "upstream_runs": [str(Path(item)) for item in upstream_runs or []],
    }

    run.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with run.manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    return run.manifest_path


__all__ = [
    "RunDir",
    "new_run",
    "resolve_output_dir",
    "resolve_run_dir",
    "write_manifest",
]
