"""Merge standard trajectory dataset runs while preserving record order."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from cli.common import add_output_arguments, resolve_output_arguments
from storage.runs import new_run, write_manifest

MODULE = "traj_dataset_merge"
INPUT_MODULE = "traj_datasets"
JSONL_FILENAME = "train_sharegpt.jsonl"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str, field_name: str) -> str:
    """Return a safe, non-empty filename component."""
    normalized = _SAFE_COMPONENT.sub("_", value.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one safe character")
    return normalized


def _resolve_input_dirs(input_dirs: Iterable[str | Path]) -> list[Path]:
    """Resolve input paths and reject duplicate or insufficient inputs."""
    resolved = [Path(path).expanduser().resolve() for path in input_dirs]
    if len(resolved) < 2:
        raise ValueError("At least two input directories are required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate input directories are not allowed")
    return resolved


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    """Load and validate an upstream trajectory dataset manifest."""
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest.json: {input_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid manifest.json in {input_dir}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    if manifest.get("module") != INPUT_MODULE:
        raise ValueError(
            f"Manifest module must be {INPUT_MODULE!r}: {manifest_path}"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"Manifest run_id must be a non-empty string: {manifest_path}")
    return manifest


def _manifest_output_filename(manifest: dict[str, Any], input_dir: Path) -> str:
    """Read the validation export filename from an upstream manifest."""
    config = manifest.get("config")
    validation = config.get("validation") if isinstance(config, dict) else None
    filename = validation.get("output_filename") if isinstance(validation, dict) else None
    if filename is None:
        return JSONL_FILENAME
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"Manifest output_filename must be a non-empty string: {input_dir}")
    basename = Path(filename).name
    if basename != filename or Path(basename).suffix.lower() != ".jsonl":
        raise ValueError(f"Manifest output_filename must be a local .jsonl filename: {input_dir}")
    return basename


def _open_jsonl(source_file: Path) -> io.StringIO:
    """Open UTF-8 JSONL content and normalize decoding failures to ValueError."""
    try:
        return io.StringIO(source_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read {source_file}: {exc}") from exc


def _image_filename(reference: str, source_file: Path, line_number: int) -> str:
    """Validate an images/<filename> reference and return its filename."""
    path = PurePosixPath(reference)
    if path.is_absolute() or not reference.startswith("images/") or "/" in reference[7:] or "\\" in reference:
        raise ValueError(
            f"Invalid image path at {source_file}:{line_number}: {reference!r}"
        )
    if len(path.parts) != 2 or path.parts[0] != "images" or path.name in {"", ".", ".."}:
        raise ValueError(
            f"Image reference must be images/<filename> at "
            f"{source_file}:{line_number}: {reference!r}"
        )
    return path.name


def _resolve_source_image(input_dir: Path, source_name: str) -> Path:
    """Resolve one regular image without following links outside the run."""
    images_dir = input_dir / "images"
    source_path = images_dir / source_name
    if images_dir.is_symlink() or source_path.is_symlink():
        raise ValueError(f"Symbolic-link images are not allowed: {source_path}")
    try:
        resolved_images = images_dir.resolve(strict=True)
        resolved_source = source_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Missing referenced image: {source_path}") from exc
    if resolved_source.parent != resolved_images or not resolved_source.is_file():
        raise ValueError(f"Referenced image escapes the run images directory: {source_path}")
    return resolved_source


def _load_run_records(
    input_dir: Path,
    run_id: str,
    source_filename: str,
    seen_ids: set[str],
    destinations: dict[str, Path],
) -> tuple[list[dict[str, Any]], int]:
    """Load one trajectory JSONL file and rewrite only validated image paths."""
    source_file = input_dir / source_filename
    if not source_file.is_file():
        raise ValueError(f"Missing {source_filename}: {input_dir}")
    records: list[dict[str, Any]] = []
    image_names: dict[str, str] = {}
    input_file = _open_jsonl(source_file)
    with input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line at {source_file}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {source_file}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {source_file}:{line_number}")
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Sample id must be a non-empty string at {source_file}:{line_number}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id {sample_id!r} at {source_file}:{line_number}")
            seen_ids.add(sample_id)

            images = record.get("images", [])
            if not isinstance(images, list) or any(not isinstance(item, str) for item in images):
                raise ValueError(f"images must be a list of strings at {source_file}:{line_number}")
            rewritten_images: list[str] = []
            for reference in images:
                source_name = _image_filename(reference, source_file, line_number)
                source_image = _resolve_source_image(input_dir, source_name)
                if reference not in image_names:
                    destination_name = (
                        f"{_safe_component(run_id, 'run_id')}_"
                        f"{_safe_component(source_name, 'image filename')}"
                    )
                    previous_source = destinations.get(destination_name)
                    if previous_source is not None and previous_source != source_image:
                        raise ValueError(f"Image destination collision: {destination_name}")
                    destinations[destination_name] = source_image
                    image_names[reference] = destination_name
                rewritten_images.append(f"images/{image_names[reference]}")
            if "images" in record:
                record["images"] = rewritten_images
            records.append(record)
    if not records:
        raise ValueError(f"No trajectory records found in {source_file}")
    return records, len(image_names)


def _write_json(path: Path, value: Any) -> None:
    """Write one UTF-8 JSON document."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_trajectory_runs(
    input_dirs: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and merge standard trajectory runs in the supplied order."""
    resolved_inputs = _resolve_input_dirs(input_dirs)
    output_path = Path(output_dir).expanduser().resolve()
    all_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    destinations: dict[str, Path] = {}
    report: dict[str, Any] = {
        "runs": [],
        "types": {JSONL_FILENAME: {"samples": 0, "images": 0}},
        "totals": {"runs": len(resolved_inputs), "types": 1, "samples": 0, "images": 0},
        "errors": [],
    }

    for input_dir in resolved_inputs:
        manifest = _load_manifest(input_dir)
        run_id = manifest["run_id"]
        source_filename = _manifest_output_filename(manifest, input_dir)
        records, image_count = _load_run_records(
            input_dir, run_id, source_filename, seen_ids, destinations
        )
        run_report = {
            "path": str(input_dir),
            "run_id": run_id,
            "source_filename": source_filename,
            "types": {
                JSONL_FILENAME: {"samples": len(records), "images": image_count}
            },
            "samples": len(records),
            "images": image_count,
            "errors": [],
        }
        report["runs"].append(run_report)
        report["types"][JSONL_FILENAME]["samples"] += len(records)
        report["types"][JSONL_FILENAME]["images"] += image_count
        all_records.extend(records)

    output_path.mkdir(parents=True, exist_ok=True)
    if destinations:
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for destination_name, source_image in destinations.items():
            shutil.copy2(source_image, images_dir / destination_name)
    output_file = output_path / JSONL_FILENAME
    with output_file.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report["totals"] = {
        "runs": len(resolved_inputs),
        "types": 1,
        "samples": len(all_records),
        "images": len(destinations),
    }
    _write_json(output_path / "merge_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Merge standard trajectory dataset runs.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        dest="input_dirs",
        help="Standard traj_datasets run directory; repeat at least twice",
    )
    output_group = parser.add_argument_group("Output configuration")
    add_output_arguments(output_group, legacy_options=("--output-dir",))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the trajectory merge CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    output_root, note = resolve_output_arguments(args, default_root="./outputs")
    run = new_run(MODULE, note, root_dir=output_root)
    status = "failed"
    error: str | None = None
    report: dict[str, Any] | None = None
    try:
        input_dirs = _resolve_input_dirs(args.input_dirs)
        report = merge_trajectory_runs(input_dirs, run.path)
        status = "completed"
        print(f"Merged trajectory dataset saved to: {run.path / JSONL_FILENAME}")
        return_code = 0
    except (OSError, ValueError, TypeError) as exc:
        error = str(exc)
        print(f"Error: {error}")
        return_code = 1
        input_dirs = [Path(path).expanduser().resolve() for path in args.input_dirs]
        failure_report = report or {
            "runs": [],
            "types": {JSONL_FILENAME: {"samples": 0, "images": 0}},
            "totals": {"runs": len(input_dirs), "types": 1, "samples": 0, "images": 0},
            "errors": [],
        }
        failure_report["errors"].append(error)
        try:
            _write_json(run.path / "merge_report.json", failure_report)
        except OSError:
            pass
    finally:
        materialized = [
            run.path / JSONL_FILENAME,
            run.path / "merge_report.json",
            run.images_dir,
        ]
        try:
            write_manifest(
                run,
                config_snapshot={
                    "status": status,
                    "error": error,
                    "note": note,
                },
                inputs={"input_dirs": [str(path) for path in input_dirs]},
                outputs=[path for path in materialized if path.exists()],
                upstream_runs=input_dirs,
            )
        except (OSError, ValueError, TypeError):
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
