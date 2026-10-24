"""Validate image references in a ShareGPT JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cli.common import add_output_arguments, resolve_output_arguments
from storage.runs import new_run, write_manifest

MODULE = "validate_images"


def validate_images(jsonl_path: str | Path) -> dict[str, Any]:
    """Validate image references without terminating the importing process."""
    dataset_path = Path(jsonl_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    images_dir = dataset_path.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    total_samples = 0
    total_images = 0
    invalid_json_lines: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    found_images: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json_lines.append({"line": line_number, "error": str(exc)})
                print(f"Warning: Skipping invalid JSON at line {line_number}: {exc}")
                continue

            total_samples += 1
            sample_id = sample.get("id", f"line_{line_number}")
            for image_path in sample.get("images", []):
                total_images += 1
                filename = image_path[7:] if image_path.startswith("images/") else image_path
                full_path = images_dir / filename
                if full_path.exists():
                    found_images.add(filename)
                else:
                    missing_images.append(
                        {
                            "sample_id": sample_id,
                            "line": line_number,
                            "image": image_path,
                            "expected_path": str(full_path),
                        }
                    )

    actual_images = {path.name for path in images_dir.iterdir() if path.is_file()}
    unused_images = sorted(actual_images - found_images)
    return {
        "dataset": str(dataset_path),
        "images_dir": str(images_dir),
        "total_samples": total_samples,
        "total_image_references": total_images,
        "found_images": len(found_images),
        "missing_images": missing_images,
        "invalid_json_lines": invalid_json_lines,
        "unused_images": unused_images,
        "valid": not missing_images,
    }


def print_result(result: dict[str, Any]) -> None:
    """Print a human-readable validation summary."""
    print(f"Dataset: {result['dataset']}")
    print(f"Images directory: {result['images_dir']}")
    print("-" * 60)
    print(f"Total samples: {result['total_samples']}")
    print(f"Total image references: {result['total_image_references']}")
    print(f"Found images: {result['found_images']}")
    print(f"Missing image references: {len(result['missing_images'])}")

    if result["missing_images"]:
        print("\nMissing images:")
        by_image: dict[str, list[str]] = defaultdict(list)
        for item in result["missing_images"]:
            by_image[item["image"]].append(str(item["sample_id"]))
        for image, samples in sorted(by_image.items()):
            print(f"  - {image}")
            if len(samples) <= 3:
                for sample in samples:
                    print(f"      Referenced by: {sample}")
            else:
                print(f"      Referenced by {len(samples)} samples")
        print(f"\nValidation failed: {len(result['missing_images'])} invalid references")
        return

    print("\nValidation passed: all image references are valid.")
    if result["unused_images"]:
        print(f"\nNotice: {len(result['unused_images'])} images are not referenced:")
        for image in result["unused_images"][:10]:
            print(f"  - {image}")
        if len(result["unused_images"]) > 10:
            print(f"  ... and {len(result['unused_images']) - 10} more")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Validate image references in ShareGPT JSONL.")
    parser.add_argument("jsonl_file", help="Path to the ShareGPT JSONL file")
    output_group = parser.add_argument_group("Output configuration")
    add_output_arguments(output_group, legacy_options=("--output-dir",))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run image validation and return a process status code."""
    args = build_parser().parse_args(argv)
    output_root, note = resolve_output_arguments(args, default_root="./outputs")
    run = new_run(MODULE, note, root_dir=output_root)
    dataset_path = Path(args.jsonl_file).expanduser()
    report_path = run.path / "validation_report.json"
    outputs: list[Path] = []
    status = "failed"
    error: str | None = None
    result: dict[str, Any] | None = None
    try:
        result = validate_images(dataset_path)
        print_result(result)
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        outputs.append(report_path)
        status = "completed" if result["valid"] else "failed"
        return_code = 0 if result["valid"] else 1
    except (OSError, ValueError, TypeError) as exc:
        error = str(exc)
        print(f"Error: {error}")
        return_code = 1
    finally:
        write_manifest(
            run,
            config_snapshot={
                "status": status,
                "note": note,
                "valid": result["valid"] if result else False,
                "error": error,
            },
            inputs={"jsonl_file": str(dataset_path)},
            outputs=[path for path in outputs if path.exists()],
            upstream_runs=[dataset_path.parent],
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
