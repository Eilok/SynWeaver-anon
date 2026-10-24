"""Command-line entry point for trajectory validation and ShareGPT export."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cli.common import (
    SUPPRESS,
    add_llm_arguments,
    add_logging_arguments,
    add_worker_argument,
    value_or,
    warn_deprecated_option,
)
from config import ValidationRunConfig, get_settings
from logging_utils import configure_logging

if TYPE_CHECKING:
    from storage.runs import RunDir
    from tasks.validation import TrajectoryValidationPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser using centralized validation defaults."""
    parser = argparse.ArgumentParser(
        description="Validate trajectories and export ShareGPT training samples"
    )
    parser.add_argument("--input-dir", required=True, help="Execution run directory")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=("validate", "reconstruct", "export"),
        default=SUPPRESS,
    )
    mode_group.add_argument(
        "--enable-reconstruction",
        dest="legacy_reconstruct",
        action="store_true",
        default=SUPPRESS,
    )
    mode_group.add_argument(
        "--skip-validation",
        dest="legacy_export",
        action="store_true",
        default=SUPPRESS,
    )

    processing_group = parser.add_argument_group("Validation settings")
    processing_group.add_argument("--max-steps", type=int, default=SUPPRESS)
    processing_group.add_argument("--min-loop-repeats", type=int, default=SUPPRESS)
    add_worker_argument(
        processing_group,
        help_text="Number of trajectory validation worker threads",
    )
    processing_group.add_argument("--max-validation-loops", type=int, default=SUPPRESS)

    llm_group = parser.add_argument_group("Advanced T2 LLM overrides")
    add_llm_arguments(
        llm_group,
        include_model=True,
        include_temperature=False,
    )

    output_group = parser.add_argument_group("Output settings")
    output_group.add_argument("--output-root", default=SUPPRESS)
    output_group.add_argument(
        "--output-dir", dest="legacy_output_dir", default=SUPPRESS
    )
    output_group.add_argument("--output-filename", default=SUPPRESS)
    output_group.add_argument("--note", default=SUPPRESS)

    logging_group = parser.add_argument_group("Logging overrides")
    add_logging_arguments(logging_group)
    return parser


def _normalize_compat_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map legacy validation flags to one explicit mode."""
    if hasattr(args, "legacy_reconstruct"):
        warn_deprecated_option(
            args,
            "legacy_reconstruct",
            "--enable-reconstruction",
            "--mode reconstruct",
        )
        args.mode = "reconstruct"
    if hasattr(args, "legacy_export"):
        warn_deprecated_option(
            args, "legacy_export", "--skip-validation", "--mode export"
        )
        args.mode = "export"
    if hasattr(args, "legacy_config"):
        warn_deprecated_option(args, "legacy_config", "--config", "--llm-config")
        if not hasattr(args, "llm_config"):
            args.llm_config = args.legacy_config
    if hasattr(args, "legacy_output_dir"):
        warn_deprecated_option(
            args, "legacy_output_dir", "--output-dir", "--output-root"
        )
        if not hasattr(args, "output_root"):
            args.output_root = args.legacy_output_dir
    if hasattr(args, "legacy_no_file"):
        warn_deprecated_option(args, "legacy_no_file", "--no-file", "--no-file-log")
        if not hasattr(args, "file_output"):
            args.file_output = False
    return args


def build_config(args: argparse.Namespace, run: RunDir) -> ValidationRunConfig:
    """Apply CLI values as replacements over centralized validation settings."""
    settings = get_settings()
    mode = value_or(args, "mode", settings.validation.mode)
    validation = replace(
        settings.validation,
        mode=mode,
        min_loop_repeats=value_or(
            args, "min_loop_repeats", settings.validation.min_loop_repeats
        ),
        num_workers=value_or(args, "workers", settings.validation.num_workers),
        max_validation_loops=value_or(
            args,
            "max_validation_loops",
            settings.validation.max_validation_loops,
        ),
        output_filename=value_or(
            args, "output_filename", settings.validation.output_filename
        ),
        config_path=value_or(args, "llm_config", settings.validation.config_path),
        log_level=value_or(args, "log_level", settings.validation.log_level),
        log_dir=value_or(args, "log_dir", settings.validation.log_dir),
        console_output=value_or(
            args, "console_output", settings.validation.console_output
        ),
        file_output=value_or(args, "file_output", settings.validation.file_output),
    )
    output = replace(settings.output, root_dir=str(run.path))
    config = ValidationRunConfig(
        validation=validation,
        output=output,
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=run.path,
        max_steps=value_or(args, "max_steps", settings.execution.max_steps),
        model_override=value_or(args, "model", None),
    )
    config.validate()
    return config


def _existing_outputs(run: RunDir) -> list[str]:
    """List materialized artifacts, excluding the manifest itself."""
    return [
        str(path)
        for path in sorted(run.path.rglob("*"))
        if path.is_file() and path != run.manifest_path
    ]


def _upstream_runs(input_dir: Path) -> list[str]:
    """Return the valid input directory as the upstream execution run."""
    resolved = input_dir.expanduser().resolve()
    return [str(resolved)] if resolved.is_dir() else []


def main(argv: list[str] | None = None) -> int:
    """Run validation and always persist a manifest."""
    args = _normalize_compat_args(build_parser().parse_args(argv))
    from storage.runs import new_run, write_manifest
    from tasks.validation import TrajectoryValidationPipeline

    settings = get_settings()
    run = new_run(
        "traj_datasets",
        note=value_or(args, "note", settings.output.note),
        root_dir=value_or(args, "output_root", settings.output.root_dir),
    )

    config: ValidationRunConfig | None = None
    pipeline: TrajectoryValidationPipeline | None = None
    status: dict[str, Any] = {"interrupted": False, "success": False}
    exit_code = 0
    try:
        config = build_config(args, run)
        effective_log_dir = config.validation.log_dir or run.path / "logs"
        config.validation.log_dir = str(effective_log_dir)
        configure_logging(
            level=config.validation.log_level,
            log_dir=effective_log_dir,
            filename_prefix="trajectory_validation",
            console=config.validation.console_output,
            file=config.validation.file_output,
        )
        if not config.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {config.input_dir}")
        pipeline = TrajectoryValidationPipeline(config)
        stats = pipeline.run()
        status = {
            "interrupted": False,
            "success": True,
            "stats": stats.to_dict(),
        }
    except KeyboardInterrupt:
        status = {"interrupted": True, "success": False}
        logger.warning("Trajectory validation interrupted by user")
        exit_code = 130
    except Exception as error:  # noqa: BLE001
        status = {
            "interrupted": False,
            "success": False,
            "error": str(error),
        }
        logger.exception("Trajectory validation failed")
        exit_code = 1
    finally:
        if pipeline is not None:
            pipeline.close()
        input_dir = Path(args.input_dir).expanduser()
        config_snapshot = config.to_dict() if config is not None else vars(args)
        write_manifest(
            run,
            config_snapshot=config_snapshot,
            inputs={"input_dir": str(input_dir.resolve()), "status": status},
            outputs=_existing_outputs(run),
            upstream_runs=_upstream_runs(input_dir),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
