"""Command-line entry point for UI dataset generation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from cli.common import (
    SUPPRESS,
    add_boolean_override,
    add_llm_arguments,
    add_logging_arguments,
    add_neo4j_arguments,
    add_worker_argument,
    value_or,
    warn_deprecated_option,
)
from config import UiDatasetRunConfig, get_settings
from logging_utils import configure_logging
from storage.runs import RunDir, resolve_run_dir, write_manifest
from ui_dataset.generators import PageDatasetGenerator, TransitionDatasetGenerator

logger = logging.getLogger(__name__)

_DATASET_FILES = (
    "page_description.jsonl",
    "page_qa.jsonl",
    "element_description.jsonl",
    "state_transition.jsonl",
    "action_prediction.jsonl",
    "dataset_stats.json",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser while retaining omitted CLI override state."""
    parser = argparse.ArgumentParser(
        description="Generate VLM training datasets from a Neo4j state graph",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("all", "page", "transition"),
        default="all",
        help="Dataset group to generate (default: all)",
    )
    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Neo4j namespace containing crawler states",
    )

    neo4j_group = parser.add_argument_group("Neo4j connection overrides")
    add_neo4j_arguments(neo4j_group, legacy_user_alias=True)

    llm_group = parser.add_argument_group("Advanced T1 LLM overrides")
    add_llm_arguments(
        llm_group,
        include_model=False,
        include_temperature=True,
        include_max_tokens=True,
    )
    llm_group.add_argument("--retry-attempts", type=int, default=SUPPRESS)

    output_group = parser.add_argument_group("Output configuration")
    run_group = output_group.add_mutually_exclusive_group()
    run_group.add_argument("--output-root", default=SUPPRESS)
    run_group.add_argument("--resume-dir", default=SUPPRESS)
    run_group.add_argument(
        "--output-dir", "-o", dest="legacy_output_dir", default=SUPPRESS,
        help="Deprecated alias for --resume-dir",
    )
    output_group.add_argument("--note", default=SUPPRESS)
    add_boolean_override(
        output_group,
        "--skip-existing",
        help="Skip sample IDs already present in append outputs",
    )
    output_group.add_argument(
        "--overwrite",
        dest="legacy_overwrite",
        action="store_true",
        default=SUPPRESS,
        help="Deprecated alias for --no-skip-existing",
    )

    processing_group = parser.add_argument_group("Processing options")
    processing_group.add_argument("--max-samples", type=int, default=SUPPRESS)
    add_worker_argument(
        processing_group,
        help_text="Number of dataset generation worker threads",
    )
    add_boolean_override(processing_group, "--dedup-states-by-url")
    add_boolean_override(processing_group, "--sample-transitions")
    processing_group.add_argument("--scroll-cap", type=int, default=SUPPRESS)
    processing_group.add_argument("--dry-run", action="store_true", default=SUPPRESS)

    logging_group = parser.add_argument_group("Logging overrides")
    add_logging_arguments(logging_group)
    return parser


def _normalize_compat_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map deprecated options without overriding canonical values."""
    if hasattr(args, "legacy_neo4j_user"):
        warn_deprecated_option(
            args, "legacy_neo4j_user", "--neo4j-user", "--neo4j-username"
        )
        if not hasattr(args, "neo4j_username"):
            args.neo4j_username = args.legacy_neo4j_user
    if hasattr(args, "legacy_config"):
        warn_deprecated_option(args, "legacy_config", "--config", "--llm-config")
        if not hasattr(args, "llm_config"):
            args.llm_config = args.legacy_config
    if hasattr(args, "legacy_output_dir"):
        warn_deprecated_option(
            args, "legacy_output_dir", "--output-dir", "--resume-dir"
        )
        if not hasattr(args, "resume_dir"):
            args.resume_dir = args.legacy_output_dir
    if hasattr(args, "legacy_overwrite"):
        warn_deprecated_option(
            args, "legacy_overwrite", "--overwrite", "--no-skip-existing"
        )
        if not hasattr(args, "skip_existing"):
            args.skip_existing = False
    if hasattr(args, "legacy_no_file"):
        warn_deprecated_option(args, "legacy_no_file", "--no-file", "--no-file-log")
        if not hasattr(args, "file_output"):
            args.file_output = False
    return args


def _create_run_config(args: argparse.Namespace, run: RunDir) -> UiDatasetRunConfig:
    """Apply CLI overrides to the centralized UI dataset settings."""
    settings = get_settings(args.namespace)
    neo4j = replace(
        settings.neo4j,
        url=value_or(args, "neo4j_url", settings.neo4j.url),
        username=value_or(args, "neo4j_username", settings.neo4j.username),
        password=value_or(args, "neo4j_password", settings.neo4j.password),
        database=value_or(args, "neo4j_database", settings.neo4j.database),
        namespace=args.namespace,
    )
    sampling = replace(
        settings.ui_sampling,
        max_samples=value_or(args, "max_samples", settings.ui_sampling.max_samples),
        dedup_states_by_url=value_or(
            args, "dedup_states_by_url", settings.ui_sampling.dedup_states_by_url
        ),
        sample_transitions=value_or(
            args, "sample_transitions", settings.ui_sampling.sample_transitions
        ),
        scroll_cap=value_or(args, "scroll_cap", settings.ui_sampling.scroll_cap),
    )
    dataset = replace(
        settings.ui_dataset,
        skip_existing=value_or(
            args, "skip_existing", settings.ui_dataset.skip_existing
        ),
        num_workers=value_or(args, "workers", settings.ui_dataset.num_workers),
        retry_attempts=value_or(
            args, "retry_attempts", settings.ui_dataset.retry_attempts
        ),
        config_path=value_or(
            args, "llm_config", settings.ui_dataset.config_path
        ),
        generate_page_description=args.command in ("all", "page"),
        generate_page_qa=args.command in ("all", "page"),
        generate_element_description=args.command in ("all", "transition"),
        generate_state_transition=args.command in ("all", "transition"),
        log_level=value_or(args, "log_level", settings.ui_dataset.log_level),
        log_dir=value_or(args, "log_dir", settings.ui_dataset.log_dir),
        console_output=value_or(
            args, "console_output", settings.ui_dataset.console_output
        ),
        file_output=value_or(args, "file_output", settings.ui_dataset.file_output),
    )
    config = UiDatasetRunConfig(
        neo4j=neo4j,
        sampling=sampling,
        dataset=dataset,
        output_dir=run.path,
        temperature=value_or(args, "temperature", None),
        max_tokens=value_or(args, "max_tokens", None),
    )
    config.validate()
    if not config.neo4j.password:
        raise ValueError(
            "Neo4j password is required. Set --neo4j-password or NEO4J_PASSWORD."
        )
    return config


def _artifact_outputs(run: RunDir, log_path: Path | None) -> list[str]:
    """Return only artifacts that were materialized during this run."""
    candidates = [run.path / filename for filename in _DATASET_FILES]
    candidates.append(run.path / "images")
    if log_path is not None:
        candidates.append(log_path)
    return [str(path) for path in candidates if path.exists()]


def _run_generation(
    args: argparse.Namespace,
    config: UiDatasetRunConfig,
) -> dict[str, dict[str, Any]]:
    """Run the selected generators in the original page-then-transition order."""
    statistics: dict[str, dict[str, Any]] = {}
    if args.command in ("all", "page"):
        logger.info("Starting page-level dataset generation")
        statistics["page"] = PageDatasetGenerator(config).generate_all(
            skip_existing=config.dataset.skip_existing,
            max_samples=config.sampling.max_samples,
        )
    if args.command in ("all", "transition"):
        logger.info("Starting transition-level dataset generation")
        statistics["transition"] = TransitionDatasetGenerator(config).generate_all(
            skip_existing=config.dataset.skip_existing,
            max_samples=config.sampling.max_samples,
        )
    return statistics


def main(argv: list[str] | None = None) -> int:
    """Run one UI dataset generation invocation and persist its manifest."""
    args = _normalize_compat_args(build_parser().parse_args(argv))
    base_settings = get_settings(args.namespace)
    run = resolve_run_dir(
        "ui_dataset",
        getattr(args, "resume_dir", None),
        note=value_or(args, "note", base_settings.output.note),
        root_dir=value_or(args, "output_root", base_settings.output.root_dir),
    )
    config = _create_run_config(args, run)
    effective_log_dir = config.dataset.log_dir or run.path / "logs"
    config.dataset.log_dir = str(effective_log_dir)
    log_path = configure_logging(
        level=config.dataset.log_level,
        log_dir=effective_log_dir,
        filename_prefix=f"ui_dataset_{args.namespace}",
        console=config.dataset.console_output,
        file=config.dataset.file_output,
    )

    statistics: dict[str, dict[str, Any]] = {}
    error: str | None = None
    exit_code = 0
    try:
        logger.info("UI dataset generation starting")
        logger.info("Neo4j namespace: %s", config.neo4j.namespace)
        logger.info("Output directory: %s", run.path)
        if value_or(args, "dry_run", False):
            logger.info("Dry run completed; configuration is valid")
        else:
            statistics = _run_generation(args, config)
            failures = sum(item.get("failures", 0) for item in statistics.values())
            successes = sum(item.get("success", 0) for item in statistics.values())
            logger.info(
                "UI dataset generation completed: %s success, %s failures",
                successes,
                failures,
            )
            if failures > 0:
                exit_code = 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("UI dataset generation failed")
        exit_code = 1
    finally:
        inputs: dict[str, Any] = {
            "namespace": config.neo4j.namespace,
            "command": args.command,
            "model_role": "t1",
            "upstream_crawler_namespace": config.neo4j.namespace,
            "dry_run": value_or(args, "dry_run", False),
        }
        if statistics:
            inputs["statistics"] = statistics
        if error:
            inputs["error"] = error
        try:
            write_manifest(
                run,
                config_snapshot=config,
                inputs=inputs,
                outputs=_artifact_outputs(run, log_path),
            )
        except Exception:
            logger.exception("Unable to write UI dataset manifest")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
