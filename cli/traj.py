"""Command-line entry point for WebArena trajectory execution."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from cli.common import (
    SUPPRESS,
    add_boolean_override,
    add_llm_arguments,
    add_logging_arguments,
    add_worker_argument,
    value_or,
    warn_deprecated_option,
)
from config import ExecutionRunConfig, get_settings
from logging_utils import configure_logging

if TYPE_CHECKING:
    from storage.runs import RunDir

logger = logging.getLogger(__name__)


def _url_override_arg(value: str) -> str:
    """Validate one required PLACEHOLDER=URL command-line mapping."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected PLACEHOLDER=URL")
    placeholder, url = value.split("=", 1)
    if not placeholder.strip():
        raise argparse.ArgumentTypeError("URL override placeholder cannot be empty")
    normalized_url = url.strip()
    if not normalized_url:
        raise argparse.ArgumentTypeError("URL override URL cannot be empty")
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise argparse.ArgumentTypeError(
            "URL override must be an absolute HTTP(S) URL"
        )
    return value


def parse_url_overrides(values: list[list[str]] | list[str] | None) -> dict[str, str]:
    """Parse repeated or grouped PLACEHOLDER=URL arguments."""
    overrides: dict[str, str] = {}
    for group in values or []:
        items = [group] if isinstance(group, str) else group
        for item in items:
            _url_override_arg(item)
            placeholder, url = item.split("=", 1)
            placeholder = placeholder.strip().upper()
            if not placeholder.startswith("__"):
                placeholder = f"__{placeholder}__"
            overrides[placeholder] = url.strip()
    if not overrides:
        raise ValueError("At least one valid URL override is required")
    return overrides


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL tasks, assigning IDs from their physical input lines."""
    tasks: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as task_file:
        for line_number, line in enumerate(task_file, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning("Skipping invalid JSON on input line %d: %s", line_number, error)
                continue
            if not isinstance(task, dict):
                logger.warning("Skipping non-object task on input line %d", line_number)
                continue
            task.setdefault("id", f"task_{line_number:04d}")
            tasks.append(task)
    return tasks


def resolve_task_urls(
    tasks: list[dict[str, Any]],
    environment: Any,
) -> list[dict[str, Any]]:
    """Resolve placeholders without losing the original URL."""
    for task in tasks:
        start_url = task.get("start_url")
        if not isinstance(start_url, str):
            continue
        resolved = environment.resolve_url(start_url)
        if resolved != start_url:
            task["_original_start_url"] = start_url
            task["start_url"] = resolved
    return tasks


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser using centralized execution defaults."""
    parser = argparse.ArgumentParser(
        description="Execute web agent tasks from a JSONL file"
    )
    parser.add_argument("--tasks", "-t", required=True, help="Path to the task JSONL file")

    execution_group = parser.add_argument_group("Execution settings")
    execution_group.add_argument(
        "--single", "-s", dest="legacy_single", action="store_true", default=SUPPRESS
    )
    add_worker_argument(
        execution_group,
        help_text="Number of task execution worker processes",
    )
    execution_group.add_argument("--max-steps", type=int, default=SUPPRESS)
    execution_group.add_argument("--failed-retry", type=int, default=SUPPRESS)
    add_boolean_override(execution_group, "--vision")
    add_boolean_override(execution_group, "--headless")
    add_boolean_override(execution_group, "--auto-login")
    add_boolean_override(execution_group, "--refine")

    environment_group = parser.add_argument_group("Environment settings")
    environment_group.add_argument("--env-domain", default=SUPPRESS)
    environment_group.add_argument(
        "--url-override",
        action="append",
        type=_url_override_arg,
        required=True,
        default=SUPPRESS,
        metavar="PLACEHOLDER=URL",
    )
    environment_group.add_argument(
        "--allowed-domain",
        action="append",
        default=SUPPRESS,
        metavar="DOMAIN",
    )

    llm_group = parser.add_argument_group("Advanced T1/T2 LLM overrides")
    add_llm_arguments(
        llm_group,
        include_model=False,
        include_temperature=False,
    )
    llm_group.add_argument("--t1-temperature", type=float, default=SUPPRESS)
    llm_group.add_argument("--t2-temperature", type=float, default=SUPPRESS)
    llm_group.add_argument(
        "--temperature", dest="legacy_temperature", type=float, default=SUPPRESS
    )

    output_group = parser.add_argument_group("Output settings")
    output_group.add_argument("--output-root", default=SUPPRESS)
    output_group.add_argument(
        "--output", "-o", dest="legacy_output", default=SUPPRESS
    )
    output_group.add_argument("--note", default=SUPPRESS)

    debug_group = parser.add_argument_group("Debug settings")
    add_boolean_override(debug_group, "--debug")
    add_boolean_override(debug_group, "--verbose")

    logging_group = parser.add_argument_group("Logging overrides")
    add_logging_arguments(logging_group)
    return parser


def _normalize_compat_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map deprecated execution options to canonical fields."""
    if hasattr(args, "legacy_single"):
        warn_deprecated_option(args, "legacy_single", "--single/-s", "--workers 1")
        if not hasattr(args, "workers"):
            args.workers = 1
    if hasattr(args, "legacy_config"):
        warn_deprecated_option(args, "legacy_config", "--config/-c", "--llm-config")
        if not hasattr(args, "llm_config"):
            args.llm_config = args.legacy_config
    if hasattr(args, "legacy_temperature"):
        warn_deprecated_option(
            args,
            "legacy_temperature",
            "--temperature",
            "--t1-temperature and --t2-temperature",
        )
        if not hasattr(args, "t1_temperature"):
            args.t1_temperature = args.legacy_temperature
        if not hasattr(args, "t2_temperature"):
            args.t2_temperature = args.legacy_temperature
    if hasattr(args, "legacy_output"):
        warn_deprecated_option(args, "legacy_output", "--output", "--output-root")
        if not hasattr(args, "output_root"):
            args.output_root = args.legacy_output
    if hasattr(args, "legacy_no_file"):
        warn_deprecated_option(args, "legacy_no_file", "--no-file", "--no-file-log")
        if not hasattr(args, "file_output"):
            args.file_output = False
    return args


def build_config(args: argparse.Namespace, run: RunDir) -> ExecutionRunConfig:
    """Apply CLI values as replacements over the centralized settings."""
    settings = get_settings()
    url_overrides = parse_url_overrides(value_or(args, "url_override", []))
    environment = replace(
        settings.environment,
        env_domain=value_or(args, "env_domain", settings.environment.env_domain),
        auto_login=value_or(args, "auto_login", settings.environment.auto_login),
        url_overrides=url_overrides,
        allowed_domains=value_or(
            args, "allowed_domain", settings.environment.allowed_domains
        ),
    )
    execution = replace(
        settings.execution,
        max_steps=value_or(args, "max_steps", settings.execution.max_steps),
        failed_retry=value_or(
            args, "failed_retry", settings.execution.failed_retry
        ),
        refine_enabled=value_or(
            args, "refine", settings.execution.refine_enabled
        ),
        enable_vision=value_or(
            args, "vision", settings.execution.enable_vision
        ),
        num_workers=value_or(args, "workers", settings.execution.num_workers),
        t1_temperature=value_or(
            args, "t1_temperature", settings.execution.t1_temperature
        ),
        t2_temperature=value_or(
            args, "t2_temperature", settings.execution.t2_temperature
        ),
        config_path=value_or(args, "llm_config", settings.execution.config_path),
        log_level=value_or(args, "log_level", settings.execution.log_level),
        log_dir=value_or(args, "log_dir", settings.execution.log_dir),
        console_output=value_or(
            args, "console_output", settings.execution.console_output
        ),
        file_output=value_or(args, "file_output", settings.execution.file_output),
    )
    browser = replace(
        settings.browser,
        headless=value_or(args, "headless", settings.browser.headless),
    )
    run_output = replace(settings.output, root_dir=str(run.path))
    debug = replace(
        settings.debug,
        debug=value_or(args, "debug", settings.debug.debug),
        verbose=value_or(args, "verbose", settings.debug.verbose),
    )
    config = ExecutionRunConfig(
        execution=execution,
        browser=browser,
        environment=environment,
        debug=debug,
        output=run_output,
        output_dir=run.path,
    )
    config.validate()
    config.environment.pre_process()
    return config


def _result_dict(task_id: str, result: Any) -> dict[str, Any]:
    """Convert the single-agent result to the source-compatible schema."""
    return {
        "task_id": task_id,
        "success": result.end_reason == "completed",
        "end_reason": result.end_reason,
        "steps": result.steps,
        "refine_count": result.refine_count,
        "auto_eval": result.auto_eval,
        "task": result.task,
        "original_task": result.original_task,
    }


def run_single_worker(
    tasks: list[dict[str, Any]],
    config: ExecutionRunConfig,
) -> list[dict[str, Any]]:
    """Execute tasks sequentially and checkpoint after every task."""
    from tasks.execution.collaborative_agent import CollaborativeAgent
    from tasks.execution.execution_agent import ExecutionAgent

    agent_class = (
        CollaborativeAgent if config.execution.refine_enabled else ExecutionAgent
    )
    agent = agent_class(config, enable_vision=config.execution.enable_vision)
    results: list[dict[str, Any]] = []
    try:
        for task in tasks:
            task_id = str(task.get("id"))
            try:
                results.append(_result_dict(task_id, agent.execute_task(task)))
            except Exception as error:  # noqa: BLE001
                logger.exception("Task %s failed", task_id)
                results.append(
                    {
                        "task_id": task_id,
                        "success": False,
                        "end_reason": "error",
                        "error": str(error),
                        "steps": 0,
                    }
                )
            agent.save_state()
    finally:
        agent.save_state()
        agent.close()
    return results


def run_multi_worker(
    tasks: list[dict[str, Any]],
    config: ExecutionRunConfig,
) -> list[dict[str, Any]]:
    """Execute tasks through the existing multiprocessing executor."""
    from tasks.execution.multi_executor import MultiExecutor

    executor = MultiExecutor(
        config=config,
        num_workers=config.execution.num_workers,
        enable_collaborative=config.execution.refine_enabled,
    )
    return [
        {
            "task_id": result.task_id,
            "worker_id": result.worker_id,
            "success": result.success,
            "error": result.error,
            "execution_time": result.execution_time,
        }
        for result in executor.execute_tasks(tasks)
    ]


def save_results(results: list[dict[str, Any]], run: RunDir) -> Path:
    """Write the source-style timestamped execution summary."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = run.path / f"results_{timestamp}.json"
    successful = sum(bool(item.get("success")) for item in results)
    from storage.io import write_json

    write_json(
        {
            "timestamp": timestamp,
            "total_tasks": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "success_rate": (
                f"{successful / len(results) * 100:.1f}%" if results else "N/A"
            ),
            "results": results,
        },
        path,
    )
    return path


def _existing_outputs(run: RunDir) -> list[str]:
    """List materialized artifacts, excluding the manifest itself."""
    paths = [
        path
        for path in run.path.rglob("*")
        if path.is_file() and path != run.manifest_path
    ]
    return [str(path) for path in sorted(paths)]


def main(argv: list[str] | None = None) -> int:
    """Execute a trajectory run and always persist its manifest."""
    args = _normalize_compat_args(build_parser().parse_args(argv))
    from storage.runs import new_run, write_manifest

    settings = get_settings()
    run = new_run(
        "task_execution",
        note=value_or(args, "note", settings.output.note),
        root_dir=value_or(args, "output_root", settings.output.root_dir),
    )
    config: ExecutionRunConfig | None = None
    results: list[dict[str, Any]] = []
    status: dict[str, Any] = {"interrupted": False, "success": False}
    exit_code = 0
    try:
        config = build_config(args, run)
        effective_log_dir = config.execution.log_dir or run.path / "logs"
        config.execution.log_dir = str(effective_log_dir)
        configure_logging(
            level=config.execution.log_level,
            log_dir=effective_log_dir,
            filename_prefix="task_execution",
            console=config.execution.console_output,
            file=config.execution.file_output,
        )
        tasks = resolve_task_urls(load_tasks(args.tasks), config.environment)
        if not tasks:
            raise ValueError("No valid tasks to execute")
        results = (
            run_single_worker(tasks, config)
            if config.execution.num_workers == 1
            else run_multi_worker(tasks, config)
        )
        save_results(results, run)
        status = {
            "interrupted": False,
            "success": all(item.get("success", False) for item in results),
            "results": results,
        }
        exit_code = 0 if status["success"] else 1
    except KeyboardInterrupt:
        status = {"interrupted": True, "success": False, "results": results}
        logger.warning("Execution interrupted by user")
        exit_code = 130
    except Exception as error:  # noqa: BLE001
        status = {
            "interrupted": False,
            "success": False,
            "error": str(error),
            "results": results,
        }
        logger.exception("Trajectory execution failed")
        exit_code = 1
    finally:
        task_path = Path(args.tasks).expanduser().resolve()
        config_snapshot = config.to_dict() if config is not None else vars(args)
        write_manifest(
            run,
            config_snapshot=config_snapshot,
            inputs={"tasks": str(task_path), "status": status},
            outputs=_existing_outputs(run),
            upstream_runs=(
                [str(task_path.parent)]
                if (task_path.parent / "manifest.json").exists()
                else []
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
