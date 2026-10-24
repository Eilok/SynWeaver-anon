"""Command-line entry point for high-level task generation."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from config import TaskGenRunConfig, get_settings
from llm import get_client
from logging_utils import configure_logging
from storage.io import write_json
from storage.runs import RunDir, new_run, resolve_run_dir, write_manifest
from tasks.generation import GraphSampler, TaskGenerator, write_extended_jsonl

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the task generation parser from centralized defaults."""
    parser = argparse.ArgumentParser(
        description="Generate high-level tasks from Neo4j state graph triplets",
    )
    parser.add_argument("--namespace", required=True, help="Neo4j graph namespace")

    neo4j_group = parser.add_argument_group("Neo4j connection overrides")
    add_neo4j_arguments(neo4j_group)

    website_group = parser.add_argument_group("Website settings")
    website_group.add_argument(
        "--website-name",
        default=SUPPRESS,
        help="Display name; defaults to the Neo4j namespace",
    )
    website_group.add_argument("--target-sites", nargs="+", default=SUPPRESS)

    sampling_group = parser.add_argument_group("Sampling settings")
    sampling_group.add_argument("--max-samples", type=int, default=SUPPRESS)
    sampling_group.add_argument("--tasks-per-transition", type=int, default=SUPPRESS)
    add_worker_argument(
        sampling_group,
        help_text="Number of task generation worker threads",
    )
    sampling_group.add_argument("--min-depth", type=int, default=SUPPRESS)
    sampling_group.add_argument("--max-depth", type=int, default=SUPPRESS)
    add_boolean_override(sampling_group, "--depth-priority")
    add_boolean_override(sampling_group, "--degree-priority")
    add_boolean_override(sampling_group, "--sample-transitions")
    sampling_group.add_argument("--scroll-cap", type=int, default=SUPPRESS)

    llm_group = parser.add_argument_group("Advanced TASK_GEN_MODEL overrides")
    add_llm_arguments(llm_group)

    output_group = parser.add_argument_group("Output settings")
    run_group = output_group.add_mutually_exclusive_group()
    run_group.add_argument("--output-root", default=SUPPRESS)
    run_group.add_argument("--resume-dir", default=SUPPRESS)
    run_group.add_argument(
        "--output-dir",
        dest="legacy_output_dir",
        default=SUPPRESS,
        help="Deprecated alias for --resume-dir",
    )
    add_boolean_override(output_group, "--save-extended")
    output_group.add_argument("--note", default=SUPPRESS)

    log_group = parser.add_argument_group("Logging overrides")
    add_logging_arguments(log_group)
    return parser


def _normalize_compat_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map deprecated task generation aliases to canonical fields."""
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
    if hasattr(args, "legacy_no_file"):
        warn_deprecated_option(args, "legacy_no_file", "--no-file", "--no-file-log")
        if not hasattr(args, "file_output"):
            args.file_output = False
    return args


def _effective_resume_dir(
    args: argparse.Namespace,
    default_resume_dir: str | None,
) -> str | None:
    """Resolve resume mode while allowing an explicit output root to disable it."""
    if hasattr(args, "resume_dir"):
        return args.resume_dir
    if hasattr(args, "output_root"):
        return None
    return default_resume_dir


def _resolve_run(
    args: argparse.Namespace,
    output_root: str,
    default_note: str | None = None,
    default_resume_dir: str | None = None,
) -> RunDir | None:
    """Resolve a fresh or existing run while preserving resume validation."""
    resume_dir = _effective_resume_dir(args, default_resume_dir)
    if resume_dir:
        resume_path = Path(resume_dir).expanduser()
        if not resume_path.exists() or not resume_path.is_dir():
            logger.error("resume-dir not found: %s", resume_path)
            return None
        return resolve_run_dir("task_gen", resume_path)
    return new_run(
        "task_gen",
        note=value_or(args, "note", default_note),
        root_dir=value_or(args, "output_root", output_root),
    )


def _create_run_config(args: argparse.Namespace, run: RunDir) -> TaskGenRunConfig:
    """Apply CLI overrides to authoritative task generation settings."""
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
        settings.task_sampling,
        max_samples=value_or(args, "max_samples", settings.task_sampling.max_samples),
        depth_priority=value_or(
            args, "depth_priority", settings.task_sampling.depth_priority
        ),
        degree_priority=value_or(
            args, "degree_priority", settings.task_sampling.degree_priority
        ),
        min_depth=value_or(args, "min_depth", settings.task_sampling.min_depth),
        max_depth=value_or(args, "max_depth", settings.task_sampling.max_depth),
        sample_transitions=value_or(
            args, "sample_transitions", settings.task_sampling.sample_transitions
        ),
        scroll_cap=value_or(args, "scroll_cap", settings.task_sampling.scroll_cap),
    )
    website_name = value_or(args, "website_name", args.namespace)
    resume_dir = _effective_resume_dir(args, settings.task_generation.resume_dir)
    generation = replace(
        settings.task_generation,
        website_name=website_name,
        target_sites=value_or(args, "target_sites", [website_name]),
        tasks_per_transition=value_or(
            args,
            "tasks_per_transition",
            settings.task_generation.tasks_per_transition,
        ),
        num_workers=value_or(args, "workers", settings.task_generation.num_workers),
        save_extended=value_or(
            args, "save_extended", settings.task_generation.save_extended
        ),
        resume_dir=resume_dir,
        config_path=value_or(
            args, "llm_config", settings.task_generation.config_path
        ),
        log_level=value_or(args, "log_level", settings.task_generation.log_level),
        log_dir=value_or(args, "log_dir", settings.task_generation.log_dir),
        console_output=value_or(
            args, "console_output", settings.task_generation.console_output
        ),
        file_output=value_or(
            args, "file_output", settings.task_generation.file_output
        ),
    )
    config = TaskGenRunConfig(
        neo4j=neo4j,
        sampling=sampling,
        generation=generation,
        output_dir=run.path,
        model=value_or(args, "model", None),
        temperature=value_or(args, "temperature", None),
    )
    config.validate()
    return config


def _artifact_outputs(
    run: RunDir,
    website_name: str,
    save_extended: bool,
    log_path: Path | None,
) -> list[str]:
    """Return only artifacts materialized by the current invocation."""
    names = [
        f"task_{website_name}.jsonl",
        "transition_context.json",
        "sampling_stats.json",
        "generation_summary.json",
    ]
    if save_extended:
        names.append(f"extended_{website_name}.jsonl")
    candidates = [run.path / name for name in names]
    if log_path is not None:
        candidates.append(log_path)
    return [str(path) for path in candidates if path.exists()]


def _generate_tasks(
    generator: TaskGenerator,
    triplets: list[Any],
    target_count: int,
    workers: int,
    resume: bool,
) -> tuple[list[Any], int, int]:
    """Top up sampled transitions sequentially or with the original thread model."""
    task_items: list[tuple[Any, int]] = []
    for triplet in triplets:
        have = generator.task_count_for(triplet)
        needed = max(0, target_count - have)
        task_items.extend((triplet, task_number) for task_number in range(needed))

    if resume:
        logger.info(
            "Resume: topping up %d triplets to at least %d task(s) each; "
            "%d new task(s) to generate",
            len(triplets),
            target_count,
            len(task_items),
        )
    else:
        logger.info(
            "Generating tasks: %d triplets x %d tasks/transition",
            len(triplets),
            target_count,
        )

    results: list[Any] = []
    successful = 0
    failed = 0
    total_tasks = len(task_items)

    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(generator.generate_task, triplet): (triplet, task_number)
            for triplet, task_number in task_items
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        successful += 1
                        preview = result.task[:60] if result.task else ""
                        logger.info("[%d/%d] Generated: %s...", completed, total_tasks, preview)
                    else:
                        failed += 1
                        logger.warning("[%d/%d] Failed: %s", completed, total_tasks, result.error)
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    logger.error("[%d/%d] Exception: %s", completed, total_tasks, error)
                if completed % 10 == 0:
                    generator.save_context()
                    logger.info("Saved context after %d tasks", completed)
        except KeyboardInterrupt:
            pending = sum(future.cancel() for future in futures if not future.done())
            close = getattr(generator.llm_client, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Cancelled %d pending task(s)", pending)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        for index, (triplet, task_number) in enumerate(task_items, start=1):
            logger.info(
                "Processing task %d/%d (triplet task #%d)",
                index,
                total_tasks,
                task_number + 1,
            )
            result = generator.generate_task(triplet)
            results.append(result)
            if result.success:
                successful += 1
                preview = result.task[:60] if result.task else ""
                logger.info("Generated: %s...", preview)
            else:
                failed += 1
                logger.warning("Failed: %s", result.error)
            if index % 10 == 0:
                generator.save_context()
                logger.info("Saved context after %d tasks", index)

    return results, successful, failed


def run_task_generation(
    config: TaskGenRunConfig,
    run: RunDir,
) -> dict[str, Any]:
    """Run sampling, generation, resume top-up, and artifact writing."""
    overrides: dict[str, Any] = {}
    if config.model is not None:
        overrides["model"] = config.model
    if config.temperature is not None:
        overrides["temperature"] = config.temperature
    client = get_client(
        "task_gen",
        conf_path=config.generation.config_path,
        overrides=overrides or None,
        track_tokens=True,
        fresh=True,
    )

    logger.info(
        "Neo4j configuration: url=%s, namespace=%s",
        config.neo4j.url,
        config.neo4j.namespace,
    )
    logger.info(
        "Sampling configuration: depth_priority=%s, degree_priority=%s",
        config.sampling.depth_priority,
        config.sampling.degree_priority,
    )

    with GraphSampler(config.neo4j, config.sampling) as sampler:
        total_transitions = sampler.count_transitions()
        logger.info("Total transitions in graph: %d", total_transitions)
        triplets = sampler.sample(config.sampling.max_samples)
        logger.info("Sampled %d triplets", len(triplets))
        if not triplets:
            logger.warning("No triplets sampled. Check namespace and graph data.")
            return {
                "success": False,
                "error": "No triplets sampled",
                "total_transitions": total_transitions,
            }

        context_path = run.path / "transition_context.json"
        generator = TaskGenerator(
            llm_client=client,
            graph_sampler=sampler,
            website_name=config.generation.website_name or "",
            target_sites=config.generation.target_sites,
            context_save_path=context_path,
        )
        logger.info("Using %d parallel worker(s)", config.generation.num_workers)
        target_count = max(1, config.generation.tasks_per_transition)
        results, successful, failed = _generate_tasks(
            generator,
            triplets,
            target_count,
            config.generation.num_workers,
            bool(config.generation.resume_dir),
        )

        generator.save_context()
        website_name = config.generation.website_name or ""
        output_file = run.path / f"task_{website_name}.jsonl"
        written_count = generator.write_context_jsonl(output_file)
        logger.info("Wrote %d tasks from full context to %s", written_count, output_file)

        if config.generation.save_extended:
            extended_file = run.path / f"extended_{website_name}.jsonl"
            write_extended_jsonl(results, extended_file)

        stats = sampler.get_statistics()
        write_json(stats, run.path / "sampling_stats.json")
        total_generated = successful + failed
        summary = {
            "timestamp": run.run_id,
            "namespace": config.neo4j.namespace,
            "website_name": website_name,
            "start_url_source": "neo4j_source_node",
            "target_sites": config.generation.target_sites,
            "triplets_sampled": len(triplets),
            "tasks_per_transition": config.generation.tasks_per_transition,
            "parallel_workers": config.generation.num_workers,
            "total_generated": total_generated,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_generated if total_generated > 0 else 0,
            "output_file": str(output_file),
            "sampling_stats": stats,
            "context_summary": generator.get_context_summary(),
        }
        write_json(summary, run.path / "generation_summary.json")
        logger.info(
            "Task generation complete: %d successful, %d failed, output=%s",
            successful,
            failed,
            output_file,
        )
        return {"success": True, **summary}


def main(argv: list[str] | None = None) -> int:
    """Run task generation and persist a manifest for fresh and resumed runs."""
    args = _normalize_compat_args(build_parser().parse_args(argv))
    settings = get_settings(args.namespace)
    run = _resolve_run(
        args,
        settings.output.root_dir,
        settings.output.note,
        settings.task_generation.resume_dir,
    )
    if run is None:
        return 1
    config = _create_run_config(args, run)
    effective_log_dir = config.generation.log_dir or run.path / "logs"
    config.generation.log_dir = str(effective_log_dir)
    log_path = configure_logging(
        level=config.generation.log_level,
        log_dir=effective_log_dir,
        filename_prefix=f"task_gen_{config.generation.website_name}",
        console=config.generation.console_output,
        file=config.generation.file_output,
    )

    result: dict[str, Any] = {}
    exit_code = 0
    try:
        logger.info("Task generation pipeline starting")
        logger.info("Namespace: %s", config.neo4j.namespace)
        logger.info("Website: %s", config.generation.website_name)
        logger.info("Output directory: %s", run.path)
        result = run_task_generation(config, run)
        if not result.get("success"):
            logger.error("Task generation failed: %s", result.get("error", "Unknown error"))
            exit_code = 1
    except KeyboardInterrupt:
        result = {"success": False, "error": "Interrupted by user"}
        logger.info("Task generation interrupted by user")
        exit_code = 130
    except Exception as error:  # noqa: BLE001
        result = {"success": False, "error": f"{type(error).__name__}: {error}"}
        logger.exception("Task generation failed")
        exit_code = 1
    finally:
        inputs: dict[str, Any] = {
            "namespace": config.neo4j.namespace,
            "website_name": config.generation.website_name,
            "upstream_crawler_namespace": config.neo4j.namespace,
            "resume_dir": config.generation.resume_dir,
            "result": result,
        }
        upstream_runs: list[str] = []
        try:
            write_manifest(
                run,
                config_snapshot=config,
                inputs=inputs,
                outputs=_artifact_outputs(
                    run,
                    config.generation.website_name or "",
                    config.generation.save_extended,
                    log_path,
                ),
                upstream_runs=upstream_runs,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unable to write task generation manifest")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
