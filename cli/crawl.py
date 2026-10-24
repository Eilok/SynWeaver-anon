"""Command-line entry point for website crawling."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from cli.common import (
    SUPPRESS,
    add_boolean_override,
    add_logging_arguments,
    add_neo4j_arguments,
    add_output_arguments,
    apply_explicit_overrides,
    provided,
    warn_deprecated_option,
)
from config import get_settings
from config.crawler import (
    CRAWLER_SITE_PROFILES,
    apply_crawler_site_profile,
    get_crawler_site_profile,
)
from crawler.core.artifacts import CrawlerPaths
from crawler.core.crawler import Crawler
from logging_utils import configure_logging
from storage.runs import new_run, write_manifest

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the crawler parser without duplicating centralized defaults."""
    parser = argparse.ArgumentParser(description="Website crawler")
    required_group = parser.add_argument_group("Target")
    required_group.add_argument("--url", required=True)
    required_group.add_argument("--site", choices=tuple(CRAWLER_SITE_PROFILES), default=SUPPRESS)
    required_group.add_argument("--neo4j-namespace", default=SUPPRESS)

    browser_group = parser.add_argument_group("Browser and crawl overrides")
    add_boolean_override(browser_group, "--headless")
    browser_group.add_argument("--browser", dest="browser_type", default=SUPPRESS)
    browser_group.add_argument("--viewport-width", type=int, default=SUPPRESS)
    browser_group.add_argument("--viewport-height", type=int, default=SUPPRESS)
    browser_group.add_argument("--storage-state-path", default=SUPPRESS)
    browser_group.add_argument("--exit-btn", dest="exit_buttons", nargs="+", default=SUPPRESS)
    add_boolean_override(browser_group, "--human-review", dest="enable_human_review")
    browser_group.add_argument("--max-depth", type=int, default=SUPPRESS)
    browser_group.add_argument("--max-states", type=int, default=SUPPRESS)
    add_boolean_override(browser_group, "--allow-external-domains")
    browser_group.add_argument("--video-dir", default=SUPPRESS)
    browser_group.add_argument("--exclude-url-patterns", nargs="*", default=SUPPRESS)
    browser_group.add_argument("--login-script", default=SUPPRESS)
    browser_group.add_argument("--skip-action-types", nargs="*", default=SUPPRESS)
    add_boolean_override(browser_group, "--aggressive-url-dedup")

    neo4j_group = parser.add_argument_group("Neo4j overrides")
    add_neo4j_arguments(neo4j_group)

    output_group = parser.add_argument_group("Output settings")
    add_output_arguments(output_group, include_note=False)

    logging_group = parser.add_argument_group("Logging overrides")
    logging_group.add_argument("--log-file", default=SUPPRESS)
    add_logging_arguments(logging_group)
    return parser


def resolve_settings(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    """Merge defaults, an optional site profile, and explicit CLI values."""
    profile = get_crawler_site_profile(args.site) if provided(args, "site") else None
    namespace = getattr(args, "neo4j_namespace", None) or (profile.namespace if profile else None)
    if not namespace:
        parser.error("--neo4j-namespace is required when --site is omitted")

    settings = get_settings(namespace)
    if profile is not None:
        settings.crawler = apply_crawler_site_profile(settings.crawler, profile)

    settings.neo4j = apply_explicit_overrides(
        settings.neo4j,
        args,
        {
            "neo4j_url": "url",
            "neo4j_username": "username",
            "neo4j_password": "password",
            "neo4j_database": "database",
        },
    )
    settings.neo4j = replace(settings.neo4j, namespace=namespace)
    settings.crawler = apply_explicit_overrides(settings.crawler, args)
    if provided(args, "output_root"):
        settings.output = replace(settings.output, root_dir=args.output_root)

    if provided(args, "legacy_no_file"):
        warn_deprecated_option(args, "legacy_no_file", "--no-file", "--no-file-log")
        settings.crawler = replace(settings.crawler, file_output=False)
    return settings, namespace


def _artifact_outputs(paths: CrawlerPaths, log_path: Path | None,
                      video_path: str | None) -> list[str]:
    candidates = [Path(paths.screenshot_dir), Path(paths.labelled_screenshot_dir),
                  Path(paths.operation_screenshot_dir), Path(paths.element_screenshot_dir),
                  Path(paths.accessibility_tree_dir), paths.logs_dir]
    if log_path:
        candidates.append(log_path)
    if video_path:
        candidates.append(Path(video_path))
    return [str(path) for path in candidates if path.exists()]


def main(argv: list[str] | None = None) -> int:
    """Run one crawler invocation and persist its manifest."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings, namespace = resolve_settings(args, parser)
    run = new_run("crawler", note=namespace, root_dir=settings.output.root_dir)
    paths = CrawlerPaths(run)
    effective_video_dir = settings.crawler.video_dir or paths.video_dir
    settings.crawler.video_dir = effective_video_dir
    effective_log_dir = settings.crawler.log_dir or paths.logs_dir
    settings.crawler.log_dir = str(effective_log_dir)
    log_path = configure_logging(
        level=settings.crawler.log_level, log_file=settings.crawler.log_file,
        log_dir=effective_log_dir, filename_prefix="crawler",
        console=settings.crawler.console_output, file=settings.crawler.file_output,
    )
    crawler: Crawler | None = None
    statistics: dict[str, Any] | None = None
    error: str | None = None
    try:
        logger.info("Website crawler starting")
        logger.info("Target URL: %s", args.url)
        logger.info("Neo4j namespace: %s", namespace)
        crawler = Crawler(
            exit_btns=settings.crawler.exit_buttons,
            enable_human_review=settings.crawler.enable_human_review,
            max_depth=settings.crawler.max_depth,
            max_states=settings.crawler.max_states,
            allow_external_domains=settings.crawler.allow_external_domains,
            neo4j_config=settings.neo4j,
            crawler_paths=paths,
            exclude_url_patterns=settings.crawler.exclude_url_patterns,
            login_script=settings.crawler.login_script,
            skip_action_types=settings.crawler.skip_action_types,
            aggressive_url_dedup=settings.crawler.aggressive_url_dedup,
        )
        graph = crawler.crawl(args.url, {
            "headless": settings.crawler.headless,
            "browser_type": settings.crawler.browser_type,
            "viewport": {"width": settings.crawler.viewport_width,
                         "height": settings.crawler.viewport_height},
            "storage_state_path": settings.crawler.storage_state_path,
            "video_dir": effective_video_dir,
        })
        statistics = graph.get_statistics()
        logger.info("Crawling completed with %s states", graph.state_count())
        return 0
    except KeyboardInterrupt:
        error = "Crawling interrupted by user"
        logger.warning(error)
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Crawling failed")
        raise
    finally:
        if crawler is not None and statistics is None:
            try:
                statistics = crawler.state_graph.get_statistics()
            except Exception:
                logger.warning("Unable to collect graph statistics", exc_info=True)
        inputs: dict[str, Any] = {"namespace": namespace, "url": args.url}
        if statistics is not None:
            inputs["graph_statistics"] = statistics
        if error:
            inputs["error"] = error
        try:
            write_manifest(
                run, config_snapshot=settings, inputs=inputs,
                outputs=_artifact_outputs(paths, log_path,
                                          crawler.video_path if crawler else None),
            )
        except Exception:
            if error is None:
                logger.exception("Unable to write crawler manifest")
                raise
            logger.warning("Unable to write crawler manifest", exc_info=True)
        finally:
            if crawler is not None:
                try:
                    crawler.state_graph.close()
                except Exception:
                    logger.warning("Unable to close Neo4j connection", exc_info=True)
            logger.info("Crawler shutdown complete")


if __name__ == "__main__":
    raise SystemExit(main())
