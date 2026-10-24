"""Summarize WebArena execution results by website."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from cli.common import add_output_arguments, resolve_output_arguments
from storage.runs import new_run, write_manifest

MODULE = "analyze_log_results"


def parse_log_file(log_path: str | Path):
    """Parse task results from an execution log and group them by website."""
    stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})
    current_site = None
    site_pattern = re.compile(r"Sites:\s*\['([^']+)'\]")
    result_pattern = re.compile(r"-\s*Auto eval:\s*(success|failed)", re.IGNORECASE)

    with Path(log_path).open("r", encoding="utf-8") as log_file:
        for line in log_file:
            site_match = site_pattern.search(line)
            if site_match:
                current_site = site_match.group(1)
                continue

            result_match = result_pattern.search(line)
            if result_match and current_site:
                result = result_match.group(1).lower()
                stats[current_site]["total"] += 1
                stats[current_site][result] += 1
                current_site = None

    return stats


def format_stats(stats: Mapping[str, Mapping[str, int]]) -> str:
    """Format parsed statistics as the legacy console table."""
    if not stats:
        return "No task results found in the log file."

    lines = [
        f"{'Site':<20} | {'Total':<8} | {'Success':<8} | {'Failed':<8} | {'Success Rate':<12}",
        "-" * 65,
    ]
    total_success = 0
    total_failed = 0
    total_tasks = 0
    for site, data in sorted(stats.items()):
        success = data["success"]
        failed = data["failed"]
        total = data["total"]
        rate = success / total * 100 if total else 0.0
        lines.append(f"{site:<20} | {total:<8} | {success:<8} | {failed:<8} | {rate:6.2f}%")
        total_success += success
        total_failed += failed
        total_tasks += total

    lines.append("-" * 65)
    overall_rate = total_success / total_tasks * 100 if total_tasks else 0.0
    lines.append(
        f"{'OVERALL':<20} | {total_tasks:<8} | {total_success:<8} | "
        f"{total_failed:<8} | {overall_rate:6.2f}%"
    )
    return "\n".join(lines)


def print_stats(stats: Mapping[str, Mapping[str, int]]) -> None:
    """Print parsed statistics to standard output."""
    print(format_stats(stats))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Parse WebArena execution logs and summarize results by website."
    )
    parser.add_argument("log_file", help="Path to the execution log file")
    output_group = parser.add_argument_group("Output configuration")
    add_output_arguments(output_group, legacy_options=("--output-dir",))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the log analysis CLI and return a process status code."""
    args = build_parser().parse_args(argv)
    output_root, note = resolve_output_arguments(args, default_root="./outputs")
    run = new_run(MODULE, note, root_dir=output_root)
    log_path = Path(args.log_file).expanduser()
    output_path = run.path / "analysis.txt"
    outputs: list[Path] = []
    status = "failed"
    error: str | None = None

    try:
        if not log_path.is_file():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        statistics = parse_log_file(log_path)
        report = f"Results for: {log_path}\n\n{format_stats(statistics)}\n"
        output_path.write_text(report, encoding="utf-8")
        outputs.append(output_path)
        print(report, end="")
        status = "completed"
        return_code = 0
    except (OSError, ValueError) as exc:
        error = str(exc)
        print(f"Error: {error}")
        return_code = 1
    finally:
        write_manifest(
            run,
            config_snapshot={"status": status, "note": note, "error": error},
            inputs={"log_file": str(log_path)},
            outputs=outputs,
            upstream_runs=[log_path.parent],
        )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
