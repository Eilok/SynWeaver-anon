#!/usr/bin/env python3
"""Merge crawler WebM video segments in filename order."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from cli.common import LOG_LEVELS, SUPPRESS, warn_deprecated_option
from logging_utils import configure_logging

logger = logging.getLogger(__name__)


def find_video_files(directory: str) -> List[Path]:
    """Return all WebM files in a directory sorted by filename."""
    video_dir = Path(directory)
    if not video_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    video_files = sorted(video_dir.glob("*.webm"))
    if not video_files:
        raise ValueError(f"No WebM files found in directory: {directory}")
    logger.info("Found %s video files", len(video_files))
    for index, video in enumerate(video_files, 1):
        logger.info("  %s. %s", index, video.name)
    return video_files


def create_concat_file(video_files: List[Path], concat_file_path: str) -> None:
    """Create an ffmpeg concat-demuxer input file."""
    with open(concat_file_path, "w", encoding="utf-8") as concat_file:
        for video in video_files:
            concat_file.write(f"file '{video.absolute()}'\n")
    logger.info("Created concat list: %s", concat_file_path)


def merge_videos_ffmpeg(video_files: List[Path], output_path: str) -> None:
    """Merge video files without re-encoding through ffmpeg."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as temp_file:
        concat_file_path = temp_file.name
    try:
        create_concat_file(video_files, concat_file_path)
        command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i",
                   concat_file_path, "-c", "copy", output_path]
        logger.info("Merging videos")
        logger.info("Command: %s", " ".join(command))
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg failed: %s", result.stderr)
            raise RuntimeError(f"ffmpeg returned code {result.returncode}")
        logger.info("Video merge completed: %s", output_path)
        output_file = Path(output_path)
        if output_file.exists():
            logger.info("Output size: %.2f MB", output_file.stat().st_size / (1024 * 1024))
    finally:
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
            logger.debug("Deleted temporary concat file: %s", concat_file_path)


def check_ffmpeg_available() -> bool:
    """Return whether ffmpeg is available on PATH."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the standalone utility."""
    parser = argparse.ArgumentParser(description="Merge WebM files in a directory")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--debug", dest="legacy_debug", action="store_true", default=SUPPRESS)
    return parser


def resolve_log_level(args: argparse.Namespace) -> str:
    """Map the deprecated debug switch to the canonical logging level."""
    if hasattr(args, "legacy_debug"):
        warn_deprecated_option(args, "legacy_debug", "--debug", "--log-level DEBUG")
        return "DEBUG"
    return args.log_level


def main(argv: list[str] | None = None) -> int:
    """Run the standalone video merge utility."""
    args = build_parser().parse_args(argv)
    log_level = resolve_log_level(args)
    configure_logging(
        level=log_level,
        console=True,
        file=False,
    )
    if not check_ffmpeg_available():
        logger.error("ffmpeg is unavailable on PATH")
        return 1
    try:
        video_files = find_video_files(args.input_dir)
        output_path = args.output or os.path.join(args.input_dir, "merged_video.webm")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Created output directory: %s", output_dir)
        merge_videos_ffmpeg(video_files, output_path)
        return 0
    except Exception as exc:
        logger.error("Video merge failed: %s", exc)
        if log_level == "DEBUG":
            logger.exception("Video merge traceback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
