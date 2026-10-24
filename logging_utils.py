"""Shared logging configuration for all project modules."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

DEFAULT_CONSOLE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_FILE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio", "neo4j")


def parse_log_level(level: str | int) -> int:
    """Parse one of the supported logging levels."""
    if isinstance(level, int):
        if level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            return level
        raise ValueError(f"Unsupported log level: {level}")
    if not isinstance(level, str):
        raise ValueError(f"Unsupported log level: {level!r}")
    normalized = level.strip().upper()
    levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
              "WARN": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    if normalized not in levels:
        raise ValueError(f"Unsupported log level: {level!r}")
    return levels[normalized]


def create_file_handler(path: str | Path, level: str | int = logging.INFO,
                        formatter: logging.Formatter | None = None) -> logging.Handler:
    """Create a UTF-8 file handler using the shared logging policy."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output, encoding="utf-8")
    handler.setLevel(parse_log_level(level))
    handler.setFormatter(formatter or logging.Formatter(DEFAULT_FILE_FORMAT))
    return handler


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None,
                      log_dir: str | Path | None = None, filename_prefix: str = "run",
                      console: bool = True, file: bool = True, reset_handlers: bool = True,
                      console_format: str = DEFAULT_CONSOLE_FORMAT,
                      file_format: str = DEFAULT_FILE_FORMAT) -> Path | None:
    """Configure root console and file handlers and return the file path."""
    numeric_level = parse_log_level(level)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    if reset_handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
    if console:
        stream = logging.StreamHandler()
        stream.setLevel(numeric_level)
        stream.setFormatter(logging.Formatter(console_format))
        root.addHandler(stream)
    output_path: Path | None = None
    if file:
        if log_file is not None:
            output_path = Path(log_file).expanduser()
        else:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            output_path = Path(log_dir or "./logs").expanduser() / f"{filename_prefix}_{timestamp}.log"
        root.addHandler(create_file_handler(output_path, numeric_level,
                                            logging.Formatter(file_format)))
    for logger_name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    return output_path


__all__ = ["configure_logging", "create_file_handler", "parse_log_level"]
