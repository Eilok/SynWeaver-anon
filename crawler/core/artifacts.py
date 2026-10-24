"""Crawler-specific artifact paths rooted in a shared run directory."""

from dataclasses import dataclass
from pathlib import Path

from storage.runs import RunDir


@dataclass(frozen=True)
class CrawlerPaths:
    """Expose legacy crawler path names without creating another run root."""

    run: RunDir

    @property
    def root_dir(self) -> str:
        return str(self.run.path)

    @property
    def screenshot_dir(self) -> str:
        return str(self.run.path / "original")

    @property
    def labelled_screenshot_dir(self) -> str:
        return str(self.run.path / "labelled")

    @property
    def operation_screenshot_dir(self) -> str:
        return str(self.run.path / "operations")

    @property
    def element_screenshot_dir(self) -> str:
        return str(self.run.path / "elements")

    @property
    def accessibility_tree_dir(self) -> str:
        return str(self.run.path / "accessibility_trees")

    @property
    def video_dir(self) -> str:
        return str(self.run.path / "videos")

    @property
    def logs_dir(self) -> Path:
        return self.run.path / "logs"


__all__ = ["CrawlerPaths"]
