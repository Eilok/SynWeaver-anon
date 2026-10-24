"""
WebArena Path Resolution

Single source of truth for locating the external ``webarena`` source tree and
making it importable from SynWeaver.

Two distinct import styles must both work:

* External code (``browser_adapter.py``) uses ``from webarena.browser_env import ...``
  (with the ``webarena.`` prefix) -> requires webarena's *parent* dir
  (``synweaver/``) on ``sys.path``.
* WebArena's own code and the ``auto_login.py`` subprocess use
  ``from browser_env import ...`` (no prefix) -> requires the *webarena dir
  itself* on ``sys.path`` / subprocess ``cwd`` / ``PYTHONPATH``.

``ensure_webarena_importable()`` satisfies both.
"""

import os
import sys
from pathlib import Path


def _resolve_webarena_root() -> Path:
    """Resolve the webarena source directory.

    Resolution prefers an explicit environment variable, then supported local
    source layouts used by the refactored and original workspaces.
    """
    env_root = os.environ.get("WEBARENA_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"WEBARENA_ROOT is not a directory: {path}")

    source = Path(__file__).resolve()
    candidates = (
        source.parents[2] / "webarena",
        source.parents[3] / "webarena",
        source.parents[4] / "_synweaver" / "webarena",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"WebArena source not found; set WEBARENA_ROOT. Searched: {searched}"
    )


WEBARENA_ROOT: Path = _resolve_webarena_root()
SYNWEAVER_ROOT: Path = WEBARENA_ROOT.parent


def _prepend_pythonpath(path: str) -> None:
    """Prepend ``path`` to ``os.environ['PYTHONPATH']`` (deduped)."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = existing.split(os.pathsep) if existing else []
    if path not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([path, *parts]) if parts else path


def ensure_webarena_importable() -> Path:
    """Make webarena importable both with and without the ``webarena.`` prefix.

    - Inserts ``SYNWEAVER_ROOT`` on ``sys.path`` (for ``import webarena.browser_env``).
    - Inserts ``WEBARENA_ROOT`` on ``sys.path`` (for bare ``import browser_env``).
    - Prepends ``WEBARENA_ROOT`` to ``PYTHONPATH`` (for spawned subprocesses).

    Returns the resolved ``WEBARENA_ROOT``.
    """
    for p in (str(SYNWEAVER_ROOT), str(WEBARENA_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _prepend_pythonpath(str(WEBARENA_ROOT))
    return WEBARENA_ROOT


def webarena_subprocess_env() -> dict:
    """Return an ``os.environ`` copy with ``WEBARENA_ROOT`` prepended to PYTHONPATH.

    For use as the ``env`` argument to ``subprocess`` calls that run webarena
    scripts (e.g. ``browser_env/auto_login.py``) which import ``browser_env``
    as a top-level module.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = existing.split(os.pathsep) if existing else []
    root = str(WEBARENA_ROOT)
    if root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([root, *parts]) if parts else root
    return env
