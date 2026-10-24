"""
Site-specific extension hooks for the crawler.

The crawler core is intentionally website-agnostic. Any behaviour that depends on
a particular website (e.g. how to log in, or how to fill a specific form field)
lives in a *user-supplied hook script* loaded at runtime — never hardcoded here.

A hook script is an ordinary Python file that defines one or more module-level
functions following the protocol below. Point the crawler at it via the
``--login-script`` CLI flag (see ``main_crawler.py``) or the ``login_script``
argument of :class:`~crawler.core.crawler.Crawler`.

Hook protocol
-------------
``login(browser_session)``  (optional)
    Called once right after the start URL is opened, and again after every
    context reset during backtracking/replay (``browser_session.reset_context()``
    wipes cookies/session, so login must be re-applied). Use it to drive the
    login flow with Playwright. ``browser_session`` is a
    :class:`~crawler.core.browser_session.BrowserSession`; the raw
    Playwright ``Page`` is ``browser_session.page``.

    Example::

        def login(browser_session):
            page = browser_session.page
            page.get_by_placeholder("user name").fill("admin")
            page.get_by_role("button", name="Sign in").click()
            browser_session._wait_for_stability()

``override_form_value(field, default_value)``  (optional)
    Called for each form input field the crawler is about to fill. ``field`` is
    a dict describing the input (keys include ``xpath``, ``tag``, ``name``,
    ``ariaLabel``, ``text``). ``default_value`` is the value the crawler would
    otherwise use (LLM-generated). Return a replacement string to override it,
    or ``None`` to keep ``default_value``.

    Example::

        def override_form_value(field, default_value):
            if field.get("xpath") == "/html/body/.../input[1]":
                return "restaurant"
            return None

Both functions are optional. A script defining only ``login`` (or only
``override_form_value``) is valid; the absent hook is simply not used. A missing
or empty path means no hooks are loaded and the crawler behaves exactly as if no
script were provided.

See ``crawler/hooks/login_script_template.py`` for a ready-to-copy
template and ``crawler/hooks/README.md`` for full usage guidance.
"""
from typing import Callable, Optional
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)


def _load_hook_module(path: str):
    """
    Dynamically load a user-supplied hook script as a module.

    Args:
        path: Filesystem path to the .py hook script.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImportError: If the file cannot be loaded as a Python module.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hook script not found: {path}")

    module_name = f"_crawler_site_hook_{os.path.splitext(os.path.basename(path))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hook script as a module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_callable(path: Optional[str], attr: str, *, required: bool) -> Optional[Callable]:
    """
    Load a named callable from a user-supplied hook script.

    Args:
        path: Path to the hook script (None/empty -> returns None).
        attr: Name of the module-level function to extract.
        required: If True, raise when the attribute is missing or not callable.
                  If False, return None when the attribute is absent.

    Returns:
        The callable, or None when no hook is available.
    """
    if not path:
        return None

    module = _load_hook_module(path)
    hook = getattr(module, attr, None)

    if hook is None:
        if required:
            raise AttributeError(
                f"Hook script '{path}' does not define a '{attr}(...)' function. "
                f"See crawler/hooks/README.md for the hook protocol."
            )
        return None

    if not callable(hook):
        raise TypeError(f"'{attr}' in hook script '{path}' is not callable.")

    return hook


def load_login_hook(path: Optional[str]) -> Optional[Callable]:
    """
    Load the ``login(browser_session)`` hook from a user script.

    Args:
        path: Path to the hook script. None or empty -> returns None (no login).

    Returns:
        The ``login`` callable, or None if no script was provided.

    Raises:
        AttributeError: If a script is provided but defines no ``login`` function.
    """
    hook = _load_callable(path, "login", required=True)
    if hook is not None:
        logger.info(f"Loaded login hook from {path}")
    return hook


def load_form_value_hook(path: Optional[str]) -> Optional[Callable]:
    """
    Load the optional ``override_form_value(field, default_value)`` hook.

    Shares the same script file as :func:`load_login_hook`. Unlike ``login``,
    this hook is optional: a script without ``override_form_value`` simply yields
    None and no overriding is performed.

    Args:
        path: Path to the hook script. None or empty -> returns None.

    Returns:
        The ``override_form_value`` callable, or None if absent.
    """
    hook = _load_callable(path, "override_form_value", required=False)
    if hook is not None:
        logger.info(f"Loaded form-value override hook from {path}")
    return hook
