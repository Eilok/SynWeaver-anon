"""Log in to a WebArena site and save its Playwright storage state."""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
from dataclasses import replace
from pathlib import Path

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from cli.common import SUPPRESS, add_boolean_override, provided, warn_deprecated_option
from config.crawler import (
    CRAWLER_SITE_PROFILES,
    LoginSettings,
    get_crawler_site_profile,
)
from logging_utils import configure_logging

logger = logging.getLogger(__name__)


def default_login_url(site: str, host: str) -> str:
    """Build the centralized default login URL for a supported site."""
    profile = get_crawler_site_profile(site)
    if profile.login_port is None or profile.login_path is None:
        raise ValueError(f"Site {site!r} does not support the login command")
    return f"http://{host}:{profile.login_port}{profile.login_path}"


def wait_for_login(page: Page, submit_button: Locator, timeout_ms: int) -> None:
    """Wait until the login form disappears."""
    try:
        submit_button.wait_for(state="hidden", timeout=timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"Login form is still visible on {page.url}") from exc


def login_shopping(page: Page, username: str, password: str) -> Locator:
    page.get_by_label("Email", exact=True).fill(username)
    page.get_by_label("Password", exact=True).fill(password)
    button = page.get_by_role("button", name="Sign In")
    button.click()
    return button


def login_shopping_admin(page: Page, username: str, password: str) -> Locator:
    page.get_by_placeholder("user name").fill(username)
    page.get_by_placeholder("password").fill(password)
    button = page.get_by_role("button", name="Sign in")
    button.click()
    return button


def login_reddit(page: Page, username: str, password: str) -> Locator:
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password").fill(password)
    button = page.get_by_role("button", name="Log in")
    button.click()
    return button


def login_gitlab(page: Page, username: str, password: str) -> Locator:
    page.get_by_test_id("username-field").fill(username)
    page.get_by_test_id("password-field").fill(password)
    button = page.get_by_test_id("sign-in-button")
    button.click()
    return button


LOGIN_HANDLERS = {
    "shopping": login_shopping,
    "shopping_admin": login_shopping_admin,
    "reddit": login_reddit,
    "gitlab": login_gitlab,
}


def save_login_state(site: str, username: str, password: str, url: str, output: Path,
                     *, headless: bool, timeout_ms: int, slow_mo_ms: int) -> None:
    """Log in and persist cookies and local storage."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            submit_button = LOGIN_HANDLERS[site](page, username, password)
            wait_for_login(page, submit_button, timeout_ms)
            context.storage_state(path=str(output))
        finally:
            browser.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the login parser without duplicating centralized defaults."""
    parser = argparse.ArgumentParser(description="Log in and save Playwright storage state")
    parser.add_argument(
        "--site",
        required=True,
        choices=tuple(CRAWLER_SITE_PROFILES) + ("reddit",),
    )
    parser.add_argument("--username", default=SUPPRESS)
    parser.add_argument("--account", dest="legacy_account", default=SUPPRESS)
    parser.add_argument("--password", default=SUPPRESS)
    parser.add_argument("--url", default=SUPPRESS)
    parser.add_argument("--storage-state", type=Path, default=SUPPRESS)
    add_boolean_override(parser, "--headless")
    parser.add_argument("--headed", dest="legacy_headed", action="store_true", default=SUPPRESS)
    parser.add_argument("--timeout", dest="timeout_ms", type=int, default=SUPPRESS, metavar="MS")
    parser.add_argument("--slow-mo", dest="slow_mo_ms", type=int, default=SUPPRESS, metavar="MS")
    return parser


def resolve_login_options(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    """Resolve profile and explicit login values without launching a browser."""
    settings = LoginSettings.from_env()
    profile = get_crawler_site_profile(args.site)
    if profile.login_site is None:
        parser.error("The map site does not support login; use its crawler login script")

    if provided(args, "legacy_account"):
        warn_deprecated_option(args, "legacy_account", "--account", "--username")
    if provided(args, "username"):
        username = args.username
    elif provided(args, "legacy_account"):
        username = args.legacy_account
    else:
        username = settings.username
    password = getattr(args, "password", settings.password)

    if provided(args, "legacy_headed"):
        warn_deprecated_option(args, "legacy_headed", "--headed", "--no-headless")
        if provided(args, "headless"):
            parser.error("--headed cannot be combined with --headless/--no-headless")
        settings = replace(settings, headless=False)

    changes = {
        name: getattr(args, name)
        for name in ("headless", "timeout_ms", "slow_mo_ms")
        if provided(args, name)
    }
    if changes:
        settings = replace(settings, **changes)

    if provided(args, "url"):
        url = args.url
    else:
        if not settings.web_env_host:
            parser.error("WEB_ENV_HOST is required when --url is omitted")
        url = default_login_url(args.site, settings.web_env_host)

    output = getattr(args, "storage_state", None)
    if output is None:
        output = settings.auth_dir / str(profile.storage_state_name)
    return settings, profile.login_site, username, password, url, output


def main(argv: list[str] | None = None) -> int:
    configure_logging(level="INFO", console=True, file=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    settings, site, username, password, url, output = resolve_login_options(args, parser)
    if not username:
        parser.error("Specify an account with --username or LOGIN_USERNAME")
    if not password:
        if not sys.stdin.isatty():
            parser.error("Non-interactive use requires --password or LOGIN_PASSWORD")
        password = getpass.getpass("Password: ")
    try:
        save_login_state(
            site,
            username,
            password,
            url,
            output,
            headless=settings.headless,
            timeout_ms=settings.timeout_ms,
            slow_mo_ms=settings.slow_mo_ms,
        )
    except Exception:
        logger.exception("Login failed")
        return 1
    logger.info("Login state saved to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
