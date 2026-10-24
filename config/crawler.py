"""Website crawler and login runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class CrawlerSettings:
    """Configure browser exploration without duplicating shared connection settings."""

    headless: bool = False
    browser_type: str = "chromium"
    viewport_width: int = 1366
    viewport_height: int = 950
    storage_state_path: str | None = None
    video_dir: str | None = None
    exit_buttons: list[str] = field(default_factory=list)
    enable_human_review: bool = True
    max_depth: int | None = None
    max_states: int | None = None
    allow_external_domains: bool = False
    exclude_url_patterns: list[str] = field(default_factory=list)
    login_script: str | None = None
    skip_action_types: list[str] = field(default_factory=list)
    aggressive_url_dedup: bool = False
    log_file: str | None = None
    log_dir: str | None = None
    log_level: str = "INFO"
    console_output: bool = True
    file_output: bool = True


@dataclass(frozen=True)
class CrawlerSiteProfile:
    """Define one stable WebArena crawler and authentication combination."""

    namespace: str
    exit_buttons: tuple[str, ...] = ()
    exclude_url_patterns: tuple[str, ...] = ()
    login_script: str | None = None
    skip_action_types: tuple[str, ...] = ()
    aggressive_url_dedup: bool = False
    enable_human_review: bool | None = None
    auth_method: str = "none"
    login_site: str | None = None
    login_port: int | None = None
    login_path: str | None = None
    storage_state_name: str | None = None


CRAWLER_SITE_PROFILES: dict[str, CrawlerSiteProfile] = {
    "shopping": CrawlerSiteProfile(
        namespace="shopping",
        exit_buttons=("Sign Out",),
        enable_human_review=False,
        auth_method="storage_state",
        login_site="shopping",
        login_port=7770,
        login_path="/customer/account/login/",
        storage_state_name="shopping_state.json",
    ),
    "shopping_admin": CrawlerSiteProfile(
        namespace="shopping_admin",
        exit_buttons=("Sign Out", "Cancel"),
        login_script="crawler/hooks/login_script_shopping_admin.py",
        skip_action_types=("fill",),
        auth_method="login_script",
        login_site="shopping_admin",
        login_port=7780,
        login_path="/admin",
        storage_state_name="shopping_admin_state.json",
    ),
    "forum": CrawlerSiteProfile(
        namespace="forum",
        exit_buttons=("Log out",),
        aggressive_url_dedup=True,
        enable_human_review=False,
        auth_method="storage_state",
        login_site="reddit",
        login_port=9999,
        login_path="/login",
        storage_state_name="reddit_state.json",
    ),
    "gitlab": CrawlerSiteProfile(
        namespace="gitlab",
        exit_buttons=("Sign out", "Sign out and sign in with a different account"),
        exclude_url_patterns=("/help",),
        enable_human_review=False,
        auth_method="storage_state",
        login_site="gitlab",
        login_port=8023,
        login_path="/users/sign_in",
        storage_state_name="gitlab_state.json",
    ),
    "map": CrawlerSiteProfile(
        namespace="map",
        exit_buttons=("Log In", "Sign Up", "Edit", "OpenStreetMap Nominatim"),
        exclude_url_patterns=("/about", "/login"),
        login_script="crawler/hooks/login_script_map.py",
        auth_method="crawler_login_script_only",
    ),
}

SITE_ALIASES = {"forum": "reddit"}
PROFILE_ALIASES = {"reddit": "forum"}


@dataclass
class LoginSettings:
    """Configure the auxiliary Playwright login command."""

    auth_dir: Path = Path(".auth")
    web_env_host: str | None = None
    username: str | None = None
    password: str | None = None
    headless: bool = True
    timeout_ms: int = 60_000
    slow_mo_ms: int = 0

    @classmethod
    def from_env(cls) -> "LoginSettings":
        """Resolve login environment values while retaining centralized defaults."""
        defaults = cls()
        return cls(
            auth_dir=defaults.auth_dir,
            web_env_host=os.environ.get("WEB_ENV_HOST"),
            username=os.environ.get("LOGIN_USERNAME"),
            password=os.environ.get("LOGIN_PASSWORD"),
            headless=defaults.headless,
            timeout_ms=defaults.timeout_ms,
            slow_mo_ms=defaults.slow_mo_ms,
        )


def normalize_site(site: str) -> str:
    """Return the canonical login handler name for a public site name."""
    return SITE_ALIASES.get(site, site)


def get_crawler_site_profile(site: str) -> CrawlerSiteProfile:
    """Return a centralized site profile, accepting the canonical reddit name."""
    return CRAWLER_SITE_PROFILES[PROFILE_ALIASES.get(site, site)]


def apply_crawler_site_profile(
    settings: CrawlerSettings,
    profile: CrawlerSiteProfile,
) -> CrawlerSettings:
    """Apply a site profile after defaults and before explicit CLI overrides."""
    changes: dict[str, Any] = {
        "exit_buttons": list(profile.exit_buttons),
        "exclude_url_patterns": list(profile.exclude_url_patterns),
        "login_script": profile.login_script,
        "skip_action_types": list(profile.skip_action_types),
        "aggressive_url_dedup": profile.aggressive_url_dedup,
    }
    if profile.enable_human_review is not None:
        changes["enable_human_review"] = profile.enable_human_review
    return replace(settings, **changes)


__all__ = [
    "CRAWLER_SITE_PROFILES",
    "CrawlerSettings",
    "CrawlerSiteProfile",
    "LoginSettings",
    "apply_crawler_site_profile",
    "get_crawler_site_profile",
    "normalize_site",
]
