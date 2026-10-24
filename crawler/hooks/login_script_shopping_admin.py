"""
Login hook template for the website crawler.

HOW TO USE
----------
1. Copy this file somewhere (e.g. ``my_site_login.py``).
2. Fill in the body of ``login()`` with the Playwright steps that log into your
   target site. Replace the placeholder selectors below with real ones.
3. (Optional) Implement ``override_form_value()`` if the crawler needs a specific
   value for a particular input field; otherwise delete that function.
4. Run the crawler pointing at your file:

       python -m cli.crawl --url "https://example.com" \
           --neo4j-namespace "my_site" \
           --login-script ./my_site_login.py

The crawler stays website-agnostic: nothing about your site is hardcoded in its
core. All site-specific logic lives in the script you write here.

See crawler/hooks/README.md for the full guide.
"""

import os


def login(browser_session):
    """
    Log into the target website. REQUIRED if you pass --login-script.

    Called once right after the start URL is opened, and again after every
    context reset during the crawler's backtracking/replay. A context reset
    wipes cookies/session, so this function must fully (re-)establish the
    logged-in state every time it runs — keep it idempotent and self-contained.

    Args:
        browser_session: A BrowserSession instance.
            - ``browser_session.page``           -> the active Playwright Page
            - ``browser_session._wait_for_stability()`` -> wait until the page settles

    Replace the placeholder selectors below with the ones for your site.
    """
    page = browser_session.page
    username = os.environ.get("SHOPPING_ADMIN_USERNAME")
    password = os.environ.get("SHOPPING_ADMIN_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "SHOPPING_ADMIN_USERNAME and SHOPPING_ADMIN_PASSWORD are required"
        )

    page.get_by_placeholder("user name").fill(username)
    page.get_by_placeholder("password").fill(password)
    page.get_by_role("button", name="Sign in").click()

    # Always wait for the page to settle so the crawler sees the logged-in state.
    browser_session._wait_for_stability()