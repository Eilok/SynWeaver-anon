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

       python main_crawler.py --url "https://example.com" \
           --neo4j-namespace "my_site" \
           --login-script ./my_site_login.py

The crawler stays website-agnostic: nothing about your site is hardcoded in its
core. All site-specific logic lives in the script you write here.

See crawler/hooks/README.md for the full guide.
"""


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

    # --- Example: a generic username/password form ---------------------------
    page.get_by_placeholder("user name").fill("YOUR_USERNAME")
    page.get_by_placeholder("password").fill("YOUR_PASSWORD")
    page.get_by_role("button", name="Sign in").click()

    # Always wait for the page to settle so the crawler sees the logged-in state.
    browser_session._wait_for_stability()


def override_form_value(field, default_value):
    """
    Override the value the crawler fills into a specific form field. OPTIONAL.

    Delete this function entirely if you don't need it — its absence is fine.

    Called for each input field just before it is filled. Use it when a field
    needs a particular value to make the crawl meaningful (e.g. a search box that
    only returns results for a known query).

    Args:
        field: dict describing the input field. Useful keys:
            ``xpath``, ``tag``, ``name``, ``ariaLabel``, ``text``.
        default_value: the value the crawler would otherwise fill (LLM-generated).

    Returns:
        A replacement string to use instead, or ``None`` to keep ``default_value``.
    """
    # --- Example: force a known search term for a specific search box ---------
    # if field.get("xpath") == "/html/body/.../input[1]":
    #     return "restaurant"
    return None
