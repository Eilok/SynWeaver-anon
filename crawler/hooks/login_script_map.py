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
    if field.get("xpath") == "/html/body/div[1]/div[1]/div[1]/form[1]/div[1]/div[1]/div[1]/div[1]/input[1]":
        return "restaurant"
