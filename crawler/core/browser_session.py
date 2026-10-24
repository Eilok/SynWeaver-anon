from typing import Literal, Any, Optional, List
from playwright.sync_api import sync_playwright, Playwright, Browser, Page, BrowserContext, CDPSession
import time
import json
import os
import re
from collections import defaultdict
from pathlib import Path

IN_VIEWPORT_RATIO_THRESHOLD = 0.6

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)

# Load crawler-local scripts relative to this module.
_UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
with (_UTILS_DIR / "mark_page.js").open("r", encoding="utf-8") as script_file:
    mark_page_script = script_file.read()
with (_UTILS_DIR / "mark_action.js").open("r", encoding="utf-8") as script_file:
    mark_action_script = script_file.read()

class BrowserSession:
    """
    browser session
    - browser instance
    - browser operations
    """

    # Bounded wait (ms) for a navigation / DOM to settle. If a navigation hangs
    # longer than this (slow or stalled resource), we abort it via window.stop()
    # so the crawler never blocks indefinitely on a single link.
    NAV_SETTLE_TIMEOUT_MS = 8000

    def __init__(
            self, 
            browser_type: Literal["chromium", "firefox"] = "chromium",
            headless: bool = False,
            timeout: int = 30000,
            viewport: dict = {"width": 1366, "height": 950},
            wait_time: float = 0.5,
            storage_state_path: str | None = None,
            video_dir: str | None = None,
        ):
        self.browser_type = browser_type
        self.headless = headless
        self.timeout = timeout
        self.viewport = viewport
        self.wait_time = wait_time
        self.storage_state_path = storage_state_path
        self.video_dir = video_dir
        self.video_path: str | None = None
        self.video_counter: int = 0  # Track video recording sequence

        # launch the browser
        self.playwright = sync_playwright().start()
        self.browser = self._launch_browser()
        
        # Configure context with optional video recording
        context_options = {
            'storage_state': self.storage_state_path,
            'viewport': self.viewport
        }
        
        if self.video_dir:
            # Enable video recording
            os.makedirs(self.video_dir, exist_ok=True)
            context_options['record_video_dir'] = self.video_dir
            context_options['record_video_size'] = self.viewport
        
        self.context = self.browser.new_context(**context_options)
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)

        # Initialize CDP session for accessibility tree
        self.client: CDPSession = self.page.context.new_cdp_session(self.page)

        # multiple pages stack
        self.page_stack: list[Page] = []

        # the number of the opened pages
        self.page_count = len(self.context.pages)
        

    def _launch_browser(self) -> Browser:
        """launch a browser instance."""
        launcher = getattr(self.playwright, self.browser_type)
        return launcher.launch(
            headless=self.headless, 
            timeout=self.timeout,
            args=['--disable-blink-features=AutomationControlled']
        )
    
    def reset_context(self) -> None:
        """reset the browser context."""
        # Save video reference before closing context if recording is enabled
        video_obj = None
        if self.video_dir and self.page:
            try:
                video_obj = self.page.video
            except Exception:
                video_obj = None
        
        # Close context (this triggers video file save)
        self.context.close()
        
        # Rename video after context is closed
        if video_obj:
            self._rename_current_video(video_obj)
        
        # Configure context with optional video recording
        context_options = {
            'storage_state': self.storage_state_path,
            'viewport': self.viewport
        }
        
        if self.video_dir:
            context_options['record_video_dir'] = self.video_dir
            context_options['record_video_size'] = self.viewport
        
        self.context = self.browser.new_context(**context_options)
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)
        # Reinitialize CDP session
        self.client: CDPSession = self.page.context.new_cdp_session(self.page)
        self.page_stack = []
        self.page_count = len(self.context.pages)

    def navigate(self, url: str) -> None:
        """navigate to a url."""
        self.page.goto(url)
        self._wait_for_stability()

    def get_dom(self) -> str:
        """get the current page DOM."""
        self._wait_for_stability()
        return self.page.content()
    
    def get_url(self) -> str:
        """get the current page URL."""
        return self.page.url
    
    def get_screenshot(self) -> Optional[bytes]:
        """get the current page screenshot."""
        try:
            self._wait_for_stability()
            return self.page.screenshot(timeout=5000)
        except Exception as e:
            print(f"Warning: Failed to get screenshot: {e}")
            return None

    def get_triggerable_elements(self) -> tuple[list, bytes|None, bytes|None, list, list]:
        """
        get the triggerable elements and the labelled screenshot on the current page.
        
        Returns:
            - triggerable_elements (list): list of triggerable elements' bounding boxes
                - page_id (int): page id
                - tag (str): tag name
                - text (str): text content
                - ariaLabel (str): aria label
                - xpath (str): xpath
                - boundingBox (dict): bounding box with left, top, right, bottom, width, height
            - screenshot_bytes (bytes): screenshot before marking
            - screenshot_labelled_bytes (bytes): screenshot after marking
            - list_structures (list): list of list containers with their items
                - containerXPath (str): xpath of the list container
                - containerTag (str): tag name of the list container
                - items (list): list of items in the container
                    - page_id (int): page id
                    - xpath (str): xpath of the item
                    - tag (str): tag name of the item
                    - text (str): text content of the item
                    - ariaLabel (str): aria label of the item
                    
                Example:
                    [
                        {
                            "containerXPath": "/html/body/div/nav/ul",
                            "containerTag": "ul",
                            "items": [
                                {"page_id": 8, "xpath": "...", "tag": "a", "text": "...", "ariaLabel": "..."},
                                ...
                            ]
                        },
                        ...
                    ]
            - form_structures (list): list of form structures with their input fields and submit button
                - containerXPath (str): xpath of the form container
                - inputFields (list): list of input field elements
                - submitButton (dict): the submit button element
                
                Example:
                    [
                        {
                            "containerXPath": "/html/body/div/form",
                            "inputFields": [
                                {"page_id": 1, "xpath": "...", "tag": "input", "text": "", "name": "email"},
                                {"page_id": 2, "xpath": "...", "tag": "input", "text": "", "name": "password"}
                            ],
                            "submitButton": {"page_id": 3, "xpath": "...", "tag": "button", "text": "Submit"}
                        },
                        ...
                    ]
        """
        try:
            screenshot_bytes = self.page.screenshot(timeout=5000)
            self.page.evaluate(mark_page_script)
            result = self.page.evaluate("markPage()")
            triggerable_elements = result.get('elements', [])
            list_structures = result.get('listStructures', [])
            form_structures = result.get('formStructures', [])
            screenshot_labelled_bytes = self.page.screenshot(timeout=5000)
            self.page.evaluate("unmarkPage()")
            return triggerable_elements, screenshot_bytes, screenshot_labelled_bytes, list_structures, form_structures
        except Exception as e:
            error_msg = str(e).lower()
            if "context was destroyed" in error_msg or "navigation" in error_msg:
                print(f"Page navigated during element extraction, returning empty results")
            else:
                print(f"Error getting triggerable elements: {e}")
            return [], None, None, [], []
    
    def trigger_element_by_rule(
            self, 
            element_info: dict, 
            save_screenshot: bool = True
        ) -> tuple[str | None, bytes | None, str | None, bool]:
        """
        trigger an element by the given rule then return the action type and the screenshot.
        
        Returns:
            - action_type
            - action screenshot
            - filled value (for input elements)
            - is new page opened
        """
        tag = element_info.get("tag", "").lower()
        xpath = element_info.get("xpath")
        text = element_info.get("text", "").strip()
        aria_label = element_info.get("ariaLabel", "").strip()

        # validate the params
        if not xpath:
            print("No xpath provided for triggering element.")
            return None, None, None, False
        if not tag:
            print("No tag provided for triggering element.")
            return None, None, None, False

        screenshot_bytes = None
        action_type: Literal["click", "fill", "select", "hover", "fill and enter", None] = None

        # step 1: input
        if tag == "input":
            input_type = self._get_input_type(xpath)
            
            # clickable input (checkbox, radio, button, submit, reset, file)
            if input_type in ["checkbox", "radio", "button", "submit", "reset", "image"]:
                action_type = "click"
                if save_screenshot:
                    screenshot_bytes = self._highlight_action(xpath, action_type)
                if self.click_element(xpath):
                    new_page_opened = self.check_new_page_opened()
                    return action_type, screenshot_bytes, None, new_page_opened

            if input_type == "file":
                print("the file input is not supported yet.")
                return None, None, None, False
            
            if input_type == "hidden":
                print("the hidden input is not triggerable.")
                return None, None, None, False
            
            # fillable input (text, email, password, search, tel, url, number, date etc.)
            is_search_box = self._is_search_box(xpath, input_type)
            test_value = self._generate_test_value(input_type, text, aria_label)
            if is_search_box:
                # the search box: fill and enter
                action_type = "fill and enter"
                if save_screenshot:
                    screenshot_bytes = self._highlight_action(xpath, action_type)
                if self.fill_and_enter_element(xpath, test_value):
                    new_page_opened = self.check_new_page_opened()
                    return action_type, screenshot_bytes, test_value, new_page_opened
            else:
                action_type = "fill"
                if save_screenshot:
                    screenshot_bytes = self._highlight_action(xpath, action_type)
                if self.fill_element(xpath, test_value):
                    new_page_opened = self.check_new_page_opened()
                    return action_type, screenshot_bytes, test_value, new_page_opened    
                # return None, None, None, False   

        if tag == "textarea":
            test_value = self._generate_test_value("textarea", text, aria_label)
            action_type = "fill"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.fill_element(xpath, test_value):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, test_value, new_page_opened
            # return None, None, None, False
            
        # step 2: select
        if tag == "select":
            action_type = "select"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.select_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened
        
        # step 3: hover for menus and submenus
        if tag == "li":
            # check if it has submenu hints
            submenu_keywords = ["more", "更多", "menu", "菜单", "▼", "▾", "›", "»"]
            has_submenu_hint = any(keyword in text.lower() or keyword in aria_label.lower() 
                                for keyword in submenu_keywords)
            
            if has_submenu_hint:
                print("find the submenu hint, try to hover")
                action_type = "hover"
                if save_screenshot:
                    screenshot_bytes = self._highlight_action(xpath, action_type)
                if self.hover_element(xpath):
                    new_page_opened = self.check_new_page_opened()
                    return action_type, screenshot_bytes, None, new_page_opened
        
        if "menu" in aria_label.lower() or "dropdown" in aria_label.lower():
            print("find the menu/dropdown hint, try to hover")
            action_type = "hover"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.hover_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened

        # step 4: click for other elements
        clickable_tags = ["button", "a", "summary", "details", "img", "span"]
        if tag in clickable_tags:
            action_type = "click"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.click_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened
            # A known clickable tag was handled here; do not fall through to the
            # step-6 unknown-element branch and click the SAME element again
            # (that caused a second redundant timeout). The click genuinely failed.
            print(f"click failed for <{tag}> element, skipping.")
            return None, None, None, False

        # step 5: special elements handling
        # medium elements (video, audio) - click to play/pause
        if tag in ["video", "audio"]:
            action_type = "click"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.click_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened

        # iframe
        if tag == "iframe":
            print("iframe element, try to click")
            action_type = "click"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.click_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened

        # label
        if tag == "label":
            action_type = "click"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.click_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened
        
        # step 6: unknown elements handling
        if text or aria_label:
            print("unknown interactive element, try to click")
            action_type = "click"
            if save_screenshot:
                screenshot_bytes = self._highlight_action(xpath, action_type)
            if self.click_element(xpath):
                new_page_opened = self.check_new_page_opened()
                return action_type, screenshot_bytes, None, new_page_opened

        # nothing
        print("no suitable action found for the element.")
        return None, None, None, False
    
    def get_element_screenshot(self, xpath: str) -> bytes | None:
        """
        
        Args:
            
        Returns:
        """
        try:
            locator = self.page.locator(f"xpath={xpath}")
            screenshot_bytes = locator.screenshot(timeout=5000)
            return screenshot_bytes
        except Exception as e:
            print(f"Error taking screenshot of element {xpath}: {e}")
            return None

    def check_new_page_opened(self) -> bool:
        """check if a new page is opened. If so, update the global params"""
        current_pages = self.context.pages
        if len(current_pages) > self.page_count:
            self.page_count = len(current_pages)
            self.page_stack.append(self.page)
            self.page = current_pages[-1]
            return True
        return False
    
    def _is_search_box(self, xpath: str, input_type: str) -> bool:
        """determine if the input element is a search box."""
        try:
            locator = self.page.locator(f"xpath={xpath}")
            
            # check type attribute
            if input_type == 'search':
                return True

            # check attributes of the element
            element_id = (locator.get_attribute("id") or "").lower()
            element_name = (locator.get_attribute("name") or "").lower()
            element_class = (locator.get_attribute("class") or "").lower()
            element_placeholder = (locator.get_attribute("placeholder") or "").lower()
            element_aria_label = (locator.get_attribute("aria-label") or "").lower()

            if any(keyword in attr for attr in [element_id, element_name, element_class, element_placeholder, element_aria_label] for keyword in ["search", "query"]):
                return True
            if 'q' in [element_id, element_name]:
                return True

            # check parent form role
            form_locator = locator.locator("xpath=ancestor::form")
            if form_locator.count() > 0:
                form_role = (form_locator.first.get_attribute("role") or "").lower()
                if form_role in ["search", "combobox"]:
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking if element is a search box for xpath {xpath}: {e}")
            return False

    def _generate_test_value(self, input_type: str, text: str, aria_label: str) -> str:
        """
        generate a test value for the input element based on its type, text, and aria label.
        """
        # predefined test values
        test_values = {
            "text": "apple",
            "search": "test search",
            "email": "test@example.com",
            "password": "Test123!@#",
            "tel": "1234567890",
            "url": "https://example.com",
            "number": "42",
            "date": "2024-01-01",
            "time": "12:00",
            "datetime-local": "2024-01-01T12:00",
            "month": "2024-01",
            "week": "2024-W01",
            "color": "#ff0000",
            "textarea": "This is a test textarea content."
        }
        
        # analyze text and aria_label for hints
        combined_hint = (text + " " + aria_label).lower()
        
        if "email" in combined_hint or "邮箱" in combined_hint:
            return "test@example.com"
        elif "password" in combined_hint or "密码" in combined_hint:
            return "Test123!@#"
        elif "phone" in combined_hint or "tel" in combined_hint or "电话" in combined_hint:
            return "1234567890"
        elif "url" in combined_hint or "网址" in combined_hint or "link" in combined_hint:
            return "https://example.com"
        elif "date" in combined_hint or "日期" in combined_hint:
            return "2024-01-01"
        elif "name" in combined_hint or "姓名" in combined_hint or "用户" in combined_hint:
            return "Test User"
        elif "search" in combined_hint or "搜索" in combined_hint:
            return "test search query"
        
        # default test value
        return test_values.get(input_type, "test_value")
    
    def _highlight_action(self, xpath: str, action_label: str, element_text: str = "") -> bytes | None:
        """Highlight an element with a border and action label, take a screenshot.
        
        Returns None if highlighting fails (e.g., due to page navigation).
        """
        try:
            self.page.evaluate(mark_action_script)
            success = self.page.evaluate(f"highlightElement('{xpath}', '{action_label}', '{element_text}')")
            if not success:
                print(f"Failed to highlight element {xpath}")
                return None
            
            screenshot_bytes = self.page.screenshot(timeout=5000)
            self.page.evaluate("removeHighlight()")
            return screenshot_bytes
        except Exception as e:
            error_msg = str(e).lower()
            if "context was destroyed" in error_msg or "navigation" in error_msg:
                print(f"Page navigated during highlight, skipping screenshot for {xpath}")
            else:
                print(f"Error highlighting element {xpath}: {e}")
            return None
            

    def _abort_navigation(self, xpath: str = "") -> None:
        """Abort an in-flight / stuck navigation — the programmatic equivalent of
        hitting the browser's Stop button.

        Uses CDP ``Page.stopLoading`` (sent directly over the debugging protocol,
        so it is NOT subject to Playwright's navigation auto-wait and returns
        immediately even while a navigation is pending). Falls back to
        ``window.stop()`` if CDP is unavailable.
        """
        try:
            self.client.send("Page.stopLoading")
        except Exception:
            try:
                self.page.evaluate("window.stop()")
            except Exception:
                pass
        print(f"Aborted stuck navigation after click on {xpath} "
              f"(>{self.NAV_SETTLE_TIMEOUT_MS}ms); continuing.")

    def click_element(self, xpath: str, timeout: int = 5000) -> bool:
        """click an element by xpath.

        Args:
            xpath: XPath of the element to click
            timeout: Maximum time to wait for the element/click action (ms)

        Robustness:
            A click on a link can kick off a navigation that never finishes on a
            slow/stuck server. Playwright then either times out the click while
            "waiting for scheduled navigations to finish", OR the navigation stays
            pending and blocks every SUBSEQUENT action (screenshot, AX tree, ...).
            In both cases the navigation is still in-flight and must be aborted, or
            the crawl hangs. We bound the wait to ``NAV_SETTLE_TIMEOUT_MS`` and
            then force-abort via :meth:`_abort_navigation`.
        """
        try:
            self.page.locator(f"xpath={xpath}").click(timeout=timeout)
        except Exception as e:
            msg = str(e).lower()
            # The click fired but its navigation didn't finish in `timeout`.
            # The navigation is still pending — abort it so it can't block later
            # actions, and report success (the click itself did happen).
            if "navigation" in msg or "navigating" in msg:
                self._abort_navigation(xpath)
                time.sleep(self.wait_time)
                return True
            print(f"Error clicking element {xpath}: {e}")
            return False

        # Click returned but a triggered navigation may still be settling.
        # Wait up to NAV_SETTLE_TIMEOUT_MS, then force-abort if still stuck.
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=self.NAV_SETTLE_TIMEOUT_MS)
        except Exception:
            self._abort_navigation(xpath)
        time.sleep(self.wait_time)
        return True


    def fill_element(self, xpath: str, value: str, timeout: int = 5000) -> bool:
        """fill an input/textarea element by xpath.
        
        Args:
            xpath: XPath of the element to fill
            value: Value to fill
            timeout: Maximum time to wait in milliseconds (default: 5000)
        """
        try:
            self.page.locator(f"xpath={xpath}").fill(value, timeout=timeout)
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error filling element {xpath} with value {value}: {e}")
            return False
    
    def fill_and_enter_element(self, xpath: str, value: str, timeout: int = 5000) -> bool:
        """fill an input/textarea element by xpath and press Enter.
        
        Args:
            xpath: XPath of the element to fill
            value: Value to fill
            timeout: Maximum time to wait in milliseconds (default: 5000)
        """
        try:
            locator = self.page.locator(f"xpath={xpath}")
            locator.fill(value, timeout=timeout)
            # locator.press("Enter", timeout=timeout)
            self.page.keyboard.press("Enter")
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error filling and entering element {xpath} with value {value}: {e}")
            return False

    def select_element(self, xpath: str, option_index: int = 1, timeout: int = 5000) -> bool:
        """select an option in a select element by xpath.
        
        Args:
            xpath: XPath of the select element
            option_index: Index of the option to select (default: 1)
            timeout: Maximum time to wait in milliseconds (default: 5000)
        """
        try:
            select_locator = self.page.locator(f"xpath={xpath}")
            options = select_locator.locator("option").all()
            if not options or len(options) <= option_index:
                print(f"No sufficient options to select in element {xpath}")
                return False

            select_locator.select_option(index=option_index, timeout=timeout)
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error selecting option in element {xpath}: {e}")
            return False

    def select_element_by_text(self, xpath: str, option_text: str, timeout: int = 5000) -> bool:
        """
        Select an option in a select element by option text.
        
        Args:
            xpath: XPath of the select element
            option_text: Text of the option to select
            timeout: Maximum time to wait in milliseconds (default: 5000)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            select_locator = self.page.locator(f"xpath={xpath}")
            select_locator.select_option(label=option_text, timeout=timeout)
            self._wait_for_stability()
            return True
        except Exception as e:
            # Fallback: try selecting by value if label fails
            try:
                select_locator = self.page.locator(f"xpath={xpath}")
                select_locator.select_option(value=option_text, timeout=timeout)
                self._wait_for_stability()
                return True
            except Exception:
                print(f"Error selecting option '{option_text}' in element {xpath}: {e}")
                return False

    def fill_form_field(self, xpath: str, tag: str, value: str) -> bool:
        """
        Fill a form field with the given value based on its tag type.
        
        This is a simplified fill method that directly applies a value without
        the complex rule-based logic of trigger_element_by_rule.
        
        Args:
            xpath: XPath of the element
            tag: Tag name of the element (input, textarea, select)
            value: Value to fill
            
        Returns:
            True if successful, False otherwise
        """
        tag_lower = tag.lower()
        
        try:
            if tag_lower == "select":
                # For select elements, try to select by option text
                return self.select_element_by_text(xpath, value)
            elif tag_lower in ["input", "textarea"]:
                # For input and textarea, use fill
                return self.fill_element(xpath, value)
            else:
                # Unknown tag, try fill as fallback
                return self.fill_element(xpath, value)
        except Exception as e:
            print(f"Error filling form field {xpath} with value '{value}': {e}")
            return False

    def hover_element(self, xpath: str, timeout: int = 5000) -> bool:
        """hover over an element by xpath.
        
        Args:
            xpath: XPath of the element to hover
            timeout: Maximum time to wait in milliseconds (default: 5000)
        """
        try:
            self.page.locator(f"xpath={xpath}").hover(timeout=timeout)
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error hovering element {xpath}: {e}")
            return False

    def navigate_back(self) -> bool:
        """navigate back in browser history."""
        try:
            self.page.go_back()
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error navigating back: {e}")
            return False
        
    def back_to_previous_page(self) -> bool:
        """close the current page and switch back to the previous page."""
        if not self.page_stack:
            print("No previous page in the stack.")
            return False
        try:
            self.page.close()
            self.page = self.page_stack.pop()
            self._wait_for_stability()
            return True
        except Exception as e:
            print(f"Error switching back to previous page: {e}")
            return False

    def scroll_page(self, scroll_y: float) -> bool:
        """
        Scroll the page by the given amount in the y direction.
        
        Args:
            scroll_y: Amount to scroll in pixels (positive = down, negative = up)
            
        Returns:
            True if scrolling was successful, False otherwise
        """
        try:
            # Get current scroll position
            current_scroll_y = self.page.evaluate("window.scrollY")
            
            # Perform scroll
            self.page.mouse.wheel(0, scroll_y)
            self._wait_for_stability()
            
            # Check if scroll actually happened
            new_scroll_y = self.page.evaluate("window.scrollY")
            
            # Return True if we scrolled (even if not the full amount)
            return new_scroll_y != current_scroll_y
        except Exception as e:
            print(f"Error scrolling page: {e}")
            return False
    
    def get_scroll_position(self) -> tuple[float, float]:
        """
        Get the current scroll position of the page.
        
        Returns:
            Tuple of (scroll_x, scroll_y)
        """
        try:
            scroll_x = self.page.evaluate("window.scrollX")
            scroll_y = self.page.evaluate("window.scrollY")
            return scroll_x, scroll_y
        except Exception as e:
            print(f"Error getting scroll position: {e}")
            return 0.0, 0.0
    
    def get_max_scroll_y(self) -> float:
        """
        Get the maximum scrollable height of the page.
        
        Returns:
            Maximum scroll Y position
        """
        try:
            max_scroll_y = self.page.evaluate(
                "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) - window.innerHeight"
            )
            return max_scroll_y
        except Exception as e:
            print(f"Error getting max scroll height: {e}")
            return 0.0
    
    def can_scroll_further(self) -> bool:
        """
        Check if the page can be scrolled further down.
        
        Returns:
            True if page can scroll further, False otherwise
        """
        try:
            current_scroll = self.page.evaluate("window.scrollY")
            max_scroll = self.get_max_scroll_y()
            # Allow 5px tolerance
            return current_scroll < (max_scroll - 5)
        except Exception as e:
            print(f"Error checking scroll capability: {e}")
            return False

    def _wait_for_stability(self) -> None:
        """wait for the page to stabilize."""
        try:
            self.page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            error_msg = str(e).lower()
            if "context was destroyed" not in error_msg and "navigation" not in error_msg:
                print(f"Wait for stability failed: {e}")
        time.sleep(self.wait_time)

    def _rename_current_video(self, video_obj) -> None:
        """Rename the video file with sequence number after context is closed.
        
        Args:
            video_obj: The video object from page.video (must be obtained before context.close())
        
        Note:
            This method must be called AFTER context.close() because Playwright only saves
            the video file to disk when the context is closed. The video object reference
            must be obtained before closing, but path() can only be called after closing.
        """
        try:
            if video_obj:
                # Get video path (only available after context.close())
                original_path = video_obj.path()
                if original_path and os.path.exists(original_path):
                    self.video_counter += 1
                    # Generate new filename with sequence number
                    video_dir = os.path.dirname(original_path)
                    new_filename = f"video_{self.video_counter:03d}.webm"
                    new_path = os.path.join(video_dir, new_filename)
                    
                    # Rename the file
                    os.rename(original_path, new_path)
                    self.video_path = new_path
                    print(f"Video segment {self.video_counter} saved to: {new_path}")
        except Exception as e:
            print(f"Failed to rename video: {e}")

    def _get_input_type(self, xpath: str) -> str:
        """get the type attribute of an input element."""
        try:
            input_type = self.page.locator(f"xpath={xpath}").get_attribute("type")
            return (input_type or "text").lower()
        except Exception as e:
            print(f"Error getting input type for xpath {xpath}: {e}")
            return "text"

    def get_accessibility_tree(self, current_viewport_only: bool) -> dict:
        """
        Get the accessibility tree of the current page.
        Returns a dict with 'tree' (list of nodes) and 'tree_str' (formatted string).
        Follows the same implementation as webarena's processors.py.
        
        Returns empty dict if page is navigating and context is destroyed.
        """
        try:
            self._wait_for_stability()
            
            # Fetch browser configuration
            browser_info = self._fetch_browser_info()
            
            # Fetch accessibility tree from CDP
            accessibility_tree = self._fetch_page_accessibility_tree(
                browser_info, 
                self.client,
                current_viewport_only=current_viewport_only
            )
            
            # Parse accessibility tree to string format
            tree_str, obs_nodes_info = self._parse_accessibility_tree(accessibility_tree)
            
            # Clean the tree string
            tree_str = self._clean_accessibility_tree(tree_str)
            
            return {
                'tree': accessibility_tree,
                'tree_str': self.page.title() + "\n\n" + tree_str,
                'obs_nodes_info': obs_nodes_info
            }
        except Exception as e:
            error_msg = str(e).lower()
            if "context was destroyed" in error_msg or "navigation" in error_msg:
                print(f"Page navigated during accessibility tree fetch, returning empty result")
            else:
                print(f"Error getting accessibility tree: {e}")
            return {
                'tree': [],
                'tree_str': '',
                'obs_nodes_info': {}
            }
    
    def _fetch_browser_info(self) -> dict:
        """
        Fetch browser viewport configuration.
        Adapted from webarena's fetch_browser_info method.
        """
        win_top_bound = self.page.evaluate("window.pageYOffset")
        win_left_bound = self.page.evaluate("window.pageXOffset")
        win_width = self.page.evaluate("window.screen.width")
        win_height = self.page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_top_bound + win_height
        device_pixel_ratio = self.page.evaluate("window.devicePixelRatio")
        
        config = {
            'win_top_bound': win_top_bound,
            'win_left_bound': win_left_bound,
            'win_width': win_width,
            'win_height': win_height,
            'win_right_bound': win_right_bound,
            'win_lower_bound': win_lower_bound,
            'device_pixel_ratio': device_pixel_ratio,
        }
        
        return {'config': config}
    
    def _get_bounding_client_rect(
        self, client: CDPSession, backend_node_id: str
    ) -> dict[str, Any]:
        try:
            remote_object = client.send(
                "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
            )
            remote_object_id = remote_object["object"]["objectId"]
            response = client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as e:
            return {"result": {"subtype": "error"}}
        
    def _get_element_in_viewport_ratio(
        self,
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: dict,
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = config["win_width"]
        win_top_bound = 0
        win_lower_bound = config["win_height"]

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound)
            - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound)
            - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / width * height
        return ratio

    def _fetch_page_accessibility_tree(
        self,
        info: dict,
        client: CDPSession,
        current_viewport_only: bool,
    ) -> list:
        accessibility_tree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            backend_node_id = str(node["backendDOMNodeId"])
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self._get_bounding_client_rect(
                    client, backend_node_id
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]

        # filter nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                # update the children of the parent node
                assert (
                    accessibility_tree[parent_cursor].get("parentId", "Root")
                    is not None
                )
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(
                    nodeid
                )
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(
                        index, child_nodeid
                    )
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor][
                        "parentId"
                    ] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            config = info["config"]
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self._get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            accessibility_tree = [
                node
                for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]

        return accessibility_tree

    def _parse_accessibility_tree(
        self, accessibility_tree: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    def _clean_accessibility_tree(self, tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            # remove statictext if the content already appears in the previous line
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText (.+)"

                match = re.search(pattern, line, re.DOTALL)
                if match:
                    static_text = match.group(1)[1:-1]  # remove the quotes
                    if static_text and all(
                        static_text not in prev_line
                        for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)
    
    def _get_parent_xpath(self, xpath: str) -> Optional[str]:
        """
        
        Args:
            
        Returns:
        """
        if not xpath or xpath == '/html' or xpath == '/html/body':
            return None
        
        last_slash = xpath.rfind('/')
        if last_slash <= 0:
            return None
        
        return xpath[:last_slash]
    
    def _get_interactive_parent_xpath(self, xpath: str, max_levels: int = 3) -> List[str]:
        """
        
        Args:
            
        Returns:
        """
        parent_xpaths = []
        current_xpath = xpath
        
        for _ in range(max_levels):
            parent = self._get_parent_xpath(current_xpath)
            if parent:
                parent_xpaths.append(parent)
                current_xpath = parent
            else:
                break
        
        return parent_xpaths
    
    def match_elements_to_accessibility_tree(
        self, 
        elements: list,
        accessibility_tree: list
    ) -> dict[str, dict]:
        """
        
        Args:
            
        Returns:
            {xpath: accessibility_node_info_dict}
        """
        backend_to_node = {}
        for node in accessibility_tree:
            if 'backendDOMNodeId' in node and not node.get('ignored', False):
                backend_to_node[node['backendDOMNodeId']] = {
                    'nodeId': node.get('nodeId'),
                    'role': node.get('role', {}).get('value', ''),
                    'name': node.get('name', {}).get('value', ''),
                    'description': node.get('description', {}).get('value', ''),
                    'properties': node.get('properties', []),
                    'backendDOMNodeId': node['backendDOMNodeId']
                }
        
        mapping = {}
        for element in elements:
            xpath = element.xpath if hasattr(element, 'xpath') else element.get('xpath')
            
            if not xpath:
                mapping[xpath] = None
                continue
            
            try:
                backend_id = self._get_backend_node_id_by_xpath(xpath)
                
                if backend_id and backend_id in backend_to_node:
                    mapping[xpath] = backend_to_node[backend_id]
                else:
                    parent_xpaths = self._get_interactive_parent_xpath(xpath, max_levels=3)
                    
                    found = False
                    for parent_xpath in parent_xpaths:
                        parent_backend_id = self._get_backend_node_id_by_xpath(parent_xpath)
                        if parent_backend_id and parent_backend_id in backend_to_node:
                            mapping[xpath] = backend_to_node[parent_backend_id]
                            found = True
                            break
                    
                    if not found:
                        mapping[xpath] = None
            except Exception as e:
                mapping[xpath] = None
        
        return mapping
    
    def _get_backend_node_id_by_xpath(self, xpath: str) -> Optional[int]:
        """
        
        Args:
            
        Returns:
        """
        try:
            doc = self.client.send('DOM.getDocument', {'depth': 0})
            root_node_id = doc['root']['nodeId']
            
            result = self.client.send('DOM.performSearch', {
                'query': xpath,
                'includeUserAgentShadowDOM': True
            })
            
            search_id = result.get('searchId')
            result_count = result.get('resultCount', 0)
            
            if result_count == 0:
                return None
            
            nodes = self.client.send('DOM.getSearchResults', {
                'searchId': search_id,
                'fromIndex': 0,
                'toIndex': 1
            })
            
            self.client.send('DOM.discardSearchResults', {'searchId': search_id})
            
            if not nodes.get('nodeIds'):
                return None
            
            node_id = nodes['nodeIds'][0]
            
            node_info = self.client.send('DOM.describeNode', {'nodeId': node_id})
            backend_node_id = node_info['node'].get('backendNodeId')
            
            return backend_node_id
            
        except Exception as e:
            return None

    def close(self) -> None:
        """close the browser session."""
        # Save video reference before closing if recording is enabled
        video_obj = None
        if self.video_dir and self.page:
            try:
                video_obj = self.page.video
            except Exception:
                video_obj = None
        
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        
        # Rename video after context is closed
        if video_obj:
            self._rename_current_video(video_obj)
        
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()




if __name__ == "__main__":
    # simple test
    # browser_session = BrowserSession(headless=True, browser_type="chromium", storage_state_path="../cache/storage_state_localhost:7770.json")
    # browser_session = BrowserSession(headless=True, browser_type="chromium", video_dir="../data/videos/")
    # browser_session.navigate("http://localhost:7770")
    # browser_session.navigate("http://localhost:7770/catalogsearch/result/?q=apple")

    # triggerable_elements, screenshot, labelled_screenshot, list_structures = browser_session.get_triggerable_elements()
    # print("triggerable_elements:\n", triggerable_elements)

    # print("list_structures:\n")
    # for list_structure in list_structures:
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")

    # with open("../debug_screens/screenshot_1.png", "wb") as f:
    #     f.write(labelled_screenshot)

    # browser_session.scroll_page(926.6875)

    # triggerable_elements, screenshot, labelled_screenshot, list_structures = browser_session.get_triggerable_elements()
    # print("after scroll, triggerable_elements:\n", triggerable_elements)
    # print("after scroll, list_structures:\n")
    # for list_structure in list_structures:
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")

    # with open("../debug_screens/screenshot_2.png", "wb") as f:
    #     f.write(labelled_screenshot)

    # max_scroll_y = browser_session.get_max_scroll_y()
    # print("max_scroll_y:", max_scroll_y)
    # print(f"Can scroll further: {browser_session.can_scroll_further()}")
    # print(f"Current scroll position: {browser_session.get_scroll_position()}")

    # browser_session.scroll_page(1000)
    # print(f"After scrolling 1000px, current scroll position: {browser_session.get_scroll_position()}")
    # print(f"Can scroll further: {browser_session.can_scroll_further()}")

    # with open("../debug_screens/screenshot_3.png", "wb") as f:
    #     f.write(browser_session.get_screenshot())



    browser_session = BrowserSession(headless=True, browser_type="chromium")
    browser_session.navigate("http://3.149.168.240:3000/search?query=restaurant#map=19/42.95290/-74.36340")

    accessibility_tree_data = browser_session.get_accessibility_tree(True)
    print(f"Number of nodes: {len(accessibility_tree_data['tree'])}")
    # for node in accessibility_tree_data['tree']:
    #     print(node)
    print("-------------------")
    print(accessibility_tree_data['tree_str'])
    print("---------------------")
    # print("Observable nodes info:\n{}".format(accessibility_tree_data['obs_nodes_info']))

    elements_raw, _, labelled_screenshot, _, _ = browser_session.get_triggerable_elements()
    print("Triggerable elements:\n", elements_raw)
    browser_session.click_element(elements_raw[18]['xpath'])
    accessibility_tree_data = browser_session.get_accessibility_tree(True)
    print(accessibility_tree_data['tree_str'])
    # labbelled_screenshot_path = "../debug_screens/labeled_accessibility_tree.png"
    # with open(labbelled_screenshot_path, "wb") as f:
    #     f.write(labelled_screenshot)
    # browser_session.click_element(elements_raw[16]['xpath'])

    # accessibility_tree_data = browser_session.get_accessibility_tree(True)
    # print(f"After scrolling, number of nodes: {len(accessibility_tree_data['tree'])}")
    # print("-------------------")
    # print(accessibility_tree_data['tree_str'])
    # print("---------------------")
    # elements_raw, _, labelled_screenshot, _, _ = browser_session.get_triggerable_elements()
    # labbelled_screenshot_path = "../debug_screens/labeled_accessibility_tree_2.png"
    # with open(labbelled_screenshot_path, "wb") as f:
    #     f.write(labelled_screenshot)

    # with open("../test/accessibility_tree_2.txt", "w", encoding="utf-8") as f:
    #     f.write(accessibility_tree_data['tree_str'])

    # with open("../test/observable_nodes_2.json", "w", encoding="utf-8") as f:
    #     json.dump(accessibility_tree_data['obs_nodes_info'], f, ensure_ascii=False, indent=2)

    # with open("../test/tree.json", "w", encoding="utf-8") as f:
    #     json.dump(accessibility_tree_data['tree'], f, ensure_ascii=False, indent=2)
    
    # elements_raw, _, _, _ = browser_session.get_triggerable_elements()
    # action_type, screenshot, _, _ = browser_session.trigger_element_by_rule(elements_raw[32])
    # with open("../debug_screens/action_highlight_click.png", "wb") as f:
    #     f.write(screenshot)
    # browser_session.close()


    # triggerable_elements, screenshot, labelled_screenshot, list_structures, form_structures = browser_session.get_triggerable_elements()
    # print("Order")
    # for idx, list_structure in enumerate(list_structures):
    #     print(list_structure['containerXPath'])
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")
    #     print("")

    # browser_session.navigate("http://localhost:7780/admin/cms/page/")
    # browser_session._wait_for_stability()

    # print("page")
    # for idx, list_structure in enumerate(list_structures):
    #     print(list_structure['containerXPath'])
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")
    #     print("")

    # # forum
    # browser_session = BrowserSession(headless=True, browser_type="chromium", storage_state_path="../../.auth/forum_state.json")
    # browser_session.navigate("http://localhost:9999/f/AskReddit/116747/how-can-you-tactfully-warn-a-coworker-that-you-know-for-a")

    # triggerable_elements, screenshot, labelled_screenshot, list_structures, form_structures = browser_session.get_triggerable_elements()
    # print("Page 1")
    # for trigerable_element in triggerable_elements:
    #     print(trigerable_element)

    # for list_structure in list_structures:
    #     print(list_structure['containerXPath'])
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")
    #     print("")

    # browser_session.navigate("http://localhost:9999/f/AskReddit/116849/who-are-the-top-3-best-rappers-of-all-time")
    # triggerable_elements, screenshot, labelled_screenshot, list_structures, form_structures = browser_session.get_triggerable_elements()
    # print("After refresh")
    # for trigerable_element in triggerable_elements:
    #     print(trigerable_element)
    # for list_structure in list_structures:
    #     print(list_structure['containerXPath'])
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")
    #     print("")


    # /html/body/div[2]/main[1]/article[2]/div[1]/div[1]/header[1]/h1[1]/span[1]/a[1]/strong[1]
    # /html/body/div[2]/main[1]/article[2]/div[1]/div[1]/header[1]/h1[1]/span[1]/a[1]/strong[1]

    # gitlab
    # browser_session = BrowserSession(headless=True, browser_type="chromium", storage_state_path="../../.auth/storage_state_localhost:8023_byteblaze.json")
    # # browser_session.navigate("http://localhost:8023/")
    # browser_session.navigate("http://localhost:8023/Roshanjossey")

    # triggerable_elements, screenshot, labelled_screenshot, list_structures, form_structures = browser_session.get_triggerable_elements()
    # for trigerable_element in triggerable_elements:
    #     print(trigerable_element)

    # for list_structure in list_structures:
    #     print(list_structure['containerXPath'])
    #     print(f"<{list_structure['containerTag']}>")
    #     for item in list_structure['items']:
    #         print(f"[{item['page_id']}]<{item['tag']}> {item['text']} </{item['tag']}>")
    #     print(f"</{list_structure['containerTag']}>")
    #     print("")

    # with open("../debug_screens/gitlab_issues.png", "wb") as f:
    #     f.write(labelled_screenshot)

    # browser_session.trigger_element_by_rule(triggerable_elements[1])
    # # xpath = "/html/body/header[1]/div[1]/div[1]/div[1]/div[3]/ul[1]/li[1]/a[1]"
    # # browser_session.click_element(xpath)
    # screenshot = browser_session.get_screenshot()
    # with open("../debug_screens/gitlab_issue_detail.png", "wb") as f:
    #     f.write(screenshot)