"""
Refactored crawler module with improved readability and maintainability.

This module implements a single-threaded web crawler that constructs a state graph
by exploring triggerable elements on web pages.
"""
from typing import Set, Optional, List, Tuple
import logging
import json
from urllib.parse import urlparse

from config.shared import Neo4jSettings
from crawler.core.artifacts import CrawlerPaths
from crawler.core.browser_session import BrowserSession
from crawler.core.neo4j_state_graph import Neo4jStateGraph
from crawler.core.models import (
    State, Transition, Element, ListStructure, FormStructure
)
from crawler.utils.dom_utils import compute_structure_hash
from crawler.core.state_comparator import compare_states_progressive, extract_url_template
from crawler.core.site_hooks import load_login_hook, load_form_value_hook
from llm import get_client
from storage.io import read_bytes, read_text
from prompts.website_crawl import (
    get_crawler_prompt,
    build_user_prompt_for_filter_list,
    build_user_prompt_for_form_fill
)

logger = logging.getLogger(__name__)


class Crawler:
    """
    Single-threaded crawler for website state graph construction.
    
    Explores web pages by triggering interactive elements and tracks state transitions
    to build a comprehensive state graph stored in Neo4j.
    """
    
    # Constants
    MAX_RETRIES = 3
    
    def __init__(
            self,
            exit_btns: Optional[List[str]] = None,
            enable_human_review: bool = True,
            max_depth: Optional[int] = None,
            max_states: Optional[int] = None,
            allow_external_domains: bool = False,
            neo4j_config: Optional[Neo4jSettings] = None,
            crawler_paths: Optional[CrawlerPaths] = None,
            exclude_url_patterns: Optional[List[str]] = None,
            login_script: Optional[str] = None,
            skip_action_types: Optional[List[str]] = None,
            aggressive_url_dedup: bool = False,
        ):
        """
        Initialize the crawler.

        Args:
            exit_btns: List of exit button texts to avoid triggering (e.g., ['Log out', 'Sign out'])
            enable_human_review: Whether to enable human review for LLM decisions on list structures
            max_depth: Maximum crawling depth (None for unlimited)
            max_states: Maximum number of states to explore (None for unlimited)
            allow_external_domains: Whether to follow links to external domains
            neo4j_config: Configuration for Neo4j database connection
            crawler_paths: Crawler output paths (screenshot dirs, etc.)
            exclude_url_patterns: List of URL path patterns to exclude (e.g., ['/help', '/admin'])
            login_script: Path to a user-supplied hook script providing site-specific
                login (and optional form-value override) logic. See
                crawler/hooks/README.md. None disables login hooks.
            skip_action_types: Action types to skip when triggering elements
                (e.g., ['fill']). Skipped actions record no transition.
            aggressive_url_dedup: If True, once a URL template has been explored,
                skip all elements on later pages sharing that template.
        """
        self.exit_btns = set(exit_btns) if exit_btns else set()
        self.enable_human_review = enable_human_review
        self.max_depth = max_depth
        self.max_states = max_states
        self.allow_external_domains = allow_external_domains
        if neo4j_config is None or crawler_paths is None:
            raise ValueError("neo4j_config and crawler_paths are required")
        self.neo4j_config = neo4j_config
        self.crawler_paths = crawler_paths
        self.exclude_url_patterns = exclude_url_patterns or []
        self.skip_action_types = set(skip_action_types or [])
        self.aggressive_url_dedup = aggressive_url_dedup
        self.browser_session: Optional[BrowserSession] = None
        self.state_graph = Neo4jStateGraph(config=self.neo4j_config, paths=self.crawler_paths)
        self.video_path: Optional[str] = None

        # Site-specific extension hooks loaded from a user-supplied script.
        # Both are None when no script is provided -> crawler stays website-agnostic.
        self._login_hook = load_login_hook(login_script)
        self._form_value_hook = load_form_value_hook(login_script)

        # Form-fill uses a bounded 2-attempt retry (then rule-based fallback),
        # distinct from the list-filter client which retries indefinitely.
        self._form_fill_llm = get_client("crawler_text", max_retries=2, fresh=True)
        # List filtering retries until it gets a valid decision (was `while True`).
        self._filter_llm = get_client("crawler_text", infinite_retry=True, fresh=True)

        # Global tracking sets to avoid duplicate exploration
        self.triggered_element_signatures: Set[str] = set()
        self.processed_list_signatures: Set[str] = set()
        self.processed_list_xpaths: Set[str] = set()
        self.processed_form_xpaths: Set[str] = set()  # Track processed form containers
        
        # Track filtered xpaths per URL
        self.filtered_xpaths_by_url: dict[str, Set[str]] = {}

    def crawl(self, url: str, browser_config: dict) -> Neo4jStateGraph:
        """
        Start crawling from the given URL and construct the state graph.
        
        Args:
            url: Starting URL for the crawl
            browser_config: Configuration dict for browser session
            
        Returns:
            Constructed Neo4jStateGraph
        """
        logger.info(f"Starting single-thread crawl for {url}")

        if not self.allow_external_domains:
            self.state_graph.set_root_domain(url)
            logger.info(f"Set root domain to {self.state_graph.root_domain}")

        # Launch browser session
        with BrowserSession(**browser_config) as browser_session:
            self.browser_session = browser_session
            self.browser_session.navigate(url)

            # Run the user-supplied login hook (no-op if no --login-script given).
            self._run_login_hook()

            self._crawl_recursive(previous_state=None, previous_transition=None, depth=0)
            
            # Save video path if recording was enabled
            if browser_config["video_dir"] and self.browser_session.video_path:
                self.video_path = self.browser_session.video_path
                logger.info(f"Video saved to {self.video_path}")

        logger.info(f"Finished crawling task for {url}")
        return self.state_graph

    def _run_login_hook(self) -> None:
        """
        Run the user-supplied login hook if one was provided.

        Called after the start URL is opened and after every context reset during
        replay (see ``_navigate_to_state``), because ``reset_context()`` clears the
        session. No-op when no ``--login-script`` was configured.
        """
        if self._login_hook is None:
            return
        logger.info("Running login hook")
        self._login_hook(self.browser_session)

    def _crawl_recursive(
        self, 
        previous_state: Optional[State], 
        previous_transition: Optional[Transition], 
        depth: int
    ):
        """
        Recursively crawl web pages and build the state graph.
        
        Args:
            previous_state: The state we transitioned from (None for root)
            previous_transition: The transition that led here (None for root)
            depth: Current crawling depth
        """
        current_url = self.browser_session.get_url()

        # Check if URL should be excluded based on configured patterns
        if self._should_skip_url(current_url):
            return

        # Get current DOM for hash calculation (duplicate detection)
        current_dom = self._get_dom_with_retry(current_url)
        if current_dom is None:
            return

        logger.info(f"Crawling URL: {current_url} at depth {depth}")

        # Compute DOM hash for duplicate detection
        current_dom_hash = compute_structure_hash(current_dom)

        # Get all triggerable elements on the page
        all_elements_raw, screenshot, labelled_screenshot, list_structures_raw, form_structures_raw = (
            self.browser_session.get_triggerable_elements()
        )
        
        # Handle empty results caused by page navigation.
        if screenshot is None:
            logger.warning("Failed to get page elements (possibly due to navigation), skipping this state")
            return
        
        # Get accessibility tree
        accessibility_tree_data = self.browser_session.get_accessibility_tree(True) # default set to True
        accessibility_tree_str = accessibility_tree_data['tree_str']
        
        # Convert to data model objects
        all_elements = [Element.from_dict(elem) for elem in all_elements_raw]
        list_structures = [ListStructure.from_dict(ls) for ls in list_structures_raw]
        form_structures = [FormStructure.from_dict(fs) for fs in form_structures_raw]

        # Add current state to the graph
        current_state_id, is_duplicate = self.state_graph.add_state(
            url=current_url,
            screenshot=screenshot,
            labelled_screenshot=labelled_screenshot,
            depth=depth,
            dom_hash=current_dom_hash,
            accessibility_tree_str=accessibility_tree_str
        )
        logger.info(f"Added visited state {current_state_id}")
        current_state = State.from_dict(self.state_graph.get_state(current_state_id))

        # Add transition from previous state if exists
        if previous_state is not None and previous_transition is not None:
            self._add_transition_to_graph(previous_state, current_state_id, previous_transition)

        # Check termination conditions
        if self._should_stop_crawling(is_duplicate, current_url, depth):
            return
        
        logger.info(f"Found {len(all_elements)} total triggerable elements on the page.")

        # Filter redundant elements
        new_list_structures, redundant_list_structures = self._partition_list_structures(
            list_structures
        )
        
        filtered_elements = self._filter_elements_from_lists(
            all_elements,
            new_list_structures,
            redundant_list_structures
        )
        
        # Filter out already triggered elements
        triggerable_elements = self._get_untriggered_elements(filtered_elements, current_url)
        
        logger.info(f"Found {len(triggerable_elements)} triggerable elements after filtering.")

        if not triggerable_elements:
            logger.info(f"No new triggerable elements found at {current_url}")
            
            # Try scrolling if no new elements found
            if self._try_scroll_and_explore(current_state, current_url, depth, accessibility_tree_str):
                return
            
            logger.info(f"No scroll possible, backtracking...")
            return
        
        # Build mapping from elements to accessibility tree nodes
        logger.info("Building element-to-accessibility-tree mapping...")
        element_to_accessibility_map = self.browser_session.match_elements_to_accessibility_tree(
            triggerable_elements,
            accessibility_tree_data['tree']
        )
        
        # Attach accessibility node info to elements
        matched_count = 0
        failed_elements = []
        for element in triggerable_elements:
            accessibility_node_info = element_to_accessibility_map.get(element.xpath)
            if accessibility_node_info:
                element.accessibility_node_info = accessibility_node_info
                matched_count += 1
            else:
                failed_elements.append(element)
        
        match_rate = (matched_count / len(triggerable_elements) * 100) if triggerable_elements else 0
        logger.info(f"Successfully matched {matched_count}/{len(triggerable_elements)} elements to accessibility tree ({match_rate:.1f}%)")
        
        if failed_elements and len(failed_elements) <= 5:
            logger.debug(f"Failed to map {len(failed_elements)} elements:")
            for elem in failed_elements[:5]:
                logger.debug(f"   - <{elem.tag}> {elem.text[:50]} | xpath: ...{elem.xpath[-60:]}")
        
        # Update global tracking sets
        self._update_global_tracking(all_elements, list_structures, current_url)
        
        # First, explore form structures as composite operations
        form_element_xpaths = set()
        new_form_structures = [fs for fs in form_structures 
                               if fs.container_xpath not in self.processed_form_xpaths]
        
        if new_form_structures:
            logger.info(f"Found {len(new_form_structures)} new form structures on the page.")
            
            for form in new_form_structures:
                # Collect all element xpaths in this form
                form_element_xpaths.update(form.get_all_element_xpaths())
                
                # Mark form as processed
                self.processed_form_xpaths.add(form.container_xpath)
                
                # Explore form as composite operation
                if not self._explore_form(form, current_state, current_url, depth, accessibility_tree_str):
                    break
        
        # Filter out elements that belong to processed forms
        triggerable_elements = [e for e in triggerable_elements 
                                if e.xpath not in form_element_xpaths]
        
        logger.info(f"Found {len(triggerable_elements)} triggerable elements after removing form elements.")
        
        # Explore each remaining triggerable element
        page_drifted = False
        for element in triggerable_elements:
            if element.text in self.exit_btns:
                logger.info(f"Skipping exit button: '{element.text}'")
                continue

            # Explore this element and check if we should continue
            if not self._explore_element(element, current_state, current_url, depth, accessibility_tree_str):
                page_drifted = True
                break

        # After exploring all elements, try scrolling to discover new content.
        # Skip when the loop broke because the page drifted away from current_url
        # (the pre-fetched elements/scroll state no longer apply); the caller will
        # backtrack to re-establish a known state.
        if not page_drifted:
            self._try_scroll_and_explore(current_state, current_url, depth, accessibility_tree_str)

    def _get_dom_with_retry(self, url: str) -> Optional[str]:
        """
        Get DOM with retry logic for error handling.
        Used only for hash calculation (duplicate detection).
        
        Args:
            url: URL being accessed
            
        Returns:
            DOM string if successful, None otherwise
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.browser_session.get_dom()
            except Exception as e:
                logger.error(f"Error getting DOM at {url}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(f"Retrying to get DOM (attempt {attempt + 2}/{self.MAX_RETRIES})...")
                else:
                    logger.error(f"Failed to get DOM after {self.MAX_RETRIES} attempts, backtracking...")
                    return None

    def _add_transition_to_graph(
        self,
        previous_state: State,
        current_state_id: str,
        transition: Transition
    ):
        """Add a transition edge to the state graph."""
        transition_dict = transition.to_dict()
        self.state_graph.add_transition(
            from_state_id=previous_state.state_id,
            to_state_id=current_state_id,
            **transition_dict
        )
        logger.info(
            f"Added transition from {previous_state.state_id} to {current_state_id} "
            f"via {transition.event_type} on {transition.get_element_html()}"
        )

    def _should_skip_url(self, url: str) -> bool:
        """
        Check if URL should be skipped based on configured exclude patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be skipped, False otherwise
        """
        if not self.exclude_url_patterns:
            return False
        
        url_path = urlparse(url).path or ""
        for pattern in self.exclude_url_patterns:
            if url_path == pattern or url_path.startswith(f"{pattern}/"):
                logger.info(f"Skipping excluded URL pattern '{pattern}': {url}")
                return True
        return False

    def _should_stop_crawling(self, is_duplicate: bool, url: str, depth: int) -> bool:
        """
        Check if crawling should stop based on various conditions.
        
        Args:
            is_duplicate: Whether the current state is a duplicate
            url: Current URL
            depth: Current depth
            
        Returns:
            True if crawling should stop, False otherwise
        """
        if is_duplicate:
            logger.info(f"Duplicate state detected at {url}, backtracking...")
            return True
        
        if not self.state_graph.is_same_domain(url):
            logger.info(f"Skip external domain: {url}")
            return True
        
        if self.max_depth is not None and depth >= self.max_depth:
            logger.info(f"Reached max depth {self.max_depth} at {url}")
            return True
        
        if self.max_states is not None and self.state_graph.state_count() >= self.max_states:
            logger.info(f"Reached max states {self.max_states} at {url}")
            return True
        
        return False

    def _partition_list_structures(
        self,
        list_structures: List[ListStructure]
    ) -> Tuple[List[ListStructure], List[ListStructure]]:
        """
        Partition list structures into new and redundant based on tracking sets.
        
        Args:
            list_structures: List of all list structures found on the page
            
        Returns:
            Tuple of (new_list_structures, redundant_list_structures)
        """
        new_list_structures = []
        redundant_list_structures = []
        
        for list_structure in list_structures:
            signature = list_structure.get_signature()
            xpath = list_structure.container_xpath
            
            if (signature not in self.processed_list_signatures and 
                xpath not in self.processed_list_xpaths):
                new_list_structures.append(list_structure)
            else:
                redundant_list_structures.append(list_structure)
        
        logger.info(f"Found {len(new_list_structures)} new list structures on the page.")
        return new_list_structures, redundant_list_structures

    def _filter_elements_from_lists(
        self,
        all_elements: List[Element],
        new_list_structures: List[ListStructure],
        redundant_list_structures: List[ListStructure]
    ) -> List[Element]:
        """
        Filter elements from list structures using LLM for new lists.
        
        Args:
            all_elements: All elements found on the page
            new_list_structures: New list structures to filter with LLM
            redundant_list_structures: Already processed list structures
            
        Returns:
            Filtered list of elements
        """
        filtered_elements = all_elements.copy()
        
        # Use LLM to filter elements from new list structures
        if new_list_structures:
            system_prompt = get_crawler_prompt("filter_lists")
            for list_structure in new_list_structures:
                # Convert back to dict for prompt building (legacy API)
                list_structure_dict = {
                    'containerXPath': list_structure.container_xpath,
                    'containerTag': list_structure.container_tag,
                    'items': [
                        {
                            'page_id': item.page_id,
                            'xpath': item.xpath,
                            'tag': item.tag,
                            'text': item.text,
                            'ariaLabel': item.aria_label
                        }
                        for item in list_structure.items
                    ]
                }
                user_prompt = build_user_prompt_for_filter_list(list_structure_dict)
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                # Convert elements to dict for legacy API
                elements_dict = [elem.to_dict() for elem in filtered_elements]
                filtered_elements_dict = self._call_model_filter_list(
                    message, list_structure_dict, elements_dict
                )
                # Convert back to Element objects
                filtered_elements = [Element.from_dict(elem) for elem in filtered_elements_dict]

        # Remove elements from redundant list structures directly
        for list_structure in redundant_list_structures:
            item_page_ids = list_structure.get_item_page_ids()
            filtered_elements = [
                elem for elem in filtered_elements 
                if elem.page_id not in item_page_ids
            ]
        
        initial_count = len(all_elements)
        filtered_count = len(filtered_elements)
        logger.info(f"Filtered {initial_count - filtered_count} elements from list structures.")
        
        return filtered_elements

    def _get_untriggered_elements(
        self,
        elements: List[Element],
        current_url: str
    ) -> List[Element]:
        """
        Filter out elements that have already been triggered.
        
        Args:
            elements: List of elements to filter
            current_url: Current page URL
            
        Returns:
            List of untriggered elements
        """
        untriggered = []
        cur_url_template = extract_url_template(current_url)
        # Aggressive URL-template dedup: once a template is seen, skip its pages entirely.
        if self.aggressive_url_dedup and cur_url_template in self.filtered_xpaths_by_url:
            logger.info(f"{cur_url_template} has been explored before.")
            return []
        url_filtered_xpaths = self.filtered_xpaths_by_url.get(cur_url_template, set())
        
        for elem in elements:
            signature = elem.get_signature()
            if (signature not in self.triggered_element_signatures and 
                elem.xpath not in url_filtered_xpaths):
                untriggered.append(elem)
        
        return untriggered

    def _update_global_tracking(
        self,
        all_elements: List[Element],
        list_structures: List[ListStructure],
        current_url: str
    ):
        """
        Update global tracking sets with current page data.
        
        Args:
            all_elements: All elements on the current page
            list_structures: All list structures on the current page
            current_url: Current page URL
        """
        # Update triggered element signatures
        element_signatures = [elem.get_signature() for elem in all_elements]
        self.triggered_element_signatures.update(element_signatures)

        # Update filtered xpaths for current URL
        element_xpaths = [elem.xpath for elem in all_elements]
        cur_url_template = extract_url_template(current_url)
        if cur_url_template not in self.filtered_xpaths_by_url:
            self.filtered_xpaths_by_url[cur_url_template] = set()
        self.filtered_xpaths_by_url[cur_url_template].update(element_xpaths)

        # Update processed list structures
        list_signatures = [ls.get_signature() for ls in list_structures]
        list_xpaths = [ls.container_xpath for ls in list_structures]
        self.processed_list_signatures.update(list_signatures)
        self.processed_list_xpaths.update(list_xpaths)

    def _explore_form(
        self,
        form: FormStructure,
        current_state: State,
        current_url: str,
        depth: int,
        current_accessibility_tree: str
    ) -> bool:
        """
        Explore a form as a composite operation: fill all fields then submit.
        Uses LLM to generate semantically appropriate fill values with rule-based fallback.
        
        Args:
            form: FormStructure to explore
            current_state: Current state before triggering
            current_url: Current URL before triggering
            depth: Current crawling depth
            current_accessibility_tree: Current accessibility tree before triggering
            
        Returns:
            True if exploration should continue, False if backtracking failed
        """
        logger.info(f"Exploring form at {form.container_xpath} with {len(form.input_fields)} input fields")
        
        # Get submit button screenshot before starting
        element_screenshot = None
        if form.submit_button:
            element_screenshot = self.browser_session.get_element_screenshot(form.submit_button.xpath)
        
        # Build sub_actions list to record all operations
        sub_actions = []
        
        # Step 1: Generate form fill values using LLM (with rule-based fallback)
        llm_values = self._generate_form_values_by_llm(
            url=current_url,
            accessibility_tree_str=current_accessibility_tree,
            input_fields=form.input_fields
        )
        
        # Step 2: Fill all input fields with LLM-generated or fallback values
        for i, input_field in enumerate(form.input_fields):
            # Get the value to fill
            if llm_values and i < len(llm_values):
                fill_value = llm_values[i]
                use_llm_value = True
            else:
                # Fallback: use None to let trigger_element_by_rule generate value
                fill_value = None
                use_llm_value = False
            
            # Fill the field
            if use_llm_value and fill_value:
                # Allow a user-supplied hook to override the value for specific
                # fields (e.g. a map search box). No-op without a form-value hook.
                if self._form_value_hook is not None:
                    overridden = self._form_value_hook(input_field.to_dict(), fill_value)
                    if overridden is not None:
                        logger.info(f"    Hook override for <{input_field.tag}> "
                                    f"{input_field.name or input_field.aria_label}: '{overridden}'")
                        fill_value = overridden
                success = self.browser_session.fill_form_field(
                    xpath=input_field.xpath,
                    tag=input_field.tag,
                    value=fill_value
                )
                if success:
                    event_type = 'fill'
                    filled_value = fill_value
                    logger.info(f"    [{i+1}/{len(form.input_fields)}] LLM fill <{input_field.tag}> "
                               f"{input_field.name or input_field.aria_label} with '{fill_value}'")
                else:
                    # LLM value fill failed, try rule-based fallback
                    logger.warning(f"    LLM value fill failed, falling back to rule-based for field {i+1}")
                    event_type, _, filled_value, _ = (
                        self.browser_session.trigger_element_by_rule(input_field.to_dict(), save_screenshot=False)
                    )
            else:
                # Rule-based fallback
                event_type, _, filled_value, _ = (
                    self.browser_session.trigger_element_by_rule(input_field.to_dict(), save_screenshot=False)
                )
                if event_type and filled_value:
                    logger.info(f"    [{i+1}/{len(form.input_fields)}] Rule fill <{input_field.tag}> "
                               f"{input_field.name or input_field.aria_label} with '{filled_value}'")
            
            if event_type and filled_value:
                sub_actions.append({
                    'sequence': i,
                    'event_type': event_type,
                    'xpath': input_field.xpath,
                    'element': f"<{input_field.tag}>{input_field.text}</{input_field.tag}>",
                    'filled_value': filled_value
                })
            else:
                logger.warning(f"    Failed to fill field: {input_field.xpath}")
        
        # Step 3: Click submit button
        if not form.submit_button:
            logger.warning(f"Form has no submit button, skipping")
            return True
        
        submit_event_type, screenshot_op, _, new_page_opened = (
            self.browser_session.trigger_element_by_rule(form.submit_button.to_dict())
        )
        
        if submit_event_type is None:
            logger.warning(f"Failed to click submit button: {form.submit_button.xpath}")
            return True
        
        logger.info(f"    Clicked submit button: <{form.submit_button.tag}>{form.submit_button.text}</{form.submit_button.tag}>")
        
        # Add submit action to sub_actions
        sub_actions.append({
            'sequence': len(form.input_fields),
            'event_type': 'click',
            'xpath': form.submit_button.xpath,
            'element': f"<{form.submit_button.tag}>{form.submit_button.text}</{form.submit_button.tag}>",
            'filled_value': None
        })
        
        if new_page_opened:
            logger.info("    New page opened after form submission")
        
        # Step 4: Check if state changed
        try:
            next_url = self.browser_session.get_url()
            next_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
            next_accessibility_tree = next_accessibility_tree_data['tree_str']
        except Exception as e:
            # Handle execution context destruction during navigation.
            error_msg = str(e).lower()
            if "context was destroyed" in error_msg or "navigation" in error_msg:
                logger.warning(f"Page still navigating after form submission, skipping state check")
                return True
            raise  # Re-raise unrelated failures.
        
        is_same, reason = compare_states_progressive(
            current_url=current_url,
            target_url=next_url,
            current_accessibility_tree=current_accessibility_tree,
            target_accessibility_tree=next_accessibility_tree,
            use_vlm=False
        )
        
        if is_same:
            logger.info(f"No state change after form submission (verified by {reason}), continuing...")
            return True
        
        # Step 5: Create composite transition
        transition = Transition(
            event_type='form_submission',
            element_tag=form.submit_button.tag,
            element_text=form.submit_button.text,
            xpath=form.submit_button.xpath,
            screenshot_operation=screenshot_op,
            element_screenshot=element_screenshot,
            filled_value=None,  # Individual values stored in sub_actions
            new_page_opened=new_page_opened,
            is_composite=True,
            composite_type='form_submission',
            sub_actions=sub_actions
        )
        
        # Step 6: Recursively explore the new state
        self._crawl_recursive(
            previous_state=current_state,
            previous_transition=transition,
            depth=depth + 1
        )
        
        # Step 7: Backtrack to current state
        success = self._backtrack(
            start_url=next_url,
            target_url=current_url,
            target_state_id=current_state.state_id,
            new_page_opened=new_page_opened
        )
        
        return success

    def _generate_form_values_by_llm(
        self,
        url: str,
        accessibility_tree_str: str,
        input_fields: List[Element]
    ) -> Optional[List[str]]:
        """
        Call LLM to generate semantically appropriate fill values for form fields.
        
        Args:
            url: Current page URL
            accessibility_tree_str: Accessibility tree string for context
            input_fields: List of Element objects representing form input fields
            
        Returns:
            List of fill values in order, or None if LLM call fails
        """
        if not input_fields:
            return None
        
        # Build input field dicts with additional info from browser
        enriched_fields = []
        for field in input_fields:
            field_dict = field.to_dict()
            # Try to get additional attributes from the browser
            try:
                locator = self.browser_session.page.locator(f"xpath={field.xpath}")
                field_dict['inputType'] = locator.get_attribute('type') or ''
                field_dict['placeholder'] = locator.get_attribute('placeholder') or ''
            except Exception:
                field_dict['inputType'] = ''
                field_dict['placeholder'] = ''
            enriched_fields.append(field_dict)
        
        # Build the prompt
        system_prompt = get_crawler_prompt("form_fill")
        user_prompt = build_user_prompt_for_form_fill(
            url=url,
            accessibility_tree_str=accessibility_tree_str,
            input_fields=enriched_fields
        )

        # Call LLM via the unified client (bounded 2-attempt retry; JSON parsed
        # for us). On failure the caller falls back to rule-based filling.
        logger.info("Calling LLM to generate form fill values")
        result = self._form_fill_llm.complete(
            system=system_prompt,
            user=user_prompt,
            json_mode=True,
        )

        if isinstance(result, dict):
            values = result.get('values', [])
            reasoning = result.get('reasoning', '')
            if values and len(values) == len(input_fields):
                logger.info(f"LLM generated {len(values)} values: {reasoning}")
                return values
            logger.warning(f"LLM returned {len(values)} values but expected {len(input_fields)}")

        logger.warning("LLM form fill generation failed, will use rule-based fallback")
        return None

    def _explore_element(
        self,
        element: Element,
        current_state: State,
        current_url: str,
        depth: int,
        current_accessibility_tree: str
    ) -> bool:
        """
        Trigger an element and recursively explore the resulting state.
        
        Args:
            element: Element to trigger
            current_state: Current state before triggering
            current_url: Current URL before triggering
            depth: Current crawling depth
            current_accessibility_tree: Current accessibility tree before triggering
            
        Returns:
            True if exploration should continue, False if backtracking failed
        """
        # Get element screenshot before triggering
        element_screenshot = self.browser_session.get_element_screenshot(element.xpath)
        
        # Trigger the element
        event_type, screenshot_op, filled_value, new_page_opened = (
            self.browser_session.trigger_element_by_rule(element.to_dict())
        )

        # Skip actions whose type the user opted out of (e.g. --skip-action-types fill).
        if event_type is None or event_type in self.skip_action_types:
            logger.info(f"Skipping element {element.get_signature()} as no action could be performed.")
            # Defensive resync: a "failed" trigger may still have navigated the page
            # (e.g. a click that fired but errored). If the browser drifted away from
            # the page these pre-fetched elements belong to, stop iterating stale
            # xpaths and let the caller backtrack to re-establish a known state.
            try:
                if self.browser_session.get_url() != current_url:
                    logger.warning(
                        f"Page drifted to {self.browser_session.get_url()} after a no-op "
                        f"trigger; stopping element loop to resync."
                    )
                    return False
            except Exception:
                pass
            return True

        logger.info(
            f"{event_type} <{element.tag}>{element.text}</{element.tag}> "
            f"{f'with {filled_value}' if filled_value else ''}"
        )
        if new_page_opened:
            logger.info("New page opened")

        # Get the new state after triggering
        try:
            next_url = self.browser_session.get_url()

            # Check if state actually changed using progressive comparison
            next_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
            next_accessibility_tree = next_accessibility_tree_data['tree_str']
        except Exception as e:
            # Handle execution context destruction during navigation.
            error_msg = str(e).lower()
            if "context was destroyed" in error_msg or "navigation" in error_msg:
                logger.warning(f"Page still navigating after {event_type}, skipping state check")
                return True
            raise  # Re-raise unrelated failures.
        
        is_same, reason = compare_states_progressive(
            current_url=current_url,
            target_url=next_url,
            current_accessibility_tree=current_accessibility_tree,
            target_accessibility_tree=next_accessibility_tree,
            use_vlm=False
        )
        
        if is_same:
            logger.info(f"No state change detected after {event_type} (verified by {reason}), continuing...")
            return True
        
        # Create transition object with accessibility node info
        transition = Transition(
            event_type=event_type,
            element_tag=element.tag,
            element_text=element.text,
            xpath=element.xpath,
            screenshot_operation=screenshot_op,
            element_screenshot=element_screenshot,
            filled_value=filled_value,
            new_page_opened=new_page_opened,
            accessibility_node_info=element.accessibility_node_info
        )
        
        # Recursively explore the new state
        self._crawl_recursive(
            previous_state=current_state,
            previous_transition=transition,
            depth=depth + 1
        )

        # Backtrack to current state
        success = self._backtrack(
            start_url=next_url,
            target_url=current_url,
            target_state_id=current_state.state_id,
            new_page_opened=new_page_opened
        )
        
        return success

    def _try_scroll_and_explore(
        self,
        current_state: State,
        current_url: str,
        depth: int,
        current_accessibility_tree: str
    ) -> bool:
        """
        Try to scroll the page and explore if new content appears.
        
        Args:
            current_state: Current state before scrolling
            current_url: Current URL
            depth: Current crawling depth
            current_accessibility_tree: Current accessibility tree before scrolling
            
        Returns:
            True if scrolling was performed, False otherwise
        """
        # Check if we can scroll further
        if not self.browser_session.can_scroll_further():
            logger.info(f"Cannot scroll further at {current_url}")
            return False
        
        try:
            # Calculate scroll amount: use viewport height to ensure viewing completely new content
            viewport_height = self.browser_session.page.viewport_size['height']
            scroll_amount = viewport_height
            logger.info(f"Scrolling by viewport height: {scroll_amount}px")
            
            # Perform scroll
            if scroll_amount <= 0:
                logger.info(f"Invalid scroll amount: {scroll_amount}")
                return False
            
            scroll_success = self.browser_session.scroll_page(scroll_amount)
            
            if not scroll_success:
                logger.info(f"Scroll failed or already at bottom")
                return False
            
            logger.info(f"Scrolled page by {scroll_amount}px")
            
            # Check if state changed significantly using progressive comparison
            new_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
            new_accessibility_tree = new_accessibility_tree_data['tree_str']
            
            is_same, reason = compare_states_progressive(
                current_url=current_url,
                target_url=current_url,  # Same URL for scroll
                current_accessibility_tree=current_accessibility_tree,
                target_accessibility_tree=new_accessibility_tree,
                use_vlm=False
            )
            
            if is_same:
                logger.info(f"No state change after scroll (verified by {reason}), continuing...")
                return False
            
            # Create a scroll transition
            scroll_transition = Transition(
                event_type="scroll",
                element_tag="page",
                element_text="N/A",
                xpath="N/A",
                screenshot_operation=None,
                element_screenshot=None,
                filled_value=f"{scroll_amount}px",
                new_page_opened=False
            )
            
            # Recursively explore the new state after scroll
            self._crawl_recursive(
                previous_state=current_state,
                previous_transition=scroll_transition,
                depth=depth + 1
            )
            
            # Backtrack: scroll back to original position
            # We need to get back to the original state
            # For simplicity, we can navigate back to the URL which will reset scroll
            logger.info(f"Backtracking after scroll exploration...")
            # Scrolling normally reaches the bottom, so navigation restores the top.
            self.browser_session.navigate(current_url)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during scroll exploration: {e}")
            return False

    def _call_model_filter_list(
        self, 
        message: list, 
        list_structure: dict, 
        all_elements: list
    ) -> list:
        """
        Call LLM to filter redundant elements from list structures, including human in loop.
        
        Args:
            message: Chat messages for the LLM
            list_structure: List structure dict
            all_elements: All elements as dicts
            
        Returns:
            Filtered list of element dicts
        """
        while True:
            logger.info("Call LLM to filter elements from list structure")
            # Unified client: infinite_retry handles transient call failures; this
            # loop additionally retries on parse/missing-key failures (returns None
            # or raises KeyError below), matching the original while-True behaviour.
            system = message[0]["content"] if message and message[0].get("role") == "system" else None
            user = message[-1]["content"]
            try:
                repaired_output = self._filter_llm.complete(
                    system=system, user=user, json_mode=True
                )
                if not isinstance(repaired_output, dict):
                    logger.error("parse JSON error, retrying LLM invocation...")
                    continue
                decision = repaired_output['decision']

                if decision.lower() == "all":
                    logger.info(
                        f"LLM decided to keep all elements in the list structure: "
                        f"{repaired_output['reason']}"
                    )
                    if not self.enable_human_review:
                        return all_elements
                    logger.info("--------------------")
                    logger.info("Begin human review ...")
                    logger.info(f"List Structure:\n{message[1]['content']}")
                    choice = input("LLM decided to keep all elements, do you want to (k)eep all or (f)ilter? [k/f]: ").strip().lower()
                    if choice == 'k':
                        return all_elements
                    else:
                        selected_indices_str = input("Please input the kept indices as a comma-separated list (e.g., 0,2,3): ").strip()
                        selected_indices = set(int(idx.strip()) for idx in selected_indices_str.split(',') if idx.strip().isdigit())
                else:
                    selected_indices = set(repaired_output['selected_indices'])
                    logger.info(
                        f"LLM selected {selected_indices} items from the list structure: "
                        f"{repaired_output['reason']}"
                    )

                original_indices = [item['page_id'] for item in list_structure['items']]
                redundant_indices = [
                    idx for idx in original_indices
                    if idx not in selected_indices
                ]

                filtered_elements = [
                    elem for elem in all_elements
                    if elem['page_id'] not in redundant_indices
                ]

                return filtered_elements

            except (KeyError, ValueError) as e:
                logger.error(f"malformed LLM response ({e}), retrying...")
            except Exception as e:
                logger.error(f"LLM call failed, retry: {e}")

    def _backtrack(
        self, 
        start_url: str, 
        target_url: str, 
        target_state_id: str, 
        new_page_opened: bool
    ) -> bool:
        """
        Backtrack to the target state using various strategies.
        
        Strategies:
        1. If new page opened, close it and return to previous page
        2. If URL changed, use browser back
        3. Navigate to target state by replaying the path from root
        
        Args:
            start_url: URL before backtracking
            target_url: Target URL to return to
            target_state_id: Target state ID
            new_page_opened: Whether a new page was opened
            
        Returns:
            True if backtracking succeeded, False otherwise
        """
        logger.info(f"Backtracking to {target_state_id}: {target_url}")
        target_state_dict = self.state_graph.get_state(target_state_id)
        
        if not target_state_dict:
            logger.error(f"Target state {target_state_id} not found.")
            return False

        # Strategy 1: Close new page if opened
        if new_page_opened:
            if self._try_backtrack_via_page_close(target_state_dict, target_url):
                return True
        
        # Strategy 2: Use browser back if URL changed
        if start_url != target_url:
            if self._try_backtrack_via_browser_back(target_state_dict, target_url):
                return True
        
        # Strategy 3: Navigate via state replay
        return self._try_backtrack_via_state_navigation(
            target_state_id, 
            target_state_dict, 
            target_url
        )

    def _try_backtrack_via_page_close(self, target_state_dict: dict, target_url: str) -> bool:
        """Try to backtrack by closing the new page."""
        if not self.browser_session.back_to_previous_page():
            return False
        
        # Use progressive state comparison (URL + Accessibility Tree only, no VLM for page close)
        current_url = self.browser_session.get_url()
        current_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
        current_accessibility_tree = current_accessibility_tree_data['tree_str']
        
        # Load the target accessibility tree
        try:
            target_accessibility_tree = read_text(target_state_dict['accessibility_tree_path'])
        except Exception as e:
            logger.error(f"Failed to load target accessibility tree: {e}")
            target_accessibility_tree = None

        # Compare without VLM (page close is usually reliable)
        is_same, reason = compare_states_progressive(
            current_url=current_url,
            target_url=target_url,
            current_accessibility_tree=current_accessibility_tree,
            target_accessibility_tree=target_accessibility_tree,
            use_vlm=False
        )
        
        if is_same:
            logger.info(f"Successfully backtracked to {target_url} via closing new page (verified by {reason})")
            return True
        return False

    def _try_backtrack_via_browser_back(self, target_state_dict: dict, target_url: str) -> bool:
        """Try to backtrack using browser back button."""
        if not self.browser_session.navigate_back():
            return False
        
        # Use progressive state comparison (URL + Accessibility Tree only, no VLM for browser back)
        current_url = self.browser_session.get_url()
        current_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
        current_accessibility_tree = current_accessibility_tree_data['tree_str']
        
        # Load the target accessibility tree
        try:
            target_accessibility_tree = read_text(target_state_dict['accessibility_tree_path'])
        except Exception as e:
            logger.error(f"Failed to load target accessibility tree: {e}")
            target_accessibility_tree = None

        # Compare without VLM (browser back is usually reliable)
        is_same, reason = compare_states_progressive(
            current_url=current_url,
            target_url=target_url,
            current_accessibility_tree=current_accessibility_tree,
            target_accessibility_tree=target_accessibility_tree,
            use_vlm=False
        )
        
        if is_same:
            logger.info(f"Successfully backtracked to {target_url} via browser back (verified by {reason})")
            return True
        return False

    def _try_backtrack_via_state_navigation(
        self,
        target_state_id: str,
        target_state_dict: dict,
        target_url: str
    ) -> bool:
        """Try to backtrack by replaying path from root state."""
        self._navigate_to_state(target_state_id)
        
        # Use progressive state comparison with full VLM fallback
        current_url = self.browser_session.get_url()
        current_accessibility_tree_data = self.browser_session.get_accessibility_tree(True)
        current_accessibility_tree = current_accessibility_tree_data['tree_str']
        
        # Get screenshots for VLM fallback (state navigation may need visual verification)
        current_screenshot = self.browser_session.get_screenshot()
        try:
            target_screenshot = read_bytes(target_state_dict['screenshot_path'])
        except Exception as e:
            logger.error(f"Failed to load target screenshot: {e}")
            target_screenshot = None

        # Load the target accessibility tree
        try:
            target_accessibility_tree = read_text(target_state_dict['accessibility_tree_path'])
        except Exception as e:
            logger.error(f"Failed to load target accessibility tree: {e}")
            target_accessibility_tree = None

        # Progressive comparison with VLM enabled (navigation is less reliable)
        is_same, reason = compare_states_progressive(
            current_url=current_url,
            target_url=target_url,
            current_accessibility_tree=current_accessibility_tree,
            target_accessibility_tree=target_accessibility_tree,
            current_screenshot=current_screenshot,
            target_screenshot=target_screenshot,
            use_vlm=True
        )
        
        if is_same:
            logger.info(f"Successfully backtracked to {target_url} via state navigation (verified by {reason})")
            return True
        
        logger.error(f"Failed to backtrack to {target_url}.")
        return False
        
    def _navigate_to_state(self, target_state_id: str) -> None:
        """
        Navigate to target state by replaying the path from root.
        
        Args:
            target_state_id: ID of the target state to navigate to
        """
        path = self.state_graph.get_path_to_state(target_state_id)

        # Navigate to the root state
        root_state = self.state_graph.get_state("S0")
        self.browser_session.reset_context()  # Restart context to avoid session interference
        self.browser_session.navigate(root_state['url'])

        # Re-apply login after the context reset wiped the session (no-op without a hook).
        self._run_login_hook()

        # Replay the path by executing each transition
        for step in path:
            transition = step['transition']
            logger.info(f"Replaying transition: {transition}")
            self._execute_transition(transition)

    def _execute_transition(self, transition: dict) -> None:
        """
        Execute a single transition action (supports composite actions like form submission).
        
        Args:
            transition: Transition dict containing event details
        """
        is_composite = transition.get('is_composite', False)
        
        if is_composite:
            # Execute composite operation (e.g., form submission)
            self._execute_composite_transition(transition)
        else:
            # Execute single atomic action
            self._execute_single_action(transition)

    def _execute_composite_transition(self, transition: dict) -> None:
        """
        Execute a composite transition (e.g., form submission with multiple actions).
        
        Args:
            transition: Transition dict containing sub_actions
        """
        composite_type = transition.get('composite_type', 'unknown')
        sub_actions_json = transition.get('sub_actions', '[]')
        
        # Parse sub_actions (may be JSON string or list)
        if isinstance(sub_actions_json, str):
            try:
                sub_actions = json.loads(sub_actions_json)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse sub_actions: {sub_actions_json}")
                return
        else:
            sub_actions = sub_actions_json or []
        
        # Sort by sequence
        sub_actions.sort(key=lambda x: x.get('sequence', 0))
        
        logger.info(f"Executing composite transition ({composite_type}) with {len(sub_actions)} sub-actions")
        
        for i, action in enumerate(sub_actions):
            event_type = action['event_type']
            xpath = action['xpath']
            filled_value = action.get('filled_value')
            element_desc = action.get('element', xpath)
            
            logger.info(f"    [{i+1}/{len(sub_actions)}] {event_type} on {element_desc}")
            
            if event_type == "fill":
                self.browser_session.fill_element(xpath, filled_value)
            elif event_type == "fill and enter":
                self.browser_session.fill_and_enter_element(xpath, filled_value)
            elif event_type == "click":
                self.browser_session.click_element(xpath)
            elif event_type == "select":
                self.browser_session.select_element(xpath)
            elif event_type == "hover":
                self.browser_session.hover_element(xpath)
        
        # Check if new page opened after composite action
        if transition.get('new_page_opened'):
            new_page_opened = self.browser_session.check_new_page_opened()
            if not new_page_opened:
                logger.error(f"    Expected new page after {composite_type}, but not detected")
            else:
                logger.info(f"    New page opened after {composite_type}")

    def _execute_single_action(self, transition: dict) -> None:
        """
        Execute a single atomic action.
        
        Args:
            transition: Transition dict containing event details
        """
        event_type = transition['event_type']
        xpath = transition['xpath']
        
        if event_type == "scroll":
            # Handle scroll operation
            filled_value = transition.get('filled_value', '')
            # Parse scroll amount from filled_value (format: "XXXpx")
            try:
                scroll_amount = float(filled_value.replace('px', ''))
                self.browser_session.scroll_page(scroll_amount)
            except (ValueError, AttributeError) as e:
                logger.error(f"    Failed to parse scroll amount from '{filled_value}': {e}")
        elif event_type == "fill":
            self.browser_session.fill_element(xpath, transition['filled_value'])
        elif event_type == "fill and enter":
            self.browser_session.fill_and_enter_element(xpath, transition['filled_value'])
        elif event_type == "hover":
            self.browser_session.hover_element(xpath)
        elif event_type == "select":
            self.browser_session.select_element(xpath)
        else:  # click
            self.browser_session.click_element(xpath)

        # Verify new page opened if expected
        if transition.get('new_page_opened'):
            new_page_opened = self.browser_session.check_new_page_opened()
            if not new_page_opened:
                logger.error(
                    f"    Expected a new page to be opened after action on "
                    f"{transition['element']}, but it was not detected."
                )
            else:
                logger.info(f"    New page opened after action on {transition['element']}")
