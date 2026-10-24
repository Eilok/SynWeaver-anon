"""
Browser Environment Adapter

This module provides adapters to integrate webarena's browser_env with the
tasks execution module. It handles:
1. Action type conversion between ActionType enum and browser_env actions
2. State extraction from browser observations
3. Browser environment wrapper for consistent interface

Design based on SynthAgent's base_explore.py browser integration.
"""

import os
import sys
import time
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field

# Make the (sibling) webarena source importable — both `from webarena.browser_env`
# (used here) and bare `browser_env` (used by webarena internals / subprocesses).
from tasks.execution.webarena_path import (
    ensure_webarena_importable,
    WEBARENA_ROOT,
    webarena_subprocess_env,
)

ensure_webarena_importable()

from tasks.models import (
    ActionType, Action, Element, RawState, StateInfo,
    ActionExecuteStatus, map_action_type
)
from config.tasks import BrowserSettings as BrowserConfig, EnvironmentSettings as EnvConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Action Conversion
# ==============================================================================

class ActionConverter:
    """
    Converts between tasks.models.Action and browser_env actions.
    
    Handles the mapping of:
    - ActionType enum values to browser_env action creation functions
    - Element IDs and values for action construction
    """
    
    @staticmethod
    def to_browser_action(action: Action) -> Any:
        """
        Convert tasks.models.Action to browser_env Action.
        
        Args:
            action: Task synthesizer Action object
            
        Returns:
            browser_env Action (TypedDict)
        """
        from webarena.browser_env import (
            create_id_based_action,
            create_scroll_action,
            create_goto_url_action,
            create_go_back_action,
            create_go_forward_action,
            create_none_action,
            create_stop_action,
            create_key_press_action,
            create_new_tab_action,
            create_page_focus_action,
            create_page_close_action,
        )
        
        action_type = action.action_type
        
        if action_type == ActionType.CLICK:
            element_id = action.target_element.id if action.target_element else ""
            return create_id_based_action(f"click [{element_id}]")
            
        elif action_type == ActionType.TYPE:
            element_id = action.target_element.id if action.target_element else ""
            value = action.value or ""
            # Default behavior: press Enter after typing
            return create_id_based_action(f"type [{element_id}] [{value}]")
            
        elif action_type == ActionType.HOVER:
            element_id = action.target_element.id if action.target_element else ""
            return create_id_based_action(f"hover [{element_id}]")
            
        elif action_type == ActionType.SCROLL:
            direction = action.value or "down"
            return create_scroll_action(direction)
            
        elif action_type == ActionType.PRESS:
            key_comb = action.value or ""
            return create_key_press_action(key_comb)
            
        elif action_type == ActionType.GOTO:
            url = action.value or ""
            return create_goto_url_action(url)
            
        elif action_type == ActionType.GO_BACK:
            return create_go_back_action()
            
        elif action_type == ActionType.GO_FORWARD:
            return create_go_forward_action()
        
        elif action_type == ActionType.NEW_TAB:
            return create_new_tab_action()
        
        elif action_type == ActionType.TAB_FOCUS:
            # value should be tab index (int)
            tab_index = int(action.value) if action.value else 0
            return create_page_focus_action(tab_index)
        
        elif action_type == ActionType.CLOSE_TAB:
            return create_page_close_action()
            
        elif action_type == ActionType.NONE:
            return create_none_action()
            
        elif action_type == ActionType.STOP:
            reason = action.value or "Task cannot be completed"
            return create_stop_action(reason)
            
        else:
            logger.warning(f"Unknown action type: {action_type}, returning none action")
            return create_none_action()
    
    @staticmethod
    def get_action_string(action: Action) -> str:
        """
        Get action string for browser_env execution.
        
        Format compatible with create_id_based_action().
        
        Args:
            action: Task synthesizer Action object
            
        Returns:
            Action string for browser_env
        """
        action_type = action.action_type
        
        if action_type == ActionType.CLICK:
            element_id = action.target_element.id if action.target_element else ""
            return f"click [{element_id}]"
            
        elif action_type == ActionType.TYPE:
            element_id = action.target_element.id if action.target_element else ""
            value = action.value or ""
            return f"type [{element_id}] [{value}]"
            
        elif action_type == ActionType.HOVER:
            element_id = action.target_element.id if action.target_element else ""
            return f"hover [{element_id}]"
            
        elif action_type == ActionType.SCROLL:
            direction = action.value or "down"
            return f"scroll [{direction}]"
            
        elif action_type == ActionType.PRESS:
            key_comb = action.value or ""
            return f"press [{key_comb}]"
            
        elif action_type == ActionType.GOTO:
            url = action.value or ""
            return f"goto [{url}]"
            
        elif action_type == ActionType.GO_BACK:
            return "go_back"
            
        elif action_type == ActionType.GO_FORWARD:
            return "go_forward"
        
        elif action_type == ActionType.NEW_TAB:
            return "new_tab"
        
        elif action_type == ActionType.TAB_FOCUS:
            tab_index = action.value or "0"
            return f"tab_focus [{tab_index}]"
        
        elif action_type == ActionType.CLOSE_TAB:
            return "close_tab"
            
        elif action_type == ActionType.NONE:
            value = action.value or ""
            return f"none [{value}]"
            
        elif action_type == ActionType.STOP:
            reason = action.value or ""
            return f"stop [{reason}]"
            
        elif action_type == ActionType.REFLECT:
            return f"reflect [{action.value or ''}]"
            
        else:
            return f"unknown [{action_type}]"


# ==============================================================================
# State Extraction
# ==============================================================================

class StateExtractor:
    """
    Extracts StateInfo from browser_env observations.
    
    Handles:
    - Raw state construction from browser observations
    - Element extraction from accessibility tree
    - Screenshot handling
    """
    
    @staticmethod
    def extract_state(
        env: Any,
        obs: Optional[Dict[str, Any]] = None,
        observation_metadata: Optional[Dict[str, Any]] = None,
    ) -> StateInfo:
        """
        Extract StateInfo from browser environment.
        
        Args:
            env: ScriptBrowserEnv instance
            obs: Observation dict (optional, will be fetched if None)
            observation_metadata: Metadata dict (optional)
            
        Returns:
            StateInfo with raw state and elements
        """
        # Get observation if not provided
        if obs is None or observation_metadata is None:
            obs = env._get_obs()
            observation_metadata = env._get_obs_metadata()
        
        # Deep copy metadata to avoid reference issues when webarena updates it later
        import copy
        observation_metadata = copy.deepcopy(observation_metadata)
        
        # Create RawState
        raw_state = RawState(
            url=env.page.url,
            accessibility_tree=obs.get('text', ''),
            observation_metadata=observation_metadata,
            screenshot=obs.get('image'),
            timestamp=time.time(),
        )
        
        # Extract elements
        elements = StateExtractor.extract_elements(raw_state)
        
        return StateInfo(
            raw_state=raw_state,
            elements=elements,
        )
    
    @staticmethod
    def extract_elements(raw_state: RawState) -> List[Element]:
        """
        Extract interactive elements from raw state.
        
        Based on SynthAgent's Explorer.extract_elements().
        
        Args:
            raw_state: RawState with observation metadata
            
        Returns:
            List of Element objects
        """
        elements = []
        
        obs_nodes_info = raw_state.observation_metadata.get('text', {}).get('obs_nodes_info', {})
        
        for ele_id, ele_info in obs_nodes_info.items():
            try:
                element = Element(
                    id=str(ele_id),
                    name=ele_info.get('text', ''),
                    role=ele_info.get('role', ''),
                    union_bound=tuple(ele_info.get('union_bound', (0, 0, 0, 0))),
                    accessibility_tree_content=ele_info.get('text', ''),
                )
                elements.append(element)
            except Exception as e:
                logger.warning(f"Failed to extract element {ele_id}: {e}")
        
        return elements


# ==============================================================================
# Browser Environment Adapter
# ==============================================================================

class BrowserEnvAdapter:
    """
    Adapter for webarena's ScriptBrowserEnv.
    
    Provides a consistent interface for:
    - Environment initialization and reset
    - Action execution
    - State observation
    - Tab management
    
    Based on SynthAgent's Explorer class browser integration.
    """
    
    def __init__(
        self,
        browser_config: BrowserConfig,
        env_config: EnvConfig,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize browser adapter.
        
        Args:
            browser_config: Browser configuration
            env_config: Environment configuration
            output_dir: Directory for saving cookies and state
        """
        self.browser_config = browser_config
        self.env_config = env_config
        self.output_dir = output_dir or "./output"
        
        self._env: Optional[Any] = None  # ScriptBrowserEnv
        self._is_initialized = False
        # Host of the current task's start_url; treated as internal so its tab
        # is never auto-closed even if it differs from env_domain.
        self._start_url_host: Optional[str] = None

        # Set up environment variables
        self._setup_env_vars()
    
    def _setup_env_vars(self):
        """Set up environment variables for service URLs."""
        self.env_config.pre_process()
    
    @property
    def env(self) -> Any:
        """Get the browser environment instance."""
        return self._env
    
    @property
    def page(self) -> Any:
        """Get the current page."""
        return self._env.page if self._env else None
    
    def init_env(self, start_url: str, sites: Optional[List[str]] = None) -> StateInfo:
        """
        Initialize browser environment and navigate to start URL.
        
        Args:
            start_url: URL to navigate to after initialization
            sites: List of site names for auto-login (e.g., ['gitlab'])
            
        Returns:
            Initial StateInfo
        """
        from webarena.browser_env import ScriptBrowserEnv

        # Remember the start_url host so its tab is recognized as internal.
        self._start_url_host = self._extract_host(start_url)

        self._env = ScriptBrowserEnv(
            headless=self.browser_config.headless,
            slow_mo=self.browser_config.slow_mo,
            observation_type=self.browser_config.observation_type,
            current_viewport_only=self.browser_config.current_viewport_only,
            viewport_size=self.browser_config.viewport_size,
            sleep_after_execution=0.5,  # Default sleep after action
        )
        
        # Reset environment
        observation, info = self._reset_env(start_url, self.env_config.auto_login, sites=sites)
        
        # Extract initial state
        observation_metadata = info.get('observation_metadata', {})
        current_state = StateExtractor.extract_state(
            self._env, observation, observation_metadata
        )
        
        self._is_initialized = True
        logger.info(f"Browser environment initialized, navigated to {start_url}")
        
        return current_state
    
    def _reset_env(
        self,
        start_url: str,
        require_login: bool = True,
        sites: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset browser environment.
        
        Args:
            start_url: URL to navigate to
            require_login: Whether to perform auto-login
            sites: List of site names for auto-login (e.g., ['gitlab'])
            
        Returns:
            Tuple of (observation, info)
        """
        storage_state = None
        
        # Get site type from sites list (use first site for auto-login)
        site_type = sites[0] if sites else None
        
        # Handle auto-login if required
        if require_login and site_type:
            cookie_folder = os.path.join(self.output_dir, "cookie")
            os.makedirs(cookie_folder, exist_ok=True)
            cookie_file = os.path.join(cookie_folder, f"{site_type}_state.json")
            
            # Run auto-login script (subprocess runs with cwd=WEBARENA_ROOT and a
            # PYTHONPATH that makes `browser_env` resolvable as a top-level module).
            auto_login_script = WEBARENA_ROOT / "browser_env" / "auto_login.py"

            if auto_login_script.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "browser_env/auto_login.py",
                        "--auth_folder", os.path.abspath(cookie_folder),
                        "--site_list", site_type,
                    ],
                    cwd=str(WEBARENA_ROOT),
                    env=webarena_subprocess_env(),
                    check=False,
                )
                
                if os.path.exists(cookie_file):
                    storage_state = cookie_file
                    logger.info(f"Loaded cookies from {cookie_file}")
        
        # Create config file for reset
        config_file = os.path.join(self.output_dir, "init_env.json")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        import json
        with open(config_file, 'w') as f:
            json.dump({
                'start_url': start_url,
                'storage_state': storage_state,
            }, f)
        
        # Reset environment
        observation, info = self._env.reset(options={'config_file': config_file})
        
        # Set timeouts
        self._env.context.set_default_timeout(120000)
        self._env.context.set_default_navigation_timeout(120000)
        
        return observation, info
    
    def reset_to_url(self, url: str, sites: Optional[List[str]] = None) -> StateInfo:
        """
        Reset all tabs and navigate to URL.
        
        Args:
            url: URL to navigate to
            sites: List of site names for auto-login
            
        Returns:
            New StateInfo
        """
        if not self._is_initialized:
            return self.init_env(url, sites=sites)
        
        from webarena.browser_env import create_goto_url_action
        
        try:
            # Close extra pages directly. Popup pages may not carry WebArena
            # private CDP clients, so closing them through env.step() can fail
            # while it attempts to create an accessibility observation.
            for page in list(self._env.context.pages)[1:]:
                try:
                    page.close()
                except Exception as e:
                    logger.warning(f"Failed to close tab during reset: {e}")
            if self._env.context.pages:
                self._env.page = self._env.context.pages[0]
                self._env.page.bring_to_front()
            
            # Navigate to URL
            obs, _, _, _, info = self._env.step(create_goto_url_action(url))
            
            return StateExtractor.extract_state(
                self._env, obs, info.get('observation_metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error resetting to URL {url}: {e}")
            # Fallback: just navigate
            obs, _, _, _, info = self._env.step(create_goto_url_action(url))
            return StateExtractor.extract_state(
                self._env, obs, info.get('observation_metadata', {})
            )
    
    def execute_action(self, action: Action, current_state: Optional[StateInfo] = None) -> StateInfo:
        """
        Execute an action and return the new state.
        
        Args:
            action: Action to execute
            current_state: Current state (for validation)
            
        Returns:
            New StateInfo after action execution
        """
        from webarena.browser_env import create_id_based_action
        
        if not self._is_initialized:
            raise RuntimeError("Browser environment not initialized")
        
        # Handle NONE action (no actual browser action)
        if action.action_type == ActionType.NONE:
            obs = self._env._get_obs()
            observation_metadata = self._env._get_obs_metadata()
            action.status = ActionExecuteStatus.SUCCESS
            return StateExtractor.extract_state(self._env, obs, observation_metadata)
        
        # Execute action
        try:
            action_str = ActionConverter.get_action_string(action)
            
            if action.action_type in {ActionType.SCROLL, ActionType.GO_BACK, ActionType.GO_FORWARD}:
                # These don't use create_id_based_action
                browser_action = ActionConverter.to_browser_action(action)
                obs, _, _, _, info = self._env.step(browser_action)
            else:
                obs, _, _, _, info = self._env.step(create_id_based_action(action_str))
            
            action.status = ActionExecuteStatus.SUCCESS
            logger.debug(f"Executed action: {action_str}")
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action.status = ActionExecuteStatus.FAILURE
            obs = self._env._get_obs()
            info = {'observation_metadata': self._env._get_obs_metadata()}
        
        # Handle new tabs (close non-local URLs)
        self._handle_new_tabs()
        
        return StateExtractor.extract_state(
            self._env, obs, info.get('observation_metadata', {})
        )
    
    def _handle_new_tabs(self):
        """Close external tabs opened during actions (not belonging to any
        internal domain).

        Protections against accidentally destroying the working page:
          - Tab 0 (webarena's main working page, ``context.pages[0]``) is never
            closed, regardless of domain matching.
          - Never close down to zero tabs (skip the close that would empty the
            context), so webarena never has to spawn a client-less page.
        """
        # Do not route popup cleanup through ScriptBrowserEnv.step(). WebArena
        # attaches a private page.client CDP session only to pages it creates;
        # browser popups do not have it. Focusing such a popup via step makes
        # observation creation fail before PAGE_CLOSE can run.
        for page in list(self._env.context.pages)[1:]:
            if self._is_env_url(page.url):
                continue
            try:
                url = page.url
                page.close()
                logger.debug(f"Closed external tab: {url}")
            except Exception as e:
                logger.warning(f"Failed to close external tab {page.url}: {e}")

        # A popup may have become the active page before cleanup. Always restore
        # the WebArena-managed first page so future observations have a CDP client.
        pages = self._env.context.pages
        if pages and (
            self._env.page not in pages or not self._is_env_url(self._env.page.url)
        ):
            self._env.page = pages[0]
            self._env.page.bring_to_front()

    @staticmethod
    def _extract_host(url: str) -> Optional[str]:
        """Extract the lowercase hostname from a URL (without port)."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            host = urlparse(url).hostname
            return host.lower() if host else None
        except Exception:
            return None

    def _is_env_url(self, url: str) -> bool:
        """Check whether ``url``'s host belongs to an internal/allowed domain.

        Internal domains are (all matched by hostname suffix, so ``webarena.test``
        matches ``a.webarena.test``):
          - ``env_config.env_domain`` (and its AWS EC2 hyphen-DNS variant)
          - each entry of ``env_config.allowed_domains``
          - the current task's ``start_url`` host (auto-included)
          - localhost loopback hosts
        """
        host = self._extract_host(url)
        if host is None:
            # Non-http schemes (about:blank, chrome://, data:) are considered
            # internal so we never try to close the working page over them.
            return True

        # Collect candidate internal domains.
        candidates: List[str] = []
        env_domain = self.env_config.env_domain
        if env_domain:
            candidates.append(env_domain.lower())
            # AWS EC2 DNS format: dots become hyphens.
            candidates.append(env_domain.replace('.', '-').lower())
        if self._start_url_host:
            candidates.append(self._start_url_host)
        extra = getattr(self.env_config, 'allowed_domains', None) or []
        candidates.extend(d.lower() for d in extra if d)

        for domain in candidates:
            if host == domain or host.endswith('.' + domain):
                return True

        # Loopback fallbacks.
        return host in {'localhost', '127.0.0.1', '0.0.0.0'}
    
    def get_current_state(self) -> StateInfo:
        """Get current browser state."""
        if not self._is_initialized:
            raise RuntimeError("Browser environment not initialized")
        
        return StateExtractor.extract_state(self._env)
    
    def goto_url(self, url: str) -> StateInfo:
        """Navigate to URL."""
        from webarena.browser_env import create_goto_url_action
        
        obs, _, _, _, info = self._env.step(create_goto_url_action(url))
        time.sleep(3)  # Wait for page load
        
        return StateExtractor.extract_state(
            self._env, obs, info.get('observation_metadata', {})
        )
    
    def can_scroll_down(self) -> bool:
        """Check if current page can scroll down."""
        if not self._is_initialized:
            return False
        
        try:
            return self._env.page.evaluate(
                """() => {
                    const doc = document.documentElement;
                    const scrolled = doc.scrollTop + window.innerHeight;
                    const total = doc.scrollHeight;
                    return scrolled < total;
                }"""
            )
        except Exception:
            return False
    
    def close(self):
        """Close the browser environment."""
        if self._env:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self._env = None
            self._is_initialized = False
            logger.info("Browser environment closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# Utility Functions
# ==============================================================================

def format_elements_for_llm(
    elements: List[Element],
    excluding_elements: Optional[set] = None,
    return_dict: bool = False,
) -> Union[str, List[Dict]]:
    """
    Format elements for LLM prompt.
    
    Based on SynthAgent's Explorer._format_elements_for_llm().
    
    Args:
        elements: List of Element objects
        excluding_elements: Set of (name, union_bound) tuples to exclude
        return_dict: Whether to return dict format
        
    Returns:
        Formatted string or list of dicts
    """
    excluding_elements = excluding_elements or set()
    
    # Filter out static text and excluded elements
    filtered_elements = [
        e for e in elements
        if not e.name.startswith('StaticText') and
        (e.name, e.union_bound) not in excluding_elements
    ]
    
    formatted = [
        {'element_id': element.id, 'text': element.name}
        for element in filtered_elements
    ]
    
    if return_dict:
        return formatted
    
    return "\n".join([str(item) for item in formatted])


def format_page_context(state: StateInfo) -> str:
    """
    Format page context (accessibility tree) for LLM.
    
    Args:
        state: StateInfo object
        
    Returns:
        Accessibility tree string
    """
    return state.raw_state.accessibility_tree
