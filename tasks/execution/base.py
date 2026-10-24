"""
Base Executor Module

This module provides the foundational execution infrastructure for running
web agent tasks. It handles:
1. Environment management (init, reset, close)
2. State tracking and trajectory recording
3. Data persistence (save/load)
4. LLM client management

Based on SynthAgent's Explorer class with optimizations for our use case.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from tasks.models import (
    ActionType, Action, Element, RawState, StateInfo,
    LowLevelTask, HighLevelTask, ExplorationTraj,
)
from config.tasks import BrowserSettings, EnvironmentSettings, ExecutionSettings
from config.shared import OutputSettings
from tasks.execution.browser_adapter import (
    BrowserEnvAdapter, StateExtractor, ActionConverter,
    format_elements_for_llm, format_page_context,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# LLM token usage persistence

from llm import get_client


class TokenUsagePersistence:
    """Persist token counters while delegating requests to UnifiedLLMClient."""

    def __init__(self, client: Any):
        self.client = client

    def save(self, path: str | Path) -> None:
        from storage.io import write_json
        write_json(self.client.get_token_usage(), path)

    def load(self, path: str | Path) -> None:
        if Path(path).exists():
            with Path(path).open(encoding="utf-8") as handle:
                data = json.load(handle)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count"):
                if key in data:
                    self.client.token_usage[key] = data[key]


# Utilities
# ==============================================================================

def serialize_dataclass(obj: Any) -> Any:
    """Recursively serialize dataclass to dict."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    else:
        return obj


def jsonl_save(data: List[Any], path: str, append: bool = False):
    """Save data to JSONL file."""
    mode = 'a' if append else 'w'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, mode) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def jsonl_load(path: str) -> List[Any]:
    """Load data from JSONL file."""
    if not os.path.exists(path):
        return []
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ==============================================================================
# Base Executor
# ==============================================================================

@dataclass
class ExecutionStats:
    """
    Statistics for execution tracking.
    
    Used by both single-worker execution (BaseExecutor/ExecutionAgent)
    and multi-process execution (MultiExecutor).
    """
    # Common fields
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Single-worker execution fields
    total_steps: int = 0
    total_refinements: int = 0
    
    # Multi-process execution fields
    successful_tasks: int = 0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_steps': self.total_steps,
            'total_refinements': self.total_refinements,
            'successful_tasks': self.successful_tasks,
            'total_time': self.total_time,
            'success_rate': (
                self.completed_tasks / self.total_tasks
                if self.total_tasks > 0 else 0
            ),
        }


class BaseExecutor:
    """
    Base executor for web agent tasks.
    
    Provides:
    - Browser environment management
    - LLM client management
    - State tracking and trajectory recording
    - Data persistence
    
    Subclass this to implement specific execution strategies.
    """
    
    def __init__(self, config: Any):
        """
        Initialize base executor.
        
        Args:
            config: Full configuration object
        """
        self.config = config
        
        # Initialize LLM clients
        conf_path = getattr(config, "config_path", "./config.yaml")
        t1_temperature = getattr(config, "t1_temperature", None)
        t2_temperature = getattr(config, "t2_temperature", None)
        t1_overrides = (
            {"temperature": t1_temperature} if t1_temperature is not None else None
        )
        t2_overrides = (
            {"temperature": t2_temperature} if t2_temperature is not None else None
        )
        self.t1_client = get_client(
            "t1",
            conf_path=conf_path,
            overrides=t1_overrides,
            track_tokens=True,
            fresh=True,
        )
        self.t2_client = get_client(
            "t2",
            conf_path=conf_path,
            overrides=t2_overrides,
            track_tokens=True,
            fresh=True,
        )
        self._t1_usage = TokenUsagePersistence(self.t1_client)
        self._t2_usage = TokenUsagePersistence(self.t2_client)
        
        # Browser adapter
        self.browser_adapter: Optional[BrowserEnvAdapter] = None

        # Trajectory storage
        self.exploration_trajs: List[ExplorationTraj] = []
        self._saved_traj_count: int = 0   # monotonic across save_state() calls
        self.stats = ExecutionStats()
        self.evaluation_results: List[Dict[str, Any]] = []
        
        # Output paths
        self.output_dir = getattr(config, "output_dir", None) or getattr(getattr(config, "output", None), "output_dir", None) or getattr(getattr(config, "output", None), "root_dir", "./outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing data
        self._load_state()
    
    def _load_state(self):
        """Load saved state from disk."""
        # Load token usage
        t1_usage_path = os.path.join(self.output_dir, 't1_token_usage.json')
        t2_usage_path = os.path.join(self.output_dir, 't2_token_usage.json')
        
        self._t1_usage.load(t1_usage_path)
        self._t2_usage.load(t2_usage_path)
        
        evaluation_path = os.path.join(self.output_dir, "evaluation_results.json")
        if os.path.exists(evaluation_path):
            with open(evaluation_path, encoding="utf-8") as handle:
                self.evaluation_results = json.load(handle)

        # Load stats
        stats_path = os.path.join(self.output_dir, 'execution_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                data = json.load(f)
                self.stats = ExecutionStats(**data)
        
        logger.info(f"Loaded state from {self.output_dir}")
    
    def save_state(self):
        """
        Save current state to disk.
        
        Screenshot optimization:
        - Only saves screenshots for the last high-level task in each trajectory
        - Uses meaningful naming: task_{idx}_step_{step_idx}.png
        """
        screenshots_dir = os.path.join(self.output_dir, 'screenshots')

        # Save trajectories with optimized screenshot saving.
        if self.exploration_trajs:
            if getattr(getattr(self.config, "output", None), "save_screenshots", True):
                os.makedirs(screenshots_dir, exist_ok=True)
            trajs_data = []
            for traj in self.exploration_trajs:
                # Monotonic, 1-indexed task number that persists across save_state()
                # calls so per-task screenshot names never collide.
                self._saved_traj_count += 1
                task_id = self._saved_traj_count

                # Save only the last high-level task's screenshots
                traj_dict = self._save_trajectory_with_screenshots(
                    traj, screenshots_dir, task_id
                )
                trajs_data.append(traj_dict)
            
            jsonl_save(trajs_data, os.path.join(self.output_dir, 'trajectories.jsonl'), append=True)
            
            # Simplified version (no screenshots)
            simplified = [traj.to_dict(simplified=True) for traj in self.exploration_trajs]
            jsonl_save(simplified, os.path.join(self.output_dir, 'trajectories_simplified.jsonl'), append=True)
            
            self.exploration_trajs = []
        
        # Save token usage
        self._t1_usage.save(os.path.join(self.output_dir, 't1_token_usage.json'))
        self._t2_usage.save(os.path.join(self.output_dir, 't2_token_usage.json'))
        
        # Save evaluation results
        with open(os.path.join(self.output_dir, "evaluation_results.json"), "w", encoding="utf-8") as handle:
            json.dump(self.evaluation_results, handle, indent=2, ensure_ascii=False)

        # Save stats
        with open(os.path.join(self.output_dir, 'execution_stats.json'), 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config.to_dict() if hasattr(self.config, "to_dict") else self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Saved state to {self.output_dir}")
    
    def _save_trajectory_with_screenshots(
        self,
        traj: ExplorationTraj,
        screenshots_dir: str,
        task_id: int,
    ) -> Dict[str, Any]:
        """
        Save trajectory with optimized screenshot handling.
        
        Only saves screenshots for the last high-level task.
        Uses meaningful naming: task_{task_id}_step_{step_idx}.png
        
        Args:
            traj: Exploration trajectory
            screenshots_dir: Directory to save screenshots
            task_id: Task identifier for naming
            
        Returns:
            Dictionary representation of trajectory
        """
        from PIL import Image
        
        # Build trajectory dict manually to control screenshot saving
        result = {
            "uuid": traj.uuid,
            "status": traj.status.value,
            "high_level_tasks": [],
        }
        
        # Save curr_state without screenshot (not needed for analysis)
        result["curr_state"] = traj.curr_state.to_dict(screenshot_dir=None)
        
        # Process high-level tasks
        for ht_idx, ht in enumerate(traj.high_level_tasks):
            is_last_task = (ht_idx == len(traj.high_level_tasks) - 1)
            
            ht_dict = {
                "task": ht.task,
                "start_url": ht.start_url,
                "trajectories": [],
            }
            
            # Process low-level tasks
            for step_idx, lt in enumerate(ht.trajectories):
                lt_dict = {
                    "task": lt.task,
                    "action": lt.action.to_dict(),
                    "task_status": lt.task_status.value,
                    "reasoning": lt.reasoning,
                }
                
                # Save screenshots only for the last high-level task
                if is_last_task and getattr(getattr(self.config, "output", None), "save_screenshots", True):
                    screenshot_name = f"task_{task_id}_step_{step_idx + 1}"
                    lt_dict["curr_state"] = self._save_state_with_named_screenshot(
                        lt.curr_state, screenshots_dir, screenshot_name
                    )
                    if lt.state_after:
                        lt_dict["state_after"] = self._save_state_with_named_screenshot(
                            lt.state_after, screenshots_dir, f"{screenshot_name}_after"
                        )
                    else:
                        lt_dict["state_after"] = None
                else:
                    # Save state without screenshot
                    lt_dict["curr_state"] = lt.curr_state.to_dict(screenshot_dir=None)
                    lt_dict["state_after"] = lt.state_after.to_dict(screenshot_dir=None) if lt.state_after else None
                
                ht_dict["trajectories"].append(lt_dict)
            
            result["high_level_tasks"].append(ht_dict)
        
        return result
    
    def _save_state_with_named_screenshot(
        self,
        state: StateInfo,
        screenshots_dir: str,
        screenshot_name: str,
    ) -> Dict[str, Any]:
        """
        Save StateInfo with a specifically named screenshot.
        
        Args:
            state: StateInfo to save
            screenshots_dir: Directory for screenshots
            screenshot_name: Base name for screenshot file
            
        Returns:
            Dictionary representation with screenshot path
        """
        from PIL import Image
        
        state_dict = {
            "raw_state": {
                "url": state.raw_state.url,
                "accessibility_tree": state.raw_state.accessibility_tree,
                "observation_metadata": state.raw_state.observation_metadata,
                "timestamp": state.raw_state.timestamp,
            },
            "elements": [e.to_dict() for e in state.elements],
            "summary": state.summary,
        }
        
        # Save screenshot with named file
        screenshot = state.raw_state.screenshot
        if isinstance(screenshot, np.ndarray):
            path = Path(screenshots_dir) / f"{screenshot_name}.png"
            if not path.exists():
                img = Image.fromarray(screenshot)
                img.save(str(path), format='PNG')
            state_dict["raw_state"]["screenshot"] = str(path)
        elif isinstance(screenshot, str):
            state_dict["raw_state"]["screenshot"] = screenshot
        else:
            state_dict["raw_state"]["screenshot"] = None
        
        return state_dict
    
    def init_browser(self, start_url: str, sites: Optional[List[str]] = None) -> StateInfo:
        """
        Initialize browser environment.
        
        Args:
            start_url: URL to navigate to
            sites: List of site names for auto-login (e.g., ['gitlab', 'shopping'])
            
        Returns:
            Initial StateInfo
        """
        self.browser_adapter = BrowserEnvAdapter(
            browser_config=getattr(self.config, "browser", getattr(self.config, "browser_settings", None)),
            env_config=getattr(self.config, "environment", getattr(self.config, "env", None)),
            output_dir=self.output_dir,
        )
        
        return self.browser_adapter.init_env(start_url, sites=sites)
    
    def reset_browser(self, url: str, sites: Optional[List[str]] = None) -> StateInfo:
        """
        Reset browser to URL.
        
        Args:
            url: URL to navigate to
            sites: List of site names for auto-login
            
        Returns:
            New StateInfo
        """
        # Always close and reinitialize to ensure fresh login for each task
        if self.browser_adapter is not None:
            self.browser_adapter.close()
            self.browser_adapter = None
        
        return self.init_browser(url, sites=sites)
    
    def execute_action(self, action: Action, current_state: Optional[StateInfo] = None) -> StateInfo:
        """
        Execute an action in the browser.
        
        Args:
            action: Action to execute
            current_state: Current state for validation
            
        Returns:
            New StateInfo after execution
        """
        if self.browser_adapter is None:
            raise RuntimeError("Browser not initialized")

        return self.browser_adapter.execute_action(action, current_state)
    
    def get_current_state(self) -> StateInfo:
        """Get current browser state."""
        if self.browser_adapter is None:
            raise RuntimeError("Browser not initialized")
        
        return self.browser_adapter.get_current_state()
    
    def close_browser(self):
        """Close browser environment."""
        if self.browser_adapter:
            self.browser_adapter.close()
            self.browser_adapter = None

    def close(self) -> None:
        """Close browser and unified LLM HTTP resources."""
        self.close_browser()
        self.t1_client.close()
        self.t2_client.close()
    
    def format_elements(
        self,
        elements: List[Element],
        excluding: Optional[set] = None,
    ) -> str:
        """
        Format elements for LLM prompt.
        
        Args:
            elements: List of elements
            excluding: Optional set of (name, union_bound) tuples to exclude
            
        Returns:
            Formatted string
        """
        return format_elements_for_llm(elements, excluding)
    
    @staticmethod
    def format_page_context(state: StateInfo) -> str:
        """Format page context for LLM."""
        return format_page_context(state)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_state()
        self.close()
