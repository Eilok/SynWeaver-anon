"""
Data Models for Task Synthesizer Module

This module defines all data classes used in the task synthesis pipeline.
Models are aligned with SynthAgent's data structures (syn/data.py) to ensure
compatibility and consistency.

Key Design Principles:
    1. Enum-based action types for type safety
    2. Dataclass-based structures for immutability and clarity
    3. Serialization support for JSON/JSONL storage
    4. Hash implementations for deduplication
"""

import json
import copy
import time
import hashlib
import numpy as np
from uuid import uuid4
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union


# ==============================================================================
# Enums (aligned with SynthAgent's syn/data.py)
# ==============================================================================

class ActionType(Enum):
    """
    Types of actions that can be performed on web elements.
    
    Aligned with SynthAgent's ActionType enum.
    
    Values:
        CLICK: Click on an element
        TYPE: Type text into an element
        HOVER: Hover over an element
        SCROLL: Scroll up or down
        PRESS: Press a keyboard key combination
        NONE: No action (used for task completion)
        GOTO: Navigate to a URL
        GO_BACK: Go to previous page
        GO_FORWARD: Go to next page
        NEW_TAB: Open a new browser tab
        TAB_FOCUS: Switch focus to a specific tab
        CLOSE_TAB: Close the current tab
        STOP: Stop execution (task cannot be completed)
        REFLECT: Reflection action after failure
    """
    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    SCROLL = "scroll"
    PRESS = "press"
    NONE = "none"
    GOTO = "goto"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    NEW_TAB = "new_tab"
    TAB_FOCUS = "tab_focus"
    CLOSE_TAB = "close_tab"
    STOP = "stop"
    REFLECT = "reflect"
    
    @classmethod
    def from_string(cls, value: str) -> "ActionType":
        """Create ActionType from string (case-insensitive)."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown action type: {value}")
    
    @staticmethod
    def requires_element(action_type: "ActionType") -> bool:
        """Check if action type requires a target element."""
        return action_type in {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.HOVER,
        }
    
    @staticmethod
    def requires_value(action_type: "ActionType") -> bool:
        """Check if action type requires a value."""
        return action_type in {
            ActionType.TYPE,
            ActionType.SCROLL,
            ActionType.GOTO,
            ActionType.PRESS,
            ActionType.NONE,
            ActionType.STOP,
            ActionType.TAB_FOCUS,  # requires tab_index
        }


# ==============================================================================
# Action Mapping (for non-SynthAgent action types)
# ==============================================================================

# Maps external/crawler action types to SynthAgent ActionType
ACTION_TYPE_MAPPING: Dict[str, str] = {
    # Direct mappings
    "click": "click",
    "type": "type",
    "hover": "hover",
    "scroll": "scroll",
    "press": "press",
    "none": "none",
    "goto": "goto",
    "go_back": "go_back",
    "go_forward": "go_forward",
    "new_tab": "new_tab",
    "tab_focus": "tab_focus",
    "close_tab": "close_tab",
    "stop": "stop",
    "reflect": "reflect",
    # External action type mappings
    "fill": "type",           # fill -> type
    "fill and enter": "type", # fill and enter -> type
    "key": "press",           # key -> press
    "select": "click",        # select -> click
    "input": "type",          # input -> type
    "enter": "press",         # enter -> press
}


def map_action_type(action_str: str) -> str:
    """
    Map external action type string to SynthAgent-compatible action type.
    
    Args:
        action_str: The action type string from external source
        
    Returns:
        Mapped action type string compatible with SynthAgent
    """
    action_lower = action_str.lower().strip()
    return ACTION_TYPE_MAPPING.get(action_lower, action_lower)


class ActionExecuteStatus(Enum):
    """Status of action execution in the environment."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    NOT_EXECUTED = "NOT_EXECUTED"


class LowTaskStatus(Enum):
    """Status of a low-level task."""
    BEGIN = "BEGIN"
    IN_PROGRESS = "IN_PROGRESS"
    END = "END"
    NOTACHIEVEABLE = "NOT_ACHIEVABLE"


class ExplorationTrajStatus(Enum):
    """Status of an exploration trajectory."""
    IN_PROGRESS = "IN_PROGRESS"
    END = "END"
    DROP = "DROP"
    OVER_DEPTH_END = "OVER_DEPTH_END"
    EARLY_END_NO_ELEMENTS = "EARLY_END_NO_ELEMENTS_TO_INTERACT"
    EARLY_END_SYNTHESIS = "EARLY_END_DURING_SYNTHESIS"
    EARLY_END_FAILED = "EARLY_END_INTERACTIVE_FAILED"
    EARLY_END_CANCELLED = "EARLY_END_BEING_CANCELLED"


class SynthesizeStatus(Enum):
    """Status of high-level task in synthesis loop."""
    KEEP = "KEEP"
    NEW = "NEW"
    REFINED = "REFINED"
    EXPANDED = "EXPANDED"
    DONE = "DONE"


class ValidationStatus(Enum):
    """Status of trajectory validation."""
    VALID = "VALID"
    INVALID_ACTION = "INVALID_ACTION"
    MISSING_END = "MISSING_END"
    LOOP_DETECTED = "LOOP_DETECTED"
    INCOHERENT = "INCOHERENT"
    RECONSTRUCTED = "RECONSTRUCTED"


# ==============================================================================
# Core Data Classes
# ==============================================================================

@dataclass
class Element:
    """
    Represents a web page element.
    
    Aligned with SynthAgent's Element class.
    
    Attributes:
        id: Element ID (from accessibility tree)
        name: Element name/text content
        role: HTML/ARIA role (button, link, textbox, etc.)
        union_bound: Bounding box (x, y, width, height)
        accessibility_tree_content: Full accessibility tree text
        action_type: Suggested action type for this element
        value: Text content or attribute value
    """
    id: str
    name: str
    role: str
    union_bound: Tuple[int, int, int, int]  # (x, y, width, height)
    accessibility_tree_content: str
    action_type: Optional[ActionType] = None
    value: Optional[str] = None
    
    @classmethod
    def from_accessibility_content(
        cls, 
        content: str, 
        union_bound: Tuple[int, int, int, int],
        element_id: str
    ) -> "Element":
        """
        Create Element from accessibility tree content.
        
        Following SynthAgent's Element.__init__ pattern.
        
        Args:
            content: Accessibility tree content string
            union_bound: Bounding box tuple
            element_id: Element identifier
        """
        import re
        
        if len(content) == 0:
            return cls(
                id=element_id,
                name="empty",
                role="empty",
                union_bound=tuple(union_bound),
                accessibility_tree_content=content,
                action_type=ActionType.NONE,
            )
        
        # Parse name (remove [id] prefix)
        name = re.sub(r'^\[\d+\]\s*', '', content)
        
        # Extract role (first word)
        role = name.split()[0] if name else ""
        
        # Determine action type from role
        action_type = cls._determine_action_type(role)
        
        return cls(
            id=element_id,
            name=name,
            role=role,
            union_bound=tuple(union_bound),
            accessibility_tree_content=content,
            action_type=action_type,
        )
    
    @staticmethod
    def _determine_action_type(role: str) -> Optional[ActionType]:
        """
        Determine action type from element role.
        
        Following SynthAgent's Element.determine_action_type pattern.
        """
        role_lower = role.lower()
        
        # CLICK roles
        if role_lower in {
            'button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
            'option', 'tab', 'checkbox', 'radio', 'switch', 'treeitem', 'rowheader'
        }:
            return ActionType.CLICK
        
        # HOVER roles
        elif role_lower in {'tooltip', 'menubar', 'menu', 'tablist'}:
            return ActionType.HOVER
        
        # TYPE roles
        elif role_lower in {'textbox', 'searchbox', 'combobox', 'spinbutton'}:
            return ActionType.TYPE
        
        # PRESS roles
        elif role_lower in {'slider', 'timer'}:
            return ActionType.PRESS
        
        # SCROLL roles
        elif role_lower in {
            'document', 'main', 'article', 'feed', 'region', 'group',
            'list', 'listbox', 'tree', 'treegrid', 'grid', 'gridcell',
            'rowgroup', 'row', 'log', 'search', 'table', 'scrollbar', 'progressbar'
        }:
            return ActionType.SCROLL
        
        return ActionType.NONE
    
    @staticmethod
    def create_empty() -> "Element":
        """Create an empty element with no action type."""
        return Element(
            id="",
            name="empty",
            role="empty",
            union_bound=(0, 0, 0, 0),
            accessibility_tree_content="",
            action_type=ActionType.NONE,
        )
    
    def needs_value_input(self) -> bool:
        """Check if this element's action requires a value input."""
        return self.action_type in {
            ActionType.TYPE, 
            ActionType.PRESS, 
            ActionType.SCROLL, 
            ActionType.GOTO
        }
    
    def __hash__(self):
        """Hash based on name and bounding box (following SynthAgent)."""
        hash_str = f"{self.name}_{self.union_bound or None}"
        return hash(hash_str)
    
    def __eq__(self, other):
        """Equality based on hash."""
        if not isinstance(other, Element):
            return False
        return hash(self) == hash(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "union_bound": list(self.union_bound),
            "accessibility_tree_content": self.accessibility_tree_content,
            "action_type": self.action_type.value if self.action_type else None,
            "value": self.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Element":
        """Create Element from dictionary."""
        action_type = None
        if data.get("action_type"):
            action_type = ActionType(data["action_type"])
        
        return cls(
            id=data["id"],
            name=data["name"],
            role=data["role"],
            union_bound=tuple(data["union_bound"]),
            accessibility_tree_content=data.get("accessibility_tree_content", ""),
            action_type=action_type,
            value=data.get("value"),
        )


@dataclass
class RawState:
    """
    Raw state information from the environment.
    
    Aligned with SynthAgent's RawState class.
    
    Attributes:
        url: Current page URL
        accessibility_tree: Full accessibility tree string
        observation_metadata: Additional observation metadata
        screenshot: Screenshot (numpy array or path string)
        timestamp: State capture timestamp
    """
    url: str
    accessibility_tree: str
    observation_metadata: Dict[str, Any]
    screenshot: Union[np.ndarray, str]  # Can be array or path
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Normalize URL (remove trailing slash)."""
        if self.url.endswith('/'):
            self.url = self.url[:-1]
    
    def __hash__(self):
        """Hash based on URL and accessibility tree."""
        hash_str = f"{self.url}_{self.accessibility_tree}"
        h = hashlib.sha256()
        h.update(hash_str.encode('utf-8'))
        
        if isinstance(self.screenshot, np.ndarray):
            h.update(self.screenshot.tobytes())
        
        return int(h.hexdigest(), 16)
    
    def hash_by_screenshot(self) -> int:
        """Get hash based on screenshot only."""
        if isinstance(self.screenshot, np.ndarray):
            h = hashlib.sha256()
            h.update(self.screenshot.tobytes())
            return int(h.hexdigest(), 16)
        return hash(self.screenshot)
    
    def to_dict(self, screenshot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            screenshot_dir: Deprecated parameter, kept for backward compatibility.
                           Screenshot saving is now handled by base_executor.
        
        Note:
            Screenshot saving logic has been moved to BaseExecutor._save_state_with_named_screenshot()
            for better control over file naming and selective saving.
        """
        # Determine screenshot value
        if isinstance(self.screenshot, np.ndarray):
            # Return None for numpy arrays - actual saving handled by caller
            screenshot_value = None
        elif isinstance(self.screenshot, str):
            screenshot_value = self.screenshot
        else:
            screenshot_value = None
        
        return {
            "url": self.url,
            "accessibility_tree": self.accessibility_tree,
            "observation_metadata": self.observation_metadata,
            "screenshot": screenshot_value,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], load_screenshot: bool = False) -> "RawState":
        """
        Create RawState from dictionary.
        
        Args:
            data: Dictionary with state data
            load_screenshot: Whether to load screenshot from path
        """
        screenshot = data.get("screenshot")
        
        if load_screenshot and isinstance(screenshot, str) and Path(screenshot).exists():
            from PIL import Image
            screenshot = np.array(Image.open(screenshot))
        
        return cls(
            url=data["url"],
            accessibility_tree=data.get("accessibility_tree", ""),
            observation_metadata=data.get("observation_metadata", {}),
            screenshot=screenshot,
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class StateInfo:
    """
    State information with parsed elements.
    
    Aligned with SynthAgent's StateInfo class.
    
    Attributes:
        raw_state: Raw state data
        elements: List of interactive elements
        summary: Optional state summary (set by LLM)
    """
    raw_state: RawState
    elements: List[Element] = field(default_factory=list)
    summary: Optional[str] = None
    
    def should_terminate(self) -> bool:
        """Check if exploration should terminate (no interactive elements)."""
        return len(self.elements) == 0
    
    def __str__(self):
        return f"StateInfo(URL={self.raw_state.url}, Elements={len(self.elements)}, Summary={self.summary})"
    
    def __hash__(self):
        return hash(self.raw_state)
    
    def to_dict(self, screenshot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            screenshot_dir: Deprecated, kept for backward compatibility.
        """
        return {
            "raw_state": self.raw_state.to_dict(screenshot_dir),
            "elements": [e.to_dict() for e in self.elements],
            "summary": self.summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], load_screenshot: bool = False) -> "StateInfo":
        """Create StateInfo from dictionary."""
        raw_state = RawState.from_dict(data["raw_state"], load_screenshot)
        elements = [Element.from_dict(e) for e in data.get("elements", [])]
        return cls(
            raw_state=raw_state,
            elements=elements,
            summary=data.get("summary"),
        )


@dataclass
class Action:
    """
    Represents a low-level action to execute.
    
    Aligned with SynthAgent's Action class.
    
    Attributes:
        action_type: Type of action
        target_element: Target element (None for non-element actions)
        value: Action value (text to type, scroll direction, etc.)
        coordinates: Click coordinates (optional)
        status: Execution status
    """
    action_type: ActionType
    target_element: Optional[Element] = None
    value: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    status: ActionExecuteStatus = ActionExecuteStatus.NOT_EXECUTED
    
    def __post_init__(self):
        """Validate action configuration."""
        # Validate element requirement
        if ActionType.requires_element(self.action_type) and self.target_element is None:
            raise ValueError(f"Action type {self.action_type} requires a target element")
        
        # Set default empty value if needed
        if self.value is None or (isinstance(self.value, str) and len(self.value) == 0):
            self.value = " "
    
    def get_action_str(self) -> str:
        """
        Get action string representation.
        
        Following SynthAgent's Action.get_action_str pattern.
        
        Returns:
            Action string like "click [5]" or "type [3] 'hello'"
        """
        action_name = self.action_type.value
        
        if self.target_element:
            element_id = self.target_element.id
            
            if self.action_type == ActionType.TYPE:
                return f"{action_name} [{element_id}] {repr(self.value)}"
            else:
                return f"{action_name} [{element_id}]"
        else:
            if self.value and self.value.strip():
                return f"{action_name} {repr(self.value)}"
            return action_name
    
    def get_action_str_with_element(self) -> str:
        """
        Get action string with element's accessibility tree content.
        
        This provides more context about which element the action targets,
        which is useful for history context in prompts and training data.
        
        Returns:
            Action string like "click [5] button 'Submit Form'" or 
            "type [3] textbox 'Search' 'query text'"
        """
        action_name = self.action_type.value
        
        if self.target_element:
            element_id = self.target_element.id
            # Get element's accessibility tree content, truncate if too long
            element_content = self.target_element.accessibility_tree_content or ""
            # Remove the [id] prefix if present (it's redundant)
            if element_content.startswith(f"[{element_id}]"):
                element_content = element_content[len(f"[{element_id}]"):].strip()
            # Truncate to reasonable length
            if len(element_content) > 80:
                element_content = element_content[:77] + "..."
            
            if self.action_type == ActionType.TYPE:
                if element_content:
                    return f"{action_name} [{element_id}] {element_content} {repr(self.value)}"
                else:
                    return f"{action_name} [{element_id}] {repr(self.value)}"
            else:
                if element_content:
                    return f"{action_name} [{element_id}] {element_content}"
                else:
                    return f"{action_name} [{element_id}]"
        else:
            if self.value and self.value.strip():
                return f"{action_name} {repr(self.value)}"
            return action_name
    
    def __str__(self) -> str:
        """String representation using action_str_with_element."""
        return self.get_action_str_with_element()
    
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Get JSON-compatible action dict.
        
        Following the format expected by prompts.
        """
        return {
            "type": self.action_type.value.upper(),
            "element_id": self.target_element.id if self.target_element else "",
            "value": self.value or "",
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "target_element": self.target_element.to_dict() if self.target_element else None,
            "value": self.value,
            "coordinates": list(self.coordinates) if self.coordinates else None,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary."""
        action_type = ActionType(data["action_type"])
        target_element = None
        if data.get("target_element"):
            target_element = Element.from_dict(data["target_element"])
        
        coordinates = None
        if data.get("coordinates"):
            coordinates = tuple(data["coordinates"])
        
        return cls(
            action_type=action_type,
            target_element=target_element,
            value=data.get("value"),
            coordinates=coordinates,
            status=ActionExecuteStatus(data.get("status", "NOT_EXECUTED")),
        )


@dataclass
class LowLevelTask:
    """
    Represents a low-level task (single step in trajectory).
    
    Aligned with SynthAgent's LowLevelTask class.
    
    Attributes:
        task: Low-level task description
        curr_state: Current state before action
        action: Action to perform
        state_after: State after action execution
        task_status: Current task status
        reasoning: LLM reasoning for this action
    """
    task: Optional[str]
    curr_state: StateInfo
    action: Action
    state_after: Optional[StateInfo] = None
    task_status: LowTaskStatus = LowTaskStatus.IN_PROGRESS
    reasoning: Optional[str] = None
    
    def to_dict(self, screenshot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            screenshot_dir: Deprecated, kept for backward compatibility.
        """
        return {
            "task": self.task,
            "curr_state": self.curr_state.to_dict(screenshot_dir),
            "action": self.action.to_dict(),
            "state_after": self.state_after.to_dict(screenshot_dir) if self.state_after else None,
            "task_status": self.task_status.value,
            "reasoning": self.reasoning,
        }
    
    def to_simplified_dict(self) -> Dict[str, Any]:
        """Get simplified dict for prompt formatting."""
        return {
            "observation": self.curr_state.summary or f"URL: {self.curr_state.raw_state.url}",
            "action": self.action.to_json_dict(),
            "observation_after": self.state_after.summary if self.state_after else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], load_screenshot: bool = False) -> "LowLevelTask":
        """Create LowLevelTask from dictionary."""
        curr_state = StateInfo.from_dict(data["curr_state"], load_screenshot)
        action = Action.from_dict(data["action"])
        state_after = None
        if data.get("state_after"):
            state_after = StateInfo.from_dict(data["state_after"], load_screenshot)
        
        return cls(
            task=data.get("task"),
            curr_state=curr_state,
            action=action,
            state_after=state_after,
            task_status=LowTaskStatus(data.get("task_status", "IN_PROGRESS")),
            reasoning=data.get("reasoning"),
        )


@dataclass
class HighLevelTask:
    """
    Represents a high-level task with its trajectory.
    
    Aligned with SynthAgent's HighLevelTask class.
    
    Attributes:
        task: High-level task description
        trajectories: List of low-level task steps
        start_url: Starting URL for this task
    """
    task: str
    trajectories: List[LowLevelTask] = field(default_factory=list)
    start_url: Optional[str] = None
    
    def add_step(self, step: LowLevelTask):
        """Add a low-level step to the trajectory."""
        self.trajectories.append(step)
    
    def get_last_step(self) -> Optional[LowLevelTask]:
        """Get the last step in the trajectory."""
        return self.trajectories[-1] if self.trajectories else None
    
    def is_complete(self) -> bool:
        """Check if task is complete (ends with NONE action)."""
        last = self.get_last_step()
        return last is not None and last.action.action_type == ActionType.NONE
    
    def is_failed(self) -> bool:
        """Check if task failed (ends with STOP action)."""
        last = self.get_last_step()
        return last is not None and last.action.action_type == ActionType.STOP
    
    def to_dict(self, screenshot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            screenshot_dir: Deprecated, kept for backward compatibility.
        """
        return {
            "task": self.task,
            "trajectories": [t.to_dict(screenshot_dir) for t in self.trajectories],
            "start_url": self.start_url,
        }
    
    def to_simplified_dict(self) -> Dict[str, Any]:
        """Get simplified dict for prompt formatting."""
        return {
            "task": self.task,
            "start_url": self.start_url,
            "trajectories": [t.to_simplified_dict() for t in self.trajectories],
            "num_steps": len(self.trajectories),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], load_screenshot: bool = False) -> "HighLevelTask":
        """Create HighLevelTask from dictionary."""
        trajectories = [
            LowLevelTask.from_dict(t, load_screenshot) 
            for t in data.get("trajectories", [])
        ]
        return cls(
            task=data["task"],
            trajectories=trajectories,
            start_url=data.get("start_url"),
        )


@dataclass
class ExplorationTraj:
    """
    Exploration trajectory containing high-level tasks.
    
    Aligned with SynthAgent's ExplorationTraj class.
    
    Attributes:
        curr_state: Current state in exploration
        high_level_tasks: List of high-level tasks attempted
        status: Trajectory status
        uuid: Unique identifier
    """
    curr_state: StateInfo
    high_level_tasks: List[HighLevelTask] = field(default_factory=list)
    status: ExplorationTrajStatus = ExplorationTrajStatus.IN_PROGRESS
    uuid: str = field(default_factory=lambda: str(uuid4()))
    
    def add_low_level_task(self, task: LowLevelTask):
        """Add a low-level task to the current high-level task."""
        if not self.high_level_tasks:
            raise ValueError("No high-level task to add low-level task to")
        self.high_level_tasks[-1].add_step(task)
    
    def add_high_level_task(self, task_desc: str, state: StateInfo):
        """
        Add a new high-level task (e.g., after refinement).
        
        The new task inherits the trajectories from the previous task,
        following SynthAgent's pattern.
        """
        # Inherit trajectories from previous task (SynthAgent pattern)
        prev_trajectories = []
        if self.high_level_tasks:
            prev_trajectories = self.high_level_tasks[-1].trajectories
        
        new_task = HighLevelTask(
            task=task_desc,
            start_url=state.raw_state.url,
            trajectories=prev_trajectories,
        )
        self.high_level_tasks.append(new_task)
        self.curr_state = state
    
    def end_exploration(self):
        """Mark exploration as ended."""
        self.status = ExplorationTrajStatus.END
    
    def get_current_task(self) -> Optional[HighLevelTask]:
        """Get the current (last) high-level task."""
        return self.high_level_tasks[-1] if self.high_level_tasks else None
    
    def get_total_steps(self) -> int:
        """Get total number of low-level steps across all tasks."""
        return sum(len(t.trajectories) for t in self.high_level_tasks)
    
    def to_dict(self, screenshot_dir: Optional[str] = None, simplified: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            screenshot_dir: Deprecated, kept for backward compatibility.
                           Screenshot saving is now handled by BaseExecutor.
            simplified: Whether to use simplified format
        """
        if simplified:
            return {
                "uuid": self.uuid,
                "status": self.status.value,
                "high_level_tasks": [t.to_simplified_dict() for t in self.high_level_tasks],
                "total_steps": self.get_total_steps(),
            }
        
        return {
            "uuid": self.uuid,
            "curr_state": self.curr_state.to_dict(screenshot_dir),
            "high_level_tasks": [t.to_dict(screenshot_dir) for t in self.high_level_tasks],
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], load_screenshot: bool = False) -> "ExplorationTraj":
        """Create ExplorationTraj from dictionary."""
        curr_state = StateInfo.from_dict(data["curr_state"], load_screenshot)
        high_level_tasks = [
            HighLevelTask.from_dict(t, load_screenshot) 
            for t in data.get("high_level_tasks", [])
        ]
        
        return cls(
            curr_state=curr_state,
            high_level_tasks=high_level_tasks,
            status=ExplorationTrajStatus(data.get("status", "IN_PROGRESS")),
            uuid=data.get("uuid", str(uuid4())),
        )


# ==============================================================================
# Graph Sampling Related
# ==============================================================================

@dataclass
class GraphTriplet:
    """
    Represents a triplet sampled from the Neo4j state graph.
    
    Structure: (from_state, transition, to_state)
    
    Attributes:
        id: Unique identifier for this triplet
        from_state_id: Source state ID
        to_state_id: Target state ID
        action_type: Type of action (click, fill, scroll, etc.)
        element_info: Information about the target element
        filled_value: Value filled for input elements
        from_url: URL of the source state
        to_url: URL of the target state
        from_screenshot_path: Path to source state screenshot
        to_screenshot_path: Path to target state screenshot
        from_depth: Depth of source node in graph
        to_depth: Depth of target node in graph
        from_out_degree: Out-degree of source node
        to_out_degree: Out-degree of target node
        from_state: Source state info (loaded on demand)
        to_state: Target state info (loaded on demand)
        transition_info: Additional transition data (legacy, optional)
    """
    # Core identifiers
    id: str = ""
    from_state_id: str = ""
    to_state_id: str = ""
    
    # Action information
    action_type: str = ""
    element_info: Optional[str] = None  # HTML tag representation string
    filled_value: Optional[str] = None
    
    # URLs
    from_url: Optional[str] = None
    to_url: Optional[str] = None
    
    # File paths (screenshots only, no accessibility trees for image-based synthesis)
    from_screenshot_path: Optional[str] = None
    to_screenshot_path: Optional[str] = None
    screenshot_op_path: Optional[str] = None  # Screenshot with element highlighted
    
    # Graph metadata
    from_depth: int = 0
    to_depth: int = 0
    from_out_degree: int = 0
    to_out_degree: int = 0
    
    # State objects (loaded on demand)
    from_state: Optional[StateInfo] = None
    to_state: Optional[StateInfo] = None
    
    # Legacy field for backwards compatibility
    transition_info: Optional[Dict[str, Any]] = None
    
    # Aliases for backwards compatibility
    @property
    def depth(self) -> int:
        """Alias for from_depth."""
        return self.from_depth
    
    @property
    def out_degree(self) -> int:
        """Alias for from_out_degree."""
        return self.from_out_degree
    
    @property
    def event_type(self) -> str:
        """Get the transition event type (alias for action_type)."""
        return self.action_type
    
    @property
    def element(self) -> str:
        """Get the element description (element_info is HTML tag string)."""
        # element_info is the HTML tag representation string from the crawler
        if self.element_info and isinstance(self.element_info, str):
            return self.element_info
        return ""
    
    def is_self_loop(self) -> bool:
        """Check if this is a self-loop (same from and to state)."""
        return self.from_state_id == self.to_state_id
    
    def is_scroll_action(self) -> bool:
        """Check if the action is a scroll."""
        return self.action_type.lower() == "scroll"
    
    def get_mapped_action_type(self) -> str:
        """Get the mapped action type compatible with SynthAgent."""
        return map_action_type(self.action_type)
    
    def get_action_description(self) -> str:
        """
        Get human-readable action description for LLM prompts.
        
        Note: All scroll actions from the graph are scroll down.
        Uses mapped action types for consistency.
        Format: Action [element] following SynthAgent conventions.
        """
        mapped_action = map_action_type(self.action_type)
        # element_info is HTML tag string
        element_desc = self.element or "the page"
        
        if mapped_action == "click":
            return f"Click {element_desc}"
        elif mapped_action == "type":
            value = self.filled_value or ""
            if value:
                return f"Type {value} into {element_desc}"
            else:
                return f"Type into {element_desc}"
        elif mapped_action == "hover":
            return f"Hover over {element_desc}"
        elif mapped_action == "scroll":
            # All scroll actions from the transition graph are scroll down
            return "Scroll down"
        elif mapped_action == "press":
            return f"Press key on {element_desc}"
        elif mapped_action == "goto":
            return "Navigate to URL"
        else:
            return f"{mapped_action.capitalize()} {element_desc}"
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "action_type": self.action_type,
            "mapped_action_type": self.get_mapped_action_type(),
            "element_info": self.element_info,
            "filled_value": self.filled_value,
            "from_url": self.from_url,
            "to_url": self.to_url,
            "from_screenshot_path": self.from_screenshot_path,
            "to_screenshot_path": self.to_screenshot_path,
            "screenshot_op_path": self.screenshot_op_path,
            "from_depth": self.from_depth,
            "to_depth": self.to_depth,
            "from_out_degree": self.from_out_degree,
            "to_out_degree": self.to_out_degree,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphTriplet":
        """Create GraphTriplet from dictionary."""
        return cls(
            id=data.get("id", ""),
            from_state_id=data.get("from_state_id", ""),
            to_state_id=data.get("to_state_id", ""),
            action_type=data.get("action_type", ""),
            element_info=data.get("element_info"),
            filled_value=data.get("filled_value"),
            from_url=data.get("from_url"),
            to_url=data.get("to_url"),
            from_screenshot_path=data.get("from_screenshot_path"),
            to_screenshot_path=data.get("to_screenshot_path"),
            screenshot_op_path=data.get("screenshot_op_path"),
            from_depth=data.get("from_depth", data.get("depth", 0)),
            to_depth=data.get("to_depth", 0),
            from_out_degree=data.get("from_out_degree", data.get("out_degree", 0)),
            to_out_degree=data.get("to_out_degree", 0),
        )


# ==============================================================================
# Validation Related
# ==============================================================================

@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    
    Attributes:
        issue_type: Type of validation issue
        description: Human-readable description
        step_index: Index of problematic step (if applicable)
        severity: Issue severity (error, warning)
    """
    issue_type: ValidationStatus
    description: str
    step_index: Optional[int] = None
    severity: str = "error"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "description": self.description,
            "step_index": self.step_index,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """
    Result of trajectory validation.
    
    Attributes:
        passed: Whether validation passed
        issues: List of validation issues found
        needs_reconstruction: Whether trajectory needs LLM reconstruction
        reconstructed_traj: Reconstructed trajectory (if applicable)
    """
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    needs_reconstruction: bool = False
    reconstructed_traj: Optional[ExplorationTraj] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "needs_reconstruction": self.needs_reconstruction,
        }


# ==============================================================================
# Utility Functions
# ==============================================================================

def serialize_dataclass(obj: Any) -> Any:
    """
    Recursively serialize dataclass to dict.
    
    Handles nested dataclasses, lists, and enums.
    """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return None  # Don't serialize numpy arrays
    else:
        return obj


def deserialize_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """
    Deserialize dict to dataclass instance.
    
    Args:
        data: Dictionary data
        cls: Target dataclass type
    """
    if hasattr(cls, 'from_dict'):
        return cls.from_dict(data)
    return cls(**data)
