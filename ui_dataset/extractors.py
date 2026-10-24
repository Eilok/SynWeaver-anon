"""Extract state and transition records from the Neo4j state graph."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from config.shared import Neo4jSettings
from storage.neo4j import create_driver, create_session

logger = logging.getLogger(__name__)
IGNORED_ACTREE_PROPERTIES = ["focusable", "readonly", "level", "settable", "multiline", "invalid"]


@dataclass
class StateData:
    """Represent a state node and its captured page material."""

    state_id: str
    url: str
    screenshot_path: Optional[str] = None
    accessibility_tree_path: Optional[str] = None
    accessibility_tree_content: Optional[str] = None
    depth: int = 0
    visit_count: int = 1

    def load_accessibility_tree(self) -> Optional[str]:
        """Load the accessibility tree from its captured file."""
        if self.accessibility_tree_path and os.path.exists(self.accessibility_tree_path):
            try:
                with open(self.accessibility_tree_path, encoding="utf-8") as file:
                    self.accessibility_tree_content = file.read()
            except OSError as exc:
                logger.warning("Failed to load accessibility tree for %s: %s", self.state_id, exc)
        return self.accessibility_tree_content

    def has_valid_screenshot(self) -> bool:
        """Return whether the captured screenshot exists."""
        return bool(self.screenshot_path and os.path.exists(self.screenshot_path))

    def to_dict(self) -> Dict[str, Any]:
        """Return the legacy serializable state fields."""
        return {
            "state_id": self.state_id,
            "url": self.url,
            "screenshot_path": self.screenshot_path,
            "accessibility_tree_path": self.accessibility_tree_path,
            "depth": self.depth,
            "visit_count": self.visit_count,
        }


@dataclass
class TransitionData:
    """Represent a transition edge and its source and target material."""

    transition_id: str
    from_state_id: str
    to_state_id: str
    from_url: str
    to_url: str
    event_type: str
    element: str
    xpath: str
    accessibility_node_info: Optional[Dict[str, Any]] = None
    filled_value: Optional[str] = None
    new_page_opened: bool = False
    is_composite: bool = False
    composite_type: Optional[str] = None
    sub_actions: Optional[List[Dict[str, Any]]] = None
    from_screenshot_path: Optional[str] = None
    to_screenshot_path: Optional[str] = None
    operation_screenshot_path: Optional[str] = None
    element_screenshot_path: Optional[str] = None
    from_accessibility_tree_path: Optional[str] = None
    to_accessibility_tree_path: Optional[str] = None
    from_accessibility_tree_content: Optional[str] = None
    to_accessibility_tree_content: Optional[str] = None

    def load_accessibility_trees(self) -> None:
        """Load both accessibility trees from their captured files."""
        for path, attribute in (
            (self.from_accessibility_tree_path, "from_accessibility_tree_content"),
            (self.to_accessibility_tree_path, "to_accessibility_tree_content"),
        ):
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as file:
                    setattr(self, attribute, file.read())
            except OSError as exc:
                logger.warning("Failed to load accessibility tree for %s: %s", self.transition_id, exc)

    def has_valid_screenshots(self) -> bool:
        """Return whether both source and target screenshots exist."""
        return bool(
            self.from_screenshot_path
            and os.path.exists(self.from_screenshot_path)
            and self.to_screenshot_path
            and os.path.exists(self.to_screenshot_path)
        )

    def has_operation_screenshot(self) -> bool:
        """Return whether the operation screenshot exists."""
        return bool(self.operation_screenshot_path and os.path.exists(self.operation_screenshot_path))

    def get_element_tree_representation(self) -> str:
        """Format the target node as an accessibility-tree line."""
        info = self.accessibility_node_info
        if not info:
            return self.element
        result = f"[{info.get('nodeId', '?')}] {info.get('role', '')} {info.get('name', '')!r}"
        properties = []
        for prop in info.get("properties", []):
            if not isinstance(prop, dict) or prop.get("name") in IGNORED_ACTREE_PROPERTIES:
                continue
            value = prop.get("value", "")
            if isinstance(value, dict):
                value = value.get("value", "")
            properties.append(f"{prop.get('name', '')}: {value}")
        if properties:
            result += " " + " ".join(properties)
        return result

    def get_action_description(self) -> str:
        """Return the original human-readable action description."""
        element = self.get_element_tree_representation()
        descriptions = {
            "click": f"Click on {element}",
            "fill": f"Fill {element} with '{self.filled_value}'",
            "fill and enter": f"Fill {element} with '{self.filled_value}' and press Enter",
            "select": f"Select option in {element}",
            "hover": f"Hover over {element}",
            "form_submission": f"Submit form via {element}",
        }
        return descriptions.get(self.event_type, f"{self.event_type} on {element}")

    def to_dict(self) -> Dict[str, Any]:
        """Return the legacy serializable transition fields."""
        return {name: getattr(self, name) for name in (
            "transition_id", "from_state_id", "to_state_id", "from_url", "to_url",
            "event_type", "element", "xpath", "accessibility_node_info", "filled_value",
            "new_page_opened", "is_composite", "composite_type", "sub_actions",
            "from_screenshot_path", "to_screenshot_path", "operation_screenshot_path",
            "element_screenshot_path",
        )}


class _GraphExtractor:
    """Share Neo4j lifecycle handling between the two extractors."""

    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self.namespace = settings.namespace
        self.driver = None

    def connect(self) -> None:
        """Create and verify the shared Neo4j driver."""
        self.driver = create_driver(self.settings)
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self.settings.url)

    def close(self) -> None:
        """Close the driver once; repeated calls are harmless."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class StateExtractor(_GraphExtractor):
    """Extract state nodes for one configured namespace."""

    def __init__(self, settings: Neo4jSettings) -> None:
        super().__init__(settings)

    def count_states(self) -> int:
        """Count states in the configured namespace."""
        with create_session(self.driver, self.settings) as session:
            result = session.run(f"MATCH (s:{self.namespace}) RETURN count(s) AS count")
            return result.single()["count"]

    def _state_from_record(self, record: Any, load_accessibility_tree: bool) -> StateData:
        state = StateData(
            state_id=record["state_id"],
            url=record["url"],
            screenshot_path=record["screenshot_path"],
            accessibility_tree_path=record["accessibility_tree_path"],
            depth=record["depth"] or 0,
            visit_count=record["visit_count"] or 1,
        )
        if load_accessibility_tree:
            state.load_accessibility_tree()
        return state

    def extract_all(self, load_accessibility_tree: bool = True, dedup_by_url: bool = False) -> Generator[StateData, None, None]:
        """Extract states in state-id order, optionally deduplicating URLs."""
        query = f"""
            MATCH (s:{self.namespace})
            RETURN s.state_id AS state_id, s.url AS url,
                   s.screenshot_path AS screenshot_path,
                   s.accessibility_tree_path AS accessibility_tree_path,
                   s.depth AS depth, s.visit_count AS visit_count
            ORDER BY s.state_id
        """
        seen_urls: set[str] = set()
        with create_session(self.driver, self.settings) as session:
            for record in session.run(query):
                if dedup_by_url and record["url"] in seen_urls:
                    continue
                seen_urls.add(record["url"])
                yield self._state_from_record(record, load_accessibility_tree)

    def extract_by_id(self, state_id: str, load_accessibility_tree: bool = True) -> Optional[StateData]:
        """Extract one state with a parameterized state-id query."""
        query = f"""
            MATCH (s:{self.namespace} {{state_id: $state_id}})
            RETURN s.state_id AS state_id, s.url AS url,
                   s.screenshot_path AS screenshot_path,
                   s.accessibility_tree_path AS accessibility_tree_path,
                   s.depth AS depth, s.visit_count AS visit_count
        """
        with create_session(self.driver, self.settings) as session:
            record = session.run(query, state_id=state_id).single()
        return self._state_from_record(record, load_accessibility_tree) if record else None


class TransitionExtractor(_GraphExtractor):
    """Extract transition edges for one configured namespace."""

    def __init__(self, settings: Neo4jSettings) -> None:
        super().__init__(settings)

    def count_transitions(self) -> int:
        """Count transition edges in the configured namespace."""
        with create_session(self.driver, self.settings) as session:
            result = session.run(f"MATCH (:{self.namespace})-[t:TRANSITION]->(:{self.namespace}) RETURN count(t) AS count")
            return result.single()["count"]

    @staticmethod
    def _parse_json(value: Any) -> Any:
        """Parse graph JSON fields while retaining invalid values as None."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    def _transition_from_record(self, record: Any, transition_id: str, load_accessibility_tree: bool) -> TransitionData:
        transition = TransitionData(
            transition_id=transition_id,
            from_state_id=record["from_state_id"], to_state_id=record["to_state_id"],
            from_url=record["from_url"], to_url=record["to_url"],
            event_type=record["event_type"], element=record["element"], xpath=record["xpath"],
            accessibility_node_info=self._parse_json(record["accessibility_node_info"]),
            filled_value=record["filled_value"], new_page_opened=record["new_page_opened"] or False,
            is_composite=record["is_composite"] or False,
            composite_type=record["composite_type"],
            sub_actions=self._parse_json(record["sub_actions"]),
            from_screenshot_path=record["from_screenshot_path"], to_screenshot_path=record["to_screenshot_path"],
            operation_screenshot_path=record["operation_screenshot_path"],
            element_screenshot_path=record["element_screenshot_path"],
            from_accessibility_tree_path=record["from_accessibility_tree_path"],
            to_accessibility_tree_path=record["to_accessibility_tree_path"],
        )
        if load_accessibility_tree:
            transition.load_accessibility_trees()
        return transition

    def _query(self) -> str:
        return f"""
            MATCH (from:{self.namespace})-[t:TRANSITION]->(to:{self.namespace})
            RETURN from.state_id AS from_state_id, to.state_id AS to_state_id,
                   from.url AS from_url, to.url AS to_url,
                   from.screenshot_path AS from_screenshot_path,
                   to.screenshot_path AS to_screenshot_path,
                   from.accessibility_tree_path AS from_accessibility_tree_path,
                   to.accessibility_tree_path AS to_accessibility_tree_path,
                   t.event_type AS event_type, t.element AS element, t.xpath AS xpath,
                   t.accessibility_node_info AS accessibility_node_info,
                   t.filled_value AS filled_value, t.new_page_opened AS new_page_opened,
                   t.is_composite AS is_composite, t.composite_type AS composite_type,
                   t.sub_actions AS sub_actions, t.screenshot_op_path AS operation_screenshot_path,
                   t.element_screenshot_path AS element_screenshot_path
            ORDER BY from.state_id, to.state_id
        """

    def extract_all(self, load_accessibility_tree: bool = True, sample: bool = False, scroll_cap: int = 25) -> Generator[TransitionData, None, None]:
        """Extract transitions, applying the legacy sampling policy when requested."""
        scroll_count = 0
        transition_count = 0
        with create_session(self.driver, self.settings) as session:
            for index, record in enumerate(session.run(self._query())):
                if sample and index % 2 == 0:
                    continue
                if sample and record["event_type"] == "scroll":
                    if scroll_count >= scroll_cap:
                        continue
                    scroll_count += 1
                transition = self._transition_from_record(record, f"T{transition_count:03d}", load_accessibility_tree)
                transition_count += 1
                yield transition

    def extract_by_states(self, from_state_id: str, to_state_id: str, load_accessibility_tree: bool = True) -> Optional[TransitionData]:
        """Extract one transition with parameterized source and target IDs."""
        query = f"""
            MATCH (from:{self.namespace} {{state_id: $from_state_id}})-[t:TRANSITION]->
                  (to:{self.namespace} {{state_id: $to_state_id}})
            RETURN from.state_id AS from_state_id, to.state_id AS to_state_id,
                   from.url AS from_url, to.url AS to_url,
                   from.screenshot_path AS from_screenshot_path,
                   to.screenshot_path AS to_screenshot_path,
                   from.accessibility_tree_path AS from_accessibility_tree_path,
                   to.accessibility_tree_path AS to_accessibility_tree_path,
                   t.event_type AS event_type, t.element AS element, t.xpath AS xpath,
                   t.accessibility_node_info AS accessibility_node_info,
                   t.filled_value AS filled_value, t.new_page_opened AS new_page_opened,
                   t.is_composite AS is_composite, t.composite_type AS composite_type,
                   t.sub_actions AS sub_actions, t.screenshot_op_path AS operation_screenshot_path,
                   t.element_screenshot_path AS element_screenshot_path
        """
        with create_session(self.driver, self.settings) as session:
            record = session.run(query, from_state_id=from_state_id, to_state_id=to_state_id).single()
        if not record:
            return None
        return self._transition_from_record(record, f"T_{from_state_id}_to_{to_state_id}", load_accessibility_tree)
