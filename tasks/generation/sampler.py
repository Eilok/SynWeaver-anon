"""Sample state-transition triplets from the Neo4j state graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator

from config.shared import Neo4jSettings, SamplingSettings
from storage.neo4j import create_driver, create_session
from tasks.models import GraphTriplet

logger = logging.getLogger(__name__)


@dataclass
class SamplingStats:
    """Statistics collected during graph sampling."""

    total_transitions: int = 0
    scroll_self_loops_filtered: int = 0
    non_scroll_self_loops_kept: int = 0
    sampled_count: int = 0
    unique_from_states: int = 0
    unique_to_states: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "total_transitions": self.total_transitions,
            "scroll_self_loops_filtered": self.scroll_self_loops_filtered,
            "non_scroll_self_loops_kept": self.non_scroll_self_loops_kept,
            "sampled_count": self.sampled_count,
            "unique_from_states": self.unique_from_states,
            "unique_to_states": self.unique_to_states,
            "max_depth": self.max_depth,
        }


class GraphSampler:
    """Sample ``(from_state, transition, to_state)`` graph triplets."""

    def __init__(
        self,
        neo4j_settings: Neo4jSettings,
        sampling_settings: SamplingSettings | None = None,
    ) -> None:
        self.neo4j_settings = neo4j_settings
        self.sampling_settings = sampling_settings or SamplingSettings()
        self.driver: Any | None = None
        self.stats = SamplingStats()
        self._node_out_degrees: dict[str, int] = {}
        self._node_depths: dict[str, int] = {}

    def connect(self) -> None:
        """Establish and verify the Neo4j connection."""
        self.driver = create_driver(self.neo4j_settings)
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self.neo4j_settings.url)

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self) -> GraphSampler:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if self.driver is None:
            self.connect()

    def _fetch_node_metadata(self) -> None:
        """Fetch depth and out-degree metadata for all namespace nodes."""
        self._ensure_connected()
        namespace = self.neo4j_settings.namespace
        with create_session(self.driver, self.neo4j_settings) as session:
            result = session.run(
                f"""
                MATCH (s:{namespace})
                RETURN s.state_id AS state_id, s.depth AS depth
                """
            )
            for record in result:
                self._node_depths[record["state_id"]] = record["depth"] or 0

            result = session.run(
                f"""
                MATCH (s:{namespace})
                OPTIONAL MATCH (s)-[t:TRANSITION]->()
                RETURN s.state_id AS state_id, count(t) AS out_degree
                """
            )
            for record in result:
                self._node_out_degrees[record["state_id"]] = record["out_degree"] or 0

        logger.info("Fetched metadata for %d nodes", len(self._node_depths))

    def _fetch_all_transitions(self) -> list[dict[str, Any]]:
        """Fetch every transition and its source and target state properties."""
        self._ensure_connected()
        namespace = self.neo4j_settings.namespace
        transitions: list[dict[str, Any]] = []
        with create_session(self.driver, self.neo4j_settings) as session:
            result = session.run(
                f"""
                MATCH (from:{namespace})-[t:TRANSITION]->(to:{namespace})
                RETURN
                    from.state_id AS from_state_id,
                    from.url AS from_url,
                    from.depth AS from_depth,
                    from.screenshot_path AS from_screenshot_path,
                    to.state_id AS to_state_id,
                    to.url AS to_url,
                    to.depth AS to_depth,
                    to.screenshot_path AS to_screenshot_path,
                    properties(t) AS transition_props
                """
            )
            for record in result:
                transitions.append(
                    {
                        "from_state_id": record["from_state_id"],
                        "from_url": record["from_url"],
                        "from_depth": record["from_depth"] or 0,
                        "from_screenshot_path": record["from_screenshot_path"],
                        "to_state_id": record["to_state_id"],
                        "to_url": record["to_url"],
                        "to_depth": record["to_depth"] or 0,
                        "to_screenshot_path": record["to_screenshot_path"],
                        "transition_props": dict(record["transition_props"]),
                    }
                )

        self.stats.total_transitions = len(transitions)
        logger.info("Fetched %d total transitions from graph", len(transitions))
        return transitions

    def _filter_transitions(
        self, transitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply the original self-loop accounting and source-depth filters."""
        filtered: list[dict[str, Any]] = []
        for transition in transitions:
            from_state_id = transition["from_state_id"]
            to_state_id = transition["to_state_id"]
            event_type = transition["transition_props"].get("event_type", "").lower()
            is_self_loop = from_state_id == to_state_id
            is_scroll = event_type == "scroll"

            # Scroll self-loop filtering remains intentionally disabled to preserve
            # the source behavior.
            # if is_self_loop and is_scroll:
            #     self.stats.scroll_self_loops_filtered += 1
            #     continue

            if is_self_loop and not is_scroll:
                self.stats.non_scroll_self_loops_kept += 1

            from_depth = transition["from_depth"]
            if self.sampling_settings.min_depth > 0 and from_depth < self.sampling_settings.min_depth:
                continue
            if self.sampling_settings.max_depth > 0 and from_depth > self.sampling_settings.max_depth:
                continue
            filtered.append(transition)

        logger.info(
            "Filtered transitions: %d kept, %d scroll self-loops removed, "
            "%d non-scroll self-loops kept",
            len(filtered),
            self.stats.scroll_self_loops_filtered,
            self.stats.non_scroll_self_loops_kept,
        )
        return filtered

    def _sort_transitions(
        self, transitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Stably sort by shallow depth and then descending source out-degree."""
        def sort_key(transition: dict[str, Any]) -> tuple[int, int]:
            depth = transition["from_depth"]
            degree = self._node_out_degrees.get(transition["from_state_id"], 0)
            if self.sampling_settings.depth_priority and self.sampling_settings.degree_priority:
                return depth, -degree
            if self.sampling_settings.depth_priority:
                return depth, 0
            if self.sampling_settings.degree_priority:
                return 0, -degree
            return 0, 0

        return sorted(transitions, key=sort_key)

    def _downsample_transitions(
        self, transitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep odd-indexed sorted edges and cap retained scroll edges."""
        kept: list[dict[str, Any]] = []
        scroll_count = 0
        for index, transition in enumerate(transitions):
            if index % 2 == 0:
                continue
            if transition["transition_props"].get("event_type", "") == "scroll":
                if scroll_count >= self.sampling_settings.scroll_cap:
                    continue
                scroll_count += 1
            kept.append(transition)

        logger.info(
            "Downsampled transitions: %d -> %d (every-2-hops, scroll_cap=%d, "
            "scroll_kept=%d)",
            len(transitions),
            len(kept),
            self.sampling_settings.scroll_cap,
            scroll_count,
        )
        return kept

    def _create_triplet(self, transition: dict[str, Any]) -> GraphTriplet:
        """Create a graph triplet while preserving source field semantics."""
        from_state_id = transition["from_state_id"]
        to_state_id = transition["to_state_id"]
        props = transition.get("transition_props", {})
        triplet_id = f"{from_state_id}_{to_state_id}_{props.get('event_type', 'unknown')}"
        return GraphTriplet(
            id=triplet_id,
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            action_type=props.get("event_type", ""),
            element_info=props.get("element"),
            filled_value=props.get("filled_value"),
            from_url=transition.get("from_url"),
            to_url=transition.get("to_url"),
            from_screenshot_path=transition.get("from_screenshot_path"),
            to_screenshot_path=transition.get("to_screenshot_path"),
            from_depth=transition.get("from_depth", 0),
            to_depth=transition.get("to_depth", 0),
            from_out_degree=self._node_out_degrees.get(from_state_id, 0),
            to_out_degree=self._node_out_degrees.get(to_state_id, 0),
        )

    def _update_stats(self, triplets: list[GraphTriplet]) -> None:
        from_states: set[str] = set()
        to_states: set[str] = set()
        max_depth = 0
        for triplet in triplets:
            from_states.add(triplet.from_state_id)
            to_states.add(triplet.to_state_id)
            max_depth = max(max_depth, triplet.from_depth)
        self.stats.sampled_count = len(triplets)
        self.stats.unique_from_states = len(from_states)
        self.stats.unique_to_states = len(to_states)
        self.stats.max_depth = max_depth

    def sample_all(self) -> list[GraphTriplet]:
        """Sample all valid transitions."""
        return self.sample(max_count=None)

    def sample(self, max_count: int | None = None) -> list[GraphTriplet]:
        """Sample transitions, limiting the final sorted sequence only when positive."""
        self._fetch_node_metadata()
        transitions = self._fetch_all_transitions()
        transitions = self._filter_transitions(transitions)
        transitions = self._sort_transitions(transitions)
        if self.sampling_settings.sample_transitions:
            transitions = self._downsample_transitions(transitions)
        if max_count is not None and max_count > 0:
            transitions = transitions[:max_count]
        triplets = [self._create_triplet(transition) for transition in transitions]
        self._update_stats(triplets)
        logger.info(
            "Sampled %d triplets (max_depth=%d, unique_from=%d, unique_to=%d)",
            len(triplets),
            self.stats.max_depth,
            self.stats.unique_from_states,
            self.stats.unique_to_states,
        )
        return triplets

    def iter_triplets(self, max_count: int | None = None) -> Generator[GraphTriplet, None, None]:
        """Yield sampled triplets."""
        yield from self.sample(max_count)

    def get_triplet_by_id(
        self, from_state_id: str, to_state_id: str
    ) -> GraphTriplet | None:
        """Return the first transition between two state IDs, if present."""
        self._ensure_connected()
        namespace = self.neo4j_settings.namespace
        with create_session(self.driver, self.neo4j_settings) as session:
            result = session.run(
                f"""
                MATCH (from:{namespace} {{state_id: $from_id}})-[t:TRANSITION]->(to:{namespace} {{state_id: $to_id}})
                RETURN
                    from.url AS from_url,
                    from.depth AS from_depth,
                    from.screenshot_path AS from_screenshot_path,
                    to.url AS to_url,
                    to.depth AS to_depth,
                    to.screenshot_path AS to_screenshot_path,
                    properties(t) AS transition_props
                LIMIT 1
                """,
                from_id=from_state_id,
                to_id=to_state_id,
            )
            record = result.single()
            if not record:
                return None
            degree_result = session.run(
                f"""
                MATCH (s:{namespace} {{state_id: $state_id}})-[t:TRANSITION]->()
                RETURN count(t) AS out_degree
                """,
                state_id=from_state_id,
            )
            degree_record = degree_result.single()
            out_degree = degree_record["out_degree"] if degree_record else 0
            props = dict(record["transition_props"]) if record["transition_props"] else {}
            triplet_id = f"{from_state_id}_{to_state_id}_{props.get('event_type', 'unknown')}"
            return GraphTriplet(
                id=triplet_id,
                from_state_id=from_state_id,
                to_state_id=to_state_id,
                action_type=props.get("event_type", ""),
                element_info=props.get("element"),
                filled_value=props.get("filled_value"),
                from_url=record["from_url"],
                to_url=record["to_url"],
                from_screenshot_path=record["from_screenshot_path"],
                to_screenshot_path=record["to_screenshot_path"],
                screenshot_op_path=props.get("screenshot_op_path"),
                from_depth=record["from_depth"] or 0,
                to_depth=record["to_depth"] or 0,
                from_out_degree=out_degree,
                to_out_degree=0,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Return current sampling statistics."""
        return self.stats.to_dict()

    def count_transitions(self) -> int:
        """Count transitions before filtering."""
        self._ensure_connected()
        namespace = self.neo4j_settings.namespace
        with create_session(self.driver, self.neo4j_settings) as session:
            result = session.run(
                f"""
                MATCH (:{namespace})-[t:TRANSITION]->(:{namespace})
                RETURN count(t) AS count
                """
            )
            return result.single()["count"]
