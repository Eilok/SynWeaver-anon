"""
Neo4j-based state graph implementation with thread safety
"""
import os
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from urllib.parse import urlparse
from datetime import datetime
import shutil
import json
import logging

from neo4j import Driver
from neo4j.exceptions import ServiceUnavailable

from config.shared import Neo4jSettings
from crawler.core.artifacts import CrawlerPaths
from storage.neo4j import create_driver, create_session
logger = logging.getLogger(__name__)


class Neo4jStateGraph:
    """
    Thread-safe state graph using Neo4j as backend storage

    Features:
    - Automatic persistence (no data loss on crash)
    - Thread-safe operations
    - Real-time visualization via Neo4j Browser
    - Efficient duplicate detection
    """

    def __init__(self, config: Neo4jSettings, paths: CrawlerPaths):
        """
        Initialize Neo4j connection and create constraints

        Args:
            config: Neo4j configuration object (connection info)
            paths: Crawler output paths (screenshot dirs, etc.)
        """
        self.config = config
        self.paths = paths

        self.driver: Optional[Driver] = None
        self._connect()
        
        self._setup_schema()
        
        self.lock = threading.RLock()
        
        self.state_counter = self._get_max_state_id() + 1
        
        self.root_domain: Optional[str] = None
        
        logger.info("Neo4jStateGraph initialized successfully")
    
    def _connect(self):
        """Crawler graph operation."""
        try:
            self.driver = create_driver(self.config)
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.url}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.error("Please make sure Neo4j is running")
            raise

    def _setup_schema(self):
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            
            try:
                session.run(f"""
                    CREATE CONSTRAINT {namespace}_state_id_unique IF NOT EXISTS
                    FOR (s:{namespace}) REQUIRE s.state_id IS UNIQUE
                """)
                logger.info(f"Created constraint: {namespace}_state_id_unique")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")
            
            try:
                session.run(f"""
                    CREATE INDEX {namespace}_state_dom_hash IF NOT EXISTS
                    FOR (s:{namespace}) ON (s.dom_hash)
                """)
                session.run(f"""
                    CREATE INDEX {namespace}_state_url IF NOT EXISTS
                    FOR (s:{namespace}) ON (s.url)
                """)
                logger.info(f"Created indexes: {namespace}_state_dom_hash, {namespace}_state_url")
            except Exception as e:
                logger.warning(f"Indexes may already exist: {e}")
    
    def _get_max_state_id(self) -> int:
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"""
                MATCH (s:{namespace})
                RETURN s.state_id AS state_id
                ORDER BY s.state_id DESC
                LIMIT 1
            """)
            record = result.single()
            if record and record["state_id"]:
                state_id_str = record["state_id"]
                return int(state_id_str[1:])
            return -1
    
    def close(self) -> None:
        """Close the Neo4j driver once."""
        if self.driver is None:
            return
        driver = self.driver
        self.driver = None
        driver.close()
        logger.info("Neo4j connection closed")
    
    def set_root_domain(self, url: str):
        """Crawler graph operation."""
        parsed = urlparse(url)
        self.root_domain = parsed.netloc
    
    def is_same_domain(self, url: str) -> bool:
        """Crawler graph operation."""
        if self.root_domain is None:
            return True
        parsed = urlparse(url)
        return parsed.netloc == self.root_domain
    
    def state_count(self) -> int:
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"MATCH (s:{namespace}) RETURN count(s) AS count")
            return result.single()["count"]
    
    def _save_screenshot(self, screenshot: bytes, filename: str, subdir: str) -> str:
        """
        
        Args:
        
        Returns:
        """
        if screenshot is None:
            return None
        
        os.makedirs(subdir, exist_ok=True)

        file_path = Path(subdir) / filename
        base_name, ext = filename.split('.', 1)
        counter = 1
        while file_path.exists():
            # ensure the file path is unique
            file_path = Path(subdir) / f"{base_name}_{counter}.{ext}"
            counter += 1
        with open(file_path, 'wb') as f:
            f.write(screenshot)
        
        return str(file_path)
    
    def _save_accessibility_tree(self, accessibility_tree_str: str, filename: str) -> str:
        """
        
        Args:
        
        Returns:
        """
        if accessibility_tree_str is None:
            return None
        
        os.makedirs(self.paths.accessibility_tree_dir, exist_ok=True)
        
        file_path = Path(self.paths.accessibility_tree_dir) / filename
        base_name, ext = filename.rsplit('.', 1)
        counter = 1
        while file_path.exists():
            # ensure the file path is unique
            file_path = Path(self.paths.accessibility_tree_dir) / f"{base_name}_{counter}.{ext}"
            counter += 1
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(accessibility_tree_str)
        
        return str(file_path)
    
    def get_state(self, state_id: str) -> Optional[Dict]:
        """
        
        Returns:
        """
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"""
                MATCH (s:{namespace} {{state_id: $state_id}})
                RETURN s
            """, state_id=state_id)
            
            record = result.single()
            if record:
                state_node = record["s"]
                return dict(state_node)
            return None
    
    def add_state(
        self,
        url: str,
        screenshot: bytes,
        labelled_screenshot: bytes,
        depth: int,
        dom_hash: str,
        accessibility_tree_str: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        
        Args:
        
        Returns:
            (state_id, is_duplicate)
        """
        with self.lock:
            existing_state = self._find_state_by_hash(dom_hash)
            if existing_state:
                state_id = existing_state['state_id']
                self._increment_visit_count(state_id)
                return state_id, True
            
            state_id = f"S{self.state_counter}"
            self.state_counter += 1
            
            screenshot_path = self._save_screenshot(
                screenshot,
                f"{state_id}.png",
                self.paths.screenshot_dir
            )
            labelled_screenshot_path = self._save_screenshot(
                labelled_screenshot,
                f"{state_id}_labelled.png",
                self.paths.labelled_screenshot_dir
            )
            
            accessibility_tree_path = None
            if accessibility_tree_str:
                accessibility_tree_path = self._save_accessibility_tree(
                    accessibility_tree_str,
                    f"{state_id}_accessibility_tree.txt"
                )
            
            namespace = self.config.namespace
            with create_session(self.driver, self.config) as session:
                session.run(f"""
                    CREATE (s:{namespace} {{
                        state_id: $state_id,
                        dom_hash: $dom_hash,
                        url: $url,
                        screenshot_path: $screenshot_path,
                        labelled_screenshot_path: $labelled_screenshot_path,
                        accessibility_tree_path: $accessibility_tree_path,
                        depth: $depth,
                        visit_count: 1,
                        created_at: datetime()
                    }})
                """, 
                    state_id=state_id,
                    dom_hash=dom_hash,
                    url=url,
                    screenshot_path=screenshot_path,
                    labelled_screenshot_path=labelled_screenshot_path,
                    accessibility_tree_path=accessibility_tree_path,
                    depth=depth
                )
            
            return state_id, False
    
    def _find_state_by_hash(self, dom_hash: str) -> Optional[Dict]:
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"""
                MATCH (s:{namespace} {{dom_hash: $dom_hash}})
                RETURN s
                LIMIT 1
            """, dom_hash=dom_hash)
            
            record = result.single()
            if record:
                return dict(record["s"])
            return None
    
    def _increment_visit_count(self, state_id: str):
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            session.run(f"""
                MATCH (s:{namespace} {{state_id: $state_id}})
                SET s.visit_count = s.visit_count + 1
            """, state_id=state_id)
    
    def add_transition(
        self,
        from_state_id: str,
        to_state_id: str,
        event_type: str,
        element: str,
        xpath: str,
        screenshot_op: Optional[bytes],
        element_screenshot: Optional[bytes],
        filled_value: Optional[str],
        new_page_opened: bool,
        accessibility_node_info: Optional[Dict] = None,
        is_composite: bool = False,
        composite_type: Optional[str] = None,
        sub_actions: Optional[List[Dict]] = None
    ):
        """
        
        Args:
        """
        with self.lock:
            screenshot_op_path = None
            if screenshot_op:
                filename = f"{from_state_id}_to_{to_state_id}.png"
                screenshot_op_path = self._save_screenshot(
                    screenshot_op,
                    filename,
                    self.paths.operation_screenshot_dir
                )

            element_screenshot_path = None
            if element_screenshot:
                filename = f"element_{from_state_id}_to_{to_state_id}.png"
                element_screenshot_path = self._save_screenshot(
                    element_screenshot,
                    filename,
                    self.paths.element_screenshot_dir
                )
            
            namespace = self.config.namespace
            
            # Convert accessibility_node_info to JSON string for storage
            accessibility_json = None
            if accessibility_node_info:
                accessibility_json = json.dumps(accessibility_node_info)
            
            # Convert sub_actions to JSON string for storage
            sub_actions_json = None
            if sub_actions:
                sub_actions_json = json.dumps(sub_actions)
            
            with create_session(self.driver, self.config) as session:
                session.run(f"""
                    MATCH (from:{namespace} {{state_id: $from_state_id}})
                    MATCH (to:{namespace} {{state_id: $to_state_id}})
                    CREATE (from)-[t:TRANSITION {{
                        event_type: $event_type,
                        element: $element,
                        xpath: $xpath,
                        screenshot_op_path: $screenshot_op_path,
                        element_screenshot_path: $element_screenshot_path,
                        filled_value: $filled_value,
                        new_page_opened: $new_page_opened,
                        accessibility_node_info: $accessibility_node_info,
                        is_composite: $is_composite,
                        composite_type: $composite_type,
                        sub_actions: $sub_actions,
                        created_at: datetime()
                    }}]->(to)
                """,
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    event_type=event_type,
                    element=element,
                    xpath=xpath,
                    screenshot_op_path=screenshot_op_path,
                    element_screenshot_path=element_screenshot_path,
                    filled_value=filled_value,
                    new_page_opened=new_page_opened,
                    accessibility_node_info=accessibility_json,
                    is_composite=is_composite,
                    composite_type=composite_type,
                    sub_actions=sub_actions_json
                )
            
            
    
    def get_path_to_state(self, state_id: str) -> List[Dict]:
        """
        
        Returns:
        """
        if state_id == 'S0':
            return []
        
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"""
                MATCH path = shortestPath(
                    (root:{namespace} {{state_id: 'S0'}})-[:TRANSITION*..20]->(target:{namespace} {{state_id: $state_id}})
                )
                UNWIND relationships(path) AS rel
                RETURN 
                    startNode(rel).state_id AS from_state,
                    endNode(rel).state_id AS to_state,
                    properties(rel) AS transition
            """, state_id=state_id)
            
            path = []
            for record in result:
                path.append({
                    'from_state': record['from_state'],
                    'to_state': record['to_state'],
                    'transition': dict(record['transition'])
                })
            
            return path
    
    def get_statistics(self) -> Dict:
        """Crawler graph operation."""
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            result = session.run(f"""
                MATCH (s:{namespace})
                OPTIONAL MATCH (s)-[t:TRANSITION]->()
                RETURN 
                    count(DISTINCT s) AS state_count,
                    count(DISTINCT t) AS transition_count,
                    max(s.depth) AS max_depth,
                    sum(s.visit_count) AS total_visits
            """)
            
            record = result.single()
            return {
                'namespace': namespace,
                'state_count': record['state_count'],
                'transition_count': record['transition_count'],
                'max_depth': record['max_depth'],
                'total_visits': record['total_visits']
            }

    # ------------------- Transition Extraction ------------------ #

    def extract_transitions(self, output_dir: str | None = None) -> List[Dict]:
        """
        
        
        transitions/
        ├── T000_S0_to_S1_click/
        │   ├── from_state.png
        │   ├── from_state_labelled.png
        │   ├── to_state.png
        │   ├── to_state_labelled.png
        │   ├── operation.png
        │   ├── element.png
        │   └── metadata.json
        ├── T001_S1_to_S2_fill/
        │   └── ...
        └── transitions_summary.json
        
        Args:
        
        Returns:
        """
        if output_dir is None:
            output_dir = str(Path(self.paths.root_dir) / "transitions")
        
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting transitions to {output_dir}")
        
        transitions = []
        transition_id = 0
        
        with create_session(self.driver, self.config) as session:
            namespace = self.config.namespace
            
            result = session.run(f"""
                MATCH (from:{namespace})-[t:TRANSITION]->(to:{namespace})
                RETURN 
                    from.state_id AS from_state_id,
                    to.state_id AS to_state_id,
                    from.screenshot_path AS from_screenshot,
                    from.labelled_screenshot_path AS from_labelled_screenshot,
                    to.screenshot_path AS to_screenshot,
                    to.labelled_screenshot_path AS to_labelled_screenshot,
                    from.url AS from_url,
                    to.url AS to_url,
                    properties(t) AS transition_props
                ORDER BY from.state_id, to.state_id
            """)
            
            for record in result:
                from_state_id = record['from_state_id']
                to_state_id = record['to_state_id']
                transition_props = dict(record['transition_props'])
                event_type = transition_props.get('event_type', 'unknown')
                
                transition_folder_name = f"T{transition_id:03d}_{from_state_id}_to_{to_state_id}"
                transition_dir = output_path / transition_folder_name
                transition_dir.mkdir(exist_ok=True)
                
                self._copy_screenshot_if_exists(
                    record['from_screenshot'],
                    transition_dir / "from_state.png"
                )
                
                self._copy_screenshot_if_exists(
                   record['to_screenshot'],
                    transition_dir / "to_state.png"
                )
                
                self._copy_screenshot_if_exists(
                    transition_props.get('screenshot_op_path'),
                    transition_dir / "operation.png"
                )
                
                self._copy_screenshot_if_exists(
                    transition_props.get('element_screenshot_path'),
                    transition_dir / "element.png"
                )
                
                metadata = {
                    'transition_id': f"T{transition_id:03d}",
                    'from_state_id': from_state_id,
                    'to_state_id': to_state_id,
                    'from_url': record['from_url'],
                    'to_url': record['to_url'],
                    'event_type': event_type,
                    'element': transition_props.get('element', ''),
                    'xpath': transition_props.get('xpath', ''),
                    'filled_value': transition_props.get('filled_value'),
                    'new_page_opened': transition_props.get('new_page_opened', False),
                    'created_at': str(transition_props.get('created_at', ''))
                }
                
                metadata_path = transition_dir / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                transitions.append({
                    'transition_id': f"T{transition_id:03d}",
                    'folder': transition_folder_name,
                    'metadata': metadata
                })
                
                transition_id += 1
                
                logger.info(
                    f"Extracted transition T{transition_id-1:03d}: "
                    f"{from_state_id} --[{event_type}]--> {to_state_id}"
                )
        
        self._save_transitions_summary(transitions, output_dir)
        
        logger.info(f"Total {len(transitions)} transitions extracted to {output_dir}")
        
        return transitions
    
    def _copy_screenshot_if_exists(self, src_path: str | None, dst_path: Path) -> bool:
        """
        
        Args:
        
        Returns:
        """
        if src_path is None:
            return False
        
        src = Path(src_path)
        if not src.exists():
            logger.warning(f"Screenshot not found: {src_path}")
            return False
        
        try:
            shutil.copy(src, dst_path)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
            return False
    
    def _save_transitions_summary(self, transitions: List[Dict], output_dir: str):
        """Crawler graph operation."""
        summary = {
            'total_transitions': len(transitions),
            'namespace': self.config.namespace,
            'extracted_at': datetime.now().isoformat(),
            'transitions': [
                {
                    'transition_id': t['transition_id'],
                    'folder': t['folder'],
                    'from_state': t['metadata']['from_state_id'],
                    'to_state': t['metadata']['to_state_id'],
                    'event_type': t['metadata']['event_type'],
                    'element': t['metadata']['element'],
                    'from_url': t['metadata']['from_url'],
                    'to_url': t['metadata']['to_url']
                }
                for t in transitions
            ]
        }
        
        summary_path = Path(output_dir) / "transitions_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transitions summary to {summary_path}")
    
    def __del__(self):
        """Crawler graph operation."""
        self.close()