"""
Data models for the website crawler

Encapsulates state, transition, element, and list structure data 
to improve code readability and maintainability.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class BoundingBox:
    """Represents the bounding box of an element"""
    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BoundingBox':
        """Create BoundingBox from dictionary"""
        return cls(
            left=data.get('left', 0.0),
            top=data.get('top', 0.0),
            right=data.get('right', 0.0),
            bottom=data.get('bottom', 0.0),
            width=data.get('width', 0.0),
            height=data.get('height', 0.0)
        )
    
    def to_dict(self) -> dict:
        """Convert BoundingBox to dictionary"""
        return {
            'left': self.left,
            'top': self.top,
            'right': self.right,
            'bottom': self.bottom,
            'width': self.width,
            'height': self.height
        }


@dataclass
class Element:
    """Represents a triggerable element on a web page"""
    page_id: int
    tag: str
    text: str
    name: str
    aria_label: str
    xpath: str
    bounding_box: Optional[BoundingBox] = None
    accessibility_node_info: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Element':
        """Create Element from dictionary"""
        bbox_data = data.get('boundingBox')
        bounding_box = BoundingBox.from_dict(bbox_data) if bbox_data else None
        
        return cls(
            page_id=data.get('page_id', 0),
            tag=data.get('tag', ''),
            text=data.get('text', ''),
            name=data.get('name', ''),
            aria_label=data.get('ariaLabel', ''),
            xpath=data.get('xpath', ''),
            bounding_box=bounding_box,
            accessibility_node_info=data.get('accessibility_node_info')
        )
    
    def to_dict(self) -> dict:
        """Convert Element to dictionary"""
        result = {
            'page_id': self.page_id,
            'tag': self.tag,
            'text': self.text,
            'name': self.name,
            'ariaLabel': self.aria_label,
            'xpath': self.xpath
        }
        if self.bounding_box:
            result['boundingBox'] = self.bounding_box.to_dict()
        if self.accessibility_node_info:
            result['accessibility_node_info'] = self.accessibility_node_info
        return result
    
    def get_signature(self) -> str:
        """Get unique signature for this element"""
        return f"{self.tag}|{self.name}|{self.aria_label}|{self.text}"


@dataclass
class ListItem:
    """Represents an item in a list structure"""
    page_id: int
    xpath: str
    tag: str
    text: str
    aria_label: str
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ListItem':
        """Create ListItem from dictionary"""
        return cls(
            page_id=data.get('page_id', 0),
            xpath=data.get('xpath', ''),
            tag=data.get('tag', ''),
            text=data.get('text', ''),
            aria_label=data.get('ariaLabel', '')
        )


@dataclass
class ListStructure:
    """Represents a list structure on a web page"""
    container_xpath: str
    container_tag: str
    items: List[ListItem]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ListStructure':
        """Create ListStructure from dictionary"""
        items = [ListItem.from_dict(item) for item in data.get('items', [])]
        return cls(
            container_xpath=data.get('containerXPath', ''),
            container_tag=data.get('containerTag', ''),
            items=items
        )
    
    def get_signature(self) -> str:
        """Get unique signature for this list structure"""
        text_signatures = "".join(f"|{item.text}" for item in self.items)
        return self.container_tag + text_signatures
    
    def get_item_page_ids(self) -> List[int]:
        """Get all page IDs of items in this list"""
        return [item.page_id for item in self.items]


@dataclass
class FormStructure:
    """Represents a form structure on a web page"""
    container_xpath: str
    input_fields: List[Element]
    submit_button: Optional[Element]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FormStructure':
        """Create FormStructure from dictionary"""
        input_fields = [Element.from_dict(item) for item in data.get('inputFields', [])]
        submit_button = None
        if data.get('submitButton'):
            submit_button = Element.from_dict(data['submitButton'])
        return cls(
            container_xpath=data.get('containerXPath', ''),
            input_fields=input_fields,
            submit_button=submit_button
        )
    
    def get_all_element_xpaths(self) -> List[str]:
        """Get XPaths of all elements in this form (inputs + submit button)"""
        xpaths = [field.xpath for field in self.input_fields]
        if self.submit_button:
            xpaths.append(self.submit_button.xpath)
        return xpaths
    
    def get_signature(self) -> str:
        """Get unique signature for this form structure"""
        field_signatures = "|".join(f"{f.name or f.aria_label or f.text}" for f in self.input_fields)
        return f"form:{self.container_xpath}|{field_signatures}"


@dataclass
class Transition:
    """Represents a state transition triggered by an element action"""
    event_type: str
    element_tag: str
    element_text: str
    xpath: str
    screenshot_operation: Optional[bytes] = None
    element_screenshot: Optional[bytes] = None
    filled_value: Optional[str] = None
    new_page_opened: bool = False
    accessibility_node_info: Optional[Dict] = None
    # Composite operation support for form submissions and similar actions.
    is_composite: bool = False
    composite_type: Optional[str] = None  # 'form_submission', etc.
    sub_actions: Optional[List[Dict]] = None  # Ordered child actions.
    
    def get_element_html(self) -> str:
        """Get HTML representation of the element"""
        return f"<{self.element_tag}>{self.element_text}</{self.element_tag}>"
    
    def to_dict(self) -> dict:
        """Convert Transition to dictionary for state graph"""
        result = {
            'event_type': self.event_type,
            'element': self.get_element_html(),
            'xpath': self.xpath,
            'screenshot_op': self.screenshot_operation,
            'element_screenshot': self.element_screenshot,
            'filled_value': self.filled_value,
            'new_page_opened': self.new_page_opened,
            'is_composite': self.is_composite,
            'composite_type': self.composite_type,
            'sub_actions': self.sub_actions,
        }
        if self.accessibility_node_info:
            result['accessibility_node_info'] = self.accessibility_node_info
        return result


@dataclass
class State:
    """Represents a web page state"""
    state_id: str
    url: str
    dom_hash: str
    screenshot_path: str
    labelled_screenshot_path: str
    accessibility_tree_path: str
    depth: int
    visit_count: int = 1
    
    @classmethod
    def from_dict(cls, data: dict) -> 'State':
        """Create State from dictionary"""
        return cls(
            state_id=data.get('state_id', ''),
            url=data.get('url', ''),
            dom_hash=data.get('dom_hash', ''),
            screenshot_path=data.get('screenshot_path', ''),
            labelled_screenshot_path=data.get('labelled_screenshot_path', ''),
            accessibility_tree_path=data.get('accessibility_tree_path', ''),
            depth=data.get('depth', 0),
            visit_count=data.get('visit_count', 1)
        )
    
    def to_dict(self) -> dict:
        """Convert State to dictionary"""
        return {
            'state_id': self.state_id,
            'url': self.url,
            'dom_hash': self.dom_hash,
            'screenshot_path': self.screenshot_path,
            'labelled_screenshot_path': self.labelled_screenshot_path,
            'accessibility_tree_path': self.accessibility_tree_path,
            'depth': self.depth,
            'visit_count': self.visit_count
        }



@dataclass
class TriggerableElementsResult:
    """Result of getting triggerable elements from a page"""
    elements: List[Element]
    screenshot: bytes
    labelled_screenshot: bytes
    list_structures: List[ListStructure]
    
    @classmethod
    def from_browser_result(
        cls, 
        elements_data: list,
        screenshot: bytes,
        labelled_screenshot: bytes, 
        list_structures_data: list
    ) -> 'TriggerableElementsResult':
        """Create result from browser session output"""
        elements = [Element.from_dict(elem) for elem in elements_data]
        list_structures = [ListStructure.from_dict(ls) for ls in list_structures_data]
        return cls(
            elements=elements,
            screenshot=screenshot,
            labelled_screenshot=labelled_screenshot,
            list_structures=list_structures
        )


@dataclass
class TriggerResult:
    """Result of triggering an element"""
    event_type: Optional[str]
    screenshot_operation: Optional[bytes]
    filled_value: Optional[str]
    new_page_opened: bool
