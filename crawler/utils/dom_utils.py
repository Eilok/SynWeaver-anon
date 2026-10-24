from lxml import html, etree
from lxml.html import HtmlElement
from typing import Set, List
import hashlib
import re
import logging

# from shared.logger import get_logger

logger = logging.getLogger(__name__)

ignore_tags: Set[str] = {
    'script', 'style', 'noscript', 'meta', 'link', 'svg', 'path'
}

ignore_attributes: Set[str] = {
    'data-timestamp', 'data-session', 'data-random', 'data-analytics',
    'data-tracking', 'data-gtm', 'data-id'
}

key_attributes: Set[str] = {
    'style',
    'hidden',
    
    'disabled',
    'readonly',
    'checked',
    'selected',
    'aria-expanded',
    'aria-hidden',
    'aria-selected',
    'aria-disabled',
    
    'class',
    
    'value',
    
    'href',
    'src',
    
    'type',
    'role',
}

state_class_keywords: Set[str] = {
    'active', 'inactive', 'open', 'closed', 'expanded', 'collapsed',
    'selected', 'disabled', 'hidden', 'visible', 'show', 'hide',
    'current', 'checked', 'unchecked', 'on', 'off'
}


def is_dom_structure_same(dom1: str, dom2: str) -> bool:
    """
    
    
    """
    structure1 = _extract_dom_structure_with_key_attrs(dom1)
    structure2 = _extract_dom_structure_with_key_attrs(dom2)
    return structure1 == structure2


def _extract_dom_structure_with_key_attrs(dom: str) -> str:
    """
    
    div[style:display:block]
      ul[class:active,open]
        li[Text content]
        li[aria-expanded:true]
    """
    try:
        tree = html.fromstring(dom)
        
        for tag in ignore_tags:
            for element in tree.xpath(f'//{tag}'):
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)
        
        structure = _extract_element_with_key_attrs(tree)
        return structure
        
    except Exception as e:
        logger.error(f"Failed to extract DOM structure: {e}")
        return ""


def _extract_element_with_key_attrs(element: HtmlElement, depth: int = 0) -> str:
    """
    
    Args:
    
    Returns:
    """
    if element.tag in ignore_tags:
        return ""
    
    indent = "  " * depth
    
    tag = element.tag
    
    key_attrs = _extract_key_attributes(element)
    attrs_str = f"[{key_attrs}]" if key_attrs else ""
    
    direct_text = (element.text or "").strip()
    text_str = f'"{direct_text[:50]}"' if direct_text else ""
    
    children_strs = []
    for child in element:
        child_str = _extract_element_with_key_attrs(child, depth + 1)
        if child_str:
            children_strs.append(child_str)
    
    if children_strs:
        children_part = "\n" + "\n".join(children_strs)
        return f"{indent}{tag}{attrs_str}{text_str}{children_part}"
    else:
        return f"{indent}{tag}{attrs_str}{text_str}"


def _extract_key_attributes(element: HtmlElement) -> str:
    """
    
    
    """
    key_parts = []
    
    for attr_name in sorted(element.attrib.keys()):
        if attr_name in ignore_attributes:
            continue
        
        if attr_name not in key_attributes:
            continue
        
        attr_value = element.get(attr_name, "")
        
        # ========================================================================
        # ========================================================================
        if attr_name == "style":
            important_styles = _extract_important_styles(attr_value)
            if important_styles:
                key_parts.append(f"style:{important_styles}")
        
        # ========================================================================
        # ========================================================================
        elif attr_name == "class":
            state_classes = _extract_state_classes(attr_value)
            if state_classes:
                key_parts.append(f"class:{state_classes}")
        
        # ========================================================================
        # ========================================================================
        elif attr_name in ["href", "src"]:
            clean_value = re.sub(r'[?#].*$', '', attr_value)
            if clean_value:
                key_parts.append(f"{attr_name}:{clean_value[:50]}")
        
        # ========================================================================
        # ========================================================================
        elif attr_name == "value":
            if attr_value:
                key_parts.append(f"value:{attr_value[:30]}")
        
        # ========================================================================
        # ========================================================================
        elif attr_name in ["disabled", "readonly", "checked", "selected", "hidden"]:
            key_parts.append(attr_name)
        
        # ========================================================================
        # ========================================================================
        elif attr_name.startswith("aria-"):
            if attr_value:
                key_parts.append(f"{attr_name}:{attr_value}")
        
        # ========================================================================
        # ========================================================================
        else:
            if attr_value:
                key_parts.append(f"{attr_name}:{attr_value[:30]}")
    
    return ",".join(key_parts)


def _extract_important_styles(style_str: str) -> str:
    """
    
    - display
    - visibility
    
    """
    if not style_str:
        return ""
    
    important_styles = []
    
    for style_rule in style_str.split(';'):
        style_rule = style_rule.strip()
        if not style_rule:
            continue
        
        if ':' not in style_rule:
            continue
        
        prop, value = style_rule.split(':', 1)
        prop = prop.strip().lower()
        value = value.strip().lower()
        
        if prop == 'display':
            important_styles.append(f"display:{value}")
        elif prop == 'visibility':
            important_styles.append(f"visibility:{value}")
        elif prop == 'opacity':
            if value == '0' or value == '1':
                important_styles.append(f"opacity:{value}")
        elif prop == 'position':
            if value in ['fixed', 'absolute', 'sticky']:
                important_styles.append(f"position:{value}")
    
    return ";".join(important_styles)


def _extract_state_classes(class_str: str) -> str:
    """
    
    - active, inactive
    - open, closed
    - expanded, collapsed
    - selected
    - disabled, hidden, visible
    - show, hide
    - current
    
    """
    if not class_str:
        return ""
    
    state_classes = []
    
    for class_name in class_str.split():
        class_lower = class_name.lower()
        
        if any(keyword in class_lower for keyword in state_class_keywords):
            state_classes.append(class_name)
    
    return ",".join(sorted(state_classes))


def compute_structure_hash(dom: str) -> str:
    """
    
    Args:
    
    Returns:
    """
    structure = _extract_dom_structure_with_key_attrs(dom)
    return hashlib.sha256(structure.encode('utf-8')).hexdigest()[:16]


def get_new_xpaths(old_dom: str, new_dom: str, all_new_xpaths: List[str]) -> List[str]:
    """
    get the list of new xpaths that are present in new_dom but not in old_dom
    """
    try:
        old_tree = html.fromstring(old_dom)
        new_tree = html.fromstring(new_dom)

        # get fingerprints of old elements
        old_elements = set()
        for elem in old_tree.iter():
            if elem.tag in ignore_tags:
                continue

            fingerprint = get_element_fingerprint(elem)
            old_elements.add(fingerprint)

        # find new elements in new_dom
        new_xpaths = []
        for xpath in all_new_xpaths:
            try:
                elements = new_tree.xpath(xpath)
                if not elements:
                    logger.debug(f"No elements found for XPath: {xpath}")
                    continue

                elem = elements[0]
                fingerprint = get_element_fingerprint(elem)
                
                if fingerprint not in old_elements:
                    new_xpaths.append(xpath)
            except Exception as e:
                logger.error(f"Error processing XPath {xpath}: {e}")
                new_xpaths.append(xpath)

        return new_xpaths
    except Exception as e:
        logger.error(f"Error in get_new_xpaths: {e}, returning all xpaths as new.")
        return all_new_xpaths


def get_element_fingerprint(element: HtmlElement) -> str:
    """
    
    """
    tag = element.tag
    text = (element.text or '').strip()[:50]
    
    key_attrs = _extract_key_attributes(element)
    
    return f"{tag}|{key_attrs}|{text}"