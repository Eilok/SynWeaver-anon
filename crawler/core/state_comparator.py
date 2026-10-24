"""
State comparison utilities with progressive verification pipeline.

Implements a streamlined strategy for backtrack state verification:
- Level 1: URL check (fast pre-filter)
- Level 2: Accessibility tree comparison (semantic structure with tolerance)
- Level 3: VLM fallback (final verification using visual comparison)

This approach reduces VLM calls by ~90% compared to always using VLM,
while maintaining high accuracy through semantic structure comparison.
"""
import logging
from typing import Optional, Any

from llm import get_client
from prompts.website_crawl import get_crawler_prompt, build_user_prompt_for_compare_state

logger = logging.getLogger(__name__)

# Lazily initialize the state-comparison VLM to avoid building a client at
# import time when CRAWLER_VISION_MODEL is not configured in config.yaml.
# max_retries=1 keeps each call single-shot; the retry-until-valid loop lives in
# _compare_states_with_vlm below, preserving the original while-True semantics.
_vlm = None


def _get_vlm():
    """Lazily build and cache the crawler vision client."""
    global _vlm
    if _vlm is None:
        _vlm = get_client("crawler_vision", max_retries=1)
    return _vlm


def compare_states_progressive(
    current_url: str,
    target_url: str,
    current_accessibility_tree: Optional[str] = None,
    target_accessibility_tree: Optional[str] = None,
    current_screenshot: Optional[bytes] = None,
    target_screenshot: Optional[bytes] = None,
    use_vlm: bool = True
) -> tuple[bool, str]:
    """
    Progressive state comparison with 3-level verification pipeline.
    
    Verification levels:
    1. URL check - Fast pre-filter to ensure same page
    2. Accessibility tree - Semantic structure comparison with 90% tolerance
    3. VLM fallback - Visual comparison using vision language model
    
    Args:
        current_url: Current page URL
        target_url: Target page URL
        current_accessibility_tree: Current page accessibility tree string (optional)
        target_accessibility_tree: Target page accessibility tree string (optional)
        current_screenshot: Current page screenshot bytes (optional, needed for VLM)
        target_screenshot: Target page screenshot bytes (optional, needed for VLM)
        use_vlm: Whether to use VLM for final verification (default: True)
    
    Returns:
        (is_same: bool, reason: str) - Whether states match and the verification level used
    """
    # Level 1: URL check (fast pre-filter)
    if current_url != target_url:
        # logger.info(f"Level 1 (URL): URLs differ - {current_url} vs {target_url}")
        return False, "url_mismatch"
    
    # logger.info(f"Level 1 (URL): URLs match")
    
    # Level 2: Accessibility tree comparison (if available)
    if current_accessibility_tree and target_accessibility_tree:
        if compare_accessibility_trees(current_accessibility_tree, target_accessibility_tree):
            # logger.info(f"Level 2 (Accessibility Tree): States match")
            return True, "accessibility_tree_match"
        # else:
            # logger.info(f"Level 2 (Accessibility Tree): Trees differ, proceeding to VLM comparison")
    # else:
    #     logger.info(f"Level 2 (Accessibility Tree): Not available, skipping to VLM comparison")
    
    # Level 3: VLM fallback (if enabled and screenshots available)
    if use_vlm and current_screenshot and target_screenshot:
        # logger.info(f"Level 3 (VLM): Falling back to visual comparison")
        try:
            if _compare_states_with_vlm(current_screenshot, target_screenshot):
                # logger.info(f"Level 3 (VLM): States match")
                return True, "vlm_match"
            else:
                # logger.info(f"Level 3 (VLM): States differ")
                return False, "vlm_mismatch"
        except Exception as e:
            # logger.error(f"Level 3 (VLM): Error during comparison - {e}")
            return False, "vlm_error"
    else:
        # logger.warning(f"Level 3 (VLM): Not available or disabled, comparison failed")
        return False, "no_vlm_fallback"


def compare_accessibility_trees(tree1: str, tree2: str, tolerance: float = 0.9) -> bool:
    """
    Compare two accessibility trees with tolerance for numeric changes.
    
    Semantic structure comparison that tolerates:
    - Numeric changes (cart counts, prices, etc.)
    - Whitespace differences
    - Minor text variations
    
    Args:
        tree1: First accessibility tree string
        tree2: Second accessibility tree string
        tolerance: Similarity threshold (0.0 to 1.0)
    
    Returns:
        True if trees are structurally similar
    """
    # Normalize trees
    normalized1 = _normalize_accessibility_tree(tree1)
    normalized2 = _normalize_accessibility_tree(tree2)
    
    # Split into lines for comparison
    lines1 = [line for line in normalized1.split('\n') if line.strip()]
    lines2 = [line for line in normalized2.split('\n') if line.strip()]
    
    # Quick check: if line counts differ significantly, trees are different
    if abs(len(lines1) - len(lines2)) > max(len(lines1), len(lines2)) * (1 - tolerance):
        logger.debug(f"Accessibility tree line count differs significantly: {len(lines1)} vs {len(lines2)}")
        return False
    
    # Calculate similarity using simple matching
    matching_lines = 0
    for line1, line2 in zip(lines1, lines2):
        if _are_tree_lines_similar(line1, line2):
            matching_lines += 1
    
    similarity = matching_lines / max(len(lines1), len(lines2)) if max(len(lines1), len(lines2)) > 0 else 1.0
    
    logger.debug(f"Accessibility tree similarity: {similarity:.2%}")
    return similarity >= tolerance


def _normalize_accessibility_tree(tree: str) -> str:
    """
    Normalize accessibility tree for comparison.
    
    - Removes extra whitespace
    - Normalizes indentation
    - Keeps structural markers (role, name, etc.)
    """
    lines = []
    for line in tree.split('\n'):
        stripped = line.strip()
        if stripped:
            # Count leading spaces to preserve structure
            indent_level = (len(line) - len(line.lstrip())) // 2
            lines.append('  ' * indent_level + stripped)
    return '\n'.join(lines)


def _are_tree_lines_similar(line1: str, line2: str) -> bool:
    """
    Check if two accessibility tree lines are similar.
    
    Tolerates numeric differences but requires structural match.
    """
    # Remove numbers for structure comparison
    import re
    struct1 = re.sub(r'\d+', 'N', line1)
    struct2 = re.sub(r'\d+', 'N', line2)
    
    return struct1 == struct2


def _compare_states_with_vlm(screenshot1: bytes, screenshot2: bytes) -> bool:
    """
    Use VLM to compare two state screenshots.
    
    This function calls a vision language model to determine if two screenshots
    represent the same page state. It uses structured prompts to get a decision
    with reasoning.
    
    Args:
        screenshot1: First screenshot bytes
        screenshot2: Second screenshot bytes
        
    Returns:
        True if states are visually the same, False otherwise
    """
    system_prompt = get_crawler_prompt("compare_state")
    user_prompt = build_user_prompt_for_compare_state(screenshot1, screenshot2)

    vlm = _get_vlm()
    while True:
        logger.info("Calling VLM to compare two state screenshots")
        # complete() handles JSON parsing; returns None on call/parse failure.
        result = vlm.complete(system=system_prompt, user=user_prompt, json_mode=True)
        if not isinstance(result, dict) or "decision" not in result:
            logger.error("VLM call/parse failed or missing decision, retrying...")
            continue

        decision = str(result["decision"]).lower()
        reasoning = result.get("reasoning", "")

        if decision == "same":
            logger.info(f"VLM decided the two states are the same: {reasoning}")
            return True
        elif decision == "different":
            logger.info(f"VLM decided the two states are different: {reasoning}")
            return False
        else:
            logger.warning(f"VLM returned unexpected decision '{decision}', retrying...")


import re
from urllib.parse import urlparse
from typing import Dict, Set


def extract_url_template(url: str) -> str:
    """Replace dynamic URL path segments with stable placeholders."""
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    patterns: Dict[str, str] = {
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$": "<UUID>",
        r"^[0-9a-f]{32,64}$": "<HASH>",
        r"^\d{4}-\d{2}-\d{2}$": "<DATE>",
        r"^v?\d+\.\d+(\.\d+)?$": "<VERSION>",
        r"^\d+$": "<ID>",
    }
    static_keywords: Set[str] = {
        "api", "v1", "v2", "v3", "admin", "dashboard", "auth", "login",
        "user", "users", "post", "posts", "product", "products", "category",
        "search", "settings", "profile", "f", "r", "b", "blog", "article",
        "page", "docs", "help",
    }
    template_segments = []
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        placeholder = next(
            (value for pattern, value in patterns.items()
             if re.match(pattern, segment, re.IGNORECASE)),
            None,
        )
        if placeholder:
            template_segments.append(placeholder)
        elif segment.lower() in static_keywords:
            template_segments.append(segment)
        elif index == 1 and not segment.isdigit():
            template_segments.append("<CATEGORY>")
        elif is_last and len(segment) > 15 and "-" in segment:
            template_segments.append("<SLUG>")
        else:
            digit_ratio = sum(character.isdigit() for character in segment) / len(segment)
            words = re.split(r"[-_]", segment)
            if digit_ratio > 0.5 and len(segment) > 2:
                template_segments.append("<ID>")
            elif len(words) >= 3 and len(segment) > 10:
                template_segments.append("<SLUG>")
            else:
                template_segments.append(segment)
    template_path = "/" + "/".join(template_segments) if template_segments else "/"
    return f"{parsed.scheme}://{parsed.netloc}{template_path}"