"""
Unified JSON parsing for LLM output.

Consolidates three previously-duplicated implementations:

- ``website_crawler.utils.json_utils.repair_json_output`` (json_repair based,
  returns a normalized string)
- the former UI generator response parser (three-tier regex fallback,
  returns dict|None)
- ``task_synthesizer.base_executor.robust_json_loads`` (strips ``//`` comments,
  three-tier fallback, returns dict|None)

The :func:`parse_json` strategy below is the union of all three behaviours, so a
single function can replace every call site without losing any prior capability.
"""
from typing import Optional, Union
import json
import logging
import re

logger = logging.getLogger(__name__)

# Matches a fenced ```json ... ``` / ```ts ... ``` / ``` ... ``` block.
_FENCE_RE = re.compile(r"```(?:json|ts)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
# Matches a single-line // comment (used by some models in JSON-ish output).
_LINE_COMMENT_RE = re.compile(r"//.*")
# Matches the first {...} object or [...] array spanning the string.
_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_ARR_RE = re.compile(r"\[[\s\S]*\]")


def _strip_fences(content: str) -> str:
    """Remove a leading/trailing ```json|ts``` code fence if present."""
    content = content.strip()
    # Fast path matching the original repair_json_output prefix/suffix handling.
    if content.startswith("```json"):
        content = content.removeprefix("```json")
    elif content.startswith("```ts"):
        content = content.removeprefix("```ts")
    elif content.startswith("```"):
        content = content.removeprefix("```")
    if content.endswith("```"):
        content = content.removesuffix("```")
    return content.strip()


def parse_json(
    content: str,
    *,
    repair: bool = True,
    strip_line_comments: bool = True,
) -> Optional[Union[dict, list]]:
    """
    Parse JSON from raw LLM output, tolerating common formatting noise.

    Strategy (tried in order, first success wins):
      1. Strip ```json``` / ```ts``` / ``` fences and (optionally) ``//`` comments.
      2. ``json.loads`` directly.
      3. If ``repair`` is True, try ``json_repair.loads`` (fixes trailing commas,
         single quotes, etc.).
      4. Extract a fenced block and parse it.
      5. Extract the first ``{...}`` object or ``[...]`` array and parse it
         (with repair fallback when enabled).

    Args:
        content: Raw response string from the model.
        repair: Whether to attempt json_repair on lightly-malformed JSON.
        strip_line_comments: Whether to strip ``//`` single-line comments first.

    Returns:
        Parsed dict/list, or None if every strategy fails (never raises).
    """
    if not content or not content.strip():
        return None

    cleaned = _strip_fences(content)
    if strip_line_comments:
        cleaned = _LINE_COMMENT_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    # 2. direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. repair the whole cleaned string
    if repair:
        repaired = _try_repair(cleaned)
        if repaired is not None:
            return repaired

    # 4. fenced block extracted from the ORIGINAL content
    fence_match = _FENCE_RE.search(content)
    if fence_match:
        block = fence_match.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            if repair:
                repaired = _try_repair(block)
                if repaired is not None:
                    return repaired

    # 5. first {...} object or [...] array
    for pattern in (_OBJ_RE, _ARR_RE):
        match = pattern.search(cleaned)
        if match:
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                if repair:
                    repaired = _try_repair(snippet)
                    if repaired is not None:
                        return repaired

    logger.warning(f"Failed to parse JSON from response: {content[:200]}...")
    return None


def _try_repair(text: str) -> Optional[Union[dict, list]]:
    """Attempt json_repair.loads, returning a dict/list or None."""
    try:
        import json_repair
    except ImportError:
        return None
    try:
        result = json_repair.loads(text)
    except Exception:
        return None
    # json_repair returns "" for hopeless input; treat empty/scalar as failure.
    if isinstance(result, (dict, list)) and result != {} and result != []:
        return result
    if isinstance(result, (dict, list)):
        # Preserve genuinely-empty containers only when the input looked like one.
        stripped = text.strip()
        if stripped in ("{}", "[]"):
            return result
    return None


def parse_json_str(content: str) -> str:
    """
    Normalize LLM output to a JSON string, preserving the legacy
    ``repair_json_output`` contract (return a string; on failure return the
    cleaned input unchanged so an outer ``json.loads`` raises as before).

    Provided for minimal-diff replacement of call sites that did
    ``json.loads(repair_json_output(resp))``.
    """
    parsed = parse_json(content)
    if parsed is not None:
        return json.dumps(parsed, ensure_ascii=False)
    return _strip_fences(content)
