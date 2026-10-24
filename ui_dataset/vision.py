"""Prompt assembly for UI dataset VLM generation."""

from __future__ import annotations

from typing import List, Optional, Tuple

from llm import get_client
from prompts.ui_dataset import (
    ELEMENT_AND_TRANSITION_SYSTEM_PROMPT,
    build_element_transition_prompt,
    build_page_description_qa_prompt,
    get_ui_dataset_prompt,
)


class VisionGenerator:
    """Generate UI dataset annotations with the shared T1 client."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_attempts: int = 3,
        config_path: str = "./config.yaml",
    ) -> None:
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        self._client = get_client(
            "t1",
            conf_path=config_path,
            overrides=overrides or None,
            max_retries=retry_attempts,
            fresh=True,
        )

    def generate_page_description_and_qa(
        self,
        image_paths: List[str],
        accessibility_tree: str,
        url: str,
    ) -> Optional[Tuple[str, str, str]]:
        """Generate a page description and one question-answer pair."""
        result = self._client.complete(
            system=get_ui_dataset_prompt("page_description_qa"),
            user=build_page_description_qa_prompt(url, accessibility_tree),
            images=image_paths,
            json_mode=True,
        )
        if not isinstance(result, dict):
            return None
        return (
            result.get("page_description", ""),
            result.get("question", ""),
            result.get("answer", ""),
        )

    def generate_element_and_transition(
        self,
        before_image_path: str,
        after_image_path: str,
        action: str,
        element: str,
        from_url: str,
        to_url: str,
    ) -> Optional[Tuple[str, str]]:
        """Generate an element description and observable state transition."""
        result = self._client.complete(
            system=ELEMENT_AND_TRANSITION_SYSTEM_PROMPT,
            user=build_element_transition_prompt(action, element, from_url, to_url),
            images=[before_image_path, after_image_path],
            json_mode=True,
        )
        if not isinstance(result, dict):
            return None
        return result.get("element_description", ""), result.get("state_transition", "")
