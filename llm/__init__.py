"""Unified LLM infrastructure."""

from llm.client import UnifiedLLMClient, get_client
from llm.image_encoding import build_user_content, image_block, to_data_url
from llm.json_parsing import parse_json, parse_json_str
from llm.roles import ModelSpec, resolve_spec

__all__ = [
    "parse_json",
    "parse_json_str",
    "to_data_url",
    "image_block",
    "build_user_content",
    "ModelSpec",
    "resolve_spec",
    "UnifiedLLMClient",
    "get_client",
]
