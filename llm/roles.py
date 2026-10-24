"""
Configuration resolution for the unified LLM client.

Maps a logical role to a concrete :class:`ModelSpec` describing how to call
that model. Resolution layers the three configuration sources used across the
codebase, with later sources overriding earlier ones:

    1. config.yaml ``{ROLE}_MODEL`` (api_key / base_url / model_id).
    2. Environment variables ``{ROLE}_API_KEY`` / ``{ROLE}_API_BASE`` /
       ``{ROLE}_MODEL`` / ``{ROLE}_TEMPERATURE`` / ``{ROLE}_MAX_TOKENS``.
    3. Explicit ``overrides`` passed by the caller.

``temperature`` and ``max_tokens`` default to ``None``, which the client treats
as "do not send this parameter". Each module can set them through overrides or
per-call arguments.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

from storage.io import load_yaml_config


# Single source of truth for supported logical roles.
ROLE_REGISTRY: Dict[str, str] = {
    "crawler_text": "Crawler list filtering and form-value generation (text).",
    "crawler_vision": "Crawler visual state comparison (vision).",
    "t1": "T1 execution agent and UI dataset generation (vision).",
    "t2": "T2 collaborative optimizer and validator reconstruction.",
    "task_gen": "Task generation from before/after visual states.",
}


def conf_key_for_role(role: str) -> str:
    """Return the config.yaml key for a role using the ``{ROLE}_MODEL`` form."""
    return f"{role.upper()}_MODEL"


@dataclass
class ModelSpec:
    """
    Concrete parameters for instantiating and calling one model.

    Attributes:
        model: Model identifier mapped from config.yaml ``model_id``.
        api_key: API key.
        base_url: OpenAI-compatible base URL.
        temperature: Sampling temperature. ``None`` means the parameter is omitted.
        max_tokens: Maximum completion tokens. ``None`` means the parameter is omitted.
        timeout: Request timeout in seconds.
    """

    model: str
    api_key: str
    base_url: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: int = 1800
    token_parameter: str = "max_completion_tokens"
    max_image_payload_bytes: Optional[int] = None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def resolve_spec(
    role: str,
    *,
    conf_path: str = "./config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> ModelSpec:
    """
    Resolve a role name to a :class:`ModelSpec`.

    Args:
        role: Logical role, case-insensitive. Supported roles are
            ``crawler_text``, ``crawler_vision``, ``t1``, ``t2``, and
            ``task_gen``. The role maps to config key ``{ROLE}_MODEL`` and
            environment prefix ``{ROLE}_``.
        conf_path: Path to config.yaml. Defaults to ``./config.yaml``.
        overrides: Explicit values that override config.yaml and environment
            values. Recognized keys are model, api_key, base_url, temperature,
            max_tokens, and timeout.

    Returns:
        A populated ModelSpec. Missing values fall back to an empty string or
        None so the client fails only when the model is actually called.
    """
    normalized_role = role.lower()
    if normalized_role not in ROLE_REGISTRY:
        supported = ", ".join(sorted(ROLE_REGISTRY))
        raise ValueError(f"Unknown LLM role '{role}'. Supported roles: {supported}")

    role_upper = normalized_role.upper()
    overrides = overrides or {}

    # Layer 1: config.yaml {ROLE}_MODEL.
    yaml_conf = load_yaml_config(conf_path)
    entry: Dict[str, Any] = yaml_conf.get(conf_key_for_role(normalized_role), {}) or {}

    model = entry.get("model_id", "")
    api_key = entry.get("api_key", "")
    base_url = entry.get("base_url", "")
    temperature = _coerce_float(entry.get("temperature"))
    max_tokens = _coerce_int(entry.get("max_tokens"))
    timeout = _coerce_int(entry.get("timeout")) or 1800
    token_parameter = entry.get("token_parameter", "max_completion_tokens")
    max_image_payload_bytes = _coerce_int(entry.get("max_image_payload_bytes"))

    # Layer 2: environment variables {ROLE}_*.
    model = os.environ.get(f"{role_upper}_MODEL", model)
    api_key = os.environ.get(f"{role_upper}_API_KEY", api_key)
    base_url = os.environ.get(f"{role_upper}_API_BASE", base_url)
    if f"{role_upper}_TEMPERATURE" in os.environ:
        temperature = _coerce_float(os.environ[f"{role_upper}_TEMPERATURE"])
    if f"{role_upper}_MAX_TOKENS" in os.environ:
        max_tokens = _coerce_int(os.environ[f"{role_upper}_MAX_TOKENS"])
    token_parameter = os.environ.get(f"{role_upper}_TOKEN_PARAMETER", token_parameter)
    if f"{role_upper}_MAX_IMAGE_PAYLOAD_BYTES" in os.environ:
        max_image_payload_bytes = _coerce_int(
            os.environ[f"{role_upper}_MAX_IMAGE_PAYLOAD_BYTES"]
        )

    # Layer 3: explicit overrides.
    if "model" in overrides and overrides["model"] is not None:
        model = overrides["model"]
    if "api_key" in overrides and overrides["api_key"] is not None:
        api_key = overrides["api_key"]
    if "base_url" in overrides and overrides["base_url"] is not None:
        base_url = overrides["base_url"]
    if "temperature" in overrides:
        temperature = overrides["temperature"]
    if "max_tokens" in overrides:
        max_tokens = overrides["max_tokens"]
    if overrides.get("timeout") is not None:
        timeout = overrides["timeout"]
    if overrides.get("token_parameter") is not None:
        token_parameter = overrides["token_parameter"]
    if "max_image_payload_bytes" in overrides:
        max_image_payload_bytes = overrides["max_image_payload_bytes"]

    if token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(
            "token_parameter must be 'max_tokens' or 'max_completion_tokens'"
        )

    return ModelSpec(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        token_parameter=token_parameter,
        max_image_payload_bytes=max_image_payload_bytes,
    )
