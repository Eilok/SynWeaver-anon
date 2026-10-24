"""
Unified LLM client.

A single OpenAI-compatible chat-completion client shared by the crawler,
UI dataset, and task modules. It uses the native ``openai`` SDK synchronously
and composes the unified JSON parsing and image encoding primitives.

Every required behavior is available through explicit per-instance or per-call
switches:

  - temperature / max_tokens are optional; ``None`` means the parameter is not
    sent. Each module can set them through the spec or per ``complete()`` call.
  - bounded retry with exponential backoff, or ``infinite_retry``.
  - ``json_mode`` requests a JSON object and parses it with
    :func:`llm.json_parsing.parse_json`. OpenRouter reasoning output is used only
    when the standard response content field is empty.
  - ``track_tokens`` accumulates ``response.usage``.
  - ``review_hook`` post-processes the parsed or text result.
  - images supplied as bytes, paths, or ndarrays are merged into user content
    with :func:`llm.image_encoding.build_user_content`.

Logical roles are ``crawler_text``, ``crawler_vision``, ``t1``, ``t2``, and
``task_gen``.
"""
from typing import Any, Callable, Dict, List, Optional, Union
import copy
import logging
import random
import time

from llm.roles import ModelSpec, resolve_spec
from llm.image_encoding import build_user_content, fit_data_urls_to_total_bytes
from llm.json_parsing import parse_json

logger = logging.getLogger(__name__)

# A user payload is either plain text or a preassembled content-block list.
UserContent = Union[str, List[Dict[str, Any]]]


class UnifiedLLMClient:
    """OpenAI-compatible chat-completion client with unified behavior switches."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        track_tokens: bool = False,
        max_retries: int = 3,
        infinite_retry: bool = False,
        review_hook: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Args:
            spec: Model parameters including model, credentials, generation
                options, and timeout.
            track_tokens: Accumulate response usage into ``token_usage``.
            max_retries: Number of bounded retry attempts. Ignored when
                ``infinite_retry`` is enabled.
            infinite_retry: Retry until a request succeeds.
            review_hook: Optional callable applied to the parsed or text result.
                Its return value becomes the completion result.
        """
        self.spec = spec
        self.track_tokens = track_tokens
        self.max_retries = max_retries
        self.infinite_retry = infinite_retry
        self.review_hook = review_hook

        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }
        self._client = None

    @property
    def client(self):
        """Lazily construct the underlying openai.OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.spec.api_key,
                base_url=self.spec.base_url,
                timeout=self.spec.timeout,
            )
        return self._client

    def _build_messages(
        self,
        system: Optional[str],
        user: UserContent,
        images: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """Assemble OpenAI-style messages from system, user, and image inputs."""
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if images:
            # Merge images with text or append them to existing content blocks.
            if isinstance(user, str):
                content = build_user_content(user, images)
            else:
                from llm.image_encoding import image_block

                content = list(user)
                for img in images:
                    block = image_block(img)
                    if block is not None:
                        content.append(block)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})

        return messages

    def _fit_image_payloads(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply an optional provider payload limit to embedded image data URLs."""
        limit = self.spec.max_image_payload_bytes
        if limit is None:
            return messages

        fitted = copy.deepcopy(messages)
        image_blocks: List[Dict[str, Any]] = []
        data_urls: List[str] = []
        for message in fitted:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if isinstance(url, str) and url.startswith("data:image/"):
                    image_blocks.append(block)
                    data_urls.append(url)

        if not data_urls:
            return fitted
        resized_urls = fit_data_urls_to_total_bytes(data_urls, limit)
        for block, url in zip(image_blocks, resized_urls):
            block["image_url"]["url"] = url
        return fitted

    def _build_kwargs(
        self,
        messages: List[Dict[str, Any]],
        json_mode: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build chat completion arguments while omitting unset parameters."""
        kwargs: Dict[str, Any] = {
            "model": model or self.spec.model,
            "messages": self._fit_image_payloads(messages),
        }
        # Per-call values take precedence over the spec; None means do not send.
        temp = temperature if temperature is not None else self.spec.temperature
        if temp is not None:
            kwargs["temperature"] = temp
        max_t = max_tokens if max_tokens is not None else self.spec.max_tokens
        if max_t is not None:
            kwargs[self.spec.token_parameter] = max_t
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _record_usage(self, response: Any) -> None:
        """Accumulate token usage from a response when tracking is enabled."""
        if not self.track_tokens:
            return
        usage = getattr(response, "usage", None)
        if usage:
            self.token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self.token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        self.token_usage["call_count"] += 1

    def complete(
        self,
        *,
        system: Optional[str] = None,
        user: UserContent,
        images: Optional[List[Any]] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Union[str, dict, list]]:
        """
        Run one chat completion.

        Args:
            system: Optional system prompt.
            user: Plain text or a preassembled content-block list.
            images: Optional image bytes, file paths, or ndarrays merged into the
                user content as image URL blocks.
            json_mode: Request and parse JSON, returning a dict or list, or None
                when parsing fails.
            temperature: Per-call temperature. None falls back to the spec, which
                may also be None and therefore omitted.
            max_tokens: Per-call maximum completion tokens with the same fallback
                semantics as temperature.

        Returns:
            Raw response text when JSON mode is disabled, parsed JSON when it is
            enabled, or None on failure. A configured review hook can replace the
            result with its own return value.
        """
        messages = self._build_messages(system, user, images)
        kwargs = self._build_kwargs(messages, json_mode, temperature, max_tokens)

        content = self._request_with_retry(kwargs)
        if content is None:
            return None

        result: Any = parse_json(content) if json_mode else content

        if self.review_hook is not None:
            result = self.review_hook(result)

        return result

    def _request_with_retry(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Call the SDK with bounded or infinite retry and return response text."""
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._record_usage(response)
                choice = response.choices[0]
                message = choice.message
                content = message.content
                if content is not None:
                    return content

                # Some OpenRouter reasoning models place JSON-mode output in the
                # reasoning field while leaving the standard content field empty.
                reasoning = getattr(message, "reasoning", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    logger.debug("Using reasoning field because LLM content is empty")
                    return reasoning

                logger.warning(
                    "LLM response contained no text content (finish_reason=%s)",
                    choice.finish_reason,
                )
                return None
            except Exception as error:  # noqa: BLE001
                attempt += 1
                status = getattr(error, "status_code", None)
                logger.warning(
                    "LLM request failed: type=%s status=%s model=%s base_url=%s "
                    "attempt=%d/%s",
                    type(error).__name__,
                    status,
                    self.spec.model,
                    self.spec.base_url,
                    attempt,
                    "unbounded" if self.infinite_retry else self.max_retries,
                )
                if self.infinite_retry or attempt < self.max_retries:
                    delay = min(2 ** min(attempt - 1, 5), 30)
                    time.sleep(delay + random.uniform(0, delay * 0.25))
                    continue
                logger.error("LLM request failed after %d attempts", self.max_retries)
                return None

    def close(self) -> None:
        """Close the underlying HTTP client if it has been constructed."""
        if self._client is not None:
            self._client.close()

    def get_token_usage(self) -> Dict[str, Any]:
        """Return accumulated token usage and average tokens per call."""
        calls = self.token_usage["call_count"]
        return {
            **self.token_usage,
            "avg_tokens_per_call": (
                self.token_usage["total_tokens"] / calls if calls > 0 else 0
            ),
        }


# Cache singleton clients by logical role, matching the original manager behavior.
_client_cache: Dict[str, UnifiedLLMClient] = {}


def get_client(
    role: str,
    *,
    conf_path: str = "./config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    track_tokens: bool = False,
    max_retries: int = 3,
    infinite_retry: bool = False,
    review_hook: Optional[Callable[[Any], Any]] = None,
    fresh: bool = False,
) -> UnifiedLLMClient:
    """
    Get a cached UnifiedLLMClient for a logical role.

    Args:
        role: One of ``crawler_text``, ``crawler_vision``, ``t1``, ``t2``, or
            ``task_gen``.
        conf_path: config.yaml path passed to :func:`resolve_spec`.
        overrides: Explicit ModelSpec overrides such as model, temperature, or
            max_tokens.
        track_tokens: Enable token usage accumulation.
        max_retries: Number of bounded retry attempts.
        infinite_retry: Retry indefinitely until success.
        review_hook: Optional result post-processing callable.
        fresh: Bypass the cache and build a new client. Use this when per-instance
            behavior switches differ from those of the cached client.

    Returns:
        A UnifiedLLMClient. The cache key is the role only; use ``fresh=True``
        when a differently configured client is required for the same role.
    """
    if not fresh and overrides is None and role in _client_cache:
        return _client_cache[role]

    spec = resolve_spec(role, conf_path=conf_path, overrides=overrides)
    client = UnifiedLLMClient(
        spec,
        track_tokens=track_tokens,
        max_retries=max_retries,
        infinite_retry=infinite_retry,
        review_hook=review_hook,
    )
    if not fresh and overrides is None:
        _client_cache[role] = client
    return client
