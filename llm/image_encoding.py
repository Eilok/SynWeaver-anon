"""
Unified image -> OpenAI ``image_url`` content-block encoding.

Consolidates three previously-duplicated encoders that differed only in their
input source:

- ``website_crawler`` / ``prompts.website_crawl`` — in-memory ``bytes``
- ``ui_generator.llm_client`` — image file path on disk (with MIME detection)
- ``task_synthesizer`` / ``prompts.execution`` / ``prompts.task_synthesis`` —
  ``numpy.ndarray`` (encoded as PNG via PIL)

All three ultimately emit the identical OpenAI block shape::

    {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<...>"}}

``numpy``/``PIL`` are imported lazily so that callers passing only ``bytes`` or
file paths do not pull in those dependencies.
"""
from typing import Any, Dict, List, Optional, Union
import base64
import logging
import math
import os

logger = logging.getLogger(__name__)

# Type of any accepted image source. ndarray is referenced lazily/by name.
ImageSource = Union[bytes, str, "Any"]

# MIME lookup by file extension (carried over from ui_generator's encoder).
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _mime_for_path(path: str) -> str:
    """Detect MIME type from a file extension, defaulting to image/png."""
    ext = os.path.splitext(path)[1].lower()
    return _MIME_BY_EXT.get(ext, "image/png")


def _ndarray_to_png_bytes(img: "Any") -> bytes:
    """Encode a numpy ndarray to PNG bytes via PIL (lazy import)."""
    from io import BytesIO
    from PIL import Image

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _is_ndarray(img: Any) -> bool:
    """True if img is a numpy ndarray, without importing numpy eagerly."""
    if "numpy" not in str(type(img).__mro__ if hasattr(type(img), "__mro__") else ""):
        # Cheap structural check: numpy arrays expose .dtype and .shape.
        return hasattr(img, "dtype") and hasattr(img, "shape") and hasattr(img, "astype")
    return True


def to_data_url(img: ImageSource, mime: str = "image/png") -> Optional[str]:
    """
    Encode an image from any supported source to a ``data:<mime>;base64,<...>`` URL.

    Args:
        img: One of:
            - ``bytes``: raw image bytes, encoded as-is under ``mime``.
            - ``str``: a file path; read from disk, MIME detected by extension
              (``mime`` arg ignored in favour of detection).
            - ``numpy.ndarray``: encoded to PNG via PIL (``mime`` forced to png).
        mime: MIME type to use for the ``bytes`` branch (default image/png).

    Returns:
        The data URL string, or None if the source is missing/unreadable
        (mirrors the lenient behaviour of the originals).
    """
    if img is None:
        return None

    # bytes -> direct
    if isinstance(img, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(img)).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    # str -> file path
    if isinstance(img, str):
        if not img or not os.path.exists(img):
            logger.warning(f"Image not found: {img}")
            return None
        try:
            with open(img, "rb") as f:
                data = f.read()
        except Exception as e:
            logger.error(f"Failed to load image {img}: {e}")
            return None
        detected = _mime_for_path(img)
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{detected};base64,{b64}"

    # ndarray -> PNG via PIL
    if _is_ndarray(img):
        try:
            data = _ndarray_to_png_bytes(img)
        except Exception as e:
            logger.error(f"Failed to encode ndarray image: {e}")
            return None
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    logger.warning(f"Unsupported image source type: {type(img)!r}")
    return None


def fit_data_urls_to_total_bytes(
    data_urls: List[str],
    max_total_bytes: int,
) -> List[str]:
    """Resize embedded images proportionally until their encoded total fits a limit."""
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive")
    original_total = sum(len(url.encode("utf-8")) for url in data_urls)
    if original_total <= max_total_bytes:
        return data_urls

    from io import BytesIO
    from PIL import Image

    images: List[Image.Image] = []
    formats: List[str] = []
    for url in data_urls:
        if not url.startswith("data:image/") or ";base64," not in url:
            raise ValueError("Image payload limiting requires base64 image data URLs")
        header, encoded = url.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        images.append(image)
        formats.append("JPEG" if header.startswith("data:image/jpeg") else "PNG")

    scale = min(0.95, math.sqrt(max_total_bytes / original_total) * 0.95)
    resized_urls: List[str] = []
    for _ in range(8):
        resized_urls = []
        for image, image_format in zip(images, formats):
            width = max(1, int(image.width * scale))
            height = max(1, int(image.height * scale))
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            save_kwargs = {"quality": 90} if image_format == "JPEG" else {"optimize": True}
            resized.save(buffer, format=image_format, **save_kwargs)
            mime = "image/jpeg" if image_format == "JPEG" else "image/png"
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            resized_urls.append(f"data:{mime};base64,{encoded}")
        total = sum(len(url.encode("utf-8")) for url in resized_urls)
        if total <= max_total_bytes:
            logger.info(
                "Reduced embedded image payload from %d to %d bytes",
                original_total,
                total,
            )
            return resized_urls
        scale *= math.sqrt(max_total_bytes / total) * 0.95

    raise ValueError(
        f"Unable to reduce embedded image payload below {max_total_bytes} bytes"
    )


def image_block(img: ImageSource, mime: str = "image/png") -> Optional[Dict[str, Any]]:
    """
    Build an OpenAI ``image_url`` content block from an image source.

    Returns:
        ``{"type": "image_url", "image_url": {"url": <data-url>}}``, or None if
        the image could not be encoded.
    """
    data_url = to_data_url(img, mime=mime)
    if data_url is None:
        return None
    return {"type": "image_url", "image_url": {"url": data_url}}


def build_user_content(
    text: str,
    images: Optional[List[ImageSource]] = None,
    *,
    text_first: bool = False,
) -> List[Dict[str, Any]]:
    """
    Assemble an OpenAI multimodal user-content list from text + optional images.

    Args:
        text: The text portion of the message.
        images: Optional list of image sources (bytes / path / ndarray).
            Unencodable entries are skipped.
        text_first: If True, place the text block before the image blocks.
            Default False matches the prevailing convention in this codebase
            (images first, trailing text).

    Returns:
        A list of content blocks suitable for a ``{"role": "user", "content": ...}``
        message.
    """
    text_block = {"type": "text", "text": text}
    image_blocks: List[Dict[str, Any]] = []
    for img in images or []:
        block = image_block(img)
        if block is not None:
            image_blocks.append(block)

    if text_first:
        return [text_block, *image_blocks]
    return [*image_blocks, text_block]
