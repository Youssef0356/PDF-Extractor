"""Vision/OCR extraction service.

Uses the configured Ollama multimodal model to extract text from rendered PDF page images.
"""

from __future__ import annotations

import base64
import re
from typing import Optional

import ollama as ollama_client

from config import VISION_MODEL


def extract_text_from_page_image_base64(
    image_base64: str,
    *,
    language_hint: str = "fr+en",
    max_chars: int = 12000,
) -> str:
    """Extract visible text from a single page image.

    Args:
        image_base64: Base64-encoded PNG/JPEG (no data URL prefix).
        language_hint: A hint for the model about expected languages.
        max_chars: Hard cap for returned text (to keep embeddings stable).

    Returns:
        Plain text.
    """
    b64 = (image_base64 or "").strip()
    if not b64:
        return ""

    prompt = (
        "You are an OCR engine. Extract ALL visible text from the image. "
        "Preserve the reading order as best as possible. "
        "Return plain text only. Do not add commentary. "
        f"The document may contain {language_hint}."
    )

    # Ollama python client supports multimodal messages via `images`.
    # The value is expected to be base64-encoded bytes.
    resp = ollama_client.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        options={"temperature": 0.0},
        keep_alive="10m",
    )

    text = (resp.get("message") or {}).get("content") or ""
    text = _normalize_ocr_text(text)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def _normalize_ocr_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    # Collapse excessive whitespace but keep newlines.
    t = "\n".join([re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")])
    # Collapse too many blank lines.
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t
