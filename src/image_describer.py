# -*- coding: utf-8 -*-
"""Extract images from PDF slides and describe them via Groq Vision, using page text as context."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_PATH = Path("data/image_descriptions.json")
MIN_IMAGE_DIM = 100  # pixels – skip decorative icons

VISION_PROMPT_TEMPLATE = """\
This image is extracted from a lecture slide. The surrounding text on the same slide is:
---
{page_text}
---

Describe ONLY the image content (diagram, chart, figure, example, table, etc.) in English. \
Do NOT repeat the surrounding text already provided above.

Rules:
1. For example sentences or annotations, transcribe them exactly as shown in the image
2. For tables, list all columns and data
3. For flowcharts or diagrams, describe the steps and relationships
4. Preserve the original language of any text in the image (e.g. keep Chinese in Chinese)
5. Do NOT add information that is not in the image
6. Be concise but complete"""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _describe_image(image_bytes: bytes, page_text: str, api_key: str,
                     model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """Call Groq Vision API to describe an image with page text as context."""
    from groq import Groq

    client = Groq(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = VISION_PROMPT_TEMPLATE.format(page_text=page_text[:500] if page_text else "(no text)")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=1024,
                timeout=30.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                log.warning(f"Groq Vision retry in {wait}s: {err_str[:60]}")
                time.sleep(wait)
            else:
                log.warning(f"Groq Vision failed: {e}")
                return ""


def describe_pdf_images(
    pdf_path: Path,
    api_key: str | None = None,
) -> dict[int, list[str]]:
    """Extract images from PDF pages, describe each using page text as context.

    Returns: {page_number: [desc1, desc2, ...]}  (0-indexed)
    """
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF not installed, skipping image extraction")
        return {}

    if not api_key:
        return {}

    cache = _load_cache()
    cache_dirty = False
    result: dict[int, list[str]] = {}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.warning(f"Cannot open PDF {pdf_path.name}: {e}")
        return {}

    # Collect pages that have real images (>= 100px)
    total_images = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text().strip()
        descriptions = []

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < MIN_IMAGE_DIM or pix.height < MIN_IMAGE_DIM:
                    continue
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
            except Exception:
                continue

            img_h = _image_hash(img_bytes)
            cache_key = f"{pdf_path.name}|p{page_num}|{img_h}"

            if cache_key in cache:
                desc = cache[cache_key]
            else:
                total_images += 1
                log.info(f"  Page {page_num+1}, image {len(descriptions)+1} ({pix.width}x{pix.height})")
                desc = _describe_image(img_bytes, page_text, api_key)
                if desc:
                    cache[cache_key] = desc
                    cache_dirty = True
                # Save incrementally
                if cache_dirty:
                    _save_cache(cache)
                    cache_dirty = False

            if desc:
                descriptions.append(desc)

        if descriptions:
            result[page_num] = descriptions

    doc.close()

    if cache_dirty:
        _save_cache(cache)

    return result
