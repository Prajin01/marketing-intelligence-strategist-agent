"""
Small helpers for the Markdown knowledge layer: turning a URL into a safe,
readable, collision-resistant filename, and deciding a document's
confidence label.

Kept separate from markdown_processor.py because these are pure, easily
unit-testable functions with no file I/O — useful on their own (e.g. the
report-generation phase can reuse `confidence_for()` without importing the
whole markdown read/write machinery).
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_SAFE_CHARS = re.compile(r"[^a-z0-9]+")


def slugify_url(url: str) -> str:
    """
    Turn a URL into a filesystem-safe, human-readable slug, e.g.:
      "https://nanavatitoyota.com/contact-su01a.html"
      -> "nanavatitoyota-com-contact-su01a-html-8f3a2b1c"

    A short hash of the full URL is appended so two different pages that
    happen to produce the same slug prefix (rare, but possible after
    aggressive character stripping) never silently overwrite each other.
    """
    parsed = urlparse(url)
    raw = f"{parsed.netloc}{parsed.path}".lower()
    slug = _SAFE_CHARS.sub("-", raw).strip("-")
    if not slug:
        slug = "page"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{url_hash}"[:150]  # keep filenames reasonably short


def confidence_for(document: dict) -> str:
    """
    A document that fetched successfully and has real body text is "high"
    confidence (it's a directly-retrieved primary source). A document that
    failed to fetch is "none" — it contributes no evidence, and should
    never be silently omitted; it's still saved, just clearly marked.
    """
    if document.get("ok") and (document.get("main_text") or "").strip():
        return "high"
    return "none"
