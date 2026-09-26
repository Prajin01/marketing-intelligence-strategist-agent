"""
Basic text cleaning applied to fetched page text before it's stored as a
knowledge document (Phase 4) or handed to Claude (Phase 5).

Kept intentionally simple: web_research.py's extract_content() already does
the heavy lifting (stripping script/nav/footer tags via BeautifulSoup).
This module handles the leftover mess that survives HTML parsing — repeated
boilerplate lines, excessive blank lines, and pages so long they'd blow out
Claude's context unnecessarily.
"""

from __future__ import annotations

DEFAULT_MAX_CHARS = 20_000


def clean_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    - Collapses consecutive duplicate lines (common artifact of poorly
      structured pages where the same nav item or heading repeats).
    - Drops blank lines entirely (they add no research value here and
      inflate stored file size / token count for no benefit).
    - Truncates to `max_chars`, appending a note so it's clear to anyone
      reading the knowledge file (or Claude, later) that this is a partial
      excerpt, not deceptive silence about missing content.
    """
    if not text:
        return ""

    lines = text.split("\n")
    deduped: list[str] = []
    previous: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == previous:
            continue
        deduped.append(stripped)
        previous = stripped

    cleaned = "\n".join(deduped)

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "\n\n[... truncated: original page text was longer ...]"

    return cleaned
