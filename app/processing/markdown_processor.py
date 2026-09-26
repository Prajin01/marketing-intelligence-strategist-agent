"""
The Markdown Knowledge Layer (Phase 4).

Converts a raw_document dict (as produced by app/research/*.py — see
FetchResult.to_raw_document in web_research.py) into a Markdown file with
YAML frontmatter, and reads such files back for later phases.

Design decisions (see Phase 1 architecture notes for full rationale):
  - One .md file per source, organized by category under
    <knowledge_dir>/<category>/ (business, competitors, customers, content,
    market — matching ResearchState's research dimensions).
  - Frontmatter carries the metadata (source, url, category, business,
    retrieved_at, confidence) that later phases (extraction, validation,
    the final report's "Evidence & Sources" section) all depend on.
  - Failed fetches ARE saved too, with confidence="none" and the error in
    the body — "information unavailable" is itself a recorded research
    finding, never silently dropped.
  - Uses the `python-frontmatter` library (the assignment's required
    Markdown-metadata approach) rather than hand-rolling YAML parsing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from app.config import Settings
from app.processing.cleaner import clean_text
from app.processing.metadata import confidence_for, slugify_url

logger = logging.getLogger(__name__)


def build_knowledge_post(document: dict, business_name: str) -> frontmatter.Post:
    """
    Convert one raw_document dict into a frontmatter.Post ready to be saved.

    Kept as a separate function from save_knowledge_document() so Phase 5's
    extraction step can build/inspect a Post in memory (e.g. in tests)
    without needing to touch the filesystem.
    """
    url = document["url"]
    category = document["category"]
    ok = document.get("ok", False)

    metadata = {
        "source": document.get("title") or url,
        "url": url,
        "category": category,
        "business": business_name,
        "retrieved_at": document.get("fetched_at") or datetime.now(timezone.utc).isoformat(),
        "confidence": confidence_for(document),
        "status": "fetched" if ok else "unavailable",
    }
    if document.get("meta_description"):
        metadata["meta_description"] = document["meta_description"]
    if not ok and document.get("error"):
        metadata["error"] = document["error"]

    if ok:
        heading = document.get("title") or url
        body_text = clean_text(document.get("main_text") or "")
        content = f"# {heading}\n\n{body_text}"
    else:
        # Explicitly record the gap rather than omitting the file — an
        # empty/missing knowledge file for a planned research target is
        # indistinguishable from "we forgot to look"; this way it's
        # indistinguishable from nothing else BUT "we looked and couldn't".
        content = (
            f"# Information unavailable\n\n"
            f"Could not retrieve content from {url}.\n\n"
            f"Reason: {document.get('error', 'unknown error')}"
        )

    return frontmatter.Post(content, **metadata)


def save_knowledge_document(post: frontmatter.Post, settings: Settings) -> Path:
    """
    Write a Post to disk under <knowledge_dir>/<category>/<slug>.md.
    Returns the path written to.
    """
    category = post.get("category", "uncategorized")
    url = post.get("url", "")
    filename = f"{slugify_url(url)}.md"

    category_dir = settings.knowledge_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    path = category_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        frontmatter.dump(post, f)

    logger.info("Saved knowledge document: %s", path)
    return path


def save_raw_documents(documents: list[dict], business_name: str, settings: Settings) -> list[Path]:
    """
    Convenience wrapper: build + save a Post for every raw_document in a
    list (i.e. the output of a research module like research_business()).
    This is what the Phase 6 orchestrator will call after each research
    stage completes.
    """
    saved_paths: list[Path] = []
    for document in documents:
        post = build_knowledge_post(document, business_name)
        saved_paths.append(save_knowledge_document(post, settings))
    return saved_paths


def load_knowledge_documents(settings: Settings, category: str | None = None) -> list[frontmatter.Post]:
    """
    Read back all saved knowledge documents, optionally filtered to one
    category. Each returned Post has an extra `.file_path` attribute (a
    plain Path, not part of the frontmatter) so callers can trace a loaded
    document back to its file — useful for the Phase 5 extraction step's
    evidence-checking, which needs to re-read the exact source text.
    """
    posts: list[frontmatter.Post] = []
    categories = [category] if category else _list_category_dirs(settings)

    for cat in categories:
        category_dir = settings.knowledge_dir / cat
        if not category_dir.exists():
            continue
        for md_file in sorted(category_dir.glob("*.md")):
            post = frontmatter.load(md_file)
            post.file_path = md_file  # type: ignore[attr-defined]
            posts.append(post)

    return posts


def _list_category_dirs(settings: Settings) -> list[str]:
    if not settings.knowledge_dir.exists():
        return []
    return [p.name for p in settings.knowledge_dir.iterdir() if p.is_dir()]