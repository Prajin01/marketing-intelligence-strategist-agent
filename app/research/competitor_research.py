"""
Competitor Researcher (Phase 3).

Fetches each competitor's homepage + relevant subpages, exactly like
business_research.py does for the business itself — same fetching logic,
different category label ("competitors" instead of "business").

Scope note: this module does NOT discover competitor URLs on its own.
Automatically finding competitors (via search-engine scraping) is the
fragile, ToS-sensitive part flagged in the Phase 1 architecture doc. For
now, competitor URLs are supplied by the caller — either typed in by the
user (MVP) or, later, suggested by the Research Planner (Phase 5). This
keeps the fetching layer solid and testable regardless of how the URLs
were sourced.
"""

from __future__ import annotations

import logging

from app.research.web_research import WebResearcher, research_website

logger = logging.getLogger(__name__)


def research_competitor(researcher: WebResearcher, homepage_url: str) -> list[dict]:
    """Fetch one competitor's homepage + relevant subpages."""
    return research_website(researcher, homepage_url, category="competitors")


def research_competitors(
    researcher: WebResearcher, homepage_urls: list[str], max_competitors: int
) -> list[dict]:
    """
    Fetch multiple competitors, capped at `max_competitors` (see
    MAX_COMPETITORS in .env — a deliberate scope decision to keep a live
    demo's runtime/cost bounded, documented in the README).
    """
    documents: list[dict] = []
    capped_urls = homepage_urls[:max_competitors]
    if len(homepage_urls) > max_competitors:
        logger.info(
            "Received %d competitor URLs, only researching the first %d (MAX_COMPETITORS).",
            len(homepage_urls),
            max_competitors,
        )
    for url in capped_urls:
        documents.extend(research_competitor(researcher, url))
    return documents
