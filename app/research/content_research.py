"""
Content Researcher (Phase 3).

Fetches an explicit list of publicly accessible content pages (blog posts,
articles, promotional landing pages) for the business and/or its
competitors. Classifying that content into categories (educational,
promotional, product-focused, etc.) is a Claude reasoning task — see
Phase 5's content analysis module — this module's only job is fetching the
raw material safely.
"""

from __future__ import annotations

from app.research.web_research import WebResearcher, research_urls


def research_content(researcher: WebResearcher, content_urls: list[str]) -> list[dict]:
    """Fetch an explicit list of public content page URLs."""
    return research_urls(researcher, content_urls, category="content")
