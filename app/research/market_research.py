"""
Market Researcher (Phase 3).

Fetches an explicit list of publicly accessible market/industry information
pages (industry association pages, published trend reports, news articles
about the sector). Same fetch-only, no-discovery approach as
customer_research.py and content_research.py, for the same reason:
automatically discovering "what market information is relevant" is a
research-planning decision (Claude's job, Phase 5), not a fetching one.
"""

from __future__ import annotations

from app.research.web_research import WebResearcher, research_urls


def research_market(researcher: WebResearcher, market_urls: list[str]) -> list[dict]:
    """Fetch an explicit list of public market/industry information page URLs."""
    return research_urls(researcher, market_urls, category="market")
