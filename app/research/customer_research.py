"""
Customer Insight Researcher (Phase 3).

Unlike business/competitor research, there's no single "homepage" to crawl
for customer feedback — it lives scattered across review pages, testimonial
pages, and public forum threads. So this module fetches an explicit list of
URLs (no link-following) rather than discovering pages itself.

Ethical/access boundary (from the assignment): only genuinely public pages
that don't require login are fetched — enforced the same way as everywhere
else, via WebResearcher.fetch()'s robots.txt check. Many review platforms
block scraping entirely; when that happens, the resulting document will
have ok=False with an explanatory error, which is an honest, reportable
finding ("insufficient public customer data") rather than something to work
around.
"""

from __future__ import annotations

from app.research.web_research import WebResearcher, research_urls


def research_customer_feedback(researcher: WebResearcher, feedback_urls: list[str]) -> list[dict]:
    """Fetch an explicit list of public customer feedback/review page URLs."""
    return research_urls(researcher, feedback_urls, category="customers")
