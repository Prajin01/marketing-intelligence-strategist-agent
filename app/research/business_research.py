"""
Business Researcher (Phase 3).

Thin wrapper around web_research.research_website(): fetches the business's
own homepage plus a handful of relevant subpages (About, Contact, Products),
tagged with category="business".

Kept as its own module (rather than inlining the call everywhere) so the
research modules stay symmetrical and easy to find: one file per research
dimension, matching the project's architecture diagram.
"""

from __future__ import annotations

from app.research.web_research import WebResearcher, research_website


def research_business(researcher: WebResearcher, homepage_url: str) -> list[dict]:
    """Fetch the business's own homepage + relevant subpages."""
    return research_website(researcher, homepage_url, category="business")