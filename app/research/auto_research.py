"""
Autonomous research layer for the Marketing Intelligence Agent.

REWIRED: previously this module called web_discovery.WebDiscovery, which is
built entirely around an injected `search_fn` that defaults to None
everywhere in the codebase — meaning discovery silently found ZERO sources
for every category, for every business, always. That path was fully
compliant with the "no external research APIs" rule (it simply never called
anything), but it was also non-functional: a real run for a new business
would produce almost no research.

This version wires in what was actually already built and unused:
research_planner.py, which asks Claude to SUGGEST candidate URLs (business
site, competitors, content, customer feedback, market pages) from its own
knowledge — explicitly treated as candidates, never as evidence (see that
module's docstring). Those candidates are then fetched for real by the
existing, robots.txt-respecting WebResearcher / research_*.py modules,
exactly as before. Nothing about the evidence pipeline downstream changes:
a Claude-suggested URL that doesn't exist or 404s just becomes an
ok=False document, same as any other failed fetch.

This keeps the exact same public function name and return shape
(`run_autonomous_research(...) -> {"discovery": ..., "documents": ...,
"all_documents": ...}`) so app/agent/orchestrator.py's call site needs only
to start passing `objective` and `homepage_url` through — nothing else
about the pipeline's contract changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import Settings
from app.research.business_research import research_business
from app.research.competitor_research import research_competitors
from app.research.content_research import research_content
from app.research.customer_research import research_customer_feedback
from app.research.market_research import research_market
from app.research.research_planner import create_research_plan
from app.research.web_research import WebResearcher

logger = logging.getLogger(__name__)


def run_autonomous_research(
    *,
    business_name: str,
    industry: str,
    location: str,
    settings: Settings,
    objective: str = "",
    homepage_url: str = "",
    search_fn=None,  # kept for call-site backward compatibility; unused now
    category_limits: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """
    Plan research with Claude (candidate URLs only), then fetch it for real.

    Returns the same shape the orchestrator already expects:
        {
            "discovery": {category: [{"url": ...}, ...], ...},
            "documents": {category: [raw_document, ...], ...},
            "all_documents": [raw_document, ...]
        }
    """
    logger.info(
        "Starting autonomous research for '%s' (%s, %s).",
        business_name,
        industry,
        location,
    )

    # ---------------------------------------------------------------
    # 1. Ask Claude to suggest candidate research targets.
    #    These are candidates, never evidence — see research_planner.py.
    # ---------------------------------------------------------------
    plan = create_research_plan(
        business_name=business_name,
        industry=industry,
        location=location,
        objective=objective,
        homepage_url=homepage_url,
        settings=settings,
    )

    if plan.get("status") != "success":
        logger.warning(
            "Research planning did not succeed (%s); continuing with whatever candidates it returned.",
            plan.get("error", "unknown reason"),
        )

    max_competitors = (category_limits or {}).get("competitor", settings.max_competitors)
    competitor_urls = plan.get("competitor_urls", [])[: max_competitors * 2]  # planner may suggest extra; cap applied again in research_competitors
    content_urls = plan.get("content_research_urls", [])
    customer_urls = plan.get("customer_research_urls", [])
    market_urls = plan.get("market_research_urls", [])

    # ---------------------------------------------------------------
    # 2. Fetch everything for real, through the existing robots.txt-
    #    respecting WebResearcher — identical to how every other phase
    #    of this project already fetches pages.
    # ---------------------------------------------------------------
    researcher = WebResearcher(settings)

    documents: dict[str, list[dict]] = {
        "business": [],
        "competitor": [],
        "content": [],
        "customer": [],
        "market": [],
    }

    if homepage_url:
        documents["business"] = research_business(researcher, homepage_url)
    else:
        logger.info("No homepage_url provided — skipping direct business-site research for '%s'.", business_name)

    documents["competitor"] = research_competitors(researcher, competitor_urls, max_competitors=max_competitors)
    documents["content"] = research_content(researcher, content_urls)
    documents["customer"] = research_customer_feedback(researcher, customer_urls)
    documents["market"] = research_market(researcher, market_urls)

    # ---------------------------------------------------------------
    # 3. "discovery" mirrors what was planned/attempted, for logging and
    #    for anyone inspecting ResearchState — kept in the same shape the
    #    old web_discovery-based version used.
    # ---------------------------------------------------------------
    discovery: dict[str, list[dict]] = {
        "business": [{"url": homepage_url}] if homepage_url else [],
        "competitor": [{"url": u} for u in competitor_urls],
        "content": [{"url": u} for u in content_urls],
        "customer": [{"url": u} for u in customer_urls],
        "market": [{"url": u} for u in market_urls],
    }

    all_documents: list[dict] = []
    for category_documents in documents.values():
        all_documents.extend(category_documents)

    logger.info(
        "Autonomous research completed: %d candidate source(s) planned, %d document(s) fetched.",
        sum(len(v) for v in discovery.values()),
        len(all_documents),
    )

    return {
        "discovery": discovery,
        "documents": documents,
        "all_documents": all_documents,
        "research_questions": plan.get("research_questions", []),
    }


__all__ = ["run_autonomous_research"]