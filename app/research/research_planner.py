"""
Research planning layer for the Marketing Intelligence Agent.

The planner converts a business/category request into a structured research
plan before the expensive research stage begins.

The planner does NOT treat Claude's suggested URLs as evidence. URLs are only
research candidates. Actual evidence enters the system only after the web
researcher successfully retrieves the source and the extraction layer
validates claims against the retrieved text.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
You are the research planner for a professional marketing intelligence agent.

Your job is to plan research for a business, brand, product, service, or
market category before a marketing strategy is generated.

The final strategy must be based on real publicly observable evidence.

Return ONLY valid JSON with this exact structure:

{
  "research_objectives": [
    "business",
    "competitors",
    "customers",
    "content",
    "market"
  ],
  "competitor_urls": [],
  "content_research_urls": [],
  "customer_research_urls": [],
  "market_research_urls": [],
  "research_questions": []
}

Rules:

1. Never invent facts about the business.
2. URLs are candidate research targets, NOT evidence.
3. Prefer official websites for the target business.
4. For competitors, identify direct competitors relevant to the specified
   industry, location, product/service and objective.
5. Prefer competitor official websites.
6. Content research should look for campaign, content, offers, landing-page,
   blog, social/content and promotional evidence when publicly accessible.
7. Customer research should target publicly accessible review, testimonial,
   FAQ, community or customer-feedback pages when possible.
8. Market research should target publicly accessible category, industry,
   trend, pricing, demand or market information.
9. Do not fabricate URLs merely to fill fields. Empty arrays are acceptable.
10. Research questions should help the later analyst understand positioning,
    customer needs, competitive gaps, acquisition channels, messaging,
    campaigns, offers and opportunities.
"""


def _clean_urls(value: Any) -> list[str]:
    """Keep only non-empty HTTP(S) URLs and remove duplicates."""
    if not isinstance(value, list):
        return []

    result: list[str] = []

    for item in value:
        if not isinstance(item, str):
            continue

        url = item.strip()

        if not url.startswith(("http://", "https://")):
            continue

        if url not in result:
            result.append(url)

    return result


def _clean_questions(value: Any) -> list[str]:
    """Normalize planner-generated research questions."""
    if not isinstance(value, list):
        return []

    result: list[str] = []

    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())

    return result


def create_research_plan(
    *,
    business_name: str,
    industry: str,
    location: str,
    objective: str,
    homepage_url: str,
    settings: Settings,
) -> dict[str, Any]:
    """
    Create a research plan for one marketing-intelligence run.

    Claude proposes research targets. It does not provide accepted evidence.
    """
    user_prompt = f"""
Target:
Business / brand / category: {business_name}
Industry: {industry}
Location: {location}
Marketing objective: {objective}
Known official homepage: {homepage_url}

Create a research plan that will allow a marketing intelligence agent to
understand:

- the target's current positioning
- products/services
- value proposition
- pricing/offers where publicly available
- competitors
- competitor positioning and messaging
- competitor campaigns and promotional patterns
- customer needs and objections
- content and acquisition patterns
- market/category dynamics
- gaps and opportunities

Return ONLY the required JSON object.
"""

    try:
        result = call_claude_json(
            settings=settings,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=settings.anthropic_model,
        )
    except ClaudeJSONError as exc:
        logger.error("Research planning failed: %s", exc)

        return {
            "research_objectives": [
                "business",
                "competitors",
                "customers",
                "content",
                "market",
            ],
            "competitor_urls": [],
            "content_research_urls": [],
            "customer_research_urls": [],
            "market_research_urls": [],
            "research_questions": [],
            "status": "failed",
            "error": str(exc),
        }

    if not isinstance(result, dict):
        logger.error("Research planner returned non-object JSON.")

        return {
            "research_objectives": [],
            "competitor_urls": [],
            "content_research_urls": [],
            "customer_research_urls": [],
            "market_research_urls": [],
            "research_questions": [],
            "status": "failed",
            "error": "Planner returned non-object JSON.",
        }

    plan = {
        "research_objectives": result.get(
            "research_objectives",
            [
                "business",
                "competitors",
                "customers",
                "content",
                "market",
            ],
        ),
        "competitor_urls": _clean_urls(result.get("competitor_urls")),
        "content_research_urls": _clean_urls(
            result.get("content_research_urls")
        ),
        "customer_research_urls": _clean_urls(
            result.get("customer_research_urls")
        ),
        "market_research_urls": _clean_urls(
            result.get("market_research_urls")
        ),
        "research_questions": _clean_questions(
            result.get("research_questions")
        ),
        "status": "success",
    }

    return plan