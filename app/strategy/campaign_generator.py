"""
Campaign generation stage.

Generates concrete campaign concepts from the evidence-backed strategy.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json


_SYSTEM_PROMPT = """
You are a senior performance and brand marketing strategist.

Create campaign concepts based ONLY on the supplied evidence-backed strategy
and claims.

Do not invent factual information about the business.

Campaign recommendations may be creative, but the strategic reasoning behind
them must reference existing claim_ids.

Return ONLY valid JSON.

Structure:

{
  "campaigns": [
    {
      "name": "...",
      "objective": "...",
      "audience": "...",
      "core_idea": "...",
      "message": "...",
      "offer_or_hook": "...",
      "recommended_channels": [],
      "content_assets": [],
      "reasoning": "...",
      "claim_ids": []
    }
  ]
}
"""


def generate_campaigns(
    *,
    business_name: str,
    objective: str,
    claims: list[dict[str, Any]],
    analysis: dict[str, Any],
    strategy: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:

    valid_ids = {
        str(c["claim_id"])
        for c in claims
        if isinstance(c, dict) and c.get("claim_id")
    }

    prompt = f"""
Business:
{business_name}

Business objective:
{objective}

Evidence-backed claims:
{claims}

Marketing analysis:
{analysis}

Approved strategic direction:
{strategy}

Generate a practical campaign portfolio.

Include a mixture of:
- acquisition campaigns
- conversion campaigns
- awareness/positioning campaigns
- retention or repeat-purchase campaigns when relevant

Campaigns should be specific to the business instead of generic templates.

Do not claim that a campaign already exists unless the evidence says so.
These are NEW recommendations.
"""

    try:
        result = call_claude_json(
            settings=settings,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
    except ClaudeJSONError as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "campaigns": [],
        }

    if not isinstance(result, dict):
        return {
            "status": "failed",
            "error": "Campaign generator returned invalid JSON.",
            "campaigns": [],
        }

    campaigns = []

    for campaign in result.get("campaigns", []):
        if not isinstance(campaign, dict):
            continue

        claim_ids = campaign.get("claim_ids", [])

        if not isinstance(claim_ids, list):
            claim_ids = []

        claim_ids = [
            str(x)
            for x in claim_ids
            if str(x) in valid_ids
        ]

        # A campaign without evidence-backed reasoning should not survive.
        if not claim_ids:
            continue

        campaign["claim_ids"] = claim_ids
        campaigns.append(campaign)

    return {
        "status": "success",
        "campaigns": campaigns,
    }