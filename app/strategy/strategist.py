"""
Phase 5 strategist.

Turns evidence-backed analysis into a coherent marketing strategy.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json


_SYSTEM_PROMPT = """
You are the senior marketing strategist for an evidence-backed marketing
intelligence system.

Your job is to turn the supplied business information and analysis into a
specific marketing strategy.

Rules:

1. Use ONLY the supplied analysis and claims.
2. Do not invent statistics, market facts, customer research, competitors,
   prices, campaign performance, or business capabilities.
3. Recommendations must reference the claim_ids that support their reasoning.
4. Do not present unsupported assumptions as facts.
5. Clearly distinguish evidence-supported observations from strategic
   recommendations.
6. The strategy must be specific to the business and objective.
7. Avoid generic advice such as "post consistently" unless the analysis
   provides a specific reason and implementation.
8. Return ONLY valid JSON.

Return:

{
  "positioning": {
    "statement": "...",
    "reasoning": "...",
    "claim_ids": []
  },
  "target_audience": [
    {
      "segment": "...",
      "reasoning": "...",
      "claim_ids": []
    }
  ],
  "messaging": [
    {
      "message": "...",
      "reasoning": "...",
      "claim_ids": []
    }
  ],
  "channels": [
    {
      "channel": "...",
      "role": "...",
      "reasoning": "...",
      "claim_ids": []
    }
  ],
  "strategic_priorities": [
    {
      "priority": "...",
      "reasoning": "...",
      "claim_ids": []
    }
  ]
}
"""

_INSUFFICIENT_EVIDENCE_POSITIONING = {
    "statement": "Insufficient evidence to support a specific positioning statement.",
    "reasoning": "",
    "claim_ids": [],
}


def _valid_claim_ids(claims: list[dict[str, Any]]) -> set[str]:
    return {
        str(claim["claim_id"])
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id")
    }


def _refs(value: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        str(x)
        for x in value
        if str(x) in valid_ids
    ]


def generate_strategy(
    *,
    business_name: str,
    industry: str,
    location: str,
    objective: str,
    claims: list[dict[str, Any]],
    analysis: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """
    Generate the central marketing strategy.
    """

    valid_ids = _valid_claim_ids(claims)

    prompt = f"""
Business:
{business_name}

Industry:
{industry}

Location:
{location}

Objective:
{objective}

Evidence-backed claims:
{claims}

Five-dimension analysis:
{analysis}

Develop a coherent marketing strategy.

The strategy should answer:

1. What positioning should the business communicate?
2. Which audience segments are supported by the evidence?
3. What messaging should be emphasized?
4. Which marketing channels have a documented strategic reason?
5. What strategic priorities should happen first?

Every strategic conclusion must reference existing claim_ids.
"""

    try:
        result = call_claude_json(
            settings=settings,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=settings.anthropic_strategy_model or settings.anthropic_model,
        )
    except ClaudeJSONError as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }

    if not isinstance(result, dict):
        return {
            "status": "failed",
            "error": "Strategist returned invalid JSON structure.",
        }

    result["status"] = "success"

    # FIX: positioning is a single object (not a list like the other four
    # sections), so it was never being dropped when it had zero valid
    # claim_ids after filtering — meaning an unsupported/fabricated
    # positioning statement could survive with just its evidence trail
    # quietly stripped, while the (unsupported) text itself stayed in the
    # output. Every other section already drops empty-evidence items; this
    # makes positioning consistent with that same rule, replacing it with
    # an honest "insufficient evidence" placeholder instead of a silently
    # unsupported claim.
    if isinstance(result.get("positioning"), dict):
        positioning_refs = _refs(result["positioning"].get("claim_ids"), valid_ids)
        if positioning_refs:
            result["positioning"]["claim_ids"] = positioning_refs
        else:
            result["positioning"] = dict(_INSUFFICIENT_EVIDENCE_POSITIONING)
    else:
        result["positioning"] = dict(_INSUFFICIENT_EVIDENCE_POSITIONING)

    for section in ("target_audience", "messaging", "channels", "strategic_priorities"):
        cleaned = []

        for item in result.get(section, []):
            if not isinstance(item, dict):
                continue

            item["claim_ids"] = _refs(
                item.get("claim_ids"),
                valid_ids,
            )

            if item["claim_ids"]:
                cleaned.append(item)

        result[section] = cleaned

    return result