"""
Shared helpers for Phase 5 analysis modules.

Analysis is allowed to create INFERENCE and RECOMMENDATION-level
interpretations, but every such interpretation must reference one or more
existing claim_ids.

FACT and OBSERVATION claims are created only by the extraction stage.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json

logger = logging.getLogger(__name__)


_ANALYSIS_SYSTEM_PROMPT = """
You are a senior marketing intelligence analyst.

You are analyzing evidence collected from publicly available sources for a
marketing intelligence system.

STRICT EVIDENCE RULES:

1. You may ONLY reason from the claims provided to you.
2. Do not invent facts, statistics, customer opinions, competitors, prices,
   campaigns, market trends, or other information that is not represented
   in the supplied claims.
3. Every inference MUST contain the claim_ids that support it.
4. Every recommendation MUST contain the claim_ids that support the reasoning
   behind the recommendation.
5. Never create a new FACT or OBSERVATION.
6. If the evidence is insufficient, explicitly say so.
7. Do not pretend that missing evidence exists.

Return ONLY valid JSON.

Use this structure:

{
  "dimension": "string",
  "summary": "string",
  "insights": [
    {
      "finding": "string",
      "type": "inference",
      "claim_ids": ["claim_id"]
    }
  ],
  "opportunities": [
    {
      "opportunity": "string",
      "reason": "string",
      "claim_ids": ["claim_id"]
    }
  ],
  "gaps": [
    {
      "gap": "string",
      "why_it_matters": "string",
      "claim_ids": ["claim_id"]
    }
  ],
  "recommendations": [
    {
      "recommendation": "string",
      "reason": "string",
      "claim_ids": ["claim_id"]
    }
  ]
}
"""


def _claim_id_set(claims: list[dict[str, Any]]) -> set[str]:
    """Return all valid claim IDs available to the analysis stage."""
    return {
        str(claim["claim_id"])
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id")
    }


def _clean_reference_list(
    claim_ids: Any,
    valid_ids: set[str],
) -> list[str]:
    """
    Keep only claim IDs that actually exist in the current evidence set.

    This prevents Claude from inventing references to nonexistent evidence.
    """
    if not isinstance(claim_ids, list):
        return []

    cleaned = []

    for claim_id in claim_ids:
        claim_id = str(claim_id)

        if claim_id in valid_ids and claim_id not in cleaned:
            cleaned.append(claim_id)

    return cleaned


def _validate_analysis_result(
    result: Any,
    claims: list[dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    """
    Normalize Claude's analysis result and enforce evidence references.

    Unsupported references are removed. Insights/opportunities/gaps/
    recommendations without evidence references are dropped.
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"{dimension} analysis must return a JSON object."
        )

    valid_ids = _claim_id_set(claims)

    cleaned: dict[str, Any] = {
        "dimension": dimension,
        "summary": str(result.get("summary", "")).strip(),
        "insights": [],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    for item in result.get("insights", []):
        if not isinstance(item, dict):
            continue

        refs = _clean_reference_list(item.get("claim_ids"), valid_ids)

        if not refs:
            continue

        cleaned["insights"].append({
            "finding": str(item.get("finding", "")).strip(),
            "type": "inference",
            "claim_ids": refs,
        })

    for item in result.get("opportunities", []):
        if not isinstance(item, dict):
            continue

        refs = _clean_reference_list(item.get("claim_ids"), valid_ids)

        if not refs:
            continue

        cleaned["opportunities"].append({
            "opportunity": str(item.get("opportunity", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "claim_ids": refs,
        })

    for item in result.get("gaps", []):
        if not isinstance(item, dict):
            continue

        refs = _clean_reference_list(item.get("claim_ids"), valid_ids)

        if not refs:
            continue

        cleaned["gaps"].append({
            "gap": str(item.get("gap", "")).strip(),
            "why_it_matters": str(item.get("why_it_matters", "")).strip(),
            "claim_ids": refs,
        })

    for item in result.get("recommendations", []):
        if not isinstance(item, dict):
            continue

        refs = _clean_reference_list(item.get("claim_ids"), valid_ids)

        if not refs:
            continue

        cleaned["recommendations"].append({
            "recommendation": str(item.get("recommendation", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "claim_ids": refs,
        })

    return cleaned


def analyze_dimension(
    *,
    dimension: str,
    objective: str,
    claims: list[dict[str, Any]],
    settings: Settings,
    instructions: str,
) -> dict[str, Any]:
    """
    Run one evidence-backed marketing analysis.

    All five analysis modules use this same implementation.
    """

    if not claims:
        return {
            "dimension": dimension,
            "summary": "Insufficient public evidence for this dimension.",
            "insights": [],
            "opportunities": [],
            "gaps": [],
            "recommendations": [],
        }

    evidence_text = "\n\n".join(
        [
            (
                f"CLAIM_ID: {claim.get('claim_id')}\n"
                f"TYPE: {claim.get('claim_type')}\n"
                f"CLAIM: {claim.get('claim')}\n"
                f"EVIDENCE: {claim.get('evidence')}\n"
                f"SOURCE: {claim.get('source_url')}"
            )
            for claim in claims
        ]
    )

    user_prompt = f"""
Business objective:
{objective}

Analysis dimension:
{dimension}

Dimension-specific instructions:
{instructions}

Evidence-backed claims:

{evidence_text}

Analyze the evidence carefully.

Do not use outside knowledge.

If there is not enough evidence to support an insight or recommendation,
leave it out rather than guessing.
"""

    try:
        raw_result = call_claude_json(
            settings=settings,
            system_prompt=_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except ClaudeJSONError:
        logger.exception("Claude analysis failed for dimension=%s", dimension)

        return {
            "dimension": dimension,
            "summary": "Analysis failed because the Claude reasoning stage did not return valid JSON.",
            "insights": [],
            "opportunities": [],
            "gaps": [],
            "recommendations": [],
        }

    try:
        return _validate_analysis_result(
            raw_result,
            claims,
            dimension,
        )
    except ValueError:
        logger.exception(
            "Invalid analysis structure returned for dimension=%s",
            dimension,
        )

        return {
            "dimension": dimension,
            "summary": "Analysis returned an invalid structure.",
            "insights": [],
            "opportunities": [],
            "gaps": [],
            "recommendations": [],
        }