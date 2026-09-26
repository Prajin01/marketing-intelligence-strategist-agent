"""
Final validation stage for the Marketing Intelligence Agent.

The validator does not generate strategy. It checks whether the generated
strategy is internally consistent and whether claims/citations used by
strategy, campaigns, and action plans actually exist.

This is intentionally deterministic wherever possible.
"""

from __future__ import annotations

from typing import Any


def _valid_claim_ids(claims: list[dict[str, Any]]) -> set[str]:
    """Return all claim IDs that actually exist in the extracted evidence."""
    return {
        str(claim["claim_id"])
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id")
    }


def _collect_claim_ids(value: Any) -> list[str]:
    """
    Recursively collect claim_ids from arbitrary strategy/campaign/action
    structures.
    """
    found: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"claim_id", "claim_ids", "source_claim_ids"}:
                if isinstance(item, str):
                    found.append(item)
                elif isinstance(item, list):
                    found.extend(
                        str(x)
                        for x in item
                        if isinstance(x, (str, int))
                    )
            else:
                found.extend(_collect_claim_ids(item))

    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_claim_ids(item))

    return found


def validate_claim_references(
    claims: list[dict[str, Any]],
    analysis: dict[str, Any],
    strategy: dict[str, Any],
    campaigns: dict[str, Any],
    action_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate that every claim reference used downstream actually exists.

    Unknown references are reported rather than silently removed because the
    validation stage should make problems visible.
    """
    valid_ids = _valid_claim_ids(claims)

    sources = {
        "analysis": analysis,
        "strategy": strategy,
        "campaigns": campaigns,
        "action_plan": action_plan,
    }

    invalid_references: list[dict[str, str]] = []
    referenced_ids: set[str] = set()

    for source_name, value in sources.items():
        for claim_id in _collect_claim_ids(value):
            referenced_ids.add(claim_id)

            if claim_id not in valid_ids:
                invalid_references.append(
                    {
                        "source": source_name,
                        "claim_id": claim_id,
                    }
                )

    return {
        "valid": len(invalid_references) == 0,
        "total_claims": len(valid_ids),
        "referenced_claims": len(referenced_ids),
        "invalid_references": invalid_references,
    }


def validate_strategy(
    claims: list[dict[str, Any]],
    analysis: dict[str, Any],
    strategy: dict[str, Any],
    campaigns: dict[str, Any],
    action_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Run final deterministic validation checks.

    Returns a serializable validation report suitable for ResearchState.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(strategy, dict) or not strategy:
        errors.append("Strategy output is empty.")

    if not isinstance(campaigns, dict):
        errors.append("Campaign output is not a dictionary.")

    if not isinstance(action_plan, dict):
        errors.append("Action-plan output is not a dictionary.")

    if not isinstance(analysis, dict):
        errors.append("Analysis output is not a dictionary.")

    claim_report = validate_claim_references(
        claims=claims,
        analysis=analysis,
        strategy=strategy,
        campaigns=campaigns,
        action_plan=action_plan,
    )

    if not claim_report["valid"]:
        errors.append(
            f"{len(claim_report['invalid_references'])} invalid claim "
            "reference(s) were found."
        )

    if not claims:
        warnings.append("No extracted claims are available.")

    if not analysis:
        warnings.append("Analysis is empty.")

    if not campaigns:
        warnings.append("No campaigns were generated.")

    if not action_plan:
        warnings.append("Action plan is empty.")

    return {
        "status": "failed" if errors else "success",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "claim_reference_check": claim_report,
    }


def validate_pipeline(
    claims: list[dict[str, Any]],
    analysis: dict[str, Any],
    strategy: dict[str, Any],
    campaigns: dict[str, Any],
    action_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Entry point the orchestrator actually calls.

    This was missing — orchestrator.py imports `validate_pipeline` by this
    exact name, but only `validate_strategy` existed, so the import silently
    failed (caught by the orchestrator's `except ImportError`) and real
    validation never ran, even though validate_strategy() itself was
    correctly implemented. This thin alias fixes that without changing the
    validation logic itself, which was already sound.
    """
    return validate_strategy(
        claims=claims,
        analysis=analysis,
        strategy=strategy,
        campaigns=campaigns,
        action_plan=action_plan,
    )