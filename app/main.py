"""
Main entry point for the Marketing Intelligence Agent.

Phase 6+ update: this now runs the REAL end-to-end pipeline (research ->
extraction -> analysis -> strategy -> campaigns -> action plan ->
validation) via app.agent.orchestrator.run_new(), rather than the Phase 2
skeleton that only created and saved an empty state. Streamlit (Phase 8)
will call this same `run()` function directly.

Usage:
    python -m app.main --business "Nanavati Toyota" --industry "Automobile" \\
        --location "Surat, Gujarat" --objective "Increase customer acquisition" \\
        --homepage-url "https://nanavatitoyota.com/"

This makes real calls to the Anthropic API (research planning, extraction,
5x analysis, strategy, campaigns, action plan) and real web requests
(robots.txt-respecting). Expect it to take a few minutes and use a modest,
real amount of API credit.
"""

from __future__ import annotations

import argparse
import logging

from app.agent.orchestrator import run_new
from app.agent.state import ResearchState
from app.config import configure_logging, load_settings

logger = logging.getLogger(__name__)


def _print_summary(state: ResearchState) -> None:
    """Print a human-readable summary of what the pipeline produced."""
    print("\n" + "=" * 70)
    print(f"RUN COMPLETE: {state.input.business_name}")
    print("=" * 70)

    print(f"\nRaw documents fetched: {len(state.raw_documents)}")
    ok_count = sum(1 for d in state.raw_documents if d.get("ok"))
    print(f"  - Successfully fetched: {ok_count}")
    print(f"  - Failed/unavailable:   {len(state.raw_documents) - ok_count}")

    print(f"\nEvidence-backed claims extracted: {len(state.claims)}")
    by_type: dict[str, int] = {}
    for claim in state.claims:
        t = claim.get("claim_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for claim_type, count in by_type.items():
        print(f"  - {claim_type}: {count}")

    positioning = state.strategy.get("positioning", {})
    print(f"\nPositioning: {positioning.get('statement', '(none)')}")

    campaigns = state.strategy.get("campaigns", {}).get("campaigns", [])
    print(f"\nCampaigns generated: {len(campaigns)}")
    for c in campaigns[:3]:
        print(f"  - {c.get('name', '(unnamed)')}: {c.get('objective', '')}")

    action_plan = state.strategy.get("action_plan", {})
    plan_phases = action_plan.get("plan", [])
    total_actions = sum(len(p.get("actions", [])) for p in plan_phases)
    print(f"\nAction plan: {len(plan_phases)} phase(s), {total_actions} action(s) total")

    validation = state.validation_report
    print(f"\nValidation status: {validation.get('status', 'unknown')}")
    if validation.get("errors"):
        print(f"  Errors: {validation['errors']}")
    if validation.get("warnings"):
        print(f"  Warnings: {validation['warnings']}")

    print("\nStage log:")
    for entry in state.stage_log:
        print(f"  [{entry.status.value:8}] {entry.stage:12} {entry.detail}")

    print("\n" + "=" * 70)
    print(f"Full state saved to: data/state_{state.run_id}.json")
    print("=" * 70 + "\n")


def run(
    business_name: str,
    industry: str,
    location: str,
    objective: str,
    homepage_url: str = "",
) -> ResearchState:
    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info("Starting full pipeline run for '%s'...", business_name)

    state = run_new(
        business_name=business_name,
        industry=industry,
        location=location,
        objective=objective,
        homepage_url=homepage_url,
        settings=settings,
    )

    _print_summary(state)
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marketing Intelligence Agent")
    parser.add_argument("--business", required=True, help="Business name, e.g. 'Nanavati Toyota'")
    parser.add_argument("--industry", required=True, help="Industry, e.g. 'Automobile'")
    parser.add_argument("--location", required=True, help="Location, e.g. 'Surat, Gujarat'")
    parser.add_argument(
        "--objective", required=True, help="Marketing objective, e.g. 'Increase customer acquisition'"
    )
    parser.add_argument(
        "--homepage-url",
        default="",
        help="The business's real official website (optional, but strongly recommended for real evidence)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        business_name=args.business,
        industry=args.industry,
        location=args.location,
        objective=args.objective,
        homepage_url=args.homepage_url,
    )