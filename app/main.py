"""
Basic entry point for the Marketing Intelligence Agent.

Phase 2 scope only: load config, accept a business as input, create the
initial ResearchState, and persist it to disk. This proves the skeleton
(config -> state -> storage) works end to end before any research or LLM
logic is added in later phases.

Usage:
    python -m app.main --business "Toyota Surat" --industry "Automobile" \\
        --location "Surat, Gujarat" --objective "Increase customer acquisition"

Once Phase 6 (agent orchestration) is implemented, this same file will
grow into the real pipeline runner. It is kept as a plain function (not
argparse-boilerplate spread everywhere) so Streamlit (Phase 8) can import
and call `run(...)` directly instead of shelling out to a CLI.
"""

from __future__ import annotations

import argparse
import logging

from app.agent.state import BusinessInput, ResearchState, StageStatus
from app.config import configure_logging, load_settings

logger = logging.getLogger(__name__)


def run(business_name: str, industry: str, location: str, objective: str) -> ResearchState:
    settings = load_settings()
    configure_logging(settings.log_level)

    from app.agent.orchestrator import run_new

    return run_new(
        business_name=business_name,
        industry=industry,
        location=location,
        objective=objective,
        settings=settings,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marketing Intelligence Agent")
    parser.add_argument("--business", required=True, help="Business name, e.g. 'Toyota Surat'")
    parser.add_argument("--industry", required=True, help="Industry, e.g. 'Automobile'")
    parser.add_argument("--location", required=True, help="Location, e.g. 'Surat, Gujarat'")
    parser.add_argument(
        "--objective", required=True, help="Marketing objective, e.g. 'Increase customer acquisition'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        business_name=args.business,
        industry=args.industry,
        location=args.location,
        objective=args.objective,
    )
