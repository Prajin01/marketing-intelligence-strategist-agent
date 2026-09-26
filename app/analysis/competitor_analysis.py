"""
Competitor marketing analysis.
"""

from __future__ import annotations

from app.config import Settings
from app.analysis.common import analyze_dimension


def analyze_competitors(
    claims: list[dict],
    settings: Settings,
    objective: str,
) -> dict:
    """
    Analyze competitor-related evidence.

    Focus:
    - competitor positioning
    - offers
    - messaging
    - differentiation
    - visible campaign patterns
    - competitive gaps
    """

    return analyze_dimension(
        dimension="competitor",
        objective=objective,
        claims=claims,
        settings=settings,
        instructions="""
Analyze the competitive landscape represented by the evidence.

Look for:
- competitor positioning
- competitor value propositions
- products/services
- publicly visible offers
- pricing where available
- campaign themes
- promotional messaging
- calls to action
- trust signals
- content themes
- similarities between competitors
- potential whitespace or differentiation opportunities

Pay particular attention to repeated patterns across competitor claims.

Do not declare that a competitor is better or worse.
Instead identify documented patterns and evidence-supported opportunities
for differentiation.
""",
    )