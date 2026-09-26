"""
Market/category analysis.
"""

from __future__ import annotations

from app.config import Settings
from app.analysis.common import analyze_dimension


def analyze_market(
    claims: list[dict],
    settings: Settings,
    objective: str,
) -> dict:
    """
    Analyze market/category-level evidence.
    """

    return analyze_dimension(
        dimension="market",
        objective=objective,
        claims=claims,
        settings=settings,
        instructions="""
Analyze the market/category evidence.

Look for:
- category characteristics
- market positioning patterns
- common customer-facing messages
- pricing patterns where evidenced
- emerging themes
- competitive density
- underserved areas
- recurring market needs
- opportunities for differentiation

Do not invent market statistics or trends.

If the collected evidence is insufficient to establish a market pattern,
say so explicitly.
""",
    )