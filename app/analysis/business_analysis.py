"""
Business-level marketing analysis.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.analysis.common import analyze_dimension


def analyze_business(
    claims: list[dict[str, Any]],
    settings: Settings,
    objective: str,
) -> dict[str, Any]:
    """
    Analyze the business itself.

    Focus:
    - current positioning
    - products/services
    - value proposition
    - offers
    - locations
    - customer-facing strengths
    - conversion opportunities
    """

    return analyze_dimension(
        dimension="business",
        objective=objective,
        claims=claims,
        settings=settings,
        instructions="""
Study the business itself.

Look for evidence about:
- what the business sells
- services/products
- stated positioning
- value propositions
- locations
- pricing or offers
- customer experience signals
- trust signals
- conversion mechanisms
- differentiators explicitly communicated by the business

Identify what the business appears to emphasize and where its marketing
communication may have opportunities or gaps.

Do not invent strengths or weaknesses that are not supported by the claims.
""",
    )