"""
Customer and audience analysis.
"""

from __future__ import annotations

from app.config import Settings
from app.analysis.common import analyze_dimension


def analyze_customers(
    claims: list[dict],
    settings: Settings,
    objective: str,
) -> dict:
    """
    Analyze publicly available customer/audience evidence.
    """

    return analyze_dimension(
        dimension="customer",
        objective=objective,
        claims=claims,
        settings=settings,
        instructions="""
Analyze evidence related to customers and audience behavior.

Look for:
- customer needs
- pain points
- desired outcomes
- objections
- frequently mentioned questions
- customer language
- reviews or feedback
- buying motivations
- customer segments
- purchase barriers

Only infer customer motivations when the supplied claims provide reasonable
support.

If customer evidence is weak or unavailable, explicitly reflect that
limitation rather than creating a hypothetical persona.
""",
    )