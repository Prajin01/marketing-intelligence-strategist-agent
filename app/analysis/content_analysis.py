"""
Content and campaign communication analysis.
"""

from __future__ import annotations

from app.config import Settings
from app.analysis.common import analyze_dimension


def analyze_content(
    claims: list[dict],
    settings: Settings,
    objective: str,
) -> dict:
    """
    Analyze publicly observable content and campaign patterns.
    """

    return analyze_dimension(
        dimension="content",
        objective=objective,
        claims=claims,
        settings=settings,
        instructions="""
Analyze the content and marketing communication represented in the evidence.

Look for:
- recurring content themes
- campaign themes
- promotional messaging
- educational content
- storytelling
- hooks
- calls to action
- offers
- landing-page messaging
- content formats where explicitly evidenced
- repeated messaging patterns
- content gaps

Identify patterns across multiple pieces of evidence when possible.

Do not claim that a particular platform, format, posting frequency, or campaign
exists unless the supplied evidence supports it.
""",
    )