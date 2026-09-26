"""
Action-plan generation stage.

Turns strategy and campaigns into an executable marketing roadmap.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json


_SYSTEM_PROMPT = """
You are a marketing operations strategist.

Turn the supplied marketing strategy and campaigns into a practical execution
plan.

Do not invent business facts, budgets, statistics, or capabilities.

Recommendations must be grounded in the supplied evidence and strategy.

Return ONLY valid JSON:

{
  "plan": [
    {
      "phase": "Week 1",
      "objective": "...",
      "actions": [
        {
          "action": "...",
          "channel": "...",
          "reason": "...",
          "claim_ids": []
        }
      ]
    }
  ],
  "kpis": [
    {
      "metric": "...",
      "purpose": "..."
    }
  ],
  "experiments": [
    {
      "experiment": "...",
      "hypothesis": "...",
      "success_signal": "..."
    }
  ]
}
"""


def generate_action_plan(
    *,
    objective: str,
    claims: list[dict[str, Any]],
    strategy: dict[str, Any],
    campaigns: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:

    valid_ids = {
        str(c["claim_id"])
        for c in claims
        if isinstance(c, dict) and c.get("claim_id")
    }

    prompt = f"""
Business objective:
{objective}

Evidence-backed claims:
{claims}

Marketing strategy:
{strategy}

Campaign portfolio:
{campaigns}

Create an execution roadmap.

Structure it into practical phases such as:
- immediate setup
- first launch
- optimization
- scaling

Do not assume a specific advertising budget.

KPIs should be framed as measurements to track rather than invented
performance targets.
"""

    try:
        result = call_claude_json(
            settings=settings,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
    except ClaudeJSONError as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "plan": [],
            "kpis": [],
            "experiments": [],
        }

    if not isinstance(result, dict):
        return {
            "status": "failed",
            "error": "Action planner returned invalid JSON.",
            "plan": [],
            "kpis": [],
            "experiments": [],
        }

    cleaned_plan = []

    for phase in result.get("plan", []):
        if not isinstance(phase, dict):
            continue

        cleaned_actions = []

        for action in phase.get("actions", []):
            if not isinstance(action, dict):
                continue

            refs = action.get("claim_ids", [])

            if not isinstance(refs, list):
                refs = []

            action["claim_ids"] = [
                str(x)
                for x in refs
                if str(x) in valid_ids
            ]

            # FIX: previously an action survived even with zero valid
            # claim_ids after filtering — meaning a fabricated action could
            # reach the final plan with its citations silently stripped,
            # exactly the same gap campaign_generator.py already avoids
            # ("a campaign without evidence-backed reasoning should not
            # survive"). Apply the same rule here for consistency, since
            # this file's own system prompt already requires every action
            # to be evidence-grounded.
            if not action["claim_ids"]:
                continue

            cleaned_actions.append(action)

        # A phase with no surviving actions still records its objective,
        # so a reader can see the phase was planned but nothing in it
        # cleared the evidence bar — consistent with never silently
        # deleting a whole section without a trace.
        phase["actions"] = cleaned_actions
        cleaned_plan.append(phase)

    return {
        "status": "success",
        "plan": cleaned_plan,
        "kpis": result.get("kpis", []),
        "experiments": result.get("experiments", []),
    }