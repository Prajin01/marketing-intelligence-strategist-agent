"""
Thin wrapper around the Anthropic API, used by every Claude-reasoning stage
in Phase 5 (extraction, the five analysis modules, and the strategist).

Kept as a single shared module rather than having each stage call the SDK
directly, so that:
  - The JSON-parsing-and-retry logic is written once, not five times.
  - Tests can mock a single function (_send) instead of patching the SDK
    in five different places.
  - Changing models, token limits, or even providers later touches one file.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import anthropic

from app.config import Settings

logger = logging.getLogger(__name__)

# One retry if Claude's response isn't valid JSON. Deliberately small: if a
# well-specified prompt asking for "ONLY JSON" still fails twice, something
# is wrong enough (a genuinely malformed request, a very unusual input) that
# silently retrying forever would hide a real problem rather than fix one.
_MAX_JSON_RETRIES = 1

DEFAULT_MAX_TOKENS = 2000


class ClaudeJSONError(Exception):
    """Raised when Claude's response could not be parsed as valid JSON after retrying."""


def _send(settings: Settings, system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """
    The only function in this module that actually calls the network.
    Isolated like this specifically so tests can monkeypatch just this one
    function rather than mocking the whole Anthropic SDK client.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def _strip_code_fences(text: str) -> str:
    """Claude sometimes wraps JSON in ```json ... ``` even when told not to; tolerate it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence line (``` or ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def call_claude_json(
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """
    Call Claude and parse its response as JSON.

    Retries once with an explicit correction message if the first response
    isn't valid JSON. If the retry also fails, raises ClaudeJSONError rather
    than returning a fabricated fallback — inventing a plausible-looking
    empty result here would itself be exactly the kind of unearned claim
    this whole project is designed to prevent; callers must handle the
    "extraction genuinely failed for this document" case explicitly.
    """
    resolved_model = model or settings.anthropic_model
    current_user_prompt = user_prompt
    last_error: Optional[Exception] = None

    for attempt in range(_MAX_JSON_RETRIES + 1):
        raw_text = _send(settings, system_prompt, current_user_prompt, resolved_model, max_tokens)
        cleaned = _strip_code_fences(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("Claude response was not valid JSON (attempt %d/%d): %s",
                            attempt + 1, _MAX_JSON_RETRIES + 1, exc)
            current_user_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response could not be parsed as JSON (error: {exc}). "
                f"Respond with ONLY valid JSON — no markdown code fences, no explanation, no other text."
            )

    raise ClaudeJSONError(
        f"Claude did not return valid JSON after {_MAX_JSON_RETRIES + 1} attempt(s). Last error: {last_error}"
    )
