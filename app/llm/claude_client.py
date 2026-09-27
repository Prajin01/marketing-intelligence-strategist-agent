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

from app.config import Settings

logger = logging.getLogger(__name__)

# Retries if the LLM's response isn't valid JSON. Raised from 1 to 2 after
# real-world testing on Groq's free-tier open-weight models: those models
# tend to write noticeably longer, more verbose JSON than Claude for the
# same structured task, and combined with the token-limit issue below, the
# first retry alone wasn't always enough headroom to succeed.
_MAX_JSON_RETRIES = 2

# Raised from 2000 to 6000 after a real live run showed a very specific,
# diagnosable failure pattern: repeated "Unterminated string starting at..."
# JSON errors — the exact signature of a response being cut off mid-string
# because it hit the token limit before finishing. 2000 was sized for
# Claude's typically more concise output; open-weight models used via the
# free Groq substitution (see README) tend to be more verbose for the same
# prompt, especially for the largest-output stages (strategy, campaigns,
# action plan, each combining many claims/analyses into one response).
DEFAULT_MAX_TOKENS = 6000


class ClaudeJSONError(Exception):
    """Raised when Claude's response could not be parsed as valid JSON after retrying."""


def _send(settings: Settings, system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """
    Dispatches to whichever LLM provider is active (see Settings.llm_provider).
    Isolated like this specifically so tests can monkeypatch just this one
    function rather than mocking a whole SDK client, regardless of provider.
    """
    if settings.llm_provider == "groq":
        return _send_groq(settings, system_prompt, user_prompt, model, max_tokens)
    return _send_anthropic(settings, system_prompt, user_prompt, model, max_tokens)


def _send_anthropic(settings: Settings, system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    import anthropic  # lazy import: only required when this provider is actually used

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def _send_groq(settings: Settings, system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """
    Groq hosts open-weight models (Llama, Mixtral, etc.) with a free tier
    and an OpenAI-compatible chat-completions API. Used here as a
    zero-cost substitute for the Claude API when no API budget is
    available — see README for the disclosure of this substitution and
    why it was made.
    """
    from groq import Groq  # lazy import: only required when this provider is actually used

    client = Groq(api_key=settings.anthropic_api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


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