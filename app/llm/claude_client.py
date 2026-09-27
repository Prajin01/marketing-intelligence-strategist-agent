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
import re
import time
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

# Groq's free tier enforces a tokens-per-minute (TPM) limit per model. A
# full research run makes many LLM calls back to back, so hitting that
# limit mid-run is expected rather than exceptional. Groq's 429 response
# says exactly how long to wait ("Please try again in 28.5s"), so the
# right behaviour is to wait that long and retry — not to fail the run.
_GROQ_RATE_LIMIT_RETRIES = 5
_GROQ_MAX_WAIT_SECONDS = 65.0  # TPM windows reset each minute; never wait longer than that
_GROQ_DEFAULT_WAIT_SECONDS = 20.0

# Guard against a single request being too large for the free-tier TPM
# window. Roughly 4 characters per token, so 16,000 chars is ~4,000 input
# tokens. The start and end of the prompt are kept (that's where the task
# instructions and output format usually live) and the middle is trimmed.
_GROQ_MAX_PROMPT_CHARS = 16000
_GROQ_PROMPT_TAIL_CHARS = 3000


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


def _groq_wait_seconds(exc: Exception) -> float:
    """
    Work out how long Groq asked us to wait. Prefers the Retry-After header,
    falls back to parsing the human-readable message (formats seen in the
    wild: "28.53s", "1m2.5s", "520ms"), and finally a safe default.
    """
    try:
        header = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        if header:
            return min(float(header) + 1.0, _GROQ_MAX_WAIT_SECONDS)
    except Exception:  # noqa: BLE001 - header parsing is best-effort
        pass

    match = re.search(r"try again in (?:(\d+)m(?!s))?([\d.]+)(ms|s)", str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        amount = float(match.group(2))
        seconds = amount / 1000.0 if match.group(3) == "ms" else amount
        return min(minutes * 60.0 + seconds + 1.0, _GROQ_MAX_WAIT_SECONDS)

    return _GROQ_DEFAULT_WAIT_SECONDS


def _trim_prompt_for_groq(user_prompt: str) -> str:
    """Keep the head and tail of an oversized prompt and trim the middle."""
    if len(user_prompt) <= _GROQ_MAX_PROMPT_CHARS:
        return user_prompt
    head_chars = _GROQ_MAX_PROMPT_CHARS - _GROQ_PROMPT_TAIL_CHARS
    logger.warning(
        "Prompt is %d chars; trimming to ~%d to fit Groq free-tier limits.",
        len(user_prompt), _GROQ_MAX_PROMPT_CHARS,
    )
    return (
        user_prompt[:head_chars]
        + "\n\n[... source text trimmed to fit model limits ...]\n\n"
        + user_prompt[-_GROQ_PROMPT_TAIL_CHARS:]
    )


def _send_groq(settings: Settings, system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """
    Groq hosts open-weight models (Llama, Mixtral, etc.) with a free tier
    and an OpenAI-compatible chat-completions API. Used here as a
    zero-cost substitute for the Claude API when no API budget is
    available — see README for the disclosure of this substitution and
    why it was made.

    Free-tier rate limits (HTTP 429) are handled by waiting the time Groq
    asks for and retrying. Daily limits are not retried, since waiting a
    minute cannot fix them.
    """
    from groq import Groq, RateLimitError  # lazy import: only required when this provider is actually used

    client = Groq(api_key=settings.anthropic_api_key, max_retries=0)
    safe_prompt = _trim_prompt_for_groq(user_prompt)

    for attempt in range(_GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            message = str(exc).lower()
            if "per day" in message or attempt == _GROQ_RATE_LIMIT_RETRIES:
                raise
            wait = _groq_wait_seconds(exc)
            logger.warning(
                "Groq rate limit hit (attempt %d/%d); waiting %.1fs before retrying.",
                attempt + 1, _GROQ_RATE_LIMIT_RETRIES + 1, wait,
            )
            time.sleep(wait)

    raise RuntimeError("unreachable")  # loop always returns or raises


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