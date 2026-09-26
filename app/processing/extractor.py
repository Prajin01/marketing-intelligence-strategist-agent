"""
Extraction stage (Phase 5): turns one source document's text into typed
claims that later stages (analysis, strategy) can build on.

This is the ONLY place FACT and OBSERVATION claims may be created — see
the ClaimType discipline in app/agent/state.py and the Phase 1 architecture
notes. Every claim Claude proposes is checked against the actual source
text before being accepted: if the "evidence" Claude cites can't genuinely
be found in the document, the claim is dropped. This is the mechanical
hallucination-prevention layer — not a prompt instruction Claude could
simply ignore, but a check the code enforces regardless of what Claude
returns.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json

logger = logging.getLogger(__name__)

_VALID_CLAIM_TYPES = {"fact", "observation"}

_SYSTEM_PROMPT = """You are a careful research analyst extracting claims from a single source document for a marketing intelligence report.

Rules:
- Extract only claims that are DIRECTLY stated in the provided text. Do not infer, guess, or add outside knowledge.
- Each claim must be either:
  - "fact": a specific, directly-stated piece of information (a location, a product, a price, a stated policy, a service offered, etc).
  - "observation": a pattern you notice ACROSS MULTIPLE PARTS of this same document (e.g. "the page repeatedly emphasizes customer service" — only use this when the pattern is genuinely visible in more than one place in the text).
- For every claim, include "evidence": a short, VERBATIM excerpt (under 200 characters) copied EXACTLY from the source text that supports the claim. Do not paraphrase the evidence field — copy it character-for-character from the source.
- If the document has little or no useful marketing-relevant information, return an empty JSON array.
- Respond with ONLY a JSON array, no other text, no markdown fences. Each item must have exactly these fields:
  {"claim": string, "claim_type": "fact" or "observation", "evidence": string, "confidence": "high", "medium", or "low"}
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _evidence_supported(evidence: str, source_text: str) -> bool:
    """
    The structural evidence check described in Phase 1: is Claude's cited
    excerpt actually present in the source? Whitespace and case differences
    are tolerated (Claude may reformat line breaks when copying), but the
    words themselves must genuinely appear, in order, in the source — this
    is a real check, not a rubber stamp.
    """
    if not evidence or not evidence.strip():
        return False
    return _normalize(evidence) in _normalize(source_text)


def extract_claims(
    source_text: str,
    source_metadata: dict[str, Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Extract FACT/OBSERVATION claims from one document's raw text.

    `source_metadata` should include at least: url, category, business.

    Returns a list of claim dicts: claim_id, claim, claim_type, evidence,
    confidence, source_url, category, business. Claims whose cited evidence
    cannot be found in source_text are silently dropped from the returned
    list but logged as rejections (not passed through with a "rejected"
    flag) — a rejected claim has zero evidentiary value and should not
    reach any later stage at all, exactly like a page that failed to fetch.

    Kept independent of the frontmatter/file-storage layer on purpose —
    see extract_claims_from_document() below for the adapter — so this
    core logic can be tested (and reasoned about) without any dependency
    on how documents happen to be stored on disk.
    """
    if not source_text or not source_text.strip():
        return []

    user_prompt = f"Source document text:\n\n{source_text}"

    try:
        raw_claims = call_claude_json(settings, _SYSTEM_PROMPT, user_prompt)
    except ClaudeJSONError as exc:
        logger.error("Extraction failed for %s: %s", source_metadata.get("url"), exc)
        return []

    if not isinstance(raw_claims, list):
        logger.error(
            "Extraction for %s returned non-list JSON (%s), discarding.",
            source_metadata.get("url"), type(raw_claims).__name__,
        )
        return []

    accepted: list[dict[str, Any]] = []
    rejected_count = 0

    for i, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            rejected_count += 1
            continue

        claim_text = item.get("claim")
        claim_type = item.get("claim_type")
        evidence = item.get("evidence", "")
        confidence = item.get("confidence", "medium")

        if not claim_text or claim_type not in _VALID_CLAIM_TYPES:
            logger.warning(
                "Rejected malformed claim from %s: claim=%r claim_type=%r",
                source_metadata.get("url"), claim_text, claim_type,
            )
            rejected_count += 1
            continue

        if not _evidence_supported(evidence, source_text):
            logger.warning(
                "Rejected claim (evidence not found in source %s): claim=%r evidence=%r",
                source_metadata.get("url"), claim_text, evidence,
            )
            rejected_count += 1
            continue

        accepted.append({
            "claim_id": f"{source_metadata.get('url', 'unknown-source')}#{i}",
            "claim": claim_text,
            "claim_type": claim_type,
            "evidence": evidence,
            "confidence": confidence,
            "source_url": source_metadata.get("url"),
            "category": source_metadata.get("category"),
            "business": source_metadata.get("business"),
        })

    if rejected_count:
        logger.info(
            "Extraction for %s: %d claim(s) accepted, %d rejected.",
            source_metadata.get("url"), len(accepted), rejected_count,
        )

    return accepted


def extract_claims_from_document(post: Any, settings: Settings) -> list[dict[str, Any]]:
    """
    Adapter from a frontmatter.Post (as loaded by markdown_processor.py)
    to extract_claims(). Kept separate so the core extraction logic above
    has zero dependency on the frontmatter library.

    A document that never fetched successfully (status != "fetched", or
    confidence == "none") is skipped entirely rather than sent to Claude —
    its content is just an "Information unavailable" placeholder, and
    sending that to an LLM risks it inventing claims from nothing, which is
    exactly what this whole extraction stage exists to prevent.
    """
    if post.get("status") != "fetched" or post.get("confidence") == "none":
        logger.debug("Skipping extraction for unavailable document: %s", post.get("url"))
        return []

    source_metadata = {
        "url": post.get("url"),
        "category": post.get("category"),
        "business": post.get("business"),
    }
    return extract_claims(post.content, source_metadata, settings)


def extract_claims_from_documents(
    posts: list[Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Convenience wrapper: run extraction across every loaded document."""
    all_claims: list[dict[str, Any]] = []
    for post in posts:
        all_claims.extend(extract_claims_from_document(post, settings))
    return all_claims
