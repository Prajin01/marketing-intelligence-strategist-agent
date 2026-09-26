"""
Tests for app/llm/claude_client.py and app/processing/extractor.py.

All tests mock the network layer (claude_client._send) — no real API key
or network access is used or required.
"""

from typing import Any
from unittest.mock import patch

import pytest

from app.config import Settings
from app.llm.claude_client import ClaudeJSONError, call_claude_json
from app.processing.extractor import (
    extract_claims,
    extract_claims_from_document,
    extract_claims_from_documents,
)


def _settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=6,
        request_delay_seconds=0.0,
        request_timeout_seconds=5.0,
        user_agent="TestAgent/0.1",
        data_dir=None,  # not used by these tests
        raw_dir=None,
        knowledge_dir=None,
        reports_dir=None,
        claims_path=None,
        log_level="INFO",
    )


# --- claude_client.py ---

def test_call_claude_json_parses_clean_json():
    with patch("app.llm.claude_client._send", return_value='[{"a": 1}]'):
        result = call_claude_json(_settings(), "system", "user")
    assert result == [{"a": 1}]


def test_call_claude_json_strips_markdown_fences():
    fenced = '```json\n[{"a": 1}]\n```'
    with patch("app.llm.claude_client._send", return_value=fenced):
        result = call_claude_json(_settings(), "system", "user")
    assert result == [{"a": 1}]


def test_call_claude_json_retries_once_on_bad_json_then_succeeds():
    responses = iter(["not json at all", '[{"a": 1}]'])
    with patch("app.llm.claude_client._send", side_effect=lambda *a, **k: next(responses)):
        result = call_claude_json(_settings(), "system", "user")
    assert result == [{"a": 1}]


def test_call_claude_json_raises_after_exhausting_retries():
    with patch("app.llm.claude_client._send", return_value="still not json"):
        with pytest.raises(ClaudeJSONError):
            call_claude_json(_settings(), "system", "user")


# --- extractor.py ---

_SOURCE_TEXT = (
    "Nanavati Toyota is a dealer in Surat, Bharuch, and Bardoli. "
    "We offer online test drives and a price list of Toyota car models. "
    "Our showroom is located on Magdalla Hazira Road, Surat."
)


def test_extract_claims_accepts_claim_with_valid_evidence():
    claude_response = [
        {
            "claim": "Nanavati Toyota has showrooms in Surat, Bharuch, and Bardoli.",
            "claim_type": "fact",
            "evidence": "Nanavati Toyota is a dealer in Surat, Bharuch, and Bardoli.",
            "confidence": "high",
        }
    ]
    metadata = {"url": "https://nanavatitoyota.com/", "category": "business", "business": "Nanavati Toyota"}

    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims(_SOURCE_TEXT, metadata, _settings())

    assert len(claims) == 1
    assert claims[0]["claim_type"] == "fact"
    assert claims[0]["source_url"] == "https://nanavatitoyota.com/"
    assert claims[0]["claim_id"] == "https://nanavatitoyota.com/#0"


def test_extract_claims_rejects_fabricated_evidence():
    """The core hallucination-prevention check: evidence not in the source must be dropped."""
    claude_response = [
        {
            "claim": "Nanavati Toyota offers free lifetime maintenance.",
            "claim_type": "fact",
            "evidence": "We offer free lifetime maintenance on all vehicles.",  # NOT in _SOURCE_TEXT
            "confidence": "high",
        }
    ]
    metadata = {"url": "https://nanavatitoyota.com/", "category": "business", "business": "Nanavati Toyota"}

    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims(_SOURCE_TEXT, metadata, _settings())

    assert claims == []  # fabricated claim must be rejected, not passed through


def test_extract_claims_rejects_invalid_claim_type():
    claude_response = [
        {
            "claim": "Something",
            "claim_type": "inference",  # not allowed at extraction time
            "evidence": "Nanavati Toyota is a dealer in Surat",
            "confidence": "high",
        }
    ]
    metadata = {"url": "https://example.com/", "category": "business", "business": "Example"}

    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims(_SOURCE_TEXT, metadata, _settings())

    assert claims == []


def test_extract_claims_handles_mixed_valid_and_invalid():
    claude_response = [
        {
            "claim": "Real claim",
            "claim_type": "fact",
            "evidence": "Our showroom is located on Magdalla Hazira Road, Surat.",
            "confidence": "high",
        },
        {
            "claim": "Fake claim",
            "claim_type": "fact",
            "evidence": "This text does not appear anywhere in the source.",
            "confidence": "high",
        },
    ]
    metadata = {"url": "https://example.com/", "category": "business", "business": "Example"}

    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims(_SOURCE_TEXT, metadata, _settings())

    assert len(claims) == 1
    assert claims[0]["claim"] == "Real claim"


def test_extract_claims_handles_empty_source_text():
    assert extract_claims("", {"url": "x"}, _settings()) == []
    assert extract_claims("   ", {"url": "x"}, _settings()) == []


def test_extract_claims_handles_claude_json_error_gracefully():
    from app.llm.claude_client import ClaudeJSONError as _Err

    with patch("app.processing.extractor.call_claude_json", side_effect=_Err("boom")):
        claims = extract_claims(_SOURCE_TEXT, {"url": "https://example.com/"}, _settings())

    assert claims == []  # a failed extraction call must not crash the pipeline


class _FakePost:
    """Minimal stand-in for a frontmatter.Post, avoiding a hard dependency on that library here."""

    def __init__(self, content: str, metadata: dict[str, Any]):
        self.content = content
        self._metadata = metadata

    def get(self, key, default=None):
        return self._metadata.get(key, default)


def test_extract_claims_from_document_skips_unavailable_documents():
    post = _FakePost(
        content="# Information unavailable\n\nCould not retrieve content.",
        metadata={"status": "unavailable", "confidence": "none", "url": "https://blocked.com/"},
    )
    with patch("app.processing.extractor.call_claude_json") as mock_call:
        claims = extract_claims_from_document(post, _settings())

    assert claims == []
    mock_call.assert_not_called()  # must not waste a Claude call on a page we never fetched


def test_extract_claims_from_document_processes_fetched_documents():
    post = _FakePost(
        content=_SOURCE_TEXT,
        metadata={
            "status": "fetched",
            "confidence": "high",
            "url": "https://nanavatitoyota.com/",
            "category": "business",
            "business": "Nanavati Toyota",
        },
    )
    claude_response = [
        {
            "claim": "Dealer in Surat, Bharuch, Bardoli",
            "claim_type": "fact",
            "evidence": "Nanavati Toyota is a dealer in Surat, Bharuch, and Bardoli.",
            "confidence": "high",
        }
    ]
    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims_from_document(post, _settings())

    assert len(claims) == 1
    assert claims[0]["business"] == "Nanavati Toyota"


def test_extract_claims_from_documents_aggregates_across_multiple_posts():
    post1 = _FakePost(
        content=_SOURCE_TEXT,
        metadata={"status": "fetched", "confidence": "high", "url": "https://a.com/", "category": "business", "business": "X"},
    )
    post2 = _FakePost(
        content="",
        metadata={"status": "unavailable", "confidence": "none", "url": "https://b.com/", "category": "business", "business": "X"},
    )
    claude_response = [
        {"claim": "c1", "claim_type": "fact", "evidence": "Nanavati Toyota is a dealer in Surat", "confidence": "high"}
    ]
    with patch("app.processing.extractor.call_claude_json", return_value=claude_response):
        claims = extract_claims_from_documents([post1, post2], _settings())

    assert len(claims) == 1  # post2 contributes nothing, but doesn't error
