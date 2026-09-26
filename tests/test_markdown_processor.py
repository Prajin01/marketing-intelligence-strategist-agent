"""
Tests for app/processing/cleaner.py, metadata.py, and markdown_processor.py.
"""

from app.processing.cleaner import clean_text
from app.processing.markdown_processor import (
    build_knowledge_post,
    load_knowledge_documents,
    save_knowledge_document,
    save_raw_documents,
)
from app.processing.metadata import confidence_for, slugify_url


# --- cleaner.py ---

def test_clean_text_removes_duplicate_consecutive_lines():
    text = "Welcome\nWelcome\nWelcome\nAbout us\nAbout us\nContact"
    cleaned = clean_text(text)
    assert cleaned == "Welcome\nAbout us\nContact"


def test_clean_text_drops_blank_lines():
    text = "Line one\n\n\nLine two\n   \nLine three"
    cleaned = clean_text(text)
    assert cleaned == "Line one\nLine two\nLine three"


def test_clean_text_truncates_long_text():
    text = "a" * 100
    cleaned = clean_text(text, max_chars=10)
    assert cleaned.startswith("a" * 10)
    assert "truncated" in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


# --- metadata.py ---

def test_slugify_url_is_filesystem_safe_and_deterministic():
    url = "https://nanavatitoyota.com/contact-su01a.html"
    slug1 = slugify_url(url)
    slug2 = slugify_url(url)
    assert slug1 == slug2  # deterministic
    assert " " not in slug1
    assert "/" not in slug1
    assert slug1.islower() or not slug1.isalpha()


def test_slugify_url_avoids_collisions_for_different_urls_same_prefix():
    slug_a = slugify_url("https://example.com/about")
    slug_b = slugify_url("https://example.com/about?ref=footer")
    # Different URLs should not collide even if their "cleaned" prefixes
    # would otherwise look identical, thanks to the hash suffix.
    assert slug_a != slug_b


def test_confidence_for_successful_document_is_high():
    doc = {"ok": True, "main_text": "Some real content here."}
    assert confidence_for(doc) == "high"


def test_confidence_for_failed_document_is_none():
    doc = {"ok": False, "error": "Information unavailable — robots.txt disallows access"}
    assert confidence_for(doc) == "none"


def test_confidence_for_ok_but_empty_text_is_none():
    doc = {"ok": True, "main_text": "   "}
    assert confidence_for(doc) == "none"


# --- markdown_processor.py ---

def _successful_document(url="https://example.com/about"):
    return {
        "url": url,
        "category": "business",
        "ok": True,
        "status_code": 200,
        "title": "About Example Motors",
        "meta_description": "Learn about us",
        "main_text": "We are a family-owned dealership since 1990.",
        "error": None,
        "fetched_at": "2026-09-24T10:00:00+00:00",
    }


def _failed_document(url="https://example.com/blocked"):
    return {
        "url": url,
        "category": "competitors",
        "ok": False,
        "status_code": None,
        "title": None,
        "meta_description": None,
        "main_text": None,
        "error": "Information unavailable — robots.txt disallows access",
        "fetched_at": "2026-09-24T10:00:00+00:00",
    }


def test_build_knowledge_post_for_successful_document():
    post = build_knowledge_post(_successful_document(), business_name="Example Motors")

    assert post["category"] == "business"
    assert post["business"] == "Example Motors"
    assert post["confidence"] == "high"
    assert post["status"] == "fetched"
    assert "error" not in post.metadata
    assert "family-owned dealership" in post.content


def test_build_knowledge_post_for_failed_document_records_the_gap():
    post = build_knowledge_post(_failed_document(), business_name="Example Motors")

    assert post["confidence"] == "none"
    assert post["status"] == "unavailable"
    assert post["error"] == "Information unavailable — robots.txt disallows access"
    assert "Information unavailable" in post.content


def test_save_and_load_knowledge_document_roundtrip(tmp_path):
    from app.config import Settings

    settings = Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=6,
        request_delay_seconds=0.0,
        request_timeout_seconds=5.0,
        user_agent="TestAgent/0.1",
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        knowledge_dir=tmp_path / "knowledge",
        reports_dir=tmp_path / "reports",
        claims_path=tmp_path / "claims.json",
        log_level="INFO",
    )

    post = build_knowledge_post(_successful_document(), business_name="Example Motors")
    saved_path = save_knowledge_document(post, settings)

    assert saved_path.exists()
    assert saved_path.suffix == ".md"
    assert saved_path.parent.name == "business"

    loaded = load_knowledge_documents(settings, category="business")
    assert len(loaded) == 1
    assert loaded[0]["business"] == "Example Motors"
    assert "family-owned dealership" in loaded[0].content
    assert loaded[0].file_path == saved_path


def test_save_raw_documents_handles_mixed_success_and_failure(tmp_path):
    from app.config import Settings

    settings = Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=6,
        request_delay_seconds=0.0,
        request_timeout_seconds=5.0,
        user_agent="TestAgent/0.1",
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        knowledge_dir=tmp_path / "knowledge",
        reports_dir=tmp_path / "reports",
        claims_path=tmp_path / "claims.json",
        log_level="INFO",
    )

    documents = [_successful_document(), _failed_document()]
    paths = save_raw_documents(documents, business_name="Example Motors", settings=settings)

    assert len(paths) == 2
    all_docs = load_knowledge_documents(settings)
    assert len(all_docs) == 2
    confidences = {doc["confidence"] for doc in all_docs}
    assert confidences == {"high", "none"}
