"""
Tests for competitor_research.py, customer_research.py, content_research.py,
and market_research.py.

These modules are thin wrappers around web_research.research_website() /
research_urls(), so tests mock WebResearcher.fetch() directly rather than
re-mocking the whole HTTP layer (that's already covered by
test_web_research.py).
"""

from unittest.mock import patch

from app.research.competitor_research import research_competitor, research_competitors
from app.research.content_research import research_content
from app.research.customer_research import research_customer_feedback
from app.research.market_research import research_market
from app.research.web_research import FetchResult, WebResearcher


def _fake_ok_result(url: str) -> FetchResult:
    return FetchResult(
        url=url,
        ok=True,
        status_code=200,
        title="Fake Page",
        main_text="Some fetched content.",
        links=[],
    )


def _make_researcher() -> WebResearcher:
    # WebResearcher.__init__ needs a Settings object with real-looking
    # fields; we only ever call .fetch() on it (which we mock), so a bare
    # object with the attributes it touches is enough here.
    class _FakeSettings:
        user_agent = "TestAgent/0.1"
        request_delay_seconds = 0.0
        request_timeout_seconds = 5.0

    return WebResearcher(_FakeSettings())


def test_research_competitor_tags_category_correctly():
    researcher = _make_researcher()
    with patch.object(researcher, "fetch", side_effect=lambda url: _fake_ok_result(url)):
        docs = research_competitor(researcher, "https://competitor-a.com")

    assert len(docs) == 1
    assert docs[0]["category"] == "competitors"
    assert docs[0]["url"] == "https://competitor-a.com"


def test_research_competitors_respects_max_competitors_cap():
    researcher = _make_researcher()
    urls = [f"https://competitor-{i}.com" for i in range(10)]

    with patch.object(researcher, "fetch", side_effect=lambda url: _fake_ok_result(url)) as mock_fetch:
        docs = research_competitors(researcher, urls, max_competitors=3)

    # Only the first 3 competitors should have been fetched at all.
    assert mock_fetch.call_count == 3
    assert len(docs) == 3
    assert all(doc["category"] == "competitors" for doc in docs)


def test_research_customer_feedback_tags_category_correctly():
    researcher = _make_researcher()
    urls = ["https://reviews.example.com/business-x"]

    with patch.object(researcher, "fetch", side_effect=lambda url: _fake_ok_result(url)):
        docs = research_customer_feedback(researcher, urls)

    assert len(docs) == 1
    assert docs[0]["category"] == "customers"


def test_research_content_tags_category_correctly():
    researcher = _make_researcher()
    urls = ["https://example.com/blog/post-1"]

    with patch.object(researcher, "fetch", side_effect=lambda url: _fake_ok_result(url)):
        docs = research_content(researcher, urls)

    assert len(docs) == 1
    assert docs[0]["category"] == "content"


def test_research_market_tags_category_correctly():
    researcher = _make_researcher()
    urls = ["https://industry-news.example.com/trends-2026"]

    with patch.object(researcher, "fetch", side_effect=lambda url: _fake_ok_result(url)):
        docs = research_market(researcher, urls)

    assert len(docs) == 1
    assert docs[0]["category"] == "market"


def test_failed_fetch_is_still_recorded_not_dropped():
    """
    A failed fetch (robots.txt block, timeout, etc.) must still appear in
    the results with ok=False — never silently disappear. This is the
    "record 'information unavailable' rather than hallucinate" requirement,
    tested at the research-module level.
    """
    researcher = _make_researcher()

    def fake_fetch(url: str) -> FetchResult:
        return FetchResult(url=url, ok=False, error="Information unavailable — robots.txt disallows access")

    with patch.object(researcher, "fetch", side_effect=fake_fetch):
        docs = research_market(researcher, ["https://blocked.example.com"])

    assert len(docs) == 1
    assert docs[0]["ok"] is False
    assert "robots.txt" in docs[0]["error"]
