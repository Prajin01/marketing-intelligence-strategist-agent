"""
Tests for the rewired app/research/auto_research.py.

Covers: the happy path (planner suggests URLs, each category gets fetched),
the no-homepage_url graceful-degrade path, and competitor-cap enforcement.
All network/Claude calls are mocked.
"""

from unittest.mock import patch

from app.config import Settings
from app.research.auto_research import run_autonomous_research


def _settings(max_competitors: int = 6) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=max_competitors,
        request_delay_seconds=0.0,
        request_timeout_seconds=5.0,
        user_agent="TestAgent/0.1",
        data_dir=None,
        raw_dir=None,
        knowledge_dir=None,
        reports_dir=None,
        claims_path=None,
        log_level="INFO",
    )


def _fake_doc(url, category):
    return {
        "url": url,
        "category": category,
        "ok": True,
        "main_text": "content",
        "title": "title",
        "meta_description": None,
        "error": None,
        "fetched_at": "now",
    }


def test_run_autonomous_research_fetches_planner_suggested_urls():
    fake_plan = {
        "competitor_urls": ["https://competitor1.com", "https://competitor2.com"],
        "content_research_urls": ["https://example.com/blog"],
        "customer_research_urls": ["https://example.com/reviews"],
        "market_research_urls": ["https://example.com/market-report"],
        "research_questions": ["What is the positioning?"],
        "status": "success",
    }

    with patch("app.research.auto_research.create_research_plan", return_value=fake_plan), \
         patch("app.research.auto_research.WebResearcher"), \
         patch("app.research.auto_research.research_business", return_value=[_fake_doc("https://nanavatitoyota.com/", "business")]), \
         patch("app.research.auto_research.research_competitors",
               side_effect=lambda r, urls, max_competitors: [_fake_doc(u, "competitors") for u in urls[:max_competitors]]), \
         patch("app.research.auto_research.research_content",
               side_effect=lambda r, urls: [_fake_doc(u, "content") for u in urls]), \
         patch("app.research.auto_research.research_customer_feedback",
               side_effect=lambda r, urls: [_fake_doc(u, "customers") for u in urls]), \
         patch("app.research.auto_research.research_market",
               side_effect=lambda r, urls: [_fake_doc(u, "market") for u in urls]):

        result = run_autonomous_research(
            business_name="Nanavati Toyota",
            industry="Automobile",
            location="Surat, Gujarat",
            objective="Increase customer acquisition",
            homepage_url="https://nanavatitoyota.com/",
            settings=_settings(max_competitors=3),
        )

    assert len(result["documents"]["business"]) == 1
    assert len(result["documents"]["competitor"]) == 2
    assert len(result["documents"]["content"]) == 1
    assert len(result["documents"]["customer"]) == 1
    assert len(result["documents"]["market"]) == 1
    assert len(result["all_documents"]) == 6
    assert result["research_questions"] == ["What is the positioning?"]


def test_run_autonomous_research_skips_business_fetch_without_homepage_url():
    fake_plan = {
        "competitor_urls": [], "content_research_urls": [],
        "customer_research_urls": [], "market_research_urls": [],
        "research_questions": [], "status": "success",
    }

    with patch("app.research.auto_research.create_research_plan", return_value=fake_plan), \
         patch("app.research.auto_research.WebResearcher"), \
         patch("app.research.auto_research.research_business") as mock_business, \
         patch("app.research.auto_research.research_competitors", return_value=[]), \
         patch("app.research.auto_research.research_content", return_value=[]), \
         patch("app.research.auto_research.research_customer_feedback", return_value=[]), \
         patch("app.research.auto_research.research_market", return_value=[]):

        result = run_autonomous_research(
            business_name="X", industry="Y", location="Z", objective="O",
            homepage_url="",  # nothing provided
            settings=_settings(),
        )

    assert result["documents"]["business"] == []
    mock_business.assert_not_called()


def test_run_autonomous_research_enforces_competitor_cap_even_if_planner_suggests_more():
    fake_plan = {
        "competitor_urls": ["https://c1.com", "https://c2.com", "https://c3.com", "https://c4.com"],
        "content_research_urls": [], "customer_research_urls": [], "market_research_urls": [],
        "research_questions": [], "status": "success",
    }

    with patch("app.research.auto_research.create_research_plan", return_value=fake_plan), \
         patch("app.research.auto_research.WebResearcher"), \
         patch("app.research.auto_research.research_business", return_value=[]), \
         patch("app.research.auto_research.research_competitors",
               side_effect=lambda r, urls, max_competitors: [_fake_doc(u, "competitors") for u in urls[:max_competitors]]), \
         patch("app.research.auto_research.research_content", return_value=[]), \
         patch("app.research.auto_research.research_customer_feedback", return_value=[]), \
         patch("app.research.auto_research.research_market", return_value=[]):

        result = run_autonomous_research(
            business_name="X", industry="Y", location="Z", objective="O",
            homepage_url="",
            settings=_settings(max_competitors=2),
        )

    assert len(result["documents"]["competitor"]) == 2  # capped, not 4


def test_run_autonomous_research_survives_planner_failure():
    """A failed planning call should still return an empty-but-valid structure, not crash."""
    failed_plan = {
        "competitor_urls": [], "content_research_urls": [],
        "customer_research_urls": [], "market_research_urls": [],
        "research_questions": [], "status": "failed", "error": "boom",
    }

    with patch("app.research.auto_research.create_research_plan", return_value=failed_plan), \
         patch("app.research.auto_research.WebResearcher"), \
         patch("app.research.auto_research.research_business", return_value=[]), \
         patch("app.research.auto_research.research_competitors", return_value=[]), \
         patch("app.research.auto_research.research_content", return_value=[]), \
         patch("app.research.auto_research.research_customer_feedback", return_value=[]), \
         patch("app.research.auto_research.research_market", return_value=[]):

        result = run_autonomous_research(
            business_name="X", industry="Y", location="Z", objective="O",
            homepage_url="", settings=_settings(),
        )

    assert result["all_documents"] == []