"""
Tests for app/research/web_research.py and business_research.py.

All network access is mocked (via unittest.mock) so these tests run
offline, deterministically, and don't depend on any real website being up.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.research.business_research import research_business
from app.research.web_research import RateLimiter, WebResearcher, extract_content


SAMPLE_HOMEPAGE_HTML = """
<html>
<head>
    <title>Example Motors</title>
    <meta name="description" content="Your trusted car dealer">
</head>
<body>
    <nav><a href="/about">About Us</a><a href="/contact">Contact</a><a href="/careers">Careers</a></nav>
    <h1>Welcome to Example Motors</h1>
    <p>We sell great cars at great prices.</p>
    <script>console.log('tracking');</script>
    <footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extract_content_pulls_title_meta_text_and_links():
    title, meta, text, links = extract_content(SAMPLE_HOMEPAGE_HTML, "https://example.com")

    assert title == "Example Motors"
    assert meta == "Your trusted car dealer"
    assert "Welcome to Example Motors" in text
    assert "We sell great cars at great prices." in text
    # Script/footer content must be stripped out.
    assert "tracking" not in text
    assert "Copyright" not in text
    assert "https://example.com/about" in links
    assert "https://example.com/contact" in links


def test_rate_limiter_waits_for_same_domain(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

    limiter = RateLimiter(delay_seconds=5.0)
    limiter.wait_if_needed("https://example.com/page1")
    limiter.wait_if_needed("https://example.com/page2")  # same domain -> should wait

    assert len(sleep_calls) == 1
    assert sleep_calls[0] <= 5.0


def _make_settings(tmp_path):
    from app.agent.state import BusinessInput  # noqa: F401  (import kept local to avoid unused warning)
    from app.config import Settings

    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=6,
        request_delay_seconds=0.0,  # no waiting in tests
        request_timeout_seconds=5.0,
        user_agent="TestAgent/0.1",
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        knowledge_dir=tmp_path / "knowledge",
        reports_dir=tmp_path / "reports",
        claims_path=tmp_path / "claims.json",
        log_level="INFO",
    )


@patch("app.research.web_research.robotparser.RobotFileParser.read", lambda self: None)
@patch("app.research.web_research.robotparser.RobotFileParser.can_fetch", lambda self, ua, url: True)
def test_fetch_success(tmp_path):
    settings = _make_settings(tmp_path)
    researcher = WebResearcher(settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.text = SAMPLE_HOMEPAGE_HTML

    with patch.object(researcher._session, "get", return_value=mock_response):
        result = researcher.fetch("https://example.com")

    assert result.ok is True
    assert result.title == "Example Motors"
    assert result.status_code == 200
    assert "great cars" in result.main_text


@patch("app.research.web_research.robotparser.RobotFileParser.read", lambda self: None)
@patch("app.research.web_research.robotparser.RobotFileParser.can_fetch", lambda self, ua, url: False)
def test_fetch_blocked_by_robots(tmp_path):
    settings = _make_settings(tmp_path)
    researcher = WebResearcher(settings)

    result = researcher.fetch("https://example.com/private")

    assert result.ok is False
    assert "robots.txt" in result.error


@patch("app.research.web_research.robotparser.RobotFileParser.read", lambda self: None)
@patch("app.research.web_research.robotparser.RobotFileParser.can_fetch", lambda self, ua, url: True)
def test_fetch_handles_404(tmp_path):
    settings = _make_settings(tmp_path)
    researcher = WebResearcher(settings)

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {"Content-Type": "text/html"}

    with patch.object(researcher._session, "get", return_value=mock_response):
        result = researcher.fetch("https://example.com/missing")

    assert result.ok is False
    assert "404" in result.error


@patch("app.research.web_research.robotparser.RobotFileParser.read", lambda self: None)
@patch("app.research.web_research.robotparser.RobotFileParser.can_fetch", lambda self, ua, url: True)
def test_research_business_fetches_homepage_and_subpages(tmp_path):
    settings = _make_settings(tmp_path)
    researcher = WebResearcher(settings)

    homepage_response = MagicMock()
    homepage_response.status_code = 200
    homepage_response.headers = {"Content-Type": "text/html"}
    homepage_response.text = SAMPLE_HOMEPAGE_HTML

    about_response = MagicMock()
    about_response.status_code = 200
    about_response.headers = {"Content-Type": "text/html"}
    about_response.text = "<html><title>About</title><body><p>We are a family business.</p></body></html>"

    contact_response = MagicMock()
    contact_response.status_code = 200
    contact_response.headers = {"Content-Type": "text/html"}
    contact_response.text = "<html><title>Contact</title><body><p>Call us at 555-1234.</p></body></html>"

    def fake_get(url, timeout):
        if url == "https://example.com":
            return homepage_response
        if url.endswith("/about"):
            return about_response
        if url.endswith("/contact"):
            return contact_response
        raise AssertionError(f"Unexpected URL fetched in test: {url}")

    with patch.object(researcher._session, "get", side_effect=fake_get):
        documents = research_business(researcher, "https://example.com")

    # Homepage + /about + /contact should all be fetched; /careers should
    # NOT be fetched since "careers" isn't in the useful-keyword list.
    urls = [doc["url"] for doc in documents]
    assert "https://example.com" in urls
    assert "https://example.com/about" in urls
    assert "https://example.com/contact" in urls
    assert not any("careers" in u for u in urls)
    assert all(doc["category"] == "business" for doc in documents)
    assert all(doc["ok"] for doc in documents)