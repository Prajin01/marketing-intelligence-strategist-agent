"""
Tests for app/research/web_discovery.py.

No real search engine or network access is used.
"""

from app.research.web_discovery import (
    WebDiscovery,
    build_discovery_queries,
    classify_url,
)


def _settings():
    from app.config import Settings

    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=3,
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


def test_build_discovery_queries_contains_all_dimensions():
    queries = build_discovery_queries(
        business_name="Nanavati Toyota",
        industry="Automobile",
        location="Surat",
    )

    assert set(queries) == {
        "business",
        "competitor",
        "campaign",
        "content",
        "customer",
        "market",
    }

    assert any("Nanavati Toyota" in q for q in queries["business"])
    assert any("competitors" in q for q in queries["competitor"])
    assert any("campaign" in q for q in queries["campaign"])
    assert any("reviews" in q for q in queries["customer"])
    assert any("market" in q for q in queries["market"])


def test_classify_url_preserves_campaign_category():
    result = classify_url(
        "https://example.com/summer-campaign",
        "campaign",
    )

    assert result == "campaign"


def test_classify_url_detects_content_path():
    result = classify_url(
        "https://example.com/blog/how-to-buy",
        "content",
    )

    assert result == "content"


def test_classify_url_detects_customer_path():
    result = classify_url(
        "https://example.com/customer-reviews",
        "customer",
    )

    assert result == "customer"


def test_classify_url_detects_market_path():
    result = classify_url(
        "https://example.com/market-report-2026",
        "market",
    )

    assert result == "market"


def test_discovery_normalizes_and_deduplicates_urls():
    search_results = [
        {
            "url": "https://example.com/page?utm_source=google",
            "title": "Example Page",
        },
        {
            "url": "https://example.com/page",
            "title": "Duplicate Page",
        },
        {
            "url": "https://example.com/other#section",
            "title": "Other Page",
        },
    ]

    def fake_search(query, max_results):
        return search_results

    discovery = WebDiscovery(
        settings=_settings(),
        search_fn=fake_search,
    )

    results = discovery.search(
        query="example",
        max_results=10,
    )

    assert len(results) == 2
    assert results[0].url == "https://example.com/page"
    assert results[1].url == "https://example.com/other"


def test_discovery_skips_search_engine_urls():
    search_results = [
        {
            "url": "https://www.google.com/search?q=toyota",
            "title": "Google",
        },
        {
            "url": "https://example.com/toyota",
            "title": "Toyota",
        },
    ]

    def fake_search(query, max_results):
        return search_results

    discovery = WebDiscovery(
        settings=_settings(),
        search_fn=fake_search,
    )

    results = discovery.search("toyota")

    assert len(results) == 1
    assert results[0].url == "https://example.com/toyota"


def test_discover_category_applies_category():
    def fake_search(query, max_results):
        return [
            {
                "url": "https://example.com/summer-campaign",
                "title": "Summer Campaign",
            }
        ]

    discovery = WebDiscovery(
        settings=_settings(),
        search_fn=fake_search,
    )

    results = discovery.discover_category(
        category="campaign",
        queries=["Toyota campaign"],
    )

    assert len(results) == 1
    assert results[0].category == "campaign"
    assert results[0].query == "Toyota campaign"


def test_discover_returns_all_research_dimensions():
    def fake_search(query, max_results):
        return [
            {
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "title": query,
            }
        ]

    discovery = WebDiscovery(
        settings=_settings(),
        search_fn=fake_search,
    )

    result = discovery.discover(
        business_name="Nanavati Toyota",
        industry="Automobile",
        location="Surat",
        max_results_per_query=2,
    )

    assert set(result) == {
        "business",
        "competitor",
        "campaign",
        "content",
        "customer",
        "market",
    }

    for category in result:
        assert result[category]


def test_discover_respects_competitor_limit():
    def fake_search(query, max_results):
        return [
            {
                "url": f"https://competitor{i}.example.com",
                "title": f"Competitor {i}",
            }
            for i in range(20)
        ]

    discovery = WebDiscovery(
        settings=_settings(),
        search_fn=fake_search,
    )

    result = discovery.discover(
        business_name="Toyota",
        industry="Automobile",
        location="Surat",
    )

    # max_competitors = 3, therefore competitor discovery cap = 12.
    assert len(result["competitor"]) <= 12


def test_no_search_provider_returns_empty_results():
    discovery = WebDiscovery(settings=_settings())

    result = discovery.search("Toyota")

    assert result == []