from unittest.mock import patch

from app.research.research_planner import create_research_plan


def _settings():
    from app.config import Settings

    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-5",
        anthropic_strategy_model="claude-sonnet-5",
        max_competitors=6,
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


def test_research_planner_normalizes_urls():
    response = {
        "research_objectives": [
            "business",
            "competitors",
            "customers",
            "content",
            "market",
        ],
        "competitor_urls": [
            "https://example.com",
            "https://example.com",
            "not-a-url",
        ],
        "content_research_urls": [
            "https://example.com/blog",
        ],
        "customer_research_urls": [],
        "market_research_urls": [],
        "research_questions": [
            "What is the current positioning?",
            "What do competitors emphasize?",
        ],
    }

    with patch(
        "app.research.research_planner.call_claude_json",
        return_value=response,
    ):
        result = create_research_plan(
            business_name="Example Brand",
            industry="Automobile",
            location="Surat, Gujarat",
            objective="Increase customer acquisition",
            homepage_url="https://example.com",
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert result["competitor_urls"] == ["https://example.com"]
    assert result["content_research_urls"] == [
        "https://example.com/blog"
    ]
    assert len(result["research_questions"]) == 2


def test_research_planner_rejects_invalid_urls():
    response = {
        "competitor_urls": [
            "example.com",
            "",
            None,
            "ftp://example.com",
            "https://valid.com",
        ],
        "content_research_urls": [],
        "customer_research_urls": [],
        "market_research_urls": [],
        "research_questions": [],
    }

    with patch(
        "app.research.research_planner.call_claude_json",
        return_value=response,
    ):
        result = create_research_plan(
            business_name="Example",
            industry="Retail",
            location="India",
            objective="Growth",
            homepage_url="https://example.com",
            settings=_settings(),
        )

    assert result["competitor_urls"] == [
        "https://valid.com"
    ]


def test_research_planner_handles_llm_failure():
    from app.llm.claude_client import ClaudeJSONError

    with patch(
        "app.research.research_planner.call_claude_json",
        side_effect=ClaudeJSONError("boom"),
    ):
        result = create_research_plan(
            business_name="Example",
            industry="Retail",
            location="India",
            objective="Growth",
            homepage_url="https://example.com",
            settings=_settings(),
        )

    assert result["status"] == "failed"
    assert result["competitor_urls"] == []