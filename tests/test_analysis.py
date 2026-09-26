"""
Tests for Phase 5 analysis and strategy modules.

All Claude calls are mocked. These tests do NOT require a real Anthropic API
key or network access.
"""

from unittest.mock import patch

from app.analysis.common import analyze_dimension
from app.analysis.business_analysis import analyze_business
from app.analysis.competitor_analysis import analyze_competitors
from app.analysis.customer_analysis import analyze_customers
from app.analysis.content_analysis import analyze_content
from app.analysis.market_analysis import analyze_market

from app.strategy.strategist import generate_strategy
from app.strategy.campaign_generator import generate_campaigns
from app.strategy.action_plan import generate_action_plan

from app.config import Settings


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

def _settings() -> Settings:
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


# ---------------------------------------------------------------------------
# Test evidence
# ---------------------------------------------------------------------------

CLAIMS = [
    {
        "claim_id": "https://example.com/#0",
        "claim": "The business offers online consultations.",
        "claim_type": "fact",
        "evidence": "We offer online consultations.",
        "confidence": "high",
        "source_url": "https://example.com/",
        "category": "business",
        "business": "Example Brand",
    },
    {
        "claim_id": "https://example.com/#1",
        "claim": "The business operates in Surat.",
        "claim_type": "fact",
        "evidence": "Our showroom is located in Surat.",
        "confidence": "high",
        "source_url": "https://example.com/",
        "category": "business",
        "business": "Example Brand",
    },
    {
        "claim_id": "https://competitor.com/#0",
        "claim": "Competitor X promotes same-day delivery.",
        "claim_type": "fact",
        "evidence": "Same-day delivery available.",
        "confidence": "high",
        "source_url": "https://competitor.com/",
        "category": "competitor",
        "business": "Competitor X",
    },
]


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

def test_analyze_dimension_returns_empty_result_for_no_claims():
    result = analyze_dimension(
        dimension="business",
        objective="Increase customer acquisition",
        claims=[],
        settings=_settings(),
        instructions="Analyze the business.",
    )

    assert result["dimension"] == "business"
    assert result["insights"] == []
    assert result["opportunities"] == []
    assert result["gaps"] == []
    assert result["recommendations"] == []


def test_analyze_dimension_accepts_valid_claim_references():
    claude_response = {
        "dimension": "business",
        "summary": "The business has an online service capability.",
        "insights": [
            {
                "finding": "The online consultation service may support digital acquisition.",
                "type": "inference",
                "claim_ids": ["https://example.com/#0"],
            }
        ],
        "opportunities": [
            {
                "opportunity": "Emphasize online consultations.",
                "reason": "The business already offers this service.",
                "claim_ids": ["https://example.com/#0"],
            }
        ],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_dimension(
            dimension="business",
            objective="Increase customer acquisition",
            claims=CLAIMS,
            settings=_settings(),
            instructions="Analyze the business.",
        )

    assert len(result["insights"]) == 1
    assert result["insights"][0]["claim_ids"] == [
        "https://example.com/#0"
    ]

    assert len(result["opportunities"]) == 1


def test_analyze_dimension_removes_fake_claim_references():
    claude_response = {
        "dimension": "business",
        "summary": "Test analysis.",
        "insights": [
            {
                "finding": "Unsupported insight.",
                "type": "inference",
                "claim_ids": ["fake-claim-id"],
            },
            {
                "finding": "Supported insight.",
                "type": "inference",
                "claim_ids": ["https://example.com/#0"],
            },
        ],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_dimension(
            dimension="business",
            objective="Increase sales",
            claims=CLAIMS,
            settings=_settings(),
            instructions="Analyze the business.",
        )

    assert len(result["insights"]) == 1
    assert result["insights"][0]["finding"] == "Supported insight."


def test_analyze_dimension_drops_items_without_claim_references():
    claude_response = {
        "dimension": "business",
        "summary": "Test analysis.",
        "insights": [
            {
                "finding": "No evidence.",
                "type": "inference",
                "claim_ids": [],
            }
        ],
        "opportunities": [
            {
                "opportunity": "No evidence.",
                "reason": "No evidence.",
                "claim_ids": [],
            }
        ],
        "gaps": [
            {
                "gap": "No evidence.",
                "why_it_matters": "No evidence.",
                "claim_ids": [],
            }
        ],
        "recommendations": [
            {
                "recommendation": "No evidence.",
                "reason": "No evidence.",
                "claim_ids": [],
            }
        ],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_dimension(
            dimension="business",
            objective="Increase sales",
            claims=CLAIMS,
            settings=_settings(),
            instructions="Analyze the business.",
        )

    assert result["insights"] == []
    assert result["opportunities"] == []
    assert result["gaps"] == []
    assert result["recommendations"] == []


# ---------------------------------------------------------------------------
# Five analysis modules
# ---------------------------------------------------------------------------

def test_business_analysis_uses_common_analysis_engine():
    claude_response = {
        "dimension": "business",
        "summary": "Business summary.",
        "insights": [
            {
                "finding": "The business offers online consultations.",
                "type": "inference",
                "claim_ids": ["https://example.com/#0"],
            }
        ],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_business(
            CLAIMS,
            _settings(),
            "Increase customer acquisition",
        )

    assert result["dimension"] == "business"
    assert len(result["insights"]) == 1


def test_competitor_analysis_works():
    claude_response = {
        "dimension": "competitor",
        "summary": "Competitor summary.",
        "insights": [
            {
                "finding": "A competitor promotes same-day delivery.",
                "type": "inference",
                "claim_ids": ["https://competitor.com/#0"],
            }
        ],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_competitors(
            CLAIMS,
            _settings(),
            "Increase customer acquisition",
        )

    assert result["dimension"] == "competitor"


def test_customer_analysis_works():
    claude_response = {
        "dimension": "customer",
        "summary": "Customer summary.",
        "insights": [],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_customers(
            CLAIMS,
            _settings(),
            "Increase customer acquisition",
        )

    assert result["dimension"] == "customer"


def test_content_analysis_works():
    claude_response = {
        "dimension": "content",
        "summary": "Content summary.",
        "insights": [],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_content(
            CLAIMS,
            _settings(),
            "Increase customer acquisition",
        )

    assert result["dimension"] == "content"


def test_market_analysis_works():
    claude_response = {
        "dimension": "market",
        "summary": "Market summary.",
        "insights": [],
        "opportunities": [],
        "gaps": [],
        "recommendations": [],
    }

    with patch(
        "app.analysis.common.call_claude_json",
        return_value=claude_response,
    ):
        result = analyze_market(
            CLAIMS,
            _settings(),
            "Increase customer acquisition",
        )

    assert result["dimension"] == "market"


# ---------------------------------------------------------------------------
# Strategist
# ---------------------------------------------------------------------------

def test_strategist_preserves_valid_claim_references():
    strategy_response = {
        "positioning": {
            "statement": "Position the business around convenient digital access.",
            "reasoning": "The business offers online consultations.",
            "claim_ids": ["https://example.com/#0"],
        },
        "target_audience": [
            {
                "segment": "Customers seeking convenient access.",
                "reasoning": "Online consultations are available.",
                "claim_ids": ["https://example.com/#0"],
            }
        ],
        "messaging": [],
        "channels": [],
        "strategic_priorities": [],
    }

    analysis = {
        "business": {
            "dimension": "business",
            "summary": "Business analysis",
            "insights": [],
            "opportunities": [],
            "gaps": [],
            "recommendations": [],
        }
    }

    with patch(
        "app.strategy.strategist.call_claude_json",
        return_value=strategy_response,
    ):
        result = generate_strategy(
            business_name="Example Brand",
            industry="Services",
            location="Surat",
            objective="Increase customer acquisition",
            claims=CLAIMS,
            analysis=analysis,
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert result["positioning"]["claim_ids"] == [
        "https://example.com/#0"
    ]


def test_strategist_removes_fake_claim_references():
    strategy_response = {
        "positioning": {
            "statement": "Test positioning.",
            "reasoning": "Test reasoning.",
            "claim_ids": ["fake-id"],
        },
        "target_audience": [],
        "messaging": [],
        "channels": [],
        "strategic_priorities": [],
    }

    with patch(
        "app.strategy.strategist.call_claude_json",
        return_value=strategy_response,
    ):
        result = generate_strategy(
            business_name="Example Brand",
            industry="Services",
            location="Surat",
            objective="Increase sales",
            claims=CLAIMS,
            analysis={},
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert result["positioning"]["claim_ids"] == []


# ---------------------------------------------------------------------------
# Campaign generator
# ---------------------------------------------------------------------------

def test_campaign_generator_accepts_evidence_backed_campaign():
    campaign_response = {
        "campaigns": [
            {
                "name": "Online Consultation Campaign",
                "objective": "Generate consultation leads.",
                "audience": "Customers seeking convenient access.",
                "core_idea": "Make online consultation the central conversion path.",
                "message": "Consult online.",
                "offer_or_hook": "Online consultation.",
                "recommended_channels": ["Website"],
                "content_assets": ["Landing page"],
                "reasoning": "The business already offers online consultations.",
                "claim_ids": ["https://example.com/#0"],
            }
        ]
    }

    with patch(
        "app.strategy.campaign_generator.call_claude_json",
        return_value=campaign_response,
    ):
        result = generate_campaigns(
            business_name="Example Brand",
            objective="Increase customer acquisition",
            claims=CLAIMS,
            analysis={},
            strategy={},
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert len(result["campaigns"]) == 1
    assert result["campaigns"][0]["claim_ids"] == [
        "https://example.com/#0"
    ]


def test_campaign_generator_rejects_campaign_without_valid_evidence():
    campaign_response = {
        "campaigns": [
            {
                "name": "Unsupported Campaign",
                "objective": "Grow sales.",
                "audience": "Everyone.",
                "core_idea": "Generic campaign.",
                "message": "Buy now.",
                "offer_or_hook": "Huge offer.",
                "recommended_channels": ["Instagram"],
                "content_assets": ["Reel"],
                "reasoning": "Made up reasoning.",
                "claim_ids": ["does-not-exist"],
            }
        ]
    }

    with patch(
        "app.strategy.campaign_generator.call_claude_json",
        return_value=campaign_response,
    ):
        result = generate_campaigns(
            business_name="Example Brand",
            objective="Increase sales",
            claims=CLAIMS,
            analysis={},
            strategy={},
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert result["campaigns"] == []


# ---------------------------------------------------------------------------
# Action plan
# ---------------------------------------------------------------------------

def test_action_plan_preserves_valid_claim_references():
    action_response = {
        "plan": [
            {
                "phase": "Week 1",
                "objective": "Prepare online consultation campaign.",
                "actions": [
                    {
                        "action": "Create a consultation landing page.",
                        "channel": "Website",
                        "reason": "The business offers online consultations.",
                        "claim_ids": ["https://example.com/#0"],
                    }
                ],
            }
        ],
        "kpis": [
            {
                "metric": "Consultation leads",
                "purpose": "Measure acquisition.",
            }
        ],
        "experiments": [],
    }

    with patch(
        "app.strategy.action_plan.call_claude_json",
        return_value=action_response,
    ):
        result = generate_action_plan(
            objective="Increase customer acquisition",
            claims=CLAIMS,
            strategy={},
            campaigns={},
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert len(result["plan"]) == 1
    assert result["plan"][0]["actions"][0]["claim_ids"] == [
        "https://example.com/#0"
    ]


def test_action_plan_removes_fake_claim_references():
    action_response = {
        "plan": [
            {
                "phase": "Week 1",
                "objective": "Test.",
                "actions": [
                    {
                        "action": "Unsupported action.",
                        "channel": "Instagram",
                        "reason": "Unsupported.",
                        "claim_ids": ["fake-id"],
                    }
                ],
            }
        ],
        "kpis": [],
        "experiments": [],
    }

    with patch(
        "app.strategy.action_plan.call_claude_json",
        return_value=action_response,
    ):
        result = generate_action_plan(
            objective="Increase sales",
            claims=CLAIMS,
            strategy={},
            campaigns={},
            settings=_settings(),
        )

    assert result["status"] == "success"
    assert result["plan"][0]["actions"] == []