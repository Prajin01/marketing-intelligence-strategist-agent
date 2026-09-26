from app.validation.validator import (
    validate_claim_references,
    validate_strategy,
)


def _claims():
    return [
        {
            "claim_id": "source-a#0",
            "claim": "Brand operates in Surat.",
            "claim_type": "fact",
        },
        {
            "claim_id": "source-a#1",
            "claim": "Brand offers online booking.",
            "claim_type": "fact",
        },
    ]


def test_valid_claim_references_pass():
    analysis = {
        "strengths": [
            {
                "finding": "Strong local presence",
                "claim_ids": ["source-a#0"],
            }
        ]
    }

    strategy = {
        "priorities": [
            {
                "recommendation": "Build local acquisition",
                "claim_ids": ["source-a#0"],
            }
        ]
    }

    campaigns = {
        "campaigns": [
            {
                "name": "Local Acquisition",
                "claim_ids": ["source-a#0"],
            }
        ]
    }

    action_plan = {
        "actions": [
            {
                "action": "Launch local campaign",
                "claim_ids": ["source-a#0"],
            }
        ]
    }

    result = validate_claim_references(
        _claims(),
        analysis,
        strategy,
        campaigns,
        action_plan,
    )

    assert result["valid"] is True
    assert result["invalid_references"] == []


def test_invalid_claim_reference_is_detected():
    result = validate_claim_references(
        _claims(),
        {},
        {
            "priorities": [
                {
                    "recommendation": "Fake recommendation",
                    "claim_ids": ["does-not-exist"],
                }
            ],
        },
        {},
        {},
    )

    assert result["valid"] is False
    assert result["invalid_references"][0]["claim_id"] == "does-not-exist"


def test_validate_strategy_succeeds_with_valid_data():
    result = validate_strategy(
        claims=_claims(),
        analysis={"market": {"finding": "Local demand"}},
        strategy={
            "priorities": [
                {
                    "recommendation": "Increase local acquisition",
                    "claim_ids": ["source-a#0"],
                }
            ]
        },
        campaigns={
            "campaigns": [
                {
                    "name": "Local Growth",
                    "claim_ids": ["source-a#0"],
                }
            ]
        },
        action_plan={
            "actions": [
                {
                    "action": "Run campaign",
                    "claim_ids": ["source-a#0"],
                }
            ]
        },
    )

    assert result["status"] == "success"
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_strategy_fails_with_invalid_claim():
    result = validate_strategy(
        claims=_claims(),
        analysis={},
        strategy={
            "priorities": [
                {
                    "recommendation": "Unsupported recommendation",
                    "claim_ids": ["fake-id"],
                }
            ]
        },
        campaigns={},
        action_plan={},
    )

    assert result["status"] == "failed"
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_empty_claims_produce_warning():
    result = validate_strategy(
        claims=[],
        analysis={},
        strategy={"priorities": []},
        campaigns={},
        action_plan={},
    )

    assert "No extracted claims are available." in result["warnings"]