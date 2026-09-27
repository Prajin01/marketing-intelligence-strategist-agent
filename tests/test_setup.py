"""
Phase 2 smoke tests: confirm config loading and state serialization work.
Later phases add tests per module (research, extraction, analysis, validation).
"""

import os

import pytest

from app.agent.state import BusinessInput, ResearchState, StageStatus


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    # Config requires an API key to load; tests don't need a real one.
    # IMPORTANT: also force LLM_PROVIDER=anthropic explicitly — without this,
    # this test isn't isolated from whatever provider a developer's real
    # .env happens to have configured (e.g. LLM_PROVIDER=groq), which would
    # make load_settings() read and return the developer's REAL Groq key
    # here instead of this fake one — a real key exposure this project hit
    # in practice, in a pytest failure diff. Every field this test depends
    # on must be pinned, not just the one it's directly asserting on.
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


def test_settings_load_with_defaults():
    from app.config import load_settings

    settings = load_settings()
    assert settings.anthropic_api_key == "test-key-not-real"
    assert settings.anthropic_model  # has a default
    assert settings.max_competitors > 0


def test_business_input_slug():
    business = BusinessInput(
        business_name="Toyota Surat",
        industry="Automobile",
        location="Surat, Gujarat",
        objective="Increase customer acquisition",
    )
    slug = business.slug()
    assert slug.startswith("toyota-surat")
    assert " " not in slug and "," not in slug


def test_state_roundtrip(tmp_path):
    business = BusinessInput(
        business_name="Toyota Surat",
        industry="Automobile",
        location="Surat, Gujarat",
        objective="Increase customer acquisition",
    )
    state = ResearchState.new(business)
    state.log_stage("init", StageStatus.SUCCESS, "created")

    path = tmp_path / "state.json"
    state.save(path)

    loaded = ResearchState.load(path)
    assert loaded.run_id == state.run_id
    assert loaded.input.business_name == "Toyota Surat"
    assert loaded.stage_log[0].stage == "init"
    assert loaded.stage_log[0].status == StageStatus.SUCCESS