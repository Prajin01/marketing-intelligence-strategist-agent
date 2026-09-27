"""
Tests for the LLM provider switch introduced to support a free, disclosed
alternative to the paid Claude API (Groq, open-weight models) when no API
budget is available. See README for the disclosure of this substitution.

Covers: Settings backward compatibility, load_settings() branching for
both providers, and that _send() correctly dispatches to the active
provider without ever calling the other one.
"""

import os
from unittest.mock import patch

import pytest

from app.config import Settings, load_settings
from app.llm.claude_client import _send


def _settings(provider: str = "anthropic") -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="test-model",
        anthropic_strategy_model="test-model",
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
        llm_provider=provider,
    )


def test_settings_defaults_to_anthropic_when_llm_provider_omitted():
    """Every existing test file constructs Settings() without llm_provider — must keep working."""
    s = Settings(
        anthropic_api_key="k", anthropic_model="m", anthropic_strategy_model="m",
        max_competitors=6, request_delay_seconds=0.0, request_timeout_seconds=5.0,
        user_agent="A", data_dir=None, raw_dir=None, knowledge_dir=None,
        reports_dir=None, claims_path=None, log_level="INFO",
    )
    assert s.llm_provider == "anthropic"


def test_load_settings_builds_groq_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings()

    assert settings.llm_provider == "groq"
    assert settings.anthropic_api_key == "gsk_test_key"  # field reused, holds the active key
    assert settings.anthropic_model == "llama-3.3-70b-versatile"


def test_load_settings_raises_when_groq_key_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        load_settings()


def test_load_settings_still_defaults_to_anthropic(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = load_settings()

    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key == "sk-ant-test"


def test_send_dispatches_to_anthropic_only():
    with patch("app.llm.claude_client._send_anthropic", return_value="FROM_ANTHROPIC") as mock_a, \
         patch("app.llm.claude_client._send_groq", return_value="FROM_GROQ") as mock_g:
        result = _send(_settings("anthropic"), "sys", "user", "model", 100)

    assert result == "FROM_ANTHROPIC"
    mock_a.assert_called_once()
    mock_g.assert_not_called()


def test_send_dispatches_to_groq_only():
    with patch("app.llm.claude_client._send_anthropic", return_value="FROM_ANTHROPIC") as mock_a, \
         patch("app.llm.claude_client._send_groq", return_value="FROM_GROQ") as mock_g:
        result = _send(_settings("groq"), "sys", "user", "model", 100)

    assert result == "FROM_GROQ"
    mock_g.assert_called_once()
    mock_a.assert_not_called()
