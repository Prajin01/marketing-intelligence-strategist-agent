"""
Centralized configuration for the Marketing Intelligence Agent.

Every other module reads settings from here rather than calling os.getenv()
directly — this keeps configuration in one place and makes it obvious what
the system depends on at a glance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, on import. Real deployments can also just set real
# environment variables; python-dotenv only fills in what's missing.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # --- Claude API ---
    anthropic_api_key: str
    anthropic_model: str
    anthropic_strategy_model: str

    # --- Research behaviour ---
    max_competitors: int
    request_delay_seconds: float
    request_timeout_seconds: float
    user_agent: str

    # --- Storage ---
    data_dir: Path
    raw_dir: Path
    knowledge_dir: Path
    reports_dir: Path
    claims_path: Path

    # --- Logging ---
    log_level: str


def load_settings() -> Settings:
    """
    Build a validated Settings object from environment variables.

    Fails loudly (raises) if the Anthropic API key is missing, since nothing
    downstream can work without it. Everything else has a sane default so a
    fresh clone can run with minimal setup.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.anthropic.com/"
        )

    data_dir = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

    settings = Settings(
        anthropic_api_key=api_key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        anthropic_strategy_model=os.getenv("ANTHROPIC_STRATEGY_MODEL")
        or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        max_competitors=int(os.getenv("MAX_COMPETITORS", "6")),
        request_delay_seconds=float(os.getenv("REQUEST_DELAY_SECONDS", "1.5")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
        user_agent=os.getenv(
            "USER_AGENT", "MarketingIntelligenceAgent/0.1 (internship project)"
        ),
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        knowledge_dir=data_dir / "knowledge",
        reports_dir=data_dir / "reports",
        claims_path=data_dir / "claims.json",
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )

    # Ensure the runtime directories exist even on a fresh clone (the
    # tracked .gitkeep files preserve empty dirs in git, but a clone into a
    # clean environment should still "just work").
    for d in (settings.raw_dir, settings.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    for sub in ("business", "competitors", "customers", "content", "market"):
        (settings.knowledge_dir / sub).mkdir(parents=True, exist_ok=True)

    return settings


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
