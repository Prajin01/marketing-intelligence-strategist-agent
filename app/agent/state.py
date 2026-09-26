"""
The shared state object that flows through every stage of the pipeline.

Design note: this is intentionally a plain, serializable Pydantic model
rather than a framework-specific "agent state" type. Every stage reads from
and writes to this object, and it gets persisted to disk after each stage
(see agent/orchestrator.py, added in Phase 6). That gives us:

  - Observability: you can inspect exactly what the system knew at any point.
  - Resumability: a failed run can be resumed from the last completed stage.
  - Debuggability: state is human-readable JSON, not opaque framework internals.

Fields are filled in incrementally as later phases are implemented (research,
claims, analysis, strategy, validation). Phase 2 only defines input + the
stage log, since that's what main.py needs to run end-to-end right now.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class BusinessInput(BaseModel):
    """What the user provides via the UI/CLI to kick off a run."""

    business_name: str
    industry: str
    location: str
    objective: str
    # Optional: the business's real official website, if known. When
    # provided, it's fetched directly for the business-research category
    # (see auto_research.py) instead of relying on Claude to guess one.
    # Defaults to "" for backward compatibility with state files saved
    # before this field existed — Pydantic fills missing fields with their
    # default on load, so old saved runs still load correctly.
    homepage_url: str = ""

    def slug(self) -> str:
        """Filesystem/ID-safe identifier for this run, e.g. 'toyota-surat'."""
        raw = f"{self.business_name}-{self.location}".lower()
        dashed = "".join(c if c.isalnum() else "-" for c in raw)
        parts = [p for p in dashed.split("-") if p]  # drop empty segments from collapsing
        return "-".join(parts)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageLogEntry(BaseModel):
    """One row of the observability log — 'what did the agent do, and when'."""

    stage: str
    status: StageStatus
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    detail: str = ""


class ClaimType(str, Enum):
    """
    The evidence-discipline vocabulary used throughout the system.
    See Phase 4/5/7 docs — FACT and OBSERVATION may only be created during
    extraction (must reference source text); INFERENCE and RECOMMENDATION
    may only be created during analysis/strategy and must reference the
    claim_ids they were derived from.
    """

    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    RECOMMENDATION = "recommendation"


class ResearchState(BaseModel):
    """
    The single object that represents "everything the agent knows and has
    done" for one run. Populated stage by stage:

      input                -> set at creation (Phase 2, this file)
      research_plan         -> Research Planner (Phase 3)
      raw_documents          -> Research modules (Phase 3)
      knowledge_documents     -> Markdown/frontmatter layer (Phase 4)
      claims                  -> Extraction (Phase 5)
      analysis                 -> Analysis modules (Phase 5)
      strategy                  -> Strategist (Phase 5)
      validation_report          -> Validator (Phase 7)

    Left as `list[dict]` / `dict` placeholders for now rather than fully
    specced sub-models, since locking those schemas down is the job of the
    phases that actually produce them — over-specifying now would mean
    re-deriving these types later anyway once the real shapes are known.
    """

    run_id: str
    input: BusinessInput
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    research_plan: list[dict[str, Any]] = Field(default_factory=list)
    raw_documents: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_documents: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    strategy: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)

    stage_log: list[StageLogEntry] = Field(default_factory=list)

    @classmethod
    def new(cls, business_input: BusinessInput) -> "ResearchState":
        return cls(run_id=business_input.slug(), input=business_input)

    def log_stage(
        self, stage: str, status: StageStatus, detail: str = ""
    ) -> None:
        self.stage_log.append(
            StageLogEntry(stage=stage, status=status, detail=detail)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ResearchState":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))